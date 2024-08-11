import json
import os
import warnings
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import pandas as pd
from hydra.core.config_store import ConfigStore
from joblib import Parallel, delayed
from tqdm import tqdm

from alphatools.conf.config import ConfigExperimentFixed

warnings.filterwarnings("ignore")
np.random.seed(42)


class DataTransformer:
    """Class to transform data before training"""

    def __init__(self, cfg: ConfigExperimentFixed):
        """Init method"""
        # get data prepare config
        self.cfg = cfg
        self.paths = self.cfg.paths
        self.params_data_prepare = self.cfg.params_data_prepare

    def scale_data(
        self, df_cur: pd.DataFrame, dict_global_mean: Dict, dict_global_std: Dict
    ) -> pd.DataFrame:
        """Method to scale data"""


        # scale data
        for log in self.params_data_prepare.list_of_logs:
            # print(dict_global_mean[log], dict_global_std[log],df_cur[log].shape )
            
            if not log in df_cur.columns.values:
                print(f'Skipping {log}')
                continue

            mean, std = dict_global_mean[log], dict_global_std[log]
            if isinstance(mean, (np.ndarray, list)):
                mean = mean[0]
                std = std[0]

            df_cur[log] = (df_cur[log] - mean) / std
        return df_cur

    def log_resistivity(self, df_cur: pd.DataFrame) -> pd.DataFrame:
        """Method to log resistivity values"""
        # get resistivity columns
        cols_resistivity = self.params_data_prepare["cols_resistivity"]
        # log resistivity
        for col in cols_resistivity:
            # check if column exists
            if col not in df_cur.columns:
                continue
            # log resistivity
            df_cur[col] = df_cur[col].apply(lambda x: np.log(x) if x > 0 else x)
        return df_cur

    def preprocess_and_select_longest_segment(
        self, df_well: pd.DataFrame, features: list, col_depth: str = "DEPTH"
    ) -> pd.DataFrame:
        """Method to preprocess and select the longest segment of the well data"""
        # Step 1: Drop NaN values across all columns to clean the data
        df_cleaned = df_well.dropna(subset=features)
        if df_cleaned.empty:
            return pd.DataFrame()
        # Step 2: Calculate the depth step between neighboring points
        df_cleaned["depth_step"] = df_cleaned[col_depth].diff().fillna(0)
        # Step 3: Calculate cumulative curve for depth delta
        df_cleaned["cumulative_depth"] = df_cleaned["depth_step"].cumsum()
        # Step 4: Identify outliers in the cumulative curve to find segment
        # boundaries assuming outliers are depth steps significantly larger
        # than the median step
        mode_step = df_cleaned["depth_step"].mode()
        # Mark rows that start a new segment based on outlier threshold
        # if mode_step.shape[0] > 0:
        df_cleaned["is_segment_start"] = df_cleaned["depth_step"] > mode_step.iloc[0]
   
            # df_cleaned["is_segment_start"] = 0
        # Identify segment IDs to facilitate later analysis
        df_cleaned["segment_id"] = df_cleaned["is_segment_start"].cumsum()
        # Step 5: Select the longest segment based on the identified boundaries
        segment_lengths = df_cleaned.groupby("segment_id")["depth_step"].sum()
        longest_segment_id = segment_lengths.idxmax()
        # Extract the longest segment for analysis
        longest_segment_df = df_cleaned[df_cleaned["segment_id"] == longest_segment_id]
        # Drop the columns used for segment identification
        segment_out = longest_segment_df.drop(
            columns=[
                "depth_step",
                "cumulative_depth",
                "is_segment_start",
                "segment_id",
            ]
        )

        return segment_out


class DataProcessing(DataTransformer):
    """Class to prepare data for training"""

    def __init__(self, cfg: ConfigExperimentFixed):
        """Init method"""
        super().__init__(cfg=cfg)
        # get experiment name
        self.experiment_name = self.cfg.log_info.experiment_name
        # create path to save
        self.path_to_save = self.create_path_to_save()

    def create_path_to_save(self) -> None:
        """Method to create path to save processed data"""
        # create folder to store tokens
        path_to_save = Path(f"{self.paths.processed}/{self.experiment_name}").resolve()
        path_to_save.mkdir(parents=True, exist_ok=True)
        return path_to_save

    def process_well_names(self, list_of_files: List) -> Dict:
        """Process well names"""
        # create dict with well names and their indexes
        dict_well_idx = {well.replace(".pkl", ""): idx for idx, well in enumerate(list_of_files)}
        # save dict with well names and their indexes to json file
        with open(f"{self.paths.processed}/well_names.json", "w") as f:
            json.dump(dict_well_idx, f)
        return dict_well_idx

    def process_chunk(
        self,
        chunk: List,
        dict_well_names: Dict,
        global_mean: Dict,
        global_std: Dict,
        path_to_raw: str,
    ) -> str:
        """Process a chunk of files"""
        paths_to_wells = []
        for file in chunk:
            # load pickle file
            df_cur = pd.read_pickle(f"{path_to_raw}/{file}")
            if not all(col in df_cur.columns.values for col in self.params_data_prepare.list_of_logs):
                continue
            # add WELL column
            df_cur["WELL"] = str(file.replace(".pkl", ""))
            # log resistivity
            if self.params_data_prepare.log_resistivity:
                df_mapped = self.log_resistivity(df_cur=df_cur)
            # substitute inf values with nan
            df_mapped = df_mapped.replace([np.inf, -np.inf], np.nan)
            # dropna from the dataframe
            features_log = self.params_data_prepare.list_of_logs
            if all(col in df_mapped.columns.values for col in features_log):
                df_mapped = df_mapped.dropna(subset=features_log, how="any")
            else:
                features_intersection = [feat for feat in features_log if feat in df_mapped.columns.values]
                df_mapped = df_mapped.dropna(subset=features_intersection, how="any")

            # scale data
            df_scaled = self.scale_data(
                df_cur=df_mapped,
                dict_global_mean=global_mean,
                dict_global_std=global_std,
            ).reset_index(drop=True)
            # keep only the columns of interest
            features_intersection = [feat for feat in features_log if feat in df_scaled.columns.values]
            # df_scaled = df_scaled[["WELL", "DEPTH"] + features_log]
            df_scaled = df_scaled[["WELL", "DEPTH"] + features_intersection]
            # encode WELL names
            df_scaled["WELL"] = df_scaled["WELL"].astype(str).map(dict_well_names)
            # preprocess and select longest segment
            df_longest_segment = self.preprocess_and_select_longest_segment(
                df_well=df_scaled, features=features_log
            )
            if df_longest_segment.empty:
                continue
            # get path to save
            path_to_save_cur = f"{self.path_to_save}/{df_longest_segment['WELL'].iloc[0]}.pkl"
            # save processed data
            df_longest_segment.to_pickle(path_to_save_cur)
            paths_to_wells.append(path_to_save_cur)
        return paths_to_wells

    def prepare_dataset(self):
        """Method to prepare dataset for training"""

        # get path to raw data
        path_to_raw = Path(self.paths.raw).resolve()
        # create path to save
        self.create_path_to_save()

        # get list of files
        list_files = [file for file in os.listdir(path_to_raw) if file.endswith(".pkl")]

        # encode well names
        dict_well_names = self.process_well_names(list_of_files=list_files)

        # split list_files into chunks
        list_chunks = np.array_split(list_files, 20)

        # Initialize dictionaries to store count, mean, and M2 for each log
        global_mean = {}
        global_M2 = {}
        global_count = {}

        # Initialize all counters to zero
        for log in self.params_data_prepare.list_of_logs:
            global_mean[log] = 0
            global_M2[log] = 0
            global_count[log] = 0

        # Iterate through each chunk to update counts, means, and M2
        for chunk in tqdm(list_chunks, desc="Calculating global mean and std", leave=False):
            for file in chunk:
                # load pickle file
                df_cur = pd.read_pickle(f"{path_to_raw}/{file}")
                # check that all columns are there
                if not all(col in df_cur.columns.values for col in self.params_data_prepare.list_of_logs):
                        print(f'Skipping: {path_to_raw}/{file}')
                        continue

                for log in self.params_data_prepare.list_of_logs:
                    if self.params_data_prepare.log_resistivity:
                        df_cur = self.log_resistivity(df_cur=df_cur)

                    
                    
                    values = df_cur[log].dropna().values
                    for value in values:
                        
                        # Update count
                        global_count[log] += 1
                        # Calculate delta between value and current mean
                        delta = value - global_mean[log]
                        # Update mean
                        global_mean[log] += delta / global_count[log]
                        # Update delta2 after mean has been updated
                        delta2 = value - global_mean[log]
                        # Update M2
                        global_M2[log] += delta * delta2

        # Calculate the global standard deviation from M2 and count
        global_std = {log: np.sqrt(global_M2[log] / global_count[log]) for log in global_M2}

        print(f"Global means: {global_mean}")
        print(f"Global std: {global_std}")

        # init pbar
        pbar = tqdm(list_chunks, desc="Processing data", leave=False)
        # process chunks in parallel
        results = Parallel(n_jobs=-1, backend="multiprocessing")(
            delayed(self.process_chunk)(
                chunk,
                dict_well_names,
                global_mean,
                global_std,
                str(path_to_raw),
            )
            for chunk in pbar
        )
        results = np.hstack(results)

        print("Data preparation is done")
        return results


# create config store
cs = ConfigStore.instance()
cs.store(name="cfg_data_preparation", node=ConfigExperimentFixed)


@hydra.main(config_path="conf", config_name="config_data_prepare_fixed")
def main(cfg: ConfigExperimentFixed):
    # create data prepare object
    data_prepare = DataProcessing(cfg=cfg)
    # prepare dataset
    paths = data_prepare.prepare_dataset()
    # open pkl file
    df_test = pd.read_pickle(paths[0])
    print(df_test.head())


if __name__ == "__main__":
    main()
