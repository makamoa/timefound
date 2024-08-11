"""This module contains classes to tokenize raw data for training the model."""

import json
import random
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import hydra
from hydra.core.config_store import ConfigStore
from joblib import Parallel, delayed
from tqdm import tqdm

from alphatools.conf.config import ConfigExperimentFixed, ConfigExperimentModular
from alphatools.processing import DataProcessing

warnings.filterwarnings("ignore")


class TokenizerFixedNextPatch:
    """
    Class to tokenize raw data
    """

    def __init__(self, cfg):
        """
        Init for tokenizer params
        :param cfg: configs of for tokenization
        """
        self.cfg = cfg

        # get data prepare config
        self.cfg_data = self.cfg.params_data_prepare

        # unroll config for token params
        self.len_of_patch = self.cfg_data.len_of_patch

        # unroll paths
        self.paths = cfg.paths

        # unroll path with dataset
        self.experiment_name = self.cfg.log_info.experiment_name

        # init empty path to save
        self.path_to_save = None

        # init empty dict to store tokens
        self.dict_tokens = {}

    def prepare_dataset(self):
        """
        Method to prepare raw dataset for tokenization
        Parameters
        ----------
        cfg

        Returns
        -------
        pandas dataframe with prepared well logging dataset for tokenization
        """
        # init DataPrepare class
        data_prepare = DataProcessing(self.cfg)

        # prepare raw dataset for tokenization
        list_of_files = data_prepare.prepare_dataset()

        return list_of_files

    def create_path_to_save(self) -> None:
        """
        Method to create folder to store tokens
        Returns
        -------

        """
        # create folder to store tokens
        path_to_save = Path(f"{self.paths.tokenized}/{self.experiment_name}").resolve()

        # check if folder exits
        if not path_to_save.exists():
            path_to_save.mkdir(parents=True, exist_ok=True)

        self.path_to_save = path_to_save

    def save_data(self, sample: Dict, file_name: str) -> Path:
        """
        Method to save sample in torch format
        Parameters
        ----------
        sample
        file_name: str name of file to save

        Returns
        -------
        path to save current sample
        """
        # generate save name
        path_to_save_cur = f"{self.path_to_save}/{file_name}.pt"

        # get path to save
        path_to_save_cur = Path(path_to_save_cur).resolve()

        # save token in torch format
        torch.save(sample, path_to_save_cur)

        return path_to_save_cur

    def process_file(self, path_to_file):
        """
        Process a single file and create tokens.
        This function is designed to be called in parallel.
        """
        local_dict_tokens = {}

        # load file
        df_well = pd.read_pickle(path_to_file)

        # get num patches for current well
        num_patches = df_well.shape[0] // self.len_of_patch

        # prepare numpy array for selected logging features
        logs = df_well[self.cfg_data.list_of_logs].to_numpy()

        # get depth column
        depths = df_well["DEPTH"].to_numpy()

        # iterate through patches and create pairs of tokens equal size
        for jx in range(num_patches - 1):
            # get current and next patch
            current_patch = logs[jx * self.len_of_patch : (jx + 1) * self.len_of_patch]
            next_patch = logs[(jx + 1) * self.len_of_patch : (jx + 2) * self.len_of_patch]

            # convert to torch tensor
            current_patch = torch.tensor(current_patch, dtype=torch.float32)
            next_patch = torch.tensor(next_patch, dtype=torch.float32)

            # generate name for saving
            well_id = df_well["WELL"].values[0]
            file_name = f"{well_id}_patch_{jx}"

            # get depth for current and next patches
            current_depth = depths[jx * self.len_of_patch : (jx + 1) * self.len_of_patch]
            next_depth = depths[(jx + 1) * self.len_of_patch : (jx + 2) * self.len_of_patch]

            # save pair of pathes
            path_to_save_sample = self.save_data(
                sample={
                    "input": current_patch,
                    "label": next_patch,
                    "depth_input": current_depth,
                    "depth_label": next_depth,
                },
                file_name=file_name,
            )

            # fill the dict_tokens
            local_dict_tokens[f"{well_id}_patch_{jx}"] = str(path_to_save_sample)
        return local_dict_tokens

    def create_tokens(self) -> None:
        """
        Method to create tokens with equal size of patches
        """

        # prepare raw dataset for tokenization
        list_of_files = self.prepare_dataset()

        # create folder to store tokens
        self.create_path_to_save()

        # init pbar
        pbar = tqdm(list_of_files)
        pbar.set_description("Tokenization is in progress..")

        # Process files in parallel
        results = Parallel(n_jobs=-1)(
            delayed(self.process_file)(path_to_file) for path_to_file in pbar
        )

        # Aggregate results into self.dict_tokens
        for result in results:
            self.dict_tokens.update(result)

        # Save dict with tokens
        with open(f"{self.path_to_save}/dict_tokens.json", "w") as f:
            json.dump(self.dict_tokens, f)

        print(f"Total N of patches: {len(self.dict_tokens)}")


class TokenizerFixedMiddlePatch(TokenizerFixedNextPatch):
    def __init__(self, cfg):
        super().__init__(cfg)
        # get data prepare config
        self.cfg = cfg

        # get data prepare config
        self.cfg_data = self.cfg.params_data_prepare

        # unroll config for token params
        self.len_of_patch = self.cfg_data.len_of_patch

        # unroll paths
        self.paths = cfg.paths

        # unroll path with dataset
        self.experiment_name = self.cfg.log_info.experiment_name

        # init empty path to save
        self.path_to_save = None

        # init empty dict to store tokens
        self.dict_tokens = {}

    def create_tokens(self) -> None:
        """
        Method to create tokens with equal size of patches
        """

        # prepare raw dataset for tokenization
        list_of_files = self.prepare_dataset()

        # create folder to store tokens
        self.create_path_to_save()

        # init pbar
        pbar = tqdm(total=len(list_of_files))
        pbar.set_description("Tokenization is in progress..")

        # init counter of patches
        idx = 0

        # iterate through wells and create tokens
        for path_to_file in list_of_files:
            # load file
            df_well = pd.read_pickle(path_to_file)

            # get num patches for current well
            num_patches = df_well.shape[0] // self.len_of_patch

            # prepare numpy array for selected logging features
            logs = df_well[self.cfg_data.list_of_logs].to_numpy()

            # get depth column
            depths = df_well["DEPTH"].to_numpy()

            # iterate through patches and create pairs of tokens equal size
            for jx in range(num_patches - 2):
                # get end position
                start_first_patch = jx * self.len_of_patch
                end_first_patch = (1 + jx) * self.len_of_patch

                # Ensure the segment doesn't exceed the array bounds
                first_patch = logs[start_first_patch:end_first_patch]

                # get second patch
                end_second_patch = int(end_first_patch + self.len_of_patch)
                second_patch = logs[end_first_patch:end_second_patch]

                # get third patch
                end_third_patch = int(end_second_patch + self.len_of_patch)
                third_patch = logs[end_second_patch:end_third_patch]
                # get first patch as a half of the current patch

                # concat current and last patch
                current_patch = np.concatenate([first_patch, third_patch])
                # convert to torch tensor
                current_patch = torch.tensor(current_patch, dtype=torch.float32)
                middle_patch = torch.tensor(second_patch, dtype=torch.float32)

                # generate name for saving
                well_id = df_well["WELL"].values[0]
                file_name = f"{well_id}_patch_{jx}"

                # get depth for all patches
                first_patch = depths[start_first_patch:end_first_patch]
                middle_depth = depths[end_first_patch:end_second_patch]
                last_patch = depths[end_second_patch:end_third_patch]

                # concat depth
                current_depth = np.concatenate([first_patch, last_patch])

                # save pair of pathes
                path_to_save_sample = self.save_data(
                    sample={
                        "input": current_patch,
                        "label": middle_patch,
                        "depth_input": current_depth,
                        "depth_label": middle_depth,
                    },
                    file_name=file_name,
                )

                # update the counter
                idx += 1

                # fill the dict_tokens
                self.dict_tokens[f"{well_id}_patch_{jx}"] = str(path_to_save_sample)

            # update pbar
            pbar.update(1)

        # save dict with tokens

        with open(f"{self.path_to_save}/dict_tokens.json", "w") as f:
            json.dump(self.dict_tokens, f)

        # close pbar
        pbar.close()

        print(f"Total N of patches: {idx}")


class TokenizerModularNextPatch(TokenizerFixedNextPatch):
    def __init__(self, cfg):
        super().__init__(cfg)
        # get data prepare config
        self.cfg = cfg

        # get data prepare config
        self.cfg_data = self.cfg.params_data_prepare

        # unroll config for token params
        self.len_of_patch = self.cfg_data.len_of_patch

        # unroll paths
        self.paths = cfg.paths

        # unroll path with dataset
        self.experiment_name = self.cfg.log_info.experiment_name

        # init empty path to save
        self.path_to_save = None

        # init empty dict to store tokens
        self.dict_tokens = {}

        # collecting len of patches
        self.lengths_of_patches = []

    @staticmethod
    def get_random_patches(
        df_well: pd.DataFrame,
        lens_of_patches: list,
    ) -> List[pd.DataFrame]:
        """
        method to make random patches from the dataframe
        Parameters
        ----------
        df_well
        lens_of_patches

        Returns
        -------
        list of dataframes with random patches
        """
        # get total len of the dataframe
        total_len = df_well.shape[0]
        start_index = 0

        # init list to store slices
        list_of_slices = []

        while start_index < total_len:
            # get random patch length
            len_of_patch = random.choice(lens_of_patches)

            # get end index
            end_index = start_index + int(len_of_patch)

            # check if end index is not less than total len
            if end_index > total_len:
                break

            # get slice
            slice_cur = df_well.iloc[start_index:end_index]

            # append slice to list
            list_of_slices.append(slice_cur)

            # update start index
            start_index = end_index

        return list_of_slices

    def process_file(self, path_to_file):
        local_dict_tokens = {}

        # load file
        df_well = pd.read_pickle(path_to_file)

        # get patches for current well
        list_of_patches = self.get_random_patches(df_well, self.len_of_patch)

        # print unique lengths
        self.lengths_of_patches += [len(ix) for ix in list_of_patches]

        # iterate through patches and create pairs of tokens of a different size
        for jx, df_patch in enumerate(list_of_patches):
            # check if jx is not the last index
            if jx == len(list_of_patches) - 1:
                break

            # prepare numpy array for selected logging features
            logs_current = df_patch[self.cfg_data.list_of_logs].to_numpy()
            logs_next = list_of_patches[jx + 1][self.cfg_data.list_of_logs].to_numpy()

            # get depth column
            depths_cur = df_patch["DEPTH"].to_numpy()
            depths_next = list_of_patches[jx + 1]["DEPTH"].to_numpy()

            # convert to torch tensor
            current_patch = torch.tensor(logs_current, dtype=torch.float32)
            next_patch = torch.tensor(logs_next, dtype=torch.float32)

            # generate name for saving
            well_id = df_patch["WELL"].values[0]
            file_name = f"{well_id}_patch_{jx}"

            # save pair of pathes
            path_to_save_sample = self.save_data(
                sample={
                    "input": current_patch,
                    "label": next_patch,
                    "depth_input": depths_cur,
                    "depth_label": depths_next,
                },
                file_name=file_name,
            )

            # fill the dict_tokens
            local_dict_tokens[f"{well_id}_patch_{jx}"] = str(path_to_save_sample)

        return local_dict_tokens

    def create_tokens(self) -> None:
        """
        Method to create tokens with equal size of patches
        """

        # prepare raw dataset for tokenization
        list_of_files = self.prepare_dataset()

        # create folder to store tokens
        self.create_path_to_save()

        # init pbar
        pbar = tqdm(list_of_files)
        pbar.set_description("Tokenization is in progress..")

        # Process files in parallel
        results = Parallel(n_jobs=4)(
            delayed(self.process_file)(path_to_file) for path_to_file in pbar
        )

        # Aggregate results into self.dict_tokens
        for result in results:
            self.dict_tokens.update(result)

        # Save dict with tokens
        with open(f"{self.path_to_save}/dict_tokens.json", "w") as f:
            json.dump(self.dict_tokens, f)

        print(f"Total N of patches: {len(self.dict_tokens)}")


class TokenizerModularMiddlePatch(TokenizerModularNextPatch):
    """Class to tokenize dataset with random patches and to predict middle"""

    def __init__(self, cfg):
        super().__init__(cfg)
        # get data prepare config
        self.cfg = cfg

        # get data prepare config
        self.cfg_data = self.cfg.params_data_prepare

        # unroll config for token params
        self.len_of_patch = self.cfg_data.len_of_patch

        # unroll paths
        self.paths = cfg.paths

        # unroll path with dataset
        self.experiment_name = self.cfg.log_info.experiment_name

        # init empty path to save
        self.path_to_save = None

        # init empty dict to store tokens
        self.dict_tokens = {}

    def create_tokens(self) -> None:
        """
        Method to create tokens with equal size of patches
        """

        # prepare raw dataset for tokenization
        list_of_files = self.prepare_dataset()

        # create folder to store tokens
        self.create_path_to_save()

        # init pbar
        pbar = tqdm(total=len(list_of_files))
        pbar.set_description("Tokenization is in progress..")

        # iterate through wells and create tokens
        for path_to_file in list_of_files:
            # load file
            df_well = pd.read_pickle(path_to_file)

            # get patches for current well
            list_of_patches = self.get_random_patches(df_well, self.len_of_patch)

            # iterate through patches and create pairs of tokens of a different size
            for jx, df_patch in enumerate(list_of_patches):
                # check if list of patches is more than 3
                if len(list_of_patches) < 3:
                    break

                # check if jx is not the last index
                if jx == len(list_of_patches) - 2:
                    break

                # prepare numpy array for selected logging features
                logs_previous = df_patch[self.cfg_data.list_of_logs].to_numpy()

                logs_current = list_of_patches[jx + 1][self.cfg_data.list_of_logs].to_numpy()

                logs_next = list_of_patches[jx + 2][self.cfg_data.list_of_logs].to_numpy()

                # get depth column
                depths_previous = df_patch["DEPTH"].to_numpy()
                depths_current = list_of_patches[jx + 1]["DEPTH"].to_numpy()
                depths_next = list_of_patches[jx + 2]["DEPTH"].to_numpy()

                # convert to torch tensor concatenated previous and next patches
                input_patch = np.concatenate([logs_previous, logs_next])
                input_patch = torch.tensor(input_patch, dtype=torch.float32)

                # convert to torch tensor current patch
                current_patch = torch.tensor(logs_current, dtype=torch.float32)

                # generate name for saving
                well_id = df_patch["WELL"].values[0]
                file_name = f"{well_id}_patch_{jx}"

                # save pair of pathes
                path_to_save_sample = self.save_data(
                    sample={
                        "input": input_patch,
                        "label": current_patch,
                        "depth_input": (depths_previous, depths_next),
                        "depth_label": depths_current,
                    },
                    file_name=file_name,
                )

                # fill the dict_tokens
                self.dict_tokens[f"{well_id}_patch_{jx}"] = str(path_to_save_sample)

            # update pbar
            pbar.update(1)

            # save dict with tokens

        with open(f"{self.path_to_save}/dict_tokens.json", "w") as f:
            json.dump(self.dict_tokens, f)

            # close pbar
        pbar.close()

        print(f"Total N of patches: {len(self.dict_tokens)}")


# create config store
cs = ConfigStore.instance()
cs.store(name="cfg_data_tokenize_fixed", node=ConfigExperimentFixed)
cs.store(name="cfg_data_tokenize_fixed_modular", node=ConfigExperimentModular)


@hydra.main(config_path="conf", config_name="config_data_prepare_fixed")
def main_next_fixed(cfg: ConfigExperimentFixed):
    # initialize Tokenizer
    tokenizer = TokenizerFixedNextPatch(cfg)

    # prepare training dataset
    tokenizer.create_tokens()
    print("Tokenization was finished!")


@hydra.main(config_path="conf", config_name="config_data_prepare_fixed")
def main_middle_fixed(cfg: ConfigExperimentFixed):
    # initialize Tokenizer
    tokenizer = TokenizerFixedMiddlePatch(cfg)

    # prepare training dataset
    tokenizer.create_tokens()
    print("Tokenization was finished!")


@hydra.main(config_path="conf", config_name="config_data_prepare_modular")
def main_next_modular(cfg: ConfigExperimentModular):
    # initialize Tokenizer
    tokenizer = TokenizerModularNextPatch(cfg)

    # prepare training dataset
    tokenizer.create_tokens()
    print("Tokenization was finished!")


@hydra.main(config_path="conf", config_name="config_data_prepare_modular")
def main_middle_modular(cfg: ConfigExperimentModular):
    # initialize Tokenizer
    tokenizer = TokenizerModularMiddlePatch(cfg)

    # prepare training dataset
    tokenizer.create_tokens()
    print("Tokenization was finished!")


if __name__ == "__main__":
    main_next_fixed()
