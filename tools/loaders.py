"""This module contains functions for loading data"""

import os
import warnings

import lasio
import pandas as pd

warnings.filterwarnings("ignore")


def load_las(file_path: str) -> pd.DataFrame:
    """Load a LAS file and return a DataFrame."""
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    # Load the file and return a DataFrame
    return lasio.read(file_path).df().reset_index(drop=False)


if __name__ == "__main__":
    from tools.mappers import map_well_logs
    from tools.qc_logs import qc_range
    from tools.visualization import logview

    # read las files in the folder
    path_to_files = "../logs"
    # get list of las files
    las_files = [f for f in os.listdir(path_to_files) if f.endswith(".las")]
    # load the first las file
    df = load_las(os.path.join(path_to_files, las_files[0]))
    print(f"Original columns: {df.columns}")
    # map well logs
    df_mapped = map_well_logs(df)
    print(f"Mapped columns: {df_mapped.columns}")
    # quality control
    df_qc = qc_range(df_mapped)
    print(df_qc["GR"].describe())
    # visualize
    fig = logview(df_qc.loc[:, df.columns[:8]])
    fig.show()
