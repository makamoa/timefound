"""Module to prepare data for training"""

import os
import pickle
from joblib import Parallel, delayed
from tqdm import tqdm
from alphatools import mappers, qc_logs, loaders


def get_chunks(path_to_data: str, chunksize: int = 20) -> list:
    """Get chunks of path to files within the list"""
    # get list of las files
    las_files = [f for f in os.listdir(path_to_data) if f.endswith(".las")]
    # get chunks of files
    chunks = [las_files[i : i + chunksize] for i in range(0, len(las_files), chunksize)]
    # get full path to files
    chunks = [[os.path.join(path_to_data, f) for f in chunk] for chunk in chunks]
    return chunks


def prepare_training_well(path_to_file: str, path_to_results: str) -> None:
    """Prepare a single well for training."""
    # load file
    df = loaders.load_las(path_to_file)
    # map column names
    df_mapped = mappers.map_well_logs(df)
    # qc well logs
    df_qc = qc_logs.qc_range(df_mapped)
    # get the file name from path_to_file
    file_name = os.path.basename(path_to_file)
    # remove the extension
    file_name = file_name.split(".")[0]
    # add pickle extension
    file_name = file_name + ".pkl"
    # save the file to pickle format
    df_qc.to_pickle(os.path.join(path_to_results, file_name))


def process_chunk(chunk: list, path_to_results: str) -> None:
    """Process a chunk of files"""
    # use joblib to parallelize the process
    Parallel(n_jobs=-1)(
        delayed(prepare_training_well)(path_to_file, path_to_results) for path_to_file in chunk
    )


def prepare_training_data(
    path_to_data: str,
    path_to_results: str,
) -> None:
    # check if path_to_file is not dir
    if not os.path.isdir(path_to_data):
        print("Path to data is not a directory..")
        return
    # check if path_to_results is not dir
    if not os.path.isdir(path_to_results):
        print("Path to results is not a directory..")
        return
    # check if path_to_file is not empty
    if os.path.getsize(path_to_data) == 0:
        print("Path to file is empty..")
        return

    """Prepare data for training."""
    # get chunks of files
    chunks = get_chunks(path_to_data)
    # loop through chunks
    pbar = tqdm(chunks)
    pbar.set_description("Processing chunks..")
    # loop through chunks
    for ix, chunk in enumerate(pbar):
        # load the first las file
        process_chunk(chunk, path_to_results)


if __name__ == "__main__":
    import pandas as pd

    path_to_data = "../logs"
    path_to_results = "../logs_pickled"

    # get list of wells in the folder
    list_of_wells = os.listdir(path_to_data)

    # prepare training data
    prepare_training_data(path_to_data, path_to_results)

    # read pickled file
    list_of_pickled = [file for file in os.listdir(path_to_results) if file.endswith(".pkl")]
    for file in list_of_pickled:
        print(file)
        with open(os.path.join(path_to_results, file), "rb") as f:
            data = pickle.load(f)
