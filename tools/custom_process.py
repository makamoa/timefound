import os, yaml
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from typing import List, Callable
import numpy as np

COLS_RESISTIVITY = ['RSHA', 'RMED', 'RDEP',  'RMIC', 'RXO']

def process_resistivity_columns(df: pd.DataFrame, resistivity_columns: list) -> pd.DataFrame:
    """
    Process resistivity columns by setting negative values to zero and then computing the logarithm of the remaining values.

    Args:
        df: The input DataFrame.
        resistivity_columns: A list of column names representing resistivity values.

    Returns:
        The DataFrame with the processed resistivity columns.
    """
    for column in resistivity_columns:
        if column in df.columns:
            # Set negative values to zero using .loc to avoid SettingWithCopyWarning
            df.loc[df[column] < 0, column] = 0
            # Compute the logarithm of the remaining values
            df[column] = np.log(df[column] + 1e-10)
    return df

def load_global_stats(yaml_file: str) -> dict:
    """
    Load global statistics from a YAML file.

    Args:
        yaml_file: Path to the YAML file containing global statistics.

    Returns:
        A dictionary with global mean, std, min, and max values for each column.
    """
    with open(yaml_file, 'r') as file:
        global_stats = yaml.safe_load(file)
    return global_stats

def normalize_column(df: pd.DataFrame, column: str, stats: dict) -> pd.DataFrame:
    """
    Normalize a specific column in the DataFrame using the global statistics.

    Args:
        df: The DataFrame containing the data to normalize.
        column: The column to normalize.
        stats: The global statistics for normalization (mean, std, min, max).

    Returns:
        The DataFrame with the normalized column.
    """
    if column == "DEPTH":
        # Special processing for DEPTH: scale min to 0 and max to 1
        min_val = stats[column]['min']
        max_val = stats[column]['max']
        df[column] = (df[column] - min_val) / (max_val - min_val)
    else:
        # Standard normalization using mean and std
        mean = stats[column]['mean']
        std = stats[column]['std']
        df[column] = (df[column] - mean) / std

    return df

def normalize_dataframe(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """
    Normalize all columns in the DataFrame using the global statistics.

    Args:
        df: The input DataFrame to normalize.
        stats: The global statistics for normalization (mean, std, min, max).

    Returns:
        The normalized DataFrame.
    """
    for column in df.columns:
        if column in stats:
            df = normalize_column(df, column, stats)
    return df

def find_longest_non_nan_segment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Finds the longest uninterrupted segment of rows where all columns have non-NaN values.
    
    Args:
        df: The input DataFrame.

    Returns:
        A DataFrame containing only the longest uninterrupted segment with non-NaN values for all columns.
    """
    # Create a mask where True indicates that all columns in a row are non-NaN
    mask = df.notna().all(axis=1)

    # Find the longest uninterrupted segment of True values in the mask
    longest_segment = []
    current_segment = []

    for i, value in enumerate(mask):
        if value:
            current_segment.append(i)
        else:
            if len(current_segment) > len(longest_segment):
                longest_segment = current_segment
            current_segment = []

    # Final check to see if the last segment is the longest
    if len(current_segment) > len(longest_segment):
        longest_segment = current_segment

    # Slice the DataFrame to keep only the longest segment
    if longest_segment:
        df = df.iloc[longest_segment]

    return df

def process_function(df: pd.DataFrame, functions: List[Callable[[pd.DataFrame], pd.DataFrame]]) -> pd.DataFrame:
    """
    Apply a sequence of processing functions to a DataFrame.

    Args:
        df: The input DataFrame to process.
        functions: A list of functions to apply sequentially to the DataFrame.

    Returns:
        A processed DataFrame after applying the functions.
    """
    for func in functions:
        df = func(df)
    
    return df

def process_csv_files(input_folder: str, output_folder: str, functions: List[Callable[[pd.DataFrame], pd.DataFrame]], save_as: str):
    """Process all CSV files in the input folder, apply a sequence of processing functions,
       and save the results as either PyTorch tensors or CSV files in the output folder."""
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get a list of all .csv files
    csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

    # Iterate over all files in the input folder with tqdm progress bar
    for filename in tqdm(csv_files, desc="Processing CSV files", unit="file"):
        file_path = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.{save_as}")

        try:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            # Apply the sequence of processing functions
            df = process_function(df, functions)

            # Save the processed data according to the specified format
            if save_as == "pt":
                tensor = torch.tensor(df.values, dtype=torch.float32)
                torch.save(tensor, output_file)
                print(f"Processed and saved as tensor: {output_file}")
            elif save_as == "csv":
                df.to_csv(output_file, index=False)
                print(f"Processed and saved as CSV: {output_file}")
            else:
                print(f"Unsupported file format: {save_as}. Skipping file.")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process CSV files and save as PyTorch tensors or CSV files.')
    parser.add_argument('input_folder', type=str, help='Path to the folder containing CSV files.')
    parser.add_argument('output_folder', type=str, help='Path to the folder where processed files will be saved.')
    parser.add_argument('yaml_file', type=str, help='Path to the YAML file containing global statistics.')
    parser.add_argument('--save_as', type=str, choices=['pt', 'csv'], default='csv',
                        help='Format to save the processed files: "pt" for PyTorch tensors, "csv" for CSV files.')
    parser.add_argument('--normalize', action='store_true', help='Enable normalization of data based on global statistics.')

    args = parser.parse_args()

    # Load global statistics from YAML file
    global_stats = load_global_stats(args.yaml_file)

    # List of functions to be applied sequentially
    processing_functions = [find_longest_non_nan_segment]
    processing_functions.append(lambda df: process_resistivity_columns(df, COLS_RESISTIVITY))

    # Add normalization function if the normalize flag is set
    if args.normalize:
        processing_functions.append(lambda df: normalize_dataframe(df, global_stats))

    process_csv_files(args.input_folder, args.output_folder, processing_functions, args.save_as)