import os
import argparse
import pandas as pd
import torch
import json
from tqdm import tqdm
import numpy as np

def create_patches(df: pd.DataFrame, patch_len: int = 512) -> torch.Tensor:
    """
    Divide the DataFrame into patches and pad with zeros if necessary.

    Args:
        df: The input DataFrame.
        patch_len: The length of each patch. Default is 512.

    Returns:
        A list of PyTorch tensors, each representing a patch.
    """
    num_rows, num_columns = df.shape
    patches = []

    for i in range(0, num_rows, patch_len):
        patch = df.iloc[i:i + patch_len].values
        
        # If the patch is smaller than the patch length, pad with zeros
        if patch.shape[0] < patch_len:
            padding = np.zeros((patch_len - patch.shape[0], num_columns))
            patch = np.vstack((patch, padding))
        
        # Convert the patch to a PyTorch tensor
        patch_tensor = torch.tensor(patch, dtype=torch.float32)
        patches.append(patch_tensor)

    return patches

def process_csv_files(input_folder: str, output_folder: str, patch_len: int, json_file: str, verbose: bool = False):
    """Process all CSV files in the input folder, divide them into patches,
       save the patches as PyTorch tensors, and store patch names in a JSON file."""
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Dictionary to store patch name mappings
    patch_names_dict = {}

    # Get a list of all .csv files
    csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

    # Iterate over all files in the input folder with tqdm progress bar
    for filename in tqdm(csv_files, desc="Processing CSV files", unit="file"):
        file_path = os.path.join(input_folder, filename)

        try:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            # Create patches from the DataFrame
            patches = create_patches(df, patch_len)

            # Save each patch as a PyTorch tensor file
            base_filename = os.path.splitext(filename)[0]
            for idx, patch in enumerate(patches):
                patch_filename = f"{base_filename}_patch_{idx+1}.pt"
                patch_filepath = os.path.join(output_folder, patch_filename)
                torch.save(patch, patch_filepath)
                patch_name_without_ext = os.path.splitext(patch_filename)[0]
                patch_names_dict[patch_name_without_ext] = base_filename
                if verbose:
                    print(f"Saved: {patch_filepath}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Save the patch names dictionary as a JSON file
    json_filepath = os.path.join(output_folder, json_file)
    with open(json_filepath, 'w') as json_out:
        json.dump(patch_names_dict, json_out, indent=4)
    print(f"Patch names mapping saved to {json_filepath}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tokenize CSV files into patches and save as PyTorch tensors.')
    parser.add_argument('input_folder', type=str, help='Path to the folder containing processed CSV files.')
    parser.add_argument('output_folder', type=str, help='Path to the folder where patch tensors will be saved.')
    parser.add_argument('--patch_len', type=int, default=512, help='Length of each patch. Default is 512.')
    parser.add_argument('--json_file', type=str, default='patch_names.json', help='Name of the JSON file to store patch names mapping. Default is patch_names.json.')

    args = parser.parse_args()

    process_csv_files(args.input_folder, args.output_folder, args.patch_len, args.json_file)
