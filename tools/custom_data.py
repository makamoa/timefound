import os
import argparse
import pandas as pd
import lasio
from tqdm import tqdm

# Dictionary of equivalent column names
equivalent_columns = {
    "GR": ["GR_ED_DM", "GR_1"],
    "RHOB": ["RHOB_ED_DM", "RHOBR_1"],
    "NPHI": ["TNPL_ED_DM", "NPL_ED_DM", "NPHI_1"],
    "DTC": ["DT_ED_DM", "DT", "DTCO_ED_DM", "DTR_1"],
    "RDEP": ["IL4_ED_DM", "ATRD_ED_DM", "ILD_ED_DM", "ILD", "ILD_1"]
}

def load_las(file_path: str) -> pd.DataFrame:
    """Load a LAS file and return a DataFrame."""
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        # Attempt to read using the normal engine
        las = lasio.read(file_path, engine="normal")
    except Exception as e:
        print(f"Error with normal engine on {file_path}: {e}")
        # Try reading with the fast engine
        las = lasio.read(file_path, engine="fast")
    
    # Return the DataFrame
    return las.df().reset_index(drop=False)

def process_las_files(input_folder: str, output_folder: str, required_columns: list):
    """Process all LAS files in the input folder, check for required columns,
       and save the valid ones as CSV in the output folder."""
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get a list of all .las files
    las_files = [f for f in os.listdir(input_folder) if f.endswith(".las")]

    # Iterate over all files in the input folder with tqdm progress bar
    for filename in tqdm(las_files, desc="Processing LAS files", unit="file"):
        output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.csv")
        
        # Check if the file has already been processed
        if os.path.exists(output_file):
            print(f"Skipping {filename}: Already processed.")
            continue

        file_path = os.path.join(input_folder, filename)
        try:
            # Load the LAS file
            df = load_las(file_path)

            # Check for equivalent columns and create a mapping for required columns
            column_mapping = {}
            for col in required_columns:
                equivalent_names = [col] + equivalent_columns.get(col, [])
                for name in equivalent_names:
                    if name in df.columns:
                        column_mapping[col] = name
                        break

            # Check if all required columns or their equivalents are present
            if set(required_columns).issubset(column_mapping.keys()):
                # Select and rename columns according to the original required columns
                df_selected = df[[column_mapping[col] for col in required_columns]]
                df_selected.columns = required_columns  # Rename columns to original names

                # Save the DataFrame as a CSV file
                df_selected.to_csv(output_file, index=False)
                print(f"Processed and saved: {output_file}")
            else:
                print(f"Skipping {filename}: Required columns or their equivalents not found.")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process LAS files and save valid data to CSV.')
    parser.add_argument('input_folder', type=str, help='Path to the folder containing LAS files.')
    parser.add_argument('output_folder', type=str, help='Path to the folder where CSV files will be saved.')
    parser.add_argument('columns', type=str, nargs='+', help='List of required columns to check in LAS files.')

    args = parser.parse_args()

    process_las_files(args.input_folder, args.output_folder, args.columns)