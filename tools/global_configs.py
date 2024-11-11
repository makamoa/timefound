import os
import argparse
import pandas as pd
import yaml
from tqdm import tqdm

def calculate_global_stats(df: pd.DataFrame) -> dict:
    """
    Calculate global statistics (mean, std, min, max) for each column in the DataFrame.

    Args:
        df: The input DataFrame.

    Returns:
        A dictionary containing the global statistics for each column.
    """
    stats = {}
    for column in df.columns:
        non_nan_values = df[column].dropna()
        std_value = non_nan_values.std()

        # Handle case where std might be nan due to insufficient variation
        if pd.isna(std_value):
            std_value = 0.0  # Set std to 0 if it's undefined

        stats[column] = {
            'mean': float(non_nan_values.mean()),  # Convert to native Python float
            'std': float(std_value),               # Convert to native Python float
            'min': float(non_nan_values.min()),    # Convert to native Python float
            'max': float(non_nan_values.max()),    # Convert to native Python float
            'count': int(non_nan_values.count())   # Convert to native Python int
        }
    return stats

def aggregate_global_stats(global_stats: dict, new_stats: dict) -> dict:
    """
    Aggregate global statistics across multiple DataFrames.

    Args:
        global_stats: The current aggregated global statistics.
        new_stats: The global statistics from the new DataFrame to aggregate.

    Returns:
        An updated global statistics dictionary.
    """
    for column, stats in new_stats.items():
        if column not in global_stats:
            global_stats[column] = stats
        else:
            # Aggregate mean and std using a weighted approach
            n_old = global_stats[column].get('count', 0)
            n_new = stats['count']
            n_total = n_old + n_new

            # Update mean
            mean_old = global_stats[column]['mean']
            mean_new = stats['mean']
            global_stats[column]['mean'] = (mean_old * n_old + mean_new * n_new) / n_total

            # Update std (using combined variance formula)
            var_old = global_stats[column]['std'] ** 2
            var_new = stats['std'] ** 2
            global_stats[column]['std'] = (
                ((n_old * var_old + n_new * var_new) / n_total) + 
                (n_old * n_new * (mean_old - mean_new)**2 / n_total**2)
            ) ** 0.5

            # Update min and max
            global_stats[column]['min'] = min(global_stats[column]['min'], stats['min'])
            global_stats[column]['max'] = max(global_stats[column]['max'], stats['max'])

            # Update count
            global_stats[column]['count'] = n_total

    return global_stats

def process_csv_files(input_folder: str) -> dict:
    """Process all CSV files in the input folder and calculate global statistics."""
    global_stats = {}

    # Get a list of all .csv files
    csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

    # Iterate over all files in the input folder with tqdm progress bar
    for filename in tqdm(csv_files, desc="Processing CSV files", unit="file"):
        file_path = os.path.join(input_folder, filename)
        
        try:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            # Calculate global statistics for this DataFrame
            file_stats = calculate_global_stats(df)
            
            # Aggregate the statistics with the global stats
            global_stats = aggregate_global_stats(global_stats, file_stats)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return global_stats

def save_global_stats_to_yaml(global_stats: dict, yaml_file: str):
    """Save the global statistics to a YAML file."""
    with open(yaml_file, 'w') as file:
        yaml.dump(global_stats, file, default_flow_style=False)
    print(f"Global statistics saved to {yaml_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract global statistics from CSV files and save to a YAML file.')
    parser.add_argument('input_folder', type=str, help='Path to the folder containing CSV files.')
    parser.add_argument('output_yaml', type=str, help='Path to the output YAML file to save global statistics.')

    args = parser.parse_args()

    # Calculate global statistics
    global_stats = process_csv_files(args.input_folder)

    # Save the global statistics to a YAML file
    save_global_stats_to_yaml(global_stats, args.output_yaml)

