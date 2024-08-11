"""Functions to map well logs to standard names"""

import json
import warnings
import os
import pandas as pd

warnings.filterwarnings("ignore")

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

# load logs_mappers.json
with open(os.path.join(__location__, "./dicts/logs_mappers.json"), "r") as f:
    LOGS_MAPPER = json.load(f)


def prepare_dict_for_mapping(dict_mapping: dict) -> dict:
    """Prepare the dictionary for mapping"""
    reverse_mapping = {}
    for standard_name, aliases in dict_mapping.items():
        for alias in aliases:
            reverse_mapping[alias] = standard_name
    return reverse_mapping


def map_well_logs(df: pd.DataFrame) -> pd.DataFrame:
    """Map well logging columns"""
    # get the gamma ray dict
    gamma_ray_dict = prepare_dict_for_mapping(LOGS_MAPPER)

    # Initialize a new DataFrame to store the results
    new_df = pd.DataFrame(index=df.index)

    # Track mapped columns to handle duplicates
    mapped_columns = {}

    for col in df.columns:
        standard_col_name = gamma_ray_dict.get(col, col)
        if standard_col_name in mapped_columns:
            # If the column has already been added, apply merge strategy
            new_df[standard_col_name] = new_df[[standard_col_name, col]].mean(axis=1)
        else:
            # Add the column to the new DataFrame
            new_df[standard_col_name] = df[col]
            mapped_columns[standard_col_name] = True
    return new_df
