"""Quality control functions for well logs."""

import json
import warnings
import os
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


# load logs_qc.json
with open(os.path.join(__location__, "./dicts/logs_qc.json"), "r") as f:
    LOGS_QC = json.load(f)


def qc_range(df: pd.DataFrame, dict_range: dict = LOGS_QC) -> pd.DataFrame:
    """Quality control a well log by setting values outside a range to NaN."""
    # get logs from the dict_range
    logs = dict_range.keys()
    # check logs in cur df
    logs_to_qc = [log for log in logs if log in df.columns]
    # iterate over the logs
    for log in logs_to_qc:
        # get the min and max values from the dictionary
        min_value, max_value = dict_range[log]
        # apply the quality control
        f_qc = lambda x: x if min_value <= x <= max_value else np.nan
        # apply the quality control to the log
        df[log] = df[log].apply(f_qc)
    return df
