"""This module contains the configuration classes for the data preparation."""

from dataclasses import dataclass
from typing import List


@dataclass
class Paths:
    """Paths to the raw, processed, and model data."""

    raw: str
    processed: str
    tokenized: str


@dataclass
class ParamsDataPreparationFixed:
    """Parameters for data preparation when fixed patch sizes."""

    list_of_logs: List
    log_resistivity: bool
    cols_resistivity: List
    scaler: str
    len_of_patch: int


@dataclass
class ParamsDataPreparationModular:
    """Parameters for data preparation when modular patch sizes."""

    list_of_logs: List
    log_resistivity: bool
    cols_resistivity: List
    scaler: str
    len_of_patch: list


@dataclass
class LogInfo:
    """Information about the experiment."""

    experiment_name: str


# ____________________DATA_PREPARATION_CONFIG____________________
@dataclass
class ConfigExperimentFixed:
    paths: Paths
    params_data_prepare: ParamsDataPreparationFixed
    log_info: LogInfo


@dataclass
class ConfigExperimentModular:
    paths: Paths
    params_data_prepare: ParamsDataPreparationModular
    log_info: LogInfo
