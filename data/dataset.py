import itertools, os
import numpy as np
import torch
import json
from torch.utils.data import Dataset, DataLoader

# # Dataset
class CustomWellLogDataset(Dataset):
    def __init__(self, 
                 root_dir, 
                 task_name: str = "imputation", 
                 data_split: str = "train", 
                 few_shot: int = 5, 
                 forecast_horizon: int = 192
                ):
        self.seq_len = 512
        self.root_dir = root_dir
        self.task_name = task_name
        self.data_split = data_split
        self.few_shot = few_shot
        self.forecast_horizon = forecast_horizon
        with open(root_dir + 'patch_names.json', 'r') as file:
            self.mapping = json.load(file)
        self._read_data()

    def __len__(self):
        return len(self.files)

    def _get_borders(self):
        train_mapping = dict(itertools.islice(self.mapping.items(), self.few_shot))
        test_mapping = dict(itertools.islice(self.mapping.items(), self.few_shot, len(self.mapping)))
        return train_mapping, test_mapping

    def _read_data(self):
        train_mapping, test_mapping = self._get_borders()
        if self.data_split == "train":
            self.files = [os.path.join(self.root_dir, f) + '.pt' for f in train_mapping.keys()]
        elif self.data_split == "test":
            self.files = [os.path.join(self.root_dir, f) + '.pt' for f in test_mapping.keys()]
        self.length_timeseries = len(self.files)
        
    def __getitem__(self, idx):
        file_name = self.files[idx]
        input_mask = np.ones(self.seq_len)
        data = torch.load(file_name)
        if self.task_name == 'imputation':
            return data.T[1:], input_mask
        else:
            pass

# # Dataset
class WellLogDataset(Dataset):
    def __init__(self, 
                 root_dir, 
                 task_name: str = "imputation", 
                 data_split: str = "train", 
                 few_shot: int = 5, 
                 forecast_horizon: int = 192
                ):
        self.seq_len = 512
        self.root_dir = root_dir
        self.task_name = task_name
        self.data_split = data_split
        self.few_shot = few_shot
        self.forecast_horizon = forecast_horizon
        with open(root_dir + 'dict_tokens.json', 'r') as file:
            self.mapping = json.load(file)
        self._read_data()

    def __len__(self):
        return len(self.files)

    def _get_borders(self):
        train_mapping = dict(itertools.islice(self.mapping.items(), self.few_shot))
        test_mapping = dict(itertools.islice(self.mapping.items(), self.few_shot, len(self.mapping)))
        return train_mapping, test_mapping

    def _read_data(self):
        train_mapping, test_mapping = self._get_borders()

        if self.data_split == "train":
               self.files = [f for f in train_mapping.values()]
        elif self.data_split == "test":
               self.files = [f for f in test_mapping.values()]
        self.length_timeseries = len(self.files)
        
    def __getitem__(self, idx):
        file_name = self.files[idx]
        input_mask = np.ones(self.seq_len)
        data_dict = torch.load(file_name)
        if self.task_name == 'imputation':
            return data_dict['input'].T, input_mask
        elif self.task_name == 'forecast':
            return  data_dict['input'].T, data_dict['label'].T[:, :self.forecast_horizon], input_mask
        else:
            pass