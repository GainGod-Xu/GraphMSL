import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import pickle

class SurfaceDataset(Dataset):
    def __init__(self, files, config):
        self.files = files
        self.surface_path = config.surface_path

    def __len__(self):
        return len(self.files)

    def get_sample_name(self, idx):
        return self.files[idx]

    def __getitem__(self, index):
        file_path = os.path.join(self.surface_path, self.files[index])

        with open(file_path, 'rb') as pickle_file:
            loaded_data = pickle.load(pickle_file)
        tensor_data = torch.Tensor(loaded_data)
        return tensor_data

    def collate_fn(self, batch):
        # Separate preprocessed NMR data and raw NMR data
        data = [item for item in batch]

        # Stack the preprocessed NMR data along a new dimension (batch dimension)
        tensor_data = torch.stack(data, dim=0)

        return tensor_data



