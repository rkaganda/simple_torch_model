import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class SimpleDataSet(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file, index_col=0)
        feature_columns = [c for c in self.df.columns if c.startswith('x_')]
        label_columns = [c for c in self.df.columns if c.startswith('y_')]

        self.input_count = len(feature_columns)
        self.label_count = len(label_columns)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # inputs from the front columns
        # labels from the end columns
        inputs = self.df.iloc[idx, :self.input_count]
        labels = self.df.iloc[idx, -self.label_count:]

        return np.array(inputs), np.array(labels)

