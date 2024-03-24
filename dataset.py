import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class ClosePriceDataset(Dataset):
    def __init__(self, file_path, sequence_length=30, number_of_training=25):
        self.data = pd.read_csv(file_path)['Close Price'].values
        self.sequence_length = sequence_length
        self.number_of_training = number_of_training
        
    def __len__(self):
        return len(self.data) - self.sequence_length + 1

    def __getitem__(self, index):
        random_start = np.random.randint(0, len(self.data) - self.sequence_length)
        sample = self.data[random_start:random_start + self.sequence_length]

        train_sample = sample[:int(self.number_of_training)]
        test_sample = sample[int(self.number_of_training):]

        return torch.tensor(train_sample, dtype=torch.float32), torch.tensor(test_sample, dtype=torch.float32)
    