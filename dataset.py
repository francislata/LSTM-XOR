from torch.utils.data import Dataset
import numpy as np
import torch

class XORDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        return self.preprocess_input(self.X[idx]), self.y[idx]

    def __len__(self):
        return len(self.X)

    def preprocess_input(self, X):
        X = torch.from_numpy(np.asarray([int(x) for x in X.split()], dtype=np.float32))

        return X
        