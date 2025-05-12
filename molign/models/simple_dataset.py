import torch
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        assert not torch.any(X.isnan()), "Features contain Nans"
        assert not torch.any(X.isnan()), "Labels contain Nans"
        self.X = torch.tensor(X, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
