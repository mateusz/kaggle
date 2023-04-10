import torch
from torch.utils.data import Dataset
import pandas as pd

class MnistDataset(Dataset):
    def __init__(self, path, device):
        df = pd.read_csv(path)
        #data = df.drop(['label'], axis=1)
        #data = (data/255.0)*2.0-1.0
        self.labels = torch.tensor(df['label'].values).to(device)
        self.data = torch.tensor(df.drop(['label'], axis=1).values).float().to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 28x28, 0-255
        return {
            'data': self.data[idx],
            'label': self.labels[idx],
        }