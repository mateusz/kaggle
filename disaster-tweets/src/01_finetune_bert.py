#%%

from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import pandas as pd
import torch
from transformers import DistilBertTokenizer

torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%


#%%


class TwitterDataset(Dataset):
    def __init__(self, path):
        d = pd.read_csv(path).iloc[1:10]
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
        t = tokenizer(d['text'].to_list(), padding='longest').convert_to_tensors('pt')
        self.input_ids = t['input_ids'].to(device)
        self.attention_masks = t['attention_mask'].to(device)
        self.labels = torch.Tensor(d['target'].values).to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'label': self.labels[idx]
        }

ds = TwitterDataset('data/train.csv')
ds[0]

#%%


#TensorDataset(torch.Tensor(d[['text', 'target']].values))

#tokens = tokenizer(d['text'].to_list())
#tokens