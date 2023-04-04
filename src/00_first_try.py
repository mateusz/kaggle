#%%

from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import os

torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%

class Dataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        self.labels = df['label']
        self.data = df.drop(['label'], axis=1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 28x28, 0-255
        return {
            'data': torch.tensor(self.data.loc[idx]).float(),
            'label': torch.tensor(self.labels.loc[idx])
        }

#%%
d = Dataset('data/train.csv')

t,v = torch.utils.data.random_split(d, [0.9, 0.1])
train = DataLoader(t, batch_size=1024, shuffle=True,
    num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=4)
val = DataLoader(v, batch_size=1024, shuffle=True,
    num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=4)

print(len(train), len(val))
#%%

net = nn.Sequential(
    nn.Unflatten(1, (1,28,28)),
    nn.Conv2d(1, 16, 3),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),   

    nn.Conv2d(16, 32, 3),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),   

    nn.Conv2d(32, 64, 3),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),   

    nn.Flatten(),
    nn.Linear(64,10),
)

net(t[0]['data'].unsqueeze(dim=0)).shape

#%%

net = net.to(device)
opt = optim.Adam(net.parameters(), lr=0.001)
lossfn = nn.CrossEntropyLoss()
report_steps = 5

for epoch in range(10):
    net.train()
    running_loss = 0.0
    for i, data in enumerate(train, 0):
        opt.zero_grad()
        pred = net(data['data'].to(device))
        loss = lossfn(pred, data['label'].to(device))

        loss.backward()
        opt.step()

        running_loss += loss.item()
        if i % report_steps == report_steps-1:
            print("train: %.4f" % (running_loss/float(report_steps)))
            running_loss = 0.0
        break

    net.eval()
    with torch.no_grad():
        total_loss = 0.0
        total_correct = 0
        for i, data in enumerate(val, 0):
            labels = data['label'].to(device)
            pred = net(data['data'].to(device))
            loss = lossfn(pred, labels)
            total_loss += loss.item()

            p = torch.argmax(torch.softmax(pred, dim=1), dim=1)
            total_correct += (labels==p).sum()

        print(total_correct, len(val))
        print("val: %.4f, acc=%.2f" % (total_loss/float(len(val)), float(total_correct)/float(len(val))))

#%%

