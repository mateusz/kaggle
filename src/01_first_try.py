#%%

import src.mnist_dataset as dataset
from torch.utils.data import DataLoader, Dataset
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

d = dataset.MnistDataset('data/train.csv', device)
t,v = torch.utils.data.random_split(d, [0.9, 0.1])
train = DataLoader(t, batch_size=1024, shuffle=True)
val = DataLoader(v, batch_size=1024, shuffle=True)

print(len(train), len(val))
#%%

net = nn.Sequential(
    nn.Unflatten(1, (1,28,28)),
    nn.Conv2d(1, 32, 3),
    nn.BatchNorm2d(32),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),   

    nn.Conv2d(32, 64, 3),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),   

    nn.Conv2d(64, 128, 3),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),   

    nn.Flatten(),

    nn.Linear(128,10),
).to(device)
mname = '01c'

net(t[0]['data'].unsqueeze(dim=0)).shape

#%%

opt = optim.Adam(net.parameters(), lr=0.001)
lossfn = nn.CrossEntropyLoss()
report_steps = 5

min_val_loss = 9999999999.0
early_stop_counter = 0
for epoch in range(9999):
    net.train()
    running_loss = 0.0
    for i, data in enumerate(train, 0):
        opt.zero_grad()
        pred = net(data['data'])
        loss = lossfn(pred, data['label'])

        loss.backward()
        opt.step()

        running_loss += loss.item()
        if i % report_steps == report_steps-1:
            print("train: %.4f" % (running_loss/float(report_steps)))
            running_loss = 0.0

    net.eval()
    with torch.no_grad():
        total_loss = 0.0
        total_correct = 0
        for i, data in enumerate(val, 0):
            labels = data['label']
            pred = net(data['data'])
            loss = lossfn(pred, labels)
            total_loss += loss.item()

            p = torch.argmax(torch.softmax(pred, dim=1), dim=1)
            total_correct += (labels==p).sum()

    vl = running_loss/float(len(val))
    print("val: %.6f, acc=%.8f" % (vl, float(total_correct)/float(len(v))))

    if vl<min_val_loss:
        min_val_loss = vl
        early_stop_counter = 0
        print('saving and exporting model...')
        torch.save(net, 'models/%s' % (mname) )
    else:
        early_stop_counter += 1

    if early_stop_counter>=3:
        break
#%%


# Notes
# 01 : 3 layers + linear, channels=16, val: 0.0077, acc=0.97
# 01a: add dense 64 with ReLU, val: 0.0134, acc=0.973333
# 01b: back to 01, double the channels to 32, val: 0.0005, acc=0.981190
# 01c: 01b, but add batch norm2d, val: 0.0000, acc=0.988095

