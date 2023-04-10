#%%

import src.mnist_dataset as dataset
import src.cbam as cbam

from torch.utils.data import DataLoader
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
    cbam.ConvBlock(1, 32),
    cbam.ConvBlock(32, 64),
    cbam.ConvBlock(64, 128),
    nn.Flatten(),
    nn.Linear(128,10),
).to(device)

mname = '02b'
net(t[0]['data'].unsqueeze(dim=0).to(device)).shape

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
            print("train: %.8f" % (running_loss/float(report_steps)))
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
    print("val: %.8f, acc=%.8f" % (vl, float(total_correct)/float(len(v))))

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
# 02a: val: 0.0010, acc=0.983810 (without batchnorm)
# 02b: add batchnorm, val: 0.00008603, acc=0.98714286 (comparable with 01, abandon)

