#%%

import src.mnist_dataset as dataset
import src.cbam as cbam
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

ch = 64
net = nn.Sequential(
    nn.Unflatten(1, (1,28,28)),

    nn.Conv2d(1, ch, 5),
    nn.BatchNorm2d(ch),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),   

    nn.Conv2d(ch, ch*2, 5),
    nn.BatchNorm2d(ch*2),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),   

    nn.Conv2d(ch*2, ch*4, 3),
    nn.BatchNorm2d(ch*4),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),   

    nn.Conv2d(ch*4, ch*8, 1),
    nn.BatchNorm2d(ch*8),
    nn.ReLU(inplace=True),

    nn.Flatten(),

    nn.Linear(ch*8,10),
).to(device)
mname = '01-' #k

net(t[0:2]['data']).shape

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

            p = torch.argmax(pred, dim=1)
            total_correct += (labels==p).sum()

    vl = running_loss/float(len(val))
    acc = float(total_correct)/float(len(v))
    print("val: %.8f, acc=%.8f" % (vl, acc))

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
# 01d: try smaller batches, val: 0.00002708, acc=0.98714286
# 01e: back to 01c, try targeting accuracy, val: 0.00183393, acc=0.98904762, not much better, probably overfits, back to 01c
# 01f: 01c, but add extra layer (1-conv), val: 0.00004327, acc=0.98976190, replicable improvement
# 01g: 01f, but one more 1-conv, val: 0.00010783, acc=0.98785714, worse.
# 01h: 01f but double the channels again, val: 0.00001253, acc=0.99119048 - nice!
# 01i: swish. Nope.
# 01j: kernel=5/5/3/1, val: 0.00001260, acc=0.99309524, nice.
# 01k: 01j, but 1-conv doubles the channels, val: 0.00001725, acc=0.99404762, also good.
# 01l: 01k, but double channels, no improvement.
# 01m: normalise data. No improvement.

# so far 01k is the best, and we are probably going into validation overfit.
# might be good to preprocess the data - move around, shear, rotate