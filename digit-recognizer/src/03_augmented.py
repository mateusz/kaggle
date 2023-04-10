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
import torchvision.transforms as T

torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%

bs = 1024
d = dataset.MnistDataset('data/train.csv', device)
t,v = torch.utils.data.random_split(d, [0.9, 0.1])
train = DataLoader(t, batch_size=bs, shuffle=True)
val = DataLoader(v, batch_size=bs, shuffle=True)

print(len(train), len(val))
#%%

ch = 64
net = nn.Sequential(
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
mname = '03-'

net(t[0:2]['data'].reshape((2,1,28,28))).shape

#%%

transform = T.Compose(
    [
        T.RandomAffine(degrees=10.0, shear=5.0, scale=(0.8,1.0), translate=(0.05, 0.05), interpolation=T.InterpolationMode.BILINEAR),
    ]
)

#%%

opt = optim.Adam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=3, cooldown=3)
lossfn = nn.CrossEntropyLoss()
report_steps = 5

min_val_loss = 9999999999.0
early_stop_counter = 0
for epoch in range(9999):
    net.train()
    running_loss = 0.0
    for i, data in enumerate(train, 0):
        d = data['data'].reshape((data['data'].shape[0],1,28,28))
        d = transform(d)
        opt.zero_grad()
        pred = net(d)
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
            pred = net(data['data'].reshape((data['data'].shape[0],1,28,28)))
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

    if early_stop_counter>=10:
        break
    
    scheduler.step(vl)

#%%


# apply learnings from https://people.idsia.ch/~juergen/ijcai2011.pdf
# 03a adaptive schedule, plus shears and rotations, val: 0.00014378, acc=0.99476190. Seems like an improvement (probaby the adaptive lr)
# 03- try extra deeply conneted layer, try cbam, nothing helps. Dones't improve things.
# 03b add translations too, and then run this a bunch of times looking for best, val: 0.00085709, acc=0.99571429 (degrees=10.0, shear=5.0, scale=(0.8,1.0), translate=(0.05, 0.05), interpolation=T.InterpolationMode.BILINEAR)
# 03c increase rotations and translations - nope, no improvement (degrees=20.0, shear=5.0, scale=(0.8,1.0), translate=(0.1, 0.1))

# 03b is best.
