#%%

import src.cbam
import src.mnist_dataset as dataset
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%

d = dataset.MnistDataset('data/train.csv', device)
t,v = torch.utils.data.random_split(d, [0.9, 0.1])

print(len(t), len(v))
#%%

net = torch.load('models/01k').eval().to(device)
w=10
h=10
fig,ax = plt.subplots(w, h, figsize=(w, h))
i=0
for x in v:
    example = x['data']
    r = torch.argmax(net(example.unsqueeze(0)), dim=1)
    if r!=x['label']:
        a = ax[i//h][i%w]
        a.imshow(example.reshape((28,28)).cpu(), interpolation='nearest')
        a.set_xticks([])
        a.set_yticks([])
        a.text(0, 4, int(r), fontsize=8, color='white')
        i += 1
        if i>=w*h:
            print('break')
            break

plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()
