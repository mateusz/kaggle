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

net = torch.load('models/03bzzz').eval().to(device)
w=10
h=10
fig,ax = plt.subplots(w, h, figsize=(w, h))
i=0
for x in v:
    example = x['data'].reshape((1,28,28))
    r = torch.argmax(net(example.unsqueeze(0)), dim=1)
    if r!=x['label']:
        a = ax[i//h][i%w]
        a.imshow(example.reshape((28,28)).cpu(), interpolation='nearest')
        a.set_xticks([])
        a.set_yticks([])
        a.text(0, 4, int(r), fontsize=8, color='white')
        a.text(20, 4, "(%d)" % x['label'], fontsize=8, color='white')
        i += 1
        if i>=w*h:
            print('break')
            break

plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()

#%%

t = pd.read_csv('data/test.csv')
test = DataLoader(torch.Tensor(t.values).to(device), batch_size=1024)

#%%

preds = torch.Tensor().to(device)
with torch.no_grad():
    for i, data in enumerate(test, 0):
        pred = net(data.reshape((data.shape[0],1,28,28)))
        p = torch.argmax(pred, dim=1)
        preds = torch.cat([preds, p])

result = pd.DataFrame(preds.cpu().numpy().astype(int), columns=['Label']) 
result.index += 1 
result.index.name = 'ImageId'
result.to_csv('data/result.csv')

#%%
plt.imshow(t.iloc[2].values.reshape((28,28)))
plt.show()

