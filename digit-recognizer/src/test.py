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
import matplotlib.pyplot as plt

torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%

d = dataset.MnistDataset('data/train.csv', device)
#%%
i = d[8]['data'].reshape((1,28,28))

transform = T.Compose(
    [
        #transforms.RandomAffine(degrees=(-5, 5), translate=(0.0, 0.1), scale=(0.75, 1.15)),
        #T.ElasticTransform(alpha=1.0, sigma=1.0),
        T.RandomAffine(degrees=10.0, shear=5.0, scale=(0.8,1.0), translate=(0.05, 0.05), interpolation=T.InterpolationMode.BILINEAR),
        #T.RandomAffine(degrees=10.0, shear=5.0, scale=(0.95,1.05), interpolation=T.InterpolationMode.BILINEAR),
        #transforms.Normalize(mean=256.0/2.0, std=128.0),
        #transforms.RandomAdjustSharpness(1.05, p=0.2),
        #transforms.RandomAdjustSharpness(0.90, p=0.25),
    ]
)

i2 = transform(i)
plt.imshow(i2.squeeze(0).log().cpu(), interpolation='nearest')
plt.show()
