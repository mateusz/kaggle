#%%

from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertForSequenceClassification 
import torch.nn.functional as F
from gdn.pytorch_gdn import GDN
import cv2
import torchvision.transforms as transforms
import math
import matplotlib.pyplot as plt
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
class ImgSet(Dataset):
    def __init__(self, path, tile=256, wtiles=-1, htiles=-1):
        super().__init__()

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        w,h=image.shape[0:2]
        padw = math.ceil(w/tile)*tile-w
        padh = math.ceil(h/tile)*tile-h
        image = cv2.copyMakeBorder(image, 0, padw, 0, padh, cv2.BORDER_CONSTANT, value=[0,0,0])

        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img = transform(image)
        # tile image
        img = img.unfold(1, tile, tile).unfold(2, tile, tile)
        self.htiles = img.shape[1]
        self.wtiles = img.shape[2]

        if wtiles>0:
            self.wtiles = wtiles
        if htiles>0:
            self.htiles = htiles
        self.img = img[:,:htiles,:wtiles].flatten(1,2).permute(1,0,2,3).to(device)

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        return self.img[idx]

tile=256
wtiles=4
htiles=4
batch_size=4
iset = ImgSet('data/STScI-01GA76Q01D09HFEV174SVMQDMV.png', tile=tile, wtiles=wtiles, htiles=htiles)
print(iset.img.shape, iset.wtiles, iset.htiles)

train = DataLoader(iset, batch_size=8, shuffle=True)

#%%
ch = 8
latent = 256
net = nn.Sequential(
        nn.Conv2d(3, ch, 5),
        nn.MaxPool2d(2),
        nn.ReLU(inplace=True),
        
        nn.Conv2d(ch, ch*2, 3),
        nn.MaxPool2d(2),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(ch*2),
        # 32,62,62

        nn.Conv2d(ch*2, ch*4, 3),
        nn.MaxPool2d(2),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(ch*4),
        # 64,30,30

        nn.Conv2d(ch*4, ch*8, 3),
        nn.MaxPool2d(2),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(ch*8),
        #128,14,14

        nn.Conv2d(ch*8, ch*16, 3),
        nn.MaxPool2d(2),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(ch*16),
        # 256,6,6

        nn.Conv2d(ch*16, ch*32, 3),
        nn.MaxPool2d(2),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(ch*32),
        # 512,2,2

        nn.Flatten(),
        nn.Linear(ch*32 * 4, latent),
        nn.ReLU(),
        nn.Linear(latent, ch*32 * 4),
        nn.ReLU(),
        nn.Unflatten(1, (ch*32,2,2)),

        nn.ConvTranspose2d(ch*32, ch*16, 4, stride=2),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(ch*16),

        nn.ConvTranspose2d(ch*16, ch*8, 4, stride=2),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(ch*8),

        nn.ConvTranspose2d(ch*8, ch*4, 4, stride=2),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(ch*4),

        nn.ConvTranspose2d(ch*4, ch*2, 4, stride=2),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(ch*2),

        nn.ConvTranspose2d(ch*2, ch, 4, stride=2),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(ch),

        nn.ConvTranspose2d(ch, 3, 6, stride=2),
)

net = net.to(device)
print(net(iset[0].unsqueeze(0)).shape)
print(sum([l.numel() for l in net.parameters()]))
#%%

opt = torch.optim.Adam(net.parameters(), lr=0.0001)

perc = SSIM(data_range=1, size_average=True, channel=3)
abs = nn.L1Loss()

perc_w = 1.0
abs_w = 1.0

net = net.to(device)
net.train()
for epoch in range(999999):
    running_loss = 0.0
    running_perc = 0.0
    running_abs = 0.0

    for i,t in enumerate(train,0):
        opt.zero_grad()

        o = net(t)

        perc_loss = 1.0 - ssim(o, t)
        abs_loss = abs(o, t)

        loss = perc_w*perc_loss + abs_w*abs_loss
        loss.backward()
        opt.step()

        running_loss += loss.item()
        running_abs += abs_loss.item()
        running_perc += perc_loss.item()

    print("[%d] l=%.4f, abs=%.4f, perc=%.4f" % (epoch, running_loss, running_abs, running_perc))
    running_loss = 0.0
    running_abs = 0.0
    running_perc = 0.0

#%%

net.eval()

def show_img(iset, data):
    h = iset.htiles*tile
    w = iset.wtiles*tile
    # Reshape into m x n
    merged = data.view(iset.htiles,iset.wtiles,3,tile,tile).permute(2,0,3,1,4)
    # Merge tiles
    merged = merged.reshape(3, h, w)
    # Prep for pyplot
    merged = merged.detach().cpu().permute(1,2,0)

    plt.imshow(merged)
    plt.show()

show_img(iset, iset[:])
show_img(iset, net(iset[:]))