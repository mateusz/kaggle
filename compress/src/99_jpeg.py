#%%

from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from gdn.pytorch_gdn import GDN
import cv2
import torchvision.transforms as transforms
import math
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os
import io
import PIL
from enum import Enum
from pytorch_msssim import MS_SSIM as MS_SSIM
from skimage.metrics import peak_signal_noise_ratio as psnr

PIL.Image.MAX_IMAGE_PIXELS = None

#%%

orig = PIL.Image.open('data/STScI-01GA76Q01D09HFEV174SVMQDMV.png')
orig.crop((512,0,1024,512)).save('out/orig-tile.png')
cropped = orig.crop((0,0,4096,4096))
cropped.save('out/test-orig.png')
cropped.save('out/test-jpeg95.jpg', quality=95)
cropped.save('out/test-jpeg70.jpg', quality=70)
cropped.save('out/test-jpeg50.jpg', quality=50)
cropped.save('out/test-jpeg20.jpg', quality=20)
orig = None
cropped = None

#%%
class Format(Enum):
    PIL = 1

class ImData():
    def __init__(self, d, format, filesize=-1):
        if format==Format.PIL:
            data = PIL.Image.open(d)
            t = transforms.ToTensor()
            self.tensor = t(data)
            if filesize>0:
                self.size = filesize
            else:
                self.size = os.path.getsize(d)
            self.w, self.h = data.size

    def numpy(self):
        return self.tensor.permute(1,2,0).numpy()

def get_metrics(orig:ImData, comp:ImData):
    msssim = MS_SSIM(data_range=1, size_average=True, channel=3)

    m_psnr = psnr(orig.numpy(), comp.numpy())
    m_msssim = msssim(orig.tensor.unsqueeze(0), comp.tensor.unsqueeze(0)).numpy()
    rate = comp.size/orig.size
    bpp = (comp.size*8)/(comp.w*comp.h)

    return m_psnr, m_msssim, rate, bpp

orig = ImData('out/test-orig.png', Format.PIL)
#jpeg95 = ImData('out/test-jpeg95.jpg', Format.PIL)
#jpeg70 = ImData('out/test-jpeg70.jpg', Format.PIL)
#jpeg50 = ImData('out/test-jpeg50.jpg', Format.PIL)
#jpeg20 = ImData('out/test-jpeg20.jpg', Format.PIL)
#zero2a = ImData('out/test-02a.png', Format.PIL, filesize=112832)
zero3a = ImData('out/test-03a.png', Format.PIL, filesize=91392)
zero3b = ImData('out/test-03b.png', Format.PIL, filesize=69376)

#print("orig: psnr=%.2f, ms-ssim=%.6f, rate=%.4f, bpp=%.2f" % get_metrics(orig, orig))
#print("jpeg95: psnr=%.2f, ms-ssim=%.6f, rate=%.4f, bpp=%.2f" % get_metrics(orig, jpeg95))
#print("jpeg70: psnr=%.2f, ms-ssim=%.6f, rate=%.4f, bpp=%.2f" % get_metrics(orig, jpeg70))
#print("jpeg50: psnr=%.2f, ms-ssim=%.6f, rate=%.4f, bpp=%.2f" % get_metrics(orig, jpeg50))
#print("jpeg20: psnr=%.2f, ms-ssim=%.6f, rate=%.4f, bpp=%.2f" % get_metrics(orig, jpeg20))
#print("02a: psnr=%.2f, ms-ssim=%.6f, rate=%.4f, bpp=%.2f" % get_metrics(orig, zero2a))
print("03a: psnr=%.2f, ms-ssim=%.6f, rate=%.4f, bpp=%.2f" % get_metrics(orig, zero3a))
print("03b: psnr=%.2f, ms-ssim=%.6f, rate=%.4f, bpp=%.2f" % get_metrics(orig, zero3b))

#orig: psnr=inf, ms-ssim=1.000000, rate=1.0000, bpp=9.53
#jpeg95: psnr=40.87, ms-ssim=0.990100, rate=0.1568, bpp=1.49
#02a: psnr=38.21, ms-ssim=0.979415, rate=0.0056, bpp=0.05
#jpeg70: psnr=37.24, ms-ssim=0.973139, rate=0.0416, bpp=0.40
#jpeg50: psnr=36.36, ms-ssim=0.963297, rate=0.0299, bpp=0.29
#jpeg20: psnr=34.05, ms-ssim=0.922156, rate=0.0189, bpp=0.18
#03a: psnr=37.69, ms-ssim=0.979635, rate=0.0046, bpp=0.04
#03b: psnr=37.60, ms-ssim=0.982884, rate=0.0035, bpp=0.03