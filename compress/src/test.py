#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import cv2
import torchvision.transforms as transforms

#%%

ch = 1
input = torch.randn(ch, 5, 5)

output = input.unfold(2, 3, 2).unfold(1, 3, 2).flatten(1,2).permute(1,0,2,3)

rec = output.permute(1,0,2,3).reshape(ch,4,9).transpose(1,2)
counter = F.fold(torch.ones_like(rec), 5, kernel_size=3, stride=2).squeeze(1).transpose(1,2)
rec = F.fold(rec, 5, kernel_size=3, stride=2).squeeze(1).transpose(1,2)

print(rec)
print(rec/counter)
print(input)

#%%
def show_tile(tile, size=5):
    tile = tile.permute(1,2,0).clamp(0.0,1.0).detach().cpu()

    fig,ax = plt.subplots(figsize=(size,size))
    ax.axis('off')
    ax.imshow(tile)
    plt.show()

class Tiler():
    def __init__(self, input, kern, overlap):
        super().__init__()
        self.device = 'cpu'
        self.kern = kern
        self.overlap = overlap
        self.h_edge = input.size(1)
        self.w_edge = input.size(2)
        self.ch = input.size(0)
        self.stride = kern-overlap
        self.data = input
        self.unfolded = False

        self.h_pad = self.get_expanded_edge(self.h_edge) - self.h_edge
        self.w_pad = self.get_expanded_edge(self.w_edge) - self.w_edge

        self.data = F.pad(self.data, (0,self.h_pad,0,self.w_pad), "constant", 0.0)
        self.h_edge=self.data.size(1)
        self.w_edge=self.data.size(2)

        self.w_tiles=(self.w_edge-self.overlap)//(self.stride)
        self.h_tiles=(self.h_edge-self.overlap)//(self.stride)

    def to(self, device):
        self.device = device
        self.data = self.data.to(device)
        return self

    def get_expanded_edge(self, edge):
        c = math.ceil((edge-self.overlap)/self.stride)
        newedge = c*self.stride+self.overlap
        return int(newedge)

    def unfold(self):
        unfolded = F.unfold(self.data, kernel_size=self.kern, stride=self.stride)
        unfolded = unfolded.permute(1,0).reshape(-1,self.ch,self.kern,self.kern)

        self.tile_count = unfolded.size(0)
        assert(self.tile_count==self.w_tiles*self.h_tiles)

        return unfolded

    def get_weights(self):
        # Calculate weights mask for interpolated merging.
        # Single tile mask with overlap 2 and kernel 5 will look like this (. is 0.33, : is 0.66 and 1 is 1)
        # .....
        # .:::.
        # .:1:.
        # .:::.
        # .....
        core_size = self.kern-2*self.overlap
        w=torch.ones(self.ch, core_size, core_size).to(self.device)
        # Step through interpolation
        for ring in torch.linspace(1.0, 0.0, self.overlap+2)[1:-1]:
            w = F.pad(w, (1,1,1,1), "constant", ring)

        # Repeat tile mask over all batches.
        w = w.unsqueeze(0).repeat(self.tile_count,1,1,1)

        # Fold and re-unfold to deal with multiple or no overlaps.
        wfolded = w.reshape(self.tile_count, -1).permute(1,0)
        wfolded = F.fold(wfolded, output_size=(self.h_edge,self.w_edge), kernel_size=self.kern, stride=self.stride)
        
        wunfolded = F.unfold(wfolded, kernel_size=self.kern, stride=self.stride)
        wunfolded = wunfolded.permute(1,0).reshape(-1,self.ch,self.kern,self.kern)

        return w/wunfolded

    def fold(self, data):
        w = self.get_weights()
        
        folded = data*w
        folded = folded.reshape(self.tile_count, -1).permute(1,0)
        folded = F.fold(folded, output_size=(self.h_edge,self.w_edge), kernel_size=self.kern, stride=self.stride)

        return folded


image = cv2.imread("out/orig-tile.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
transform = transforms.Compose([
    transforms.ToTensor()
])
img = transform(image)

overlap=2
kern=8
#input = torch.randn(ch, hedge, wedge)
input = img

til = Tiler(input, kern, overlap)
#show_tile(til.data, 5)
d = til.unfold()
output = til.fold(d)

show_tile(output, 5)

#%%
wedge=32
hedge=16
ch=3
hpad = get_expanded_edge(hedge, kern, overlap) - hedge
wpad = get_expanded_edge(wedge, kern, overlap) - wedge

input = F.pad(input, (0,hpad,0,wpad), "constant", 0.0)
hedge=input.size(1)
wedge=input.size(2)

wtiles=(wedge-overlap)//(kern-overlap)
htiles=(hedge-overlap)//(kern-overlap)
show_tile(input,2)

folded = F.unfold(input, kernel_size=kern, stride=kern-overlap).permute(1,0).reshape(-1,ch,kern,kern)

tile_count = folded.size(0)
assert(tile_count==wtiles*htiles)

w=torch.ones(ch, kern-2*overlap, kern-2*overlap)
for ring in torch.linspace(1.0, 0.0, overlap+2)[1:-1]:
    w = F.pad(w, (1,1,1,1), "constant", ring)
w = w.unsqueeze(0).unsqueeze(0).repeat(htiles,wtiles,1,1,1)
# Restore full weights of the entire image
w[0,:,:,0:overlap,:]=1.0
w[-1,:,:,-overlap:,:]=1.0
w[:,0,:,:,0:overlap]=1.0
w[:,-1,:,:,-overlap:]=1.0
w = w.reshape(tile_count,w.size(2),w.size(3),w.size(4))

wunfolded = w
wunfolded = wunfolded.reshape(tile_count, -1).permute(1,0)
wunfolded = F.fold(wunfolded, output_size=(hedge,wedge), kernel_size=kern, stride=kern-overlap)

unfolded = folded*w
unfolded = unfolded.reshape(tile_count, -1).permute(1,0)
unfolded = F.fold(unfolded, output_size=(hedge,wedge), kernel_size=kern, stride=kern-overlap)
unfolded = unfolded/wunfolded

show_tile(unfolded, 2)
show_tile(input-unfolded, 2)