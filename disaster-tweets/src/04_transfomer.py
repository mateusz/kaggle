#%%

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset, Dataset, WeightedRandomSampler
import numpy as np
import pandas as pd
from transformers import GPT2Tokenizer

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

import re

torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', do_lower_case=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.add_special_tokens({'mask_token': '[MASK]'})

#%%

def cleanup(t):
    t = re.sub(r'[\s\r\n]', ' ', t)
    t = re.sub(r' +', ' ', t)
    t = re.sub(r'http[^ ]*', '', t)
    t = re.sub(r'@[^ ]*', '', t)
    t = re.sub(r'[^A-Za-z0-9- ]', '', t)
    t = re.sub(r'^ *', '', t)
    t = re.sub(r' *$', '', t)
    t = t.lower()
    return t

class TwitterDataset(Dataset):
    def __init__(self, path, tokenizer, token_dropout=0.1):
        self.token_dropout = token_dropout
        d = pd.read_csv(path)#.iloc[:2048]

        c1w = 1.0/len(d[d['target']==0])
        c2w = 1.0/len(d[d['target']!=0])
        d.loc[d['target']==0, 'weight'] = c1w
        d.loc[d['target']!=0, 'weight'] = c2w
        self.weights = d['weight']

        t = d['text']
        t = t.apply(cleanup)
        t = tokenizer(t.to_list(), padding='longest').convert_to_tensors('pt')
        self.input_ids = t['input_ids'].to(device)
        self.attention_masks = t['attention_mask'].to(device)
        self.labels = torch.Tensor(d['target'].values).to(torch.uint8).to(device)
        self.mask_token_id = tokenizer(tokenizer.mask_token)['input_ids'][0]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        mask = torch.empty_like(self.input_ids[idx]).bernoulli_(self.token_dropout).bool()
        t = self.input_ids[idx].masked_fill(mask, self.mask_token_id)
        return {
            'input_ids': t,
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels[idx],
            'weight': self.weights[idx],
        }

d = TwitterDataset('data/train.csv', tokenizer, token_dropout=0.1)
max_len = len(d[0]['input_ids'])

batch_size = 16
t,v = torch.utils.data.random_split(d, [0.9, 0.1])
train = DataLoader(t, batch_size=batch_size, sampler=WeightedRandomSampler(t[:]['weight'], len(t), replacement=True))
val = DataLoader(v, batch_size=batch_size)

#%%

# Following 
# https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51


def scaled_dp_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    attention = query.bmm(key.transpose(1,2))
    scale = key.size(-1) ** 0.5
    attention = attention / scale
    attention = F.softmax(attention, dim=-1)

    return attention.bmm(value)

class AttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_k):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_k)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        q = self.q(query)
        k = self.k(key)
        v = self.v(value)
        return scaled_dp_attention(q, k, v)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int):
        super().__init__()
        dim_k = dim_in // num_heads
        
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_k) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return self.linear(
            torch.cat([h(query, key, value) for h in self.heads], dim=-1)
        )

def position_encoding(
    max_seq_len: int, dim_model: int, device: torch.device = torch.device("cpu"),
) -> Tensor:
    pos = torch.arange(max_seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / (1e4 ** (dim / dim_model))

    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))

def feed_forward(dim_in: int = 512, dim_feedforward: int = 2048, dropout: float = 0.1) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim_in, dim_feedforward),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_feedforward, dim_in),
    )

class Residual(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: Tensor) -> Tensor:
        return self.norm(tensors[0] + self.dropout(self.sublayer(*tensors)))


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim_model: int = 512,
        num_heads: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = Residual(
            MultiHeadAttention(num_heads, dim_model),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward, dropout),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, tgt: Tensor, memory: Tensor = None) -> Tensor:
        tgt = self.attention(tgt, tgt, tgt)
        return self.feed_forward(tgt)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        tokenizer: GPT2Tokenizer,
        max_len: int,
        device: torch.device,
        num_layers: int = 6,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dim_out: int = 1,
        dropout: float = 0.1,
        dim_head: int = 512,
    ):
        super().__init__()
        self.dim_model = dim_model
        self.embed = nn.Embedding(len(tokenizer.get_vocab()), dim_model).to(device)
        self.embed_dropout = nn.Dropout(dropout)
        self.pe = position_encoding(max_len, dim_model).to(device)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout).to(device)
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(dim_model, dim_head),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_head, dim_out),
            nn.Sigmoid(),
        ).to(device)

    def forward(self, tgt: Tensor, memory: Tensor = None) -> Tensor:
        x = self.embed(tgt)
        x = self.embed_dropout(x)
        x += self.pe
        for layer in self.layers:
            x = layer(x, memory)
        # Pool all seq elements
        x = torch.mean(x, dim=1)
        x = self.head(x)
        return x

net = TransformerEncoder(
    tokenizer=tokenizer,
    max_len=max_len,
    device=device,
    num_layers=2,
    dim_model=32,
    num_heads=2,
    dim_feedforward=32,
    dim_out=1,
    dropout=0.3,
    dim_head=32,
)
seq = torch.randint(0, len(tokenizer.get_vocab()), (1, max_len)).long().to(device)
net(seq)

print(sum([p.numel() for p in net.parameters()]))
#net

#%%

report_steps = 1000
epochs = 1000
warmup_epochs = 20
#net = torch.load('models/99_0').to(device)

#def calc_lr(step, dim_embed, warmup_steps):
#opt = torch.optim.Adam(net.parameters(), betas = (0.9, 0.98), eps = 1.0e-9)
opt = torch.optim.Adam(net.parameters(), lr=1e-4)
lossfn = nn.BCELoss()
#scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[2,4], gamma=0.1)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=10, cooldown=10)
#scheduler = SchedulerWarmupExpo(opt, warmup_steps=len(train)*warmup_epochs, verbose=True)

sched1 = torch.optim.lr_scheduler.LinearLR(
    opt, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs#, verbose=True
)
sched2 = torch.optim.lr_scheduler.ExponentialLR(
    opt, gamma=0.9, verbose=True
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    opt, schedulers=[sched1, sched2], milestones=[warmup_epochs], verbose=True,
)

#scheduler = Scheduler(
#    optimizer=opt,
#    dim_embed=4096, # Max lr = 0.0069
#    warmup_steps=20,
#    verbose=True,
#)

min_val_loss = 9999999999.0
early_stop_counter = 0
torch.cuda.empty_cache()
for epoch in range(epochs):
    net.train()
    running_loss = 0.0
    train_total_correct = 0
    train_total_loss = 0.0
    for i, data in enumerate(train, 0):
        opt.zero_grad()
        labels = data['labels']
        o = net(
            data['input_ids'],
        )

        loss = lossfn(o, data['labels'].unsqueeze(dim=1).float())
        train_total_loss += loss.item()

        loss.backward()
        opt.step()

        running_loss += loss.item()
        if i % report_steps == report_steps-1:
            print("%d train: %.8f" % (running_loss/float(report_steps)))
            running_loss = 0.0

        p = torch.where(o>0.5, 1.0, 0.0)
        train_total_correct += (labels.unsqueeze(-1)==p).sum()

    net.eval()
    total_loss = 0.0
    with torch.no_grad():
        total_correct = 0
        for i, data in enumerate(val, 0):
            labels = data['labels']
            o = net(
                data['input_ids'],
            )

            loss = lossfn(o, data['labels'].unsqueeze(dim=1).float())
            total_loss += loss.item()

            p = torch.where(o>0.5, 1.0, 0.0)
            total_correct += (labels.unsqueeze(-1)==p).sum()

    tl = train_total_loss/float(len(train))
    tacc = float(train_total_correct)/float(len(train)*batch_size)

    vl = total_loss/float(len(val))
    vacc = float(total_correct)/float(len(val)*batch_size)
    print("[%d] train: %.8f, tacc: %.8f, val: %.8f, vacc=%.8f" % (epoch, tl, tacc, vl, vacc))

    if vl<min_val_loss:
        min_val_loss = vl
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    if early_stop_counter>=1000:
        break
 
    scheduler.step()
    #scheduler.step(vl)
#%%

torch.save(net, 'models/04_0')

#%%

t = torch.Tensor([
    [1,1,1],
    [2,2,2]
])
torch.mean(t, dim=0)
n = nn.AvgPool2d()
n(t)

#%%
"""
p = position_encoding(100,512)
p = p.squeeze(dim=0)
cax = plt.matshow(p)
plt.gcf().colorbar(cax)

max_seq_len = 100
dim_model = 1000
pos = torch.arange(max_seq_len, dtype=torch.float).reshape(1, -1, 1)
dim = torch.arange(dim_model, dtype=torch.float).reshape(1, 1, -1)
print(pos)
print(1 / (1e4 ** (dim/dim_model)))
phase = pos / (1e4 ** (dim / dim_model))
plt.plot(torch.sin(phase[0][99]))
plt.plot(phase[0][99])
plt.show()
#torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))

#torch.where(dim.long() % 2 == 0, 0, 1)
"""