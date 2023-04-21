#%%

from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertForSequenceClassification 
from transformers import BertTokenizer, BertModel, BertForSequenceClassification 
import torch.nn.functional as F
import transformers.adapters as adapters

torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%


class TwitterDataset(Dataset):
    def __init__(self, path):
        d = pd.read_csv(path)#.iloc[:2048]
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
        t = tokenizer(d['text'].to_list(), padding='longest').convert_to_tensors('pt')
        self.input_ids = t['input_ids'].to(device)
        self.attention_masks = t['attention_mask'].to(device)
        self.labels = torch.Tensor(d['target'].values).to(torch.uint8).to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels[idx]
        }

d = TwitterDataset('data/train.csv')
d[0]

#%%
t,v = torch.utils.data.random_split(d, [0.9, 0.1])
train = DataLoader(t, batch_size=128, shuffle=True)
val = DataLoader(v, batch_size=128)

#%%
net = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2).to(device)

for p in net.distilbert.parameters():
    p.requires_grad = False

config = adapters.LoRAConfig(r=8, alpha=16)
#config = adapters.HoulsbyConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu")
#config = adapters.IA3Config()
net.add_adapter("adapter", config=config)
net.train_adapter("adapter")
net.set_active_adapters("adapter")

net = net.to(device)
trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
total = sum(p.numel() for p in net.parameters())

print("Percent trainable: %.2f%%" % (100.0*(trainable/total)))
#%%

report_steps = 5
opt = torch.optim.Adam(
    net.parameters(),
    lr = 1e-4
)
lossfn = nn.CrossEntropyLoss()
#scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[2,3], gamma=0.1)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=0, cooldown=0)

min_val_loss = 9999999999.0
early_stop_counter = 0
torch.cuda.empty_cache()
for epoch in range(999):
    net.train()
    running_loss = 0.0
    for i, data in enumerate(train, 0):
        opt.zero_grad()
        o = net(
            input_ids=data['input_ids'],
            attention_mask=data['attention_mask']
        ).logits
        
        loss = lossfn(o, F.one_hot(data['labels'].long(), num_classes=2).float())
        loss.backward()
        opt.step()

        running_loss += loss.item()
        if i % report_steps == report_steps-1:
            print("train: %.8f" % (running_loss/float(report_steps)))
            running_loss = 0.0

    net.eval()
    total_loss = 0.0
    with torch.no_grad():
        total_correct = 0
        for i, data in enumerate(val, 0):
            labels = data['labels']
            o = net(
                input_ids=data['input_ids'],
                attention_mask=data['attention_mask']
            ).logits

            loss = lossfn(o, F.one_hot(labels.long(), num_classes=2).float())
            total_loss += loss.item()

            p = torch.argmax(o, dim=1)
            total_correct += (labels==p).sum()

    vl = total_loss/float(len(val))
    acc = float(total_correct)/float(len(v))
    print("val: %.8f, acc=%.8f" % (vl, acc))

    if vl<min_val_loss:
        min_val_loss = vl
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    if early_stop_counter>=3:
        break
 
    #scheduler.step(vl)
    #scheduler.step()

#%%

# full finetune: 84% (5.85GB)
# finetune_bert_top, Final layer+classifier: 82.2% (3.75GB)
# Just head: 80.5% (2GB)
# Housby: 81% (4.5GB)
# LoRA: 81% (4.3GB)
# IA3: 79% (5.30GB)
# 