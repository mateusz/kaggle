#%%

from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertForSequenceClassification 
from transformers import BertTokenizer, BertModel, BertForSequenceClassification 
import torch.nn.functional as F

torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%


class TwitterDataset(Dataset):
    def __init__(self, path):
        d = pd.read_csv(path)#.iloc[:2048]
        tokenizer = BertTokenizer.from_pretrained('bert-large-cased', do_lower_case=True)
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

net = BertForSequenceClassification.from_pretrained('bert-large-cased', num_labels=2).to(device)

for p in net.bert.parameters():
    p.requires_grad = False

for p in net.bert.encoder.layer[12].parameters():
    p.requires_grad = True

for p in net.bert.encoder.layer[16].parameters():
    p.requires_grad = True

for p in net.bert.encoder.layer[20].parameters():
    p.requires_grad = True
#%%

report_steps = 5
opt = torch.optim.Adam(
    net.parameters(),
    lr = 1e-4
)
lossfn = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[2,4], gamma=0.1)
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
    scheduler.step()
# %%

# distilbert-base-uncased:
# Baseline: val: 0.39954047, acc=0.83705650 (after 1 epoch, at 1e-4, lr schedule not needed)
# Just classifier layer: val: 0.44075171, acc=0.80157687
# Final layer+classifier: val: 0.42129846, acc=0.81865966
# First layer + classfier: val: 0.43145483, acc=0.82785808 (overfits training)

# bert-base-uncased: 
# just classifier: same bad
# final 2 layers + classifier: same bad

# bert-large-cased:
# 20:24 + classifier: val: 0.43918899, acc=0.81208936
# 0 + classifier: OOM
# 12 + classifier: val: 0.44430767, acc=0.80814717, still converging
# 12: + classfier: OOM
# 12,16,20 + classifier: val: 0.44983255, acc=0.80420499