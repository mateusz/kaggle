#%%

from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import torch.nn.functional as F
from transformers import pipeline, set_seed
import re

torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(1)
model = GPT2Model.from_pretrained("gpt2")
#%%

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

pipe = pipeline('feature-extraction', model=model, tokenizer=tokenizer)
data = pipe("disaster")
emb_disaster = torch.Tensor(data[0][0])

data = pipe("regular")
emb_regular = torch.Tensor(data[0][0])

#%%

df = pd.read_csv('data/train.csv')
t0, v0 = np.split(df[df['target']==0].sample(frac=1, random_state=2), [16])
t1, v1 = np.split(df[df['target']==1].sample(frac=1, random_state=2), [16])

train = pd.concat([t0, t1]).sample(frac=1, random_state=2)
val = pd.concat([v0, v1]).sample(frac=1, random_state=2)
#%%

def cleanup(t):
    t = re.sub(r'[\s\r\n]', ' ', r['text'])
    t = re.sub(r' +', ' ', t)
    t = re.sub(r'http[^ ]*', '', t)
    t = re.sub(r'@[^ ]*', '', t)
    t = re.sub(r'[^A-Za-z0-9- ]', '', t)
    t = re.sub(r'^ *', '', t)
    t = re.sub(r' *$', '', t)
    return t

prompt = "This document gives examples how to recognise tweets about disasters.\n\n"

for i,r in train.iterrows():
    if r['target']==1:
        suffix = "is a disaster message."
    else:
        suffix = "is a regular message."

    prompt += "\"%s\" %s\n" % (cleanup(r['text']), suffix)

print(prompt)
#%%

#https://huggingface.co/blog/constrained-beam-search
force_words_ids = [
    tokenizer(["disaster", "something"], add_prefix_space=True, add_special_tokens=False).input_ids,
]
print(force_words_ids)

#%%
model = model.to(device)
emb_disaster = emb_disaster.to(device)
emb_regular = emb_regular.to(device)
for i,r in val.sample(n=4).iterrows():
    print(cleanup(r['text']))
    inp = "%s\"%s\" is a " % (prompt, cleanup(r['text']))
    
    t = tokenizer(inp, return_tensors="pt")
    outputs = model(
        t.input_ids.to(device),
        attention_mask=t.attention_mask.to(device),
        #num_beams=3,
        #do_sample=False,
        #force_words_ids=force_words_ids,
        #num_return_sequences=1,
        #no_repeat_ngram_size=1,
        #remove_invalid_values=True,
    )
    dis = F.cosine_similarity(outputs[0][0][-1], emb_disaster, dim=0)
    reg = F.cosine_similarity(outputs[0][0][-1], emb_regular, dim=0)
    if dis>reg:
        print("-> disaster (disaster=%.3f, regular=%.3f)" % (dis, reg))
    else:
        print("-> regular (disaster=%.3f, regular=%.3f)" % (dis, reg))