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
#%%

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

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

prompt = "Document classifying messages as related to disasters:\n\n"

for i,r in train.iterrows():
    if r['target']==1:
        suffix = "describes a disaster."
    else:
        suffix = "does not."

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
token_disaster = tokenizer("disaster", add_prefix_space=True, add_special_tokens=False).input_ids[0]
for i,r in val.sample(n=4).iterrows():
    inp = "%s\"%s\" describes " % (prompt, cleanup(r['text']))
    
    t = tokenizer(inp, return_tensors="pt")
    in_len = len(t.input_ids[0])
    outputs = model.generate(
        t.input_ids.to(device),
        attention_mask=t.attention_mask.to(device),
        max_new_tokens=1,
        return_dict_in_generate=True,
        output_scores=True,
        #num_beams=3,
        #do_sample=False,
        #force_words_ids=force_words_ids,
        #num_return_sequences=1,
        #no_repeat_ngram_size=1,
        #remove_invalid_values=True,
    )

    print(cleanup(r['text']))
    for s in outputs.scores:
        print(s[0][token_disaster])
    #outputs
    #print(tokenizer.decode(outputs.scores[5].argmax()))
    break