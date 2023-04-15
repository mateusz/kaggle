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
        suffix = " == disaster"
    else:
        suffix = " == other"

    prompt += "\"%s\" %s\n" % (cleanup(r['text']), suffix)

tp = 0
fp = 0
tn = 0
fn = 0
threshold = 0.1

test = val.sample(n=1024, random_state=1)
baseline = len(val[val['target']==0])/len(val)
print(baseline)

model = model.to(device)
token_disaster = tokenizer("disaster", add_prefix_space=True, add_special_tokens=False).input_ids[0]

for i,r in test.iterrows():
    inp = "%s\"%s\" == " % (prompt, cleanup(r['text']))
    
    t = tokenizer(inp, return_tensors="pt")
    in_len = len(t.input_ids[0])
    outputs = model.generate(
        t.input_ids.to(device),
        attention_mask=t.attention_mask.to(device),
        max_new_tokens=5,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.eos_token_id,
        #num_beams=3,
        #do_sample=False,
        #force_words_ids=force_words_ids,
        #num_return_sequences=1,
        #no_repeat_ngram_size=1,
        #remove_invalid_values=True,
    )

    #collected = 0.0
    #for s in outputs.scores:
    #    p = F.softmax(s[0], dim=0)[token_disaster]
    #    collected += p/5.0
    collected = F.softmax(outputs.scores[0][0], dim=0)[token_disaster]
    
    test.loc[i, 'p'] = float(collected)
    if collected>=threshold:
        test.loc[i, 'pred'] = 1
    else:
        test.loc[i, 'pred'] = 0

print("Accuracy: %.5f" % (len(test[test['pred']==test['target']])/len(test)))

#%%

# GPT2 baseline 0.5706371191135734, Accuracy: 0.58984, on "== disaster" query. Useless :D
