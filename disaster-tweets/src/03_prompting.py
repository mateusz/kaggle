#%%

from torch.utils.data import DataLoader, TensorDataset, Dataset, WeightedRandomSampler
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import torch.nn.functional as F
from transformers import pipeline, set_seed
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(1)

#%%

tokenizer = AutoTokenizer.from_pretrained("StabilityAI/stablelm-base-alpha-3b", padding_side='left')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

#%%

model = AutoModelForCausalLM.from_pretrained("StabilityAI/stablelm-base-alpha-3b", device_map="auto", load_in_8bit=True, torch_dtype=torch.float16)
#model.half().cuda()

#%%

df = pd.read_csv('data/train.csv')
t0, v0 = np.split(df[df['target']==0].sample(frac=1, random_state=2), [16])
t1, v1 = np.split(df[df['target']==1].sample(frac=1, random_state=2), [16])

train = pd.concat([t0, t1]).sample(frac=1, random_state=2)
val = pd.concat([v0, v1]).sample(frac=1, random_state=2)

print(len(train), len(val))
#%%

def cleanup(t):
    t = re.sub(r'[\r\n]', ' ', t)
    t = re.sub(r'"', '\'', t)
    return t

prompt = """The task is to find messages that talk about disasters.

DISASTER messages could be about natural or man-made situations, and will usually involve death, damaged property, fires, floods, war, famine etc. They will also be in a official tone.

OTHER messages are about sport, politics, love, gossip and other mundane daily issues, will contain emoji, and will sound informal.

You score one point for each correctly identified disaster message.

Here is a list of examples to help you along:
"""

for i,r in train.iterrows():
    if r['target']==1:
        suffix = "is categorized as DISASTER."
    else:
        suffix = "is categorized as OTHER."

    prompt += "\"%s\" %s\n" % (cleanup(r['text']), suffix)

print(prompt)
#%%

class TwitterDataset(Dataset):
    def __init__(self, d):
        c1w = 1.0/len(d[d['target']==0])
        c2w = 1.0/len(d[d['target']!=0])
        d.loc[d['target']==0, 'weight'] = c1w
        d.loc[d['target']!=0, 'weight'] = c2w
        self.weights = d['weight']

        t = d['text']
        self.texts = t.apply(cleanup)
        self.labels = torch.Tensor(d['target'].values).to(torch.uint8).to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        t = self.texts[idx]
        t = "%s\"%s\" is categorized as " % (prompt, t)
        return {
            'texts': t,
            'labels': self.labels[idx],
            'weight': self.weights[idx],
        }

val = val.reset_index()
v = TwitterDataset(val)
batch_size=3
vb = DataLoader(v, batch_size=batch_size,
    sampler=WeightedRandomSampler(v[:]['weight'].to_list(), num_samples=len(v), replacement=True)
)

print(len(val))
#%%

torch.cuda.empty_cache()
threshold = 0.1

#model = model.to(device)
token_disaster = tokenizer(" DISASTER").input_ids[0]

count = 0
correct = 0
batches = 250
for i,r in enumerate(vb, 0):
    t = tokenizer(r['texts'], return_tensors="pt", padding=True)
    outputs = model.generate(
        t.input_ids.to(device),
        attention_mask=t.attention_mask.to(device),
        max_new_tokens=5,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.pad_token_id,
        #num_beams=3,
        #do_sample=False,
        #force_words_ids=force_words_ids,
        #num_return_sequences=1,
        #no_repeat_ngram_size=1,
        #remove_invalid_values=True,
    )

    #print(tokenizer.decode(outputs.sequences[0][-25:]))
    #token = outputs.scores[-4].argmax()
    #print("%d, '%s'" % (token, tokenizer.decode(token)))

    collected = torch.zeros((batch_size)).to(device)
    for o in outputs.scores:
        collected = torch.max(collected, o.softmax(dim=1)[:,token_disaster])

    correct += torch.sum(torch.where(collected>threshold, 1.0, 0.0)==r['labels'])
    count += len(collected)

    print(batches)
    batches -= 1
    if batches<=0:
        break
    
    print("[%d] Accuracy: %.5f (%d samples)" % (251-batches, correct/count, count))

#%%

