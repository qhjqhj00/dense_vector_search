import os
from tqdm import tqdm
import json
import re
from split import split_text

data = json.loads(open('../1022.json').read())
print(len(data))
data = [d for d in data if len(d[0]) > 0]
print(len(data))
with open('1022.json', 'w') as f:
    for d in data:
        f.write(f'{d[1]}\t{d[0]}\n')
    f.truncate()

with open('1022.tsv', 'w') as f:
    f.write('id\ttext\ttitle\n')
    for p in tqdm(data):
        text = p[0]
        text = re.sub('\s+', '', text)
        splits = split_text(text, maxlen=256)[0]
        for i,s in enumerate(splits):
            f.write(f'{p[1]}_{i}\t{s}\ttitle\n')
    f.truncate()