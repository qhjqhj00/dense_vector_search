import os
from tqdm import tqdm
import json
import re
from split import split_text
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--passage_raw', type=str, required=True)
parser.add_argument('--passge_json', type=str, required=True)
parser.add_argument('--save_file', type=str, required=True)
parser.add_argument('--max_len', type=int, default=256, required=True)
args = parser.parse_args()

def get_context(file_path):
    contexts = []
    f = open(file_path)
    for line in f.readlines():
        line = json.loads(line)
        context = re.sub('\s+','', line['doc']['text'])
        idx = line['doc']['url']
        if len(context) > 110 and len(context) < 4000:
            contexts.append([idx,context])
    return contexts

passages = []
ori = open(args.passage_raw).read().strip().split('\n')[1:]
ori = [p.split('\t') for p in ori]
passages.extend(ori)


"""for pa in tqdm(list(os.walk('/home/qhj/beijKW/data/fs/'))[0][2]):
    if pa.endswith('dat'):
        a = get_context('/home/qhj/beijKW/data/fs/' + pa)
        passages.extend(a)"""

with open(args.passge_json, 'w') as f:
    json.dump(passages,f,ensure_ascii=False,indent=4)
with open(args.save_file, 'w') as f:
    f.write('id\ttext\ttitle\n')
    for p in tqdm(passages):
        text = p[1]
        splits = split_text(text, maxlen=args.max_len)[0]
        for i,s in enumerate(splits):
            f.write(f'{p[0]}_{i}\t{s}\ttitle\n')
    f.truncate()