
from flask import Flask, request, Response, make_response
from flask_cors import CORS
import argparse
import faiss
import time
import json
import requests
import numpy as np
from split import split_text

app = Flask(__name__)
CORS(app)

parser = argparse.ArgumentParser()

parser.add_argument('--save_paths', required=True, type=str)
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--encoder_url', required=True, type=str)
parser.add_argument('--port', required=True, type=str)

args = parser.parse_args()
save_paths = args.save_paths.split(',')

index_list = {}
idx_map_list = {}
ctx_list = {}
for path in save_paths:
    domain_sign = path.split('_')[1]
    print(f'loading {domain_sign} index...')
    index_list[domain_sign]=faiss.read_index(
        f'generated/{path}' + '/index')
    if args.gpu:
        index_list[domain_sign] = faiss.index_cpu_to_all_gpus(
            index_list[domain_sign])
        
    idx_map_list[domain_sign] = json.loads(
        open(f'generated/{path}' + '/idx_map.json').read())

    try:
        ctx_dict = json.loads(open(f'generated/{path}' + '/ctx.json').read()) #.strip().split('\n')
    except:
        ctx_dict=open(f'generated/{path}' + '/ctx.json').read().strip().split('\n')
        ctx_dict = [p.split('\t') for p in ctx_dict]
    ctx_dict = {k[0]: k[1] for k in ctx_dict}
    print(f'{domain_sign}: {len(ctx_dict)} loaded.')
    ctx_list.update(ctx_dict)
    
def get_res(domain, index, query, topk=10):
    D, I = index.search(query, topk)
    res = [
        [[idx_map_list[domain][I[i][k]], float(D[i][k])]
        for k in range(D.shape[1])] 
        for i in range(D.shape[0])
        ]
    return res


@app.route("/api", methods=['POST', 'GET'])
def get():
    input_json = request.get_json(force=True)
    query = input_json['query']
    domain_sign = 'general'
    domain_sign = input_json['domain']
    query = np.array(
        requests.get(f'http://{args.encoder_url}/api?text={query}'
        ).json()).astype('float32')
    res = []
    if domain_sign == 'general' or domain_sign not in index_list:
        for dom in index_list:
            print(dom)
            res.extend(get_res(dom, index_list[dom], query))
    else:
        res.extend(get_res(domain_sign, index_list[domain_sign], query))
    res.sort(key=lambda k: k[1])
    merged_ids_scores = []
    for s in res:
        tmp = {}
        for i, candidate in enumerate(s):
            full_id = '_'.join(candidate[0].split('_')[:-1])
            if full_id in tmp:
                continue
            else:
                tmp[full_id] = candidate[1]
        tmp = [[k,v] for k,v in tmp.items()]
        tmp.sort(key=lambda k: k[1])
        merged_ids_scores.extend(tmp)
    merged_ids_scores.sort(key=lambda k: k[1])
    merged_ids_scores = merged_ids_scores[:3]
    n_merged = len(merged_ids_scores)
    resp = {
        'text': [ctx_list[merged_ids_scores[i][0]]
                         for i in range(n_merged)],
        'score': [merged_ids_scores[i][1] for i in range(n_merged)],
        'id':[merged_ids_scores[i][0] for i in range(n_merged)]
    }
    return Response(
        json.dumps(resp, ensure_ascii=False), mimetype='application/json; charset=utf-8')

#TODO
@app.route("/update/", methods=['POST', 'GET'])
def update():
    """
    data format: [[passage1, id1], [passage2, id2], ...]
    """
    res = {'pre_num': index.ntotal}
    input_json = request.get_json(force=True)
    passages = input_json['batch']
    text_chunks = []
    for passage in passages:
        text = passage[0]
        idx = passage[1]
        if idx in ctx_dict:
            continue
        ctx_dict[idx] = text
        text_chunk = split_text(text, maxlen=256)[0]
        text_chunks.extend(text_chunk)
        idx_map.extend([f'{idx}_{i}' for i in range(len(text_chunk))])
    if len(text_chunks) == 0:
        res['update'] = 0
        return Response(json.dumps(res, ensure_ascii=False), mimetype='application/json; charset=utf-8')
    encoded = np.array(
        requests.post(f'http://{args.encoder_url}/batch_encode/', json.dumps({'batch':text_chunks})
        ).json()).astype('float32')
    index.add(encoded)
    res['update'] =  encoded.shape[0]
    res['current_num'] = index.ntotal
    if 'dump' in input_json:
        res['dumped'] = True
        if args.gpu:
            faiss.write_index(faiss.index_gpu_to_cpu(index), args.save_path + '/index')
        else:
            faiss.write_index(index, args.save_path + '/index')
        with open(args.save_path + '/idx_map.json', 'w') as f:
            json.dump(idx_map, f, ensure_ascii=False, indent=2)
        with open(args.ctx_file, 'w') as f:
            for k,v in ctx_dict.items():
                f.write(f'{k}\t{v}\n')
        with open('data.log', 'a') as f:
            ts = list(time.localtime(time.time()))
            ts = '-'.join([str(t) for t in ts])
            pre = res['pre_num']
            now = res['current_num']
            f.write(f'{ts}\t{pre}\t{now}\t{now-pre}\n')
    else:
        res['dumped'] = False
    return Response(
        json.dumps(res, ensure_ascii=False), mimetype='application/json; charset=utf-8')

#TODO
@app.route('/batch_retrieval/', methods=['POST', 'GET'])
def retrieval():
    input_json = request.get_json(force=True)
    data = input_json['batch']
    topk = input_json['topk']
    queries = np.array(
        requests.post(f'http://{args.encoder_url}/batch_encode/', json.dumps({'batch':data})
        ).json()).astype('float32')
    D, I = index.search(queries, topk)
    res = [
        [[idx_map[I[i][k]], float(D[i][k])] for k in range(D.shape[1])] 
        for i in range(D.shape[0])
        ]
    return Response(
        json.dumps(res, ensure_ascii=False), mimetype='application/json; charset=utf-8')

@app.route("/check", methods=['POST', 'GET'])
def check():
    query = request.args.get('id')
    domain = request.args.get('domain')
    if query in ctx_list[domain]:
        res = {'is_data': True}
    else:
        res = {'is_data': False}
    return Response(
        json.dumps(res, ensure_ascii=False), mimetype='application/json; charset=utf-8')

if __name__ == '__main__':
    app.run(host ='0.0.0.0',port=args.port,threaded=True)

