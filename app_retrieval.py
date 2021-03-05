
from flask import Flask, request, Response, make_response
from flask_cors import CORS
from retriever import get_retriever, get_idx
from dpr.options import setup_args_gpu, print_args, add_encoder_params, add_tokenizer_params, add_cuda_params
import argparse
import json

app = Flask(__name__)
CORS(app)

parser = argparse.ArgumentParser()

add_encoder_params(parser)
add_tokenizer_params(parser)
add_cuda_params(parser)

parser.add_argument('--ctx_file', required=True, type=str, default=None,
                    help="All passages file in the tsv format: id \\t passage_text ")
parser.add_argument('--encoded_ctx_file', type=str, default=None,
                    help='Glob path to encoded passages (from generate_dense_embeddings tool)')
parser.add_argument('--batch_size', type=int, default=32, help="Batch size for question encoder forward pass")
parser.add_argument('--index_buffer', type=int, default=50000,
                    help="Temporal memory data buffer size (in samples) for indexer")
parser.add_argument("--hnsw_index", action='store_true', help='If enabled, use inference time efficient HNSW index')
parser.add_argument("--save_or_load_index", action='store_true', help='If enabled, save index')

args = parser.parse_args()

assert args.model_file, 'Please specify --model_file checkpoint to init model weights'

setup_args_gpu(args)
print_args(args)
retriever = get_retriever(args)
ctx_dict = open(args.ctx_file).read().strip().split('\n')[1:]
ctx_dict = [p.split('\t') for p in ctx_dict]
ctx_dict = {k[0]: k[1] for k in ctx_dict}

def add_context(res):
    for r in res:
        for k in r:
            r[k]['content'] = ctx_dict[k]

@app.route("/api")
def get():
    text = request.args.get('text', '')
    res = get_idx(retriever, text, topk=10)
    add_context(res) 
    return Response(json.dumps(res, ensure_ascii=False), mimetype='application/json; charset=utf-8')

@app.route('/batch_retrieval/', methods=['POST', 'GET'])
def retrieval():
    input_json = request.get_json(force=True)
    data = input_json['batch']
    topk = input_json['topk']
    res = get_idx(retriever, data, topk=topk)
    add_context(res)
    return Response(json.dumps(res, ensure_ascii=False), mimetype='application/json; charset=utf-8')

if __name__ == '__main__':
    app.run(host ='0.0.0.0',port=6790,threaded=True)
