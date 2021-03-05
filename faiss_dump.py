import argparse
import faiss
import os
import numpy as np
import json


def load_encoded(path):
    import pickle
    with open(path, mode='rb') as f:
        data = pickle.load(f)
    return data

def to_dump(path, target_path):
    os.makedirs(target_path, exist_ok=True)
    data = load_encoded(path)        
    idx = [k[0] for k in data]
    encoded = np.stack([k[1] for k in data])
    print(f'ctx shape: {encoded.shape}')
    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatL2(encoded.shape[1])
    #gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    index_flat.add(encoded)
    print(index_flat.ntotal)
    faiss.write_index(index_flat, target_path + '/index')
    with open(target_path + '/idx_map.json', 'w') as f:
        json.dump(idx, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoded_ctx_file", required = True, type=str)
    parser.add_argument("--save_path", required = True, type=str)
    args = parser.parse_args()
    path = args.encoded_ctx_file
    target_path = args.save_path
    to_dump(path, target_path)


if __name__ == "__main__":
    main()
