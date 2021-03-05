ts=`date +%Y%m%d%H%M`
max_len=256
domain_sign=$1
shard_id=0
source_file=$2

json_file=generated/index_${domain_sign}/ctx.json
mkdir generated/index_${domain_sign}
enocoder_model=models/dpr_biencoder.32.139
bs=32


python generate_ctx.py \
	--passage_raw ${source_file} \
	--passge_json  ${json_file}\
	--save_file generated/index_${domain_sign}/${ts}_${max_len}.tsv \
	--max_len ${max_len}

python generate_dense_embeddings.py \
	--model_file ${enocoder_model}\
	--ctx_file  generated/index_${domain_sign}/${ts}_${max_len}.tsv \
	--shard_id ${shard_id} --num_shards 1 \
	--batch_size ${bs} \
	--out_file generated/index_${domain_sign}/${ts}_${max_len}_index 	

python faiss_dump.py  \
    --encoded_ctx_file generated/index_${domain_sign}/${ts}_${max_len}_index_${shard_id}  \
    --save_path generated/index_${domain_sign}/
	
