python generate_dense_embeddings.py \
	--model_file dpr_biencoder.32.139 \
	--ctx_file 1022.tsv \
	--shard_id 0 --num_shards 1 \
	--batch_size 128 \
	--out_file 1022_embed_256 	