python retriever.py \
	--model_file ../gov_dpr/dpr_biencoder.39.125 \
	--ctx_file  ../ctx.tsv \
	--qa_file ../gov_qa.csv \
	--encoded_ctx_file ../gov_embed_0 \
	--out_file ./gov_res \
  	--n-docs 50