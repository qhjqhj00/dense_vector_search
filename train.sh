python -m torch.distributed.launch \
	--nproc_per_node=8 train_dense_encoder.py \
	--max_grad_norm 2.0 \
	--encoder_model_type hf_bert \
	--pretrained_model_cfg ../cn_bert \
	--seed 12345 \
	--sequence_length 256 \
	--warmup_steps 1237 \
	--batch_size 4 \
	--do_lower_case \
	--train_file ../gov-train.json \
	--dev_file ../gov-dev.json\
	--output_dir ../gov_dpr_256 \
	--learning_rate 2e-05 \
	--num_train_epochs 40 \
	--dev_batch_size 4 \
	--val_av_rank_start_epoch 30 \
	--hard_negatives 4

