#!/bin/bash

set -e
# -------------- Training on BIRD with external knowledge -------------- #
# SFT Qwen2.5-Coder-1.5B on BIRD with external knowledge
# accelerate launch train_causal_lm.py \
#     --per_device_train_batch_size 1 \
#     --block_size 4096 \
#     --seed 42 \
#     --pretrained_model_name_or_path Qwen/Qwen2.5-Coder-1.5B \
#     --epochs 4 \
#     --lr 5e-6 \
#     --warmup_ratio 0.05 \
#     --checkpointing_steps 10000 \
#     --tensorboard_log_dir ./train_logs/qwen2.5-coder-1.5b-bird-with-evidence \
#     --mode sft \
#     --output_ckpt_dir /checkpoint/arthur/13884468/qwen2.5-coder-1.5b-bird-with-evidence \
#     --text2sql_data_dir ./data/sft_bird_with_evidence_train_text2sql.json \
#     --table_num 6 --column_num 10


# # SFT Qwen/Qwen2.5-Coder-7B-Instruct on BIRD with external knowledge
# accelerate launch train_causal_lm.py \
#     --per_device_train_batch_size 2 \
#     --block_size 4096 \
#     --seed 42 \
#     --pretrained_model_name_or_path Qwen/Qwen2.5-Coder-7B-Instruct \
#     --epochs 4 \
#     --lr 5e-6 \
#     --warmup_ratio 0.05 \
#     --checkpointing_steps 10000 \
#     --tensorboard_log_dir ./train_logs/qwen2.5-coder-7b-instruct-bird-with-evidence \
#     --mode sft \
#     --output_ckpt_dir /checkpoint/arthur/13910240/qwen2.5-coder-7b-instruct-bird-with-evidence \
#     --text2sql_data_dir ./data/sft_bird_with_evidence_train_text2sql.json \
#     --table_num 6 --column_num 10

# -------------- Training on BIRD with external knowledge -------------- #
# SFT codes-7b on BIRD with external knowledge
accelerate launch train_causal_lm.py \
    --per_device_train_batch_size 2 \
    --block_size 4096 \
    --seed 42 \
    --pretrained_model_name_or_path seeklhy/codes-7b \
    --epochs 4 \
    --lr 5e-6 \
    --warmup_ratio 0.05 \
    --checkpointing_steps 100000 \
    --tensorboard_log_dir ./train_logs/codes-7b-bird-with-evidence \
    --mode sft \
    --output_ckpt_dir /checkpoint/arthur/13910268/codes-7b-bird-with-evidence \
    --text2sql_data_dir ./data/sft_bird_with_evidence_train_text2sql.json \
    --table_num 6 --column_num 10


# Notes:
# 1. we set batch size to 2 because we train on a40 with 4 GPUs, the gradient accumulation steps is 8 set by accelerate config.
    # --per_device_train_batch_size 4 \
    # --block_size 4096 \
