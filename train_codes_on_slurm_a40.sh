#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=train_qwen2.5-coder-1.5b-instruct
#SBATCH --gres=gpu:a40:4
#SBATCH --time=2:30:00
#SBATCH -c 32
#SBATCH --mem=160G
#SBATCH --qos=deadline
#SBATCH --account=deadline
#SBATCH --partition=a40
#SBATCH --output=logs/train-slurm-a40-qwen2.5-coder-1.5b-instruct-%j.out
#SBATCH --error=logs/train-slurm-a40-qwen2.5-coder-1.5b-instruct-%j.err

# N_GPU=4

source /fs01/home/arthur/.zshrc
eval "$(micromamba shell hook --shell zsh)"
micromamba activate codes
cd /fs01/projects/r2llab/arthur/codes

config_file=/h/arthur/.cache/huggingface/accelerate/bird_4A40s_config.yaml
batch_size=2
epochs=12
dataset_name=$1
db_ids=("computer_student" "movie_platform" "app_store")

set -e

for db_id in ${db_ids[@]}; do
    output_ckpt_dir=/projects/r2llab/arthur/checkpoints/Text2SQL/${dataset_name}/${db_id}
    text2sql_data_dir=./data/synthetic_data_manual_prompt/${dataset_name}_${db_id}_text2sql.json
    neptune_tags="qwen2.5-coder-1.5b-instruct ${dataset_name} ${db_id}"

    # SFT Qwen/Qwen2.5-Coder-1.5B-Instruct on BIRD with external knowledge
    accelerate launch --config_file ${config_file} train_causal_lm.py \
        --per_device_train_batch_size ${batch_size} \
        --block_size 4096 \
        --seed 42 \
        --pretrained_model_name_or_path Qwen/Qwen2.5-Coder-1.5B-Instruct \
        --epochs ${epochs} \
        --lr 5e-6 \
        --warmup_ratio 0.05 \
        --checkpointing_steps 10000 \
        --mode sft \
        --output_ckpt_dir ${output_ckpt_dir} \
        --text2sql_data_dir ${text2sql_data_dir} \
        --table_num 6 --column_num 10 \
        --neptune_project arthurchen/bird-data-synthesis \
        --neptune_api_token eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmNDk2MmE1NS01Y2RiLTQ0YTAtYmJjYS01YjM3ZGQ0ZWZhMjcifQ== \
        --neptune_name qwen2.5-coder-1.5b-instruct-bird \
        --neptune_tags ${neptune_tags} \
        --prompt_mode train \
        --max_num_ckpts 1
        # --save_all_states # this is needed to resume training
done
