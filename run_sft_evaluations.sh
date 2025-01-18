set -e

# SFT Qwen2.5-Coder-7B-Instruct on BIRD's training set (w/ EK), 14060494 is with 32 batch size
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py \
    --llm_path $1 \
    --sic_path ./sic_ckpts/sic_bird_with_evidence_new_train \
    --table_num 6 \
    --column_num 10 \
    --dataset_path ./data/sft_bird_with_evidence_new_dev_text2sql.json \
    --max_tokens 4096 \
    --max_new_tokens 256 \
    --output_path $2
    
