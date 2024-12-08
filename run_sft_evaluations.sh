set -e

# # SFT Qwen2.5-Coder-1.5B on BIRD's training set (w/ EK)
# CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py \
#     --llm_path /checkpoint/arthur/13910240/qwen2.5-coder-7b-instruct-bird-with-evidence/ckpt-592 \
#     --sic_path ./sic_ckpts/sic_bird_with_evidence \
#     --table_num 6 \
#     --column_num 10 \
#     --dataset_path ./data/sft_bird_with_evidence_dev_text2sql.json \
#     --max_tokens 4096 \
#     --max_new_tokens 256


# # SFT Qwen2.5-Coder-7B-Instruct on BIRD's training set (w/ EK)
# CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py \
#     --llm_path /checkpoint/arthur/13910240/qwen2.5-coder-7b-instruct-bird-with-evidence/ckpt-592 \
#     --sic_path ./sic_ckpts/sic_bird_with_evidence \
#     --table_num 6 \
#     --column_num 10 \
#     --dataset_path ./data/sft_bird_with_evidence_dev_text2sql.json \
#     --max_tokens 4096 \
#     --max_new_tokens 256

# SFT CodeS-7B on BIRD's training set (w/ EK)
CUDA_VISIBLE_DEVICES=0 python -u text2sql_zero_shot.py \
    --llm_path /checkpoint/arthur/13910268/codes-7b-bird-with-evidence/ckpt-592 \
    --sic_path ./sic_ckpts/sic_bird_with_evidence \
    --table_num 6 \
    --column_num 10 \
    --dataset_path ./data/sft_bird_with_evidence_dev_text2sql.json \
    --max_tokens 4096 \
    --max_new_tokens 256
    
