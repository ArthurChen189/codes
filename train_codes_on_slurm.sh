#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=train_codes_7b_bird_with_evidence
#SBATCH --gres=gpu:a100:4
#SBATCH --time=8:00:00
#SBATCH -c 24
#SBATCH --mem=128G
#SBATCH --qos=a100_vzhong
#SBATCH --partition=a100
#SBATCH --output=codes_logs/slurm-a100-%j.out
#SBATCH --error=codes_logs/slurm-a100-%j.err

# N_GPU=4

source /fs01/home/arthur/.zshrc
eval "$(micromamba shell hook --shell zsh)"
micromamba activate codes
cd /fs01/projects/r2llab/arthur/codes

bash train_codes.sh
