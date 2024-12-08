#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=train_codes_7b_bird_with_evidence
#SBATCH --gres=gpu:a40:4
#SBATCH --time=8:00:00
#SBATCH -c 32
#SBATCH --mem=160G
#SBATCH --qos=normal
#SBATCH --partition=a40
#SBATCH --output=codes_logs/slurm-a40-%j.out
#SBATCH --error=codes_logs/slurm-a40-%j.err

# N_GPU=4

source /fs01/home/arthur/.zshrc
eval "$(micromamba shell hook --shell zsh)"
micromamba activate codes
cd /fs01/projects/r2llab/arthur/codes

bash train_codes.sh
