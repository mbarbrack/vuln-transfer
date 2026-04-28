#!/bin/bash
#SBATCH --job-name=vuln-gcb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=logs/graphcodebert_%j.out
#SBATCH --error=logs/graphcodebert_%j.err

export TOKENIZERS_PARALLELISM=false

/work/pi_indrani_mandal_uri_edu/mbarbrack_uri_edu/.conda/envs/nerfstudio/bin/python scripts/train.py --model graphcodebert --data_dir data/processed --output_dir results/graphcodebert --epochs 10 --batch_size 32 --lr 2e-5
