#!/bin/bash
#SBATCH --job-name=vuln-bilstm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/bilstm_%j.out
#SBATCH --error=logs/bilstm_%j.err

python scripts/train.py --model bilstm --data_dir data/processed --output_dir results/bilstm --epochs 20 --batch_size 32 --lr 1e-3
