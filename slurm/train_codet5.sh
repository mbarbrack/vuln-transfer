#!/bin/bash
#SBATCH --job-name=vuln-codet5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=logs/codet5_%j.out
#SBATCH --error=logs/codet5_%j.err

/work/pi_indrani_mandal_uri_edu/mbarbrack_uri_edu/.conda/envs/nerfstudio/bin/python scripts/train.py --model codet5 --data_dir data/processed --output_dir results/codet5 --epochs 10 --batch_size 16 --lr 2e-5
