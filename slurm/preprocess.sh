#!/bin/bash
#SBATCH --job-name=vuln-preprocess
#SBATCH --partition=cpu
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err

mkdir -p logs data/processed

python scripts/preprocess.py \
    --bigvul_path   data/raw/MSR_data_cleaned.json \
    --devign_path   data/raw/devign.json \
    --cvefixes_path data/raw/cvefixes_full_dataset.jsonl \
    --output_dir    data/processed
