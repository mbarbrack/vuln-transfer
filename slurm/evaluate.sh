#!/bin/bash
#SBATCH --job-name=vuln-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/evaluate_%j.out
#SBATCH --error=logs/evaluate_%j.err

for model in codebert codet5 graphcodebert bilstm; do
    /work/pi_indrani_mandal_uri_edu/mbarbrack_uri_edu/.conda/envs/nerfstudio/bin/python scripts/evaluate.py \
        --results_dir results/$model \
        --data_dir data/processed
done

/work/pi_indrani_mandal_uri_edu/mbarbrack_uri_edu/.conda/envs/nerfstudio/bin/python scripts/evaluate.py \
    --mcnemar \
    --preds_a results/codebert/bigvul_test_preds.json \
    --preds_b results/codet5/bigvul_test_preds.json
