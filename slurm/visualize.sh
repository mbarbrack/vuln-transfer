#!/bin/bash
#SBATCH --job-name=vuln-viz
#SBATCH --partition=cpu
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/visualize_%j.out
#SBATCH --error=logs/visualize_%j.err

mkdir -p figures

/work/pi_indrani_mandal_uri_edu/mbarbrack_uri_edu/.conda/envs/nerfstudio/bin/python scripts/visualize.py \
    --results_root results/ \
    --output_dir figures/
