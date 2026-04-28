# Vulnerability Detection: Pre-Training Coverage & Cross-Dataset Generalization

CSC 561 Final Project — University of Rhode Island  
Matthew Barbrack — mbarbrack@uri.edu

## Overview

This project investigates two questions:
1. Does it matter whether a transformer model was pretrained on C and C++ when fine-tuning on C and C++ vulnerability data?
2. Do models trained on one vulnerability dataset generalize to code from different codebases?

Four models are compared: CodeBERT, CodeT5, GraphCodeBERT, and a BiLSTM baseline. All are trained on BigVul and evaluated on two external held-out datasets: Devign and CVEFixes.

---

## Project Structure

```
vuln-transfer/
├── data/
│   ├── raw/              ← downloaded datasets go here
│   └── processed/        ← output of preprocess.py
├── models/
│   └── bilstm.py         ← BiLSTM architecture
├── scripts/
│   ├── preprocess.py     ← data cleaning + splitting
│   ├── train.py          ← fine-tuning all 4 models
│   ├── evaluate.py       ← metrics + McNemar's test
│   ├── visualize.py      ← all paper figures
│   └── slurm/            ← Unity HPC job scripts
├── results/              ← checkpoints + metrics (gitignored)
├── figures/              ← output PDFs for paper
├── requirements.txt
└── README.md
```

---

## Setup

### Local (development)
```bash
pip install -r requirements.txt
```

### Unity cluster
```bash
module load anaconda/2023
conda activate nerfstudio
pip install -r requirements.txt
```

> **Note:** Always use the full conda Python path in SLURM scripts rather than relying on conda activation. Broken `~/.local` torch installs can silently conflict with conda environments.

---

## Step 1 — Download Datasets

### BigVul (primary training dataset)
```bash
wget https://github.com/ZeoVan/MSR_20_Code_vulnerability_Search_DP/releases/download/v2.0/MSR_data_cleaned.csv \
     -O data/raw/bigvul.csv
```
188,636 C/C++ functions, 10,900 labeled vulnerable across 91 CWE types (~5.8% positive rate).

### Devign (external generalization test)
```bash
# From CodeXGLUE benchmark
wget https://raw.githubusercontent.com/microsoft/CodeXGLUE/main/Code-Code/Defect-detection/dataset/train.jsonl \
     -O data/raw/devign.jsonl
```
27,318 C functions from FFmpeg and QEMU. Never used during training or validation.

### CVEFixes (external generalization test)
```bash
# Available at: https://github.com/secureIT-project/CVEfixes
# Place the processed CSV at: data/raw/cvefixes.csv
```
16,048 C/C++ functions from CVE fix commits across diverse open source repositories. Never used during training or validation.

> **Note:** The original proposal included D2A as a cross-language Java transfer dataset. D2A became inaccessible following the deprecation of IBM DAX and was replaced by CVEFixes as a second external C/C++ generalization test.

---

## Step 2 — Preprocess

```bash
python scripts/preprocess.py \
    --bigvul_path  data/raw/bigvul.csv \
    --devign_path  data/raw/devign.jsonl \
    --cvefixes_path data/raw/cvefixes.csv \
    --output_dir   data/processed
```

Outputs:
- `data/processed/bigvul_train.jsonl`
- `data/processed/bigvul_val.jsonl`
- `data/processed/bigvul_test.jsonl`
- `data/processed/devign_test.jsonl`
- `data/processed/cvefixes_test.jsonl`

BigVul is split 70/15/15 (132,045 / 28,295 / 28,296 samples), stratified by label to preserve the positive rate across all splits.

---

## Step 3 — Test Pipeline Locally (small sample)

Before submitting to Unity, verify the pipeline runs end-to-end:

```bash
head -n 500 data/processed/bigvul_train.jsonl > data/processed/bigvul_train_small.jsonl
head -n 100 data/processed/bigvul_val.jsonl   > data/processed/bigvul_val_small.jsonl

python scripts/train.py --model codebert \
    --data_dir data/processed \
    --epochs 1 --batch_size 4
```

---

## Step 4 — Train on Unity

```bash
mkdir -p logs results

sbatch scripts/slurm/train_codebert.sh
sbatch scripts/slurm/train_codet5.sh
sbatch scripts/slurm/train_graphcodebert.sh
sbatch scripts/slurm/train_bilstm.sh
```

Monitor jobs:
```bash
squeue -u <your_username>
tail -f logs/codebert_<jobid>.out
```

All transformer models were trained on NVIDIA A100 GPUs on the Unity HPC cluster. Training time per epoch: ~50 min for CodeBERT and GraphCodeBERT, ~54 min for CodeT5, ~11 min for BiLSTM.

---

## Step 5 — Evaluate

```bash
for model in codebert codet5 graphcodebert bilstm; do
    python scripts/evaluate.py \
        --results_dir results/$model \
        --data_dir data/processed
done

# McNemar's test: CodeBERT vs CodeT5
python scripts/evaluate.py --mcnemar \
    --preds_a results/codebert/bigvul_test_preds.json \
    --preds_b results/codet5/bigvul_test_preds.json
```

---

## Step 6 — Generate Figures

```bash
python scripts/visualize.py \
    --results_root results/ \
    --output_dir   figures/
```

Produces: `auroc_curves.pdf`, `f1_comparison.pdf`, `cwe_breakdown.pdf`, `training_curves.pdf`

---

## Results Summary

### In-Domain (BigVul test set)

| Model | F1 | Precision | Recall | AUROC |
|---|---|---|---|---|
| BiLSTM baseline | 0.808 | 0.862 | 0.760 | 0.958 |
| CodeBERT | 0.835 | 0.868 | 0.804 | 0.967 |
| GraphCodeBERT | 0.838 | 0.917 | 0.771 | 0.977 |
| CodeT5 | 0.845 | 0.878 | 0.815 | 0.977 |

McNemar's test (CodeBERT vs CodeT5): χ² = 2.58, p = 0.108 — not statistically significant.

### Out-of-Domain (Devign and CVEFixes)

| Model | Devign F1 | Devign AUROC | CVEFixes F1 | CVEFixes AUROC |
|---|---|---|---|---|
| BiLSTM baseline | 0.023 | 0.544 | 0.106 | 0.511 |
| CodeBERT | 0.023 | 0.534 | 0.116 | 0.497 |
| GraphCodeBERT | 0.011 | 0.541 | 0.098 | 0.493 |
| CodeT5 | 0.011 | 0.535 | 0.103 | 0.498 |

All models collapse to near-random AUROC on both external datasets regardless of architecture or pretraining.

---

## Notes

- **Class imbalance:** BigVul is ~5.8% vulnerable. Positive class weight of 16.31 is applied in BCELoss.
- **CodeT5:** Uses `T5EncoderModel` with mean pooling (no CLS token). Handled in `train.py`.
- **GraphCodeBERT:** Used as a RoBERTa-style encoder. Full data-flow graph integration would require custom preprocessing beyond the scope of this project.
- **Devign and CVEFixes:** Held out entirely — never seen during training or validation.
- **transformers pinned to 4.38.0** to avoid huggingface_hub conflicts on Unity.
- **GraphCodeBERT DataLoader:** Uses `num_workers=0` to avoid shared memory bus errors on Unity.