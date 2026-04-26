# Vulnerability Detection: Pre-Training Coverage & Cross-Language Transfer

CSC 561 Final Project — University of Rhode Island

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
conda create -n vuln-env python=3.10
conda activate vuln-env
pip install -r requirements.txt
```

---

## Step 1 — Download Datasets

### BigVul
```bash
# Download from GitHub releases
wget https://github.com/ZeoVan/MSR_20_Code_vulnerability_Search_DP/releases/download/v2.0/MSR_data_cleaned.csv \
     -O data/raw/bigvul.csv
```

### Devign
```bash
# From CodeXGLUE benchmark
wget https://raw.githubusercontent.com/microsoft/CodeXGLUE/main/Code-Code/Defect-detection/dataset/train.jsonl \
     -O data/raw/devign_train.jsonl
# Combine all splits into one file, or use the full dataset JSON
```

### D2A
```bash
# IBM Research D2A dataset
# See: https://github.com/ibm/D2A for download instructions
# Place the JSON file at: data/raw/d2a.json
```

---

## Step 2 — Preprocess

```bash
python scripts/preprocess.py \
    --bigvul_path data/raw/bigvul.csv \
    --devign_path data/raw/devign.json \
    --d2a_path    data/raw/d2a.json \
    --output_dir  data/processed
```

Outputs:
- `data/processed/bigvul_train.jsonl`
- `data/processed/bigvul_val.jsonl`
- `data/processed/bigvul_test.jsonl`
- `data/processed/devign_test.jsonl`
- `data/processed/d2a_test.jsonl`

---

## Step 3 — Test Pipeline Locally (small sample)

Before submitting to Unity, verify the pipeline runs end-to-end:

```bash
# Quick smoke test: 500 training samples, 1 epoch
head -n 500 data/processed/bigvul_train.jsonl > data/processed/bigvul_train_small.jsonl
head -n 100 data/processed/bigvul_val.jsonl   > data/processed/bigvul_val_small.jsonl

python scripts/train.py --model codebert \
    --data_dir data/processed \  # update VulnDataset paths to _small files temporarily
    --epochs 1 --batch_size 4
```

---

## Step 4 — Train on Unity

```bash
mkdir -p logs results

sbatch scripts/slurm/train_codebert.sh
sbatch scripts/slurm/train_codet5.sh
sbatch scripts/slurm/train_graphcodebert.sh
# BiLSTM is in train_others.sh — split it into its own file first
```

Monitor jobs:
```bash
squeue -u <your_username>
tail -f logs/codebert_<jobid>.out
```

---

## Step 5 — Evaluate

```bash
# Evaluate each model
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

# Attention heatmap (pick a flagged vulnerable function)
python scripts/visualize.py --attention \
    --results_dir   results/codebert \
    --function_file data/sample_function.txt
```

---

## Notes

- **Class imbalance**: BigVul is ~5-10% vulnerable. `pos_weight` in BCELoss handles this automatically.
- **CodeT5**: Uses `T5EncoderModel` with mean pooling (no CLS token). This is handled in `train.py`.
- **GraphCodeBERT**: Used here as a RoBERTa-style encoder. Full data-flow graph integration would require custom preprocessing not in scope.
- **D2A**: Zero-shot only — never seen during training or validation.
