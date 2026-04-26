"""
preprocess.py
-------------
Prepares BigVul, Devign, and CVEFixes datasets for vulnerability detection experiments.

Usage:
    python preprocess.py --bigvul_path   data/raw/MSR_data_cleaned.json \
                         --devign_path   data/raw/devign.json \
                         --cvefixes_path data/raw/cvefixes_full_dataset.jsonl \
                         --output_dir    data/processed
"""

import argparse
import json
import re
import os
import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Comment removal (C/C++/Java)
# ---------------------------------------------------------------------------

def remove_comments(code: str) -> str:
    """Remove single-line (//) and block (/* */) comments from source code."""
    # Block comments first
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    # Single-line comments
    code = re.sub(r'//[^\n]*', '', code)
    # Collapse extra blank lines left behind
    code = re.sub(r'\n\s*\n', '\n', code)
    return code.strip()


# ---------------------------------------------------------------------------
# BigVul
# ---------------------------------------------------------------------------

def load_bigvul(path: str) -> pd.DataFrame:
    """
    Load BigVul from MSR_data_cleaned.json.
    Format: nested dict {"0": {record}, "1": {record}, ...}
    Relevant columns: 'func_before' (source code), 'vul' (1=vulnerable, 0=safe),
    'CWE ID' (for CWE breakdown). File is ~11GB — be patient.
    """
    print(f"[BigVul] Loading from {path} (this may take several minutes)...")

    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    # raw is {"0": {...}, "1": {...}, ...} — convert values to list of records
    records = list(raw.values())
    df = pd.DataFrame(records)
    print(f"[BigVul] Loaded {len(df)} records. Columns sample: {df.columns.tolist()[:10]}...")

    # Keep only the columns we need
    required = ['func_before', 'vul']
    optional = ['CWE ID', 'CVE ID']
    keep = required + [c for c in optional if c in df.columns]
    df = df[keep].copy()
    df.rename(columns={'func_before': 'code', 'vul': 'label', 'CWE ID': 'cwe'}, inplace=True)

    # Drop rows with missing code or label
    before = len(df)
    df.dropna(subset=['code', 'label'], inplace=True)
    df = df[df['code'].str.strip().astype(bool)]
    print(f"[BigVul] Dropped {before - len(df)} rows with missing data. Remaining: {len(df)}")

    df['label'] = df['label'].astype(int)
    df['code'] = df['code'].apply(remove_comments)

    pos = df['label'].sum()
    print(f"[BigVul] Vulnerable: {pos} ({100*pos/len(df):.1f}%)  |  Total: {len(df)}")
    return df


def split_bigvul(df: pd.DataFrame):
    """Stratified 70 / 15 / 15 split."""
    train, temp = train_test_split(df, test_size=0.30, stratify=df['label'], random_state=42)
    val, test   = train_test_split(temp, test_size=0.50, stratify=temp['label'], random_state=42)
    print(f"[BigVul] Split → train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


# ---------------------------------------------------------------------------
# Devign
# ---------------------------------------------------------------------------

def load_devign(path: str) -> pd.DataFrame:
    """
    Load Devign JSON.
    Each record: { "func": <source>, "target": 0|1, "project": ..., "commit_id": ... }
    Held out entirely — used only for external C/C++ generalization testing.
    """
    print(f"[Devign] Loading from {path}")
    with open(path) as f:
        records = json.load(f)

    df = pd.DataFrame(records)
    df.rename(columns={'func': 'code', 'target': 'label'}, inplace=True)
    df = df[['code', 'label']].dropna()
    df = df[df['code'].str.strip().astype(bool)]
    df['label'] = df['label'].astype(int)
    df['code'] = df['code'].apply(remove_comments)

    pos = df['label'].sum()
    print(f"[Devign] Vulnerable: {pos} ({100*pos/len(df):.1f}%)  |  Total: {len(df)}")
    return df


# ---------------------------------------------------------------------------
# CVEFixes
# ---------------------------------------------------------------------------

def load_cvefixes(path: str) -> pd.DataFrame:
    """
    Load CVEFixes dataset as external C/C++ generalization test set.
    Format: JSONL with columns 'func' (source code) and 'target' (0/1 label).
    Held out entirely — never seen during training or validation.

    Dataset: ~16k real-world C/C++ functions from CVE fix commits.
    Label distribution: ~59% safe (0), ~41% vulnerable (1).
    Source: https://huggingface.co/datasets/hitoshura25/cvefixes
    """
    print(f"[CVEFixes] Loading from {path}")

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    df = pd.DataFrame(records)

    # Columns are 'func' and 'target' — same schema as Devign
    df.rename(columns={'func': 'code', 'target': 'label'}, inplace=True)

    df = df[['code', 'label']].dropna()
    df = df[df['code'].str.strip().astype(bool)]
    df['label'] = df['label'].astype(int)

    pos = df['label'].sum()
    print(f"[CVEFixes] Vulnerable: {pos} ({100*pos/len(df):.1f}%)  |  Total: {len(df)}")
    return df


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_split(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_json(path, orient='records', lines=True)
    print(f"  Saved {len(df)} records → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bigvul_path',   required=True)
    parser.add_argument('--devign_path',   required=True)
    parser.add_argument('--cvefixes_path', required=True)
    parser.add_argument('--output_dir',    default='data/processed')
    args = parser.parse_args()

    out = args.output_dir

    # BigVul
    bigvul = load_bigvul(args.bigvul_path)
    train, val, test = split_bigvul(bigvul)
    save_split(train, f'{out}/bigvul_train.jsonl')
    save_split(val,   f'{out}/bigvul_val.jsonl')
    save_split(test,  f'{out}/bigvul_test.jsonl')

    # Devign (held out, no splitting)
    devign = load_devign(args.devign_path)
    save_split(devign, f'{out}/devign_test.jsonl')

    # CVEFixes (held out, no splitting)
    cvefixes = load_cvefixes(args.cvefixes_path)
    save_split(cvefixes, f'{out}/cvefixes_test.jsonl')

    print("\n✓ Preprocessing complete.")


if __name__ == '__main__':
    main()