import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)
from statsmodels.stats.contingency_tables import mcnemar

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.bilstm import BiLSTMClassifier
from scripts.train import TransformerClassifier, VulnDataset



# Inference
def run_inference(model, loader, device):
    """Returns (probabilities, binary predictions, true labels)."""
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for batch in loader:
            ids   = batch['input_ids'].to(device)
            mask  = batch['attention_mask'].to(device)
            probs = model(ids, mask).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            labels = batch['label'].long().numpy()

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    return np.array(all_probs), np.array(all_preds), np.array(all_labels)


def compute_metrics(probs, preds, labels, dataset_name: str) -> dict:
    f1    = f1_score(labels, preds, zero_division=0)
    pre   = precision_score(labels, preds, zero_division=0)
    rec   = recall_score(labels, preds, zero_division=0)
    # AUROC requires at least one positive sample
    try:
        auroc = roc_auc_score(labels, probs)
    except ValueError:
        auroc = float('nan')

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    result = {
        'dataset':   dataset_name,
        'f1':        round(f1,    4),
        'precision': round(pre,   4),
        'recall':    round(rec,   4),
        'auroc':     round(auroc, 4),
        'tp': int(tp), 'fp': int(fp),
        'tn': int(tn), 'fn': int(fn),
    }
    print(f"  [{dataset_name}] F1={f1:.4f} | P={pre:.4f} | R={rec:.4f} | AUROC={auroc:.4f}")
    return result



# CWE breakdown
def cwe_breakdown(preds, labels, cwe_ids, output_path: str):
    """
    Compute per-CWE F1 on BigVul test set.
    cwe_ids: list of CWE strings (or NaN) aligned with preds/labels.
    """
    import pandas as pd

    df = pd.DataFrame({'pred': preds, 'label': labels, 'cwe': cwe_ids})
    df['cwe'] = df['cwe'].fillna('Unknown')

    rows = []
    for cwe, group in df.groupby('cwe'):
        if len(group) < 10:   # skip tiny groups
            continue
        f1 = f1_score(group['label'], group['pred'], zero_division=0)
        rows.append({'cwe': cwe, 'f1': round(f1, 4), 'n': len(group),
                     'n_vuln': int(group['label'].sum())})

    rows.sort(key=lambda x: -x['n'])
    with open(output_path, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f"  CWE breakdown saved → {output_path}  ({len(rows)} CWE types)")



# McNemar's test
def run_mcnemar(preds_a_path: str, preds_b_path: str):
    """
    McNemar's test comparing two models on the same test set.
    Tests whether the difference in errors is statistically significant.
    """
    with open(preds_a_path) as f:
        a = json.load(f)
    with open(preds_b_path) as f:
        b = json.load(f)

    preds_a = np.array(a['preds'])
    preds_b = np.array(b['preds'])
    labels  = np.array(a['labels'])   # must be same test set

    correct_a = (preds_a == labels)
    correct_b = (preds_b == labels)

    # Contingency table
    # n00: both wrong | n01: A wrong B right | n10: A right B wrong | n11: both right
    n00 = int(((~correct_a) & (~correct_b)).sum())
    n01 = int(((~correct_a) &   correct_b ).sum())
    n10 = int((  correct_a  & (~correct_b)).sum())
    n11 = int((  correct_a  &   correct_b ).sum())

    table = [[n11, n10], [n01, n00]]
    result = mcnemar(table, exact=False, correction=True)

    print(f"\n[McNemar's Test]")
    print(f"  Model A: {a.get('model', preds_a_path)}")
    print(f"  Model B: {b.get('model', preds_b_path)}")
    print(f"  Contingency: both_correct={n11} | A_only={n10} | B_only={n01} | both_wrong={n00}")
    print(f"  Chi-squared={result.statistic:.4f}  p-value={result.pvalue:.4f}")
    if result.pvalue < 0.05:
        print("  → Statistically significant difference (p < 0.05)")
    else:
        print("  → No statistically significant difference (p ≥ 0.05)")

    return result



# Load model from checkpoint
def load_model(results_dir: str, device):
    with open(os.path.join(results_dir, 'meta.json')) as f:
        meta = json.load(f)

    hf_name   = meta['hf_name']
    is_bilstm = meta['is_bilstm']
    use_t5    = meta['use_t5']

    tokenizer = AutoTokenizer.from_pretrained(hf_name)

    if is_bilstm:
        model = BiLSTMClassifier(vocab_size=tokenizer.vocab_size,
                                 pad_token_id=tokenizer.pad_token_id)
    else:
        model = TransformerClassifier(hf_name, use_t5=use_t5)

    ckpt = torch.load(os.path.join(results_dir, 'best_model.pt'), map_location=device)
    model.load_state_dict(ckpt)
    model.to(device)

    return model, tokenizer, meta



# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--data_dir',    default='data/processed')
    parser.add_argument('--batch_size',  type=int, default=32)
    # McNemar mode
    parser.add_argument('--mcnemar',  action='store_true')
    parser.add_argument('--preds_a',  default=None)
    parser.add_argument('--preds_b',  default=None)
    args = parser.parse_args()

    if args.mcnemar:
        run_mcnemar(args.preds_a, args.preds_b)
        return

    assert args.results_dir, '--results_dir required for evaluation'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[eval] Device: {device}")

    model, tokenizer, meta = load_model(args.results_dir, device)
    max_length = meta['max_length']
    model_name = meta['model']

    test_sets = {
        'bigvul_test': f"{args.data_dir}/bigvul_test.jsonl",
        'devign_test': f"{args.data_dir}/devign_test.jsonl",
        'cvefixes_test': f"{args.data_dir}/cvefixes_test.jsonl",
    }

    all_results = []

    for name, path in test_sets.items():
        if not os.path.exists(path):
            print(f"  [skip] {path} not found.")
            continue

        print(f"\n[eval] Running on {name}...")
        ds     = VulnDataset(path, tokenizer, max_length)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

        probs, preds, labels = run_inference(model, loader, device)
        metrics = compute_metrics(probs, preds, labels, name)
        all_results.append(metrics)

        # Save raw predictions for McNemar's test and AUROC curves
        pred_out = {
            'model':  model_name,
            'dataset': name,
            'probs':  probs.tolist(),
            'preds':  preds.tolist(),
            'labels': labels.tolist(),
        }
        pred_path = os.path.join(args.results_dir, f'{name}_preds.json')
        with open(pred_path, 'w') as f:
            json.dump(pred_out, f)
        print(f"  Predictions saved → {pred_path}")

        # CWE breakdown for BigVul test only
        if name == 'bigvul_test':
            import pandas as pd
            df = pd.read_json(path, lines=True)
            cwe_ids = df['cwe'].tolist() if 'cwe' in df.columns else ['Unknown'] * len(preds)
            cwe_path = os.path.join(args.results_dir, 'cwe_breakdown.json')
            cwe_breakdown(preds, labels, cwe_ids, cwe_path)

    # Summary table
    summary_path = os.path.join(args.results_dir, 'metrics_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Evaluation complete. Summary → {summary_path}")


if __name__ == '__main__':
    main()
