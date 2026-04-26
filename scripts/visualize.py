"""
visualize.py
------------
Generates all figures for the paper:
  1. AUROC curves (all 4 models, all 3 datasets)
  2. F1/Precision/Recall comparison table (saved as CSV + figure)
  3. CWE-type F1 breakdown bar chart
  4. Attention weight heatmap for a selected function
  5. Training loss/F1 curves

Usage:
    python scripts/visualize.py --results_root results/ --output_dir figures/

    # Attention heatmap for a specific function
    python scripts/visualize.py --attention \
        --results_dir results/codebert \
        --function_file data/sample_function.txt
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from sklearn.metrics import roc_curve, auc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

MODEL_COLORS = {
    'codebert':      '#2196F3',   # blue
    'codet5':        '#4CAF50',   # green
    'graphcodebert': '#FF9800',   # orange
    'bilstm':        '#9E9E9E',   # grey
}

MODEL_LABELS = {
    'codebert':      'CodeBERT',
    'codet5':        'CodeT5',
    'graphcodebert': 'GraphCodeBERT',
    'bilstm':        'BiLSTM (baseline)',
}

DATASET_LABELS = {
    'bigvul_test': 'BigVul (in-domain)',
    'devign_test': 'Devign (ext. C/C++)',
    'd2a_test':    'D2A (Java, zero-shot)',
}

plt.rcParams.update({
    'font.family':  'DejaVu Sans',
    'font.size':    11,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})


# ---------------------------------------------------------------------------
# 1. AUROC curves
# ---------------------------------------------------------------------------

def plot_auroc_curves(results_root: str, output_dir: str):
    """One subplot per dataset, one line per model."""
    models   = ['codebert', 'codet5', 'graphcodebert', 'bilstm']
    datasets = ['bigvul_test', 'devign_test', 'd2a_test']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, ds in zip(axes, datasets):
        for model in models:
            pred_path = os.path.join(results_root, model, f'{ds}_preds.json')
            if not os.path.exists(pred_path):
                continue
            with open(pred_path) as f:
                data = json.load(f)

            fpr, tpr, _ = roc_curve(data['labels'], data['probs'])
            roc_auc     = auc(fpr, tpr)
            ax.plot(fpr, tpr,
                    color=MODEL_COLORS[model],
                    label=f"{MODEL_LABELS[model]} (AUC={roc_auc:.3f})",
                    linewidth=2)

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.legend(fontsize=8, loc='lower right')

    fig.suptitle('AUROC Curves — Vulnerability Detection', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(output_dir, 'auroc_curves.pdf')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  [saved] {out}")


# ---------------------------------------------------------------------------
# 2. F1 comparison table & figure
# ---------------------------------------------------------------------------

def plot_f1_comparison(results_root: str, output_dir: str):
    """Bar chart comparing F1 across models and datasets."""
    models   = ['codebert', 'codet5', 'graphcodebert', 'bilstm']
    datasets = ['bigvul_test', 'devign_test', 'd2a_test']

    data = {}
    for model in models:
        summary_path = os.path.join(results_root, model, 'metrics_summary.json')
        if not os.path.exists(summary_path):
            continue
        with open(summary_path) as f:
            metrics = json.load(f)
        data[model] = {m['dataset']: m for m in metrics}

    # CSV table
    rows = []
    for model in models:
        if model not in data:
            continue
        row = {'Model': MODEL_LABELS[model]}
        for ds in datasets:
            if ds in data[model]:
                row[DATASET_LABELS[ds] + ' F1']  = data[model][ds]['f1']
                row[DATASET_LABELS[ds] + ' AUROC'] = data[model][ds]['auroc']
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'metrics_table.csv')
    df.to_csv(csv_path, index=False)
    print(f"  [saved] {csv_path}")

    # Bar chart (F1 only)
    x      = np.arange(len(datasets))
    width  = 0.18
    fig, ax = plt.subplots(figsize=(11, 5))

    for i, model in enumerate(models):
        if model not in data:
            continue
        f1s = [data[model].get(ds, {}).get('f1', 0) for ds in datasets]
        ax.bar(x + i * width, f1s,
               width=width,
               color=MODEL_COLORS[model],
               label=MODEL_LABELS[model])

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([DATASET_LABELS[d] for d in datasets])
    ax.set_ylabel('F1 Score')
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.set_title('F1 Score Comparison Across Models and Datasets')
    ax.legend()
    plt.tight_layout()

    out = os.path.join(output_dir, 'f1_comparison.pdf')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  [saved] {out}")


# ---------------------------------------------------------------------------
# 3. CWE breakdown
# ---------------------------------------------------------------------------

def plot_cwe_breakdown(results_root: str, output_dir: str, top_n: int = 15):
    """
    Bar chart of per-CWE F1 for the best model (codet5 or codebert — whichever you specify).
    Compares CodeBERT vs CodeT5 side by side.
    """
    models_to_compare = ['codebert', 'codet5']
    cwe_data = {}

    for model in models_to_compare:
        path = os.path.join(results_root, model, 'cwe_breakdown.json')
        if not os.path.exists(path):
            continue
        with open(path) as f:
            cwe_data[model] = {row['cwe']: row for row in json.load(f)}

    if not cwe_data:
        print("  [skip] No CWE breakdown files found.")
        return

    # Get top N CWEs by sample count (from first available model)
    ref_model = list(cwe_data.keys())[0]
    top_cwes  = sorted(cwe_data[ref_model].values(), key=lambda x: -x['n'])[:top_n]
    top_cwe_names = [r['cwe'] for r in top_cwes]

    x     = np.arange(len(top_cwe_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 5))

    for i, model in enumerate(models_to_compare):
        if model not in cwe_data:
            continue
        f1s = [cwe_data[model].get(cwe, {}).get('f1', 0) for cwe in top_cwe_names]
        ax.bar(x + i * width, f1s, width, label=MODEL_LABELS[model],
               color=MODEL_COLORS[model])

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(top_cwe_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('F1 Score')
    ax.set_ylim(0, 1.0)
    ax.set_title(f'Per-CWE F1: CodeBERT vs CodeT5 (Top {top_n} by sample count)')
    ax.legend()
    plt.tight_layout()

    out = os.path.join(output_dir, 'cwe_breakdown.pdf')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  [saved] {out}")


# ---------------------------------------------------------------------------
# 4. Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(results_root: str, output_dir: str):
    """Loss and val F1 over epochs for each model."""
    models = ['codebert', 'codet5', 'graphcodebert', 'bilstm']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for model in models:
        hist_path = os.path.join(results_root, model, 'train_history.json')
        if not os.path.exists(hist_path):
            continue
        with open(hist_path) as f:
            hist = json.load(f)

        epochs  = [h['epoch']   for h in hist]
        losses  = [h['loss']    for h in hist]
        val_f1s = [h['val_f1']  for h in hist]

        ax1.plot(epochs, losses,  color=MODEL_COLORS[model], label=MODEL_LABELS[model], linewidth=2)
        ax2.plot(epochs, val_f1s, color=MODEL_COLORS[model], label=MODEL_LABELS[model], linewidth=2)

    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss'); ax1.legend(fontsize=8)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Validation F1')
    ax2.set_title('Validation F1'); ax2.legend(fontsize=8)
    ax2.set_ylim(0, 1.0)

    plt.tight_layout()
    out = os.path.join(output_dir, 'training_curves.pdf')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  [saved] {out}")


# ---------------------------------------------------------------------------
# 5. Attention heatmap
# ---------------------------------------------------------------------------

def plot_attention_heatmap(results_dir: str, function_file: str, output_dir: str):
    """
    Visualize attention weights from CodeBERT/GraphCodeBERT on a single function.
    Requires the model to be loaded and attention outputs captured.
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    with open(os.path.join(results_dir, 'meta.json')) as f:
        meta = json.load(f)

    if meta['is_bilstm'] or meta['use_t5']:
        print("  [skip] Attention heatmap only supported for BERT-based models.")
        return

    device    = torch.device('cpu')  # heatmaps are done locally
    hf_name   = meta['hf_name']
    tokenizer = AutoTokenizer.from_pretrained(hf_name)

    with open(function_file) as f:
        code = f.read()

    enc = tokenizer(code, return_tensors='pt', max_length=512,
                    truncation=True, padding='max_length')
    tokens = tokenizer.convert_ids_to_tokens(enc['input_ids'][0])

    # Load model with output_attentions=True
    base_model = AutoModel.from_pretrained(hf_name, output_attentions=True)
    base_model.eval()

    with torch.no_grad():
        out = base_model(**enc)

    # Mean attention across heads, last layer
    # attentions: tuple of (batch, heads, seq, seq) per layer
    attn = out.attentions[-1]          # last layer: (1, heads, 512, 512)
    attn = attn[0].mean(dim=0)         # (512, 512) — mean over heads
    attn = attn.numpy()

    # Only show first N non-padding tokens
    non_pad = (enc['attention_mask'][0] == 1).sum().item()
    n_show  = min(non_pad, 40)
    attn    = attn[:n_show, :n_show]
    labels  = tokens[:n_show]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(attn, cmap='Blues', aspect='auto')
    ax.set_xticks(range(n_show))
    ax.set_yticks(range(n_show))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    plt.colorbar(im, ax=ax)
    ax.set_title(f'Attention Weights — {meta["model"]} (last layer, mean heads)')

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'attention_heatmap_{meta["model"]}.pdf')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"  [saved] {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_root', default='results/')
    parser.add_argument('--output_dir',   default='figures/')
    # Attention heatmap mode
    parser.add_argument('--attention',     action='store_true')
    parser.add_argument('--results_dir',   default=None)
    parser.add_argument('--function_file', default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.attention:
        assert args.results_dir and args.function_file, \
            '--results_dir and --function_file required for attention heatmap'
        plot_attention_heatmap(args.results_dir, args.function_file, args.output_dir)
        return

    print("[visualize] Generating all figures...")
    plot_auroc_curves(args.results_root, args.output_dir)
    plot_f1_comparison(args.results_root, args.output_dir)
    plot_cwe_breakdown(args.results_root, args.output_dir)
    plot_training_curves(args.results_root, args.output_dir)
    print("\n✓ All figures saved.")


if __name__ == '__main__':
    main()
