"""
train.py
--------
Fine-tunes one of four models on BigVul for binary vulnerability detection.

Usage:
    python scripts/train.py --model codebert --output_dir results/codebert
    python scripts/train.py --model codet5
    python scripts/train.py --model graphcodebert
    python scripts/train.py --model bilstm

Supported --model values: codebert | codet5 | graphcodebert | bilstm
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

from sklearn.metrics import f1_score, precision_score, recall_score

# HuggingFace
from transformers import AutoTokenizer, AutoModel, T5EncoderModel

# Local
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.bilstm import BiLSTMClassifier


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_MAP = {
    'codebert':      'microsoft/codebert-base',
    'graphcodebert': 'microsoft/graphcodebert-base',
    'codet5':        'Salesforce/codet5-base',
    'bilstm':        'microsoft/codebert-base',   # borrows tokenizer only
}

DEFAULT_CONFIG = {
    'max_length':   512,
    'batch_size':   16,
    'epochs':       10,
    'lr':           2e-5,
    'dropout':      0.3,
    'patience':     3,    # early stopping
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VulnDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int):
        self.samples = []
        with open(path) as f:
            for line in f:
                self.samples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        enc = self.tokenizer(
            item['code'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label':          torch.tensor(item['label'], dtype=torch.float),
        }


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------

class TransformerClassifier(nn.Module):
    """
    Wraps CodeBERT / GraphCodeBERT / CodeT5 with a shared classification head:
        CLS token → Linear(768, 256) → ReLU → Dropout → Linear(256, 1) → Sigmoid
    """

    def __init__(self, model_name: str, dropout: float = 0.3, use_t5: bool = False):
        super().__init__()
        if use_t5:
            self.encoder = T5EncoderModel.from_pretrained(model_name)
            hidden = self.encoder.config.d_model
        else:
            self.encoder = AutoModel.from_pretrained(model_name)
            hidden = self.encoder.config.hidden_size

        self.use_t5 = use_t5

        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_ids, attention_mask):
        if self.use_t5:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            # Mean pool over sequence (T5 has no CLS token)
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        else:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled = out.last_hidden_state[:, 0, :]   # CLS token

        return self.classifier(pooled).squeeze(-1)    # (B,)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_pos_weight(train_path: str, device) -> torch.Tensor:
    """
    Compute pos_weight for BCELoss to handle class imbalance.
    pos_weight = n_negative / n_positive
    """
    labels = []
    with open(train_path) as f:
        for line in f:
            labels.append(json.loads(line)['label'])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    weight = n_neg / max(n_pos, 1)
    print(f"[train] pos_weight = {weight:.2f}  (neg={n_neg}, pos={n_pos})")
    return torch.tensor([weight], dtype=torch.float).to(device)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            logits = model(ids, mask)
            preds = (logits > 0.5).long().cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch['label'].long().tolist())

    f1  = f1_score(all_labels, all_preds, zero_division=0)
    pre = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    return f1, pre, rec


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[train] Device: {device}")

    cfg = DEFAULT_CONFIG.copy()
    # Allow CLI overrides
    if args.epochs:    cfg['epochs']     = args.epochs
    if args.lr:        cfg['lr']         = args.lr
    if args.batch_size: cfg['batch_size'] = args.batch_size

    hf_name  = MODEL_MAP[args.model]
    use_t5   = args.model == 'codet5'
    is_bilstm = args.model == 'bilstm'

    # Tokenizer (all models share the same tokenizer interface)
    print(f"[train] Loading tokenizer: {hf_name}")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)

    # Datasets
    data_dir = args.data_dir
    train_ds = VulnDataset(f'{data_dir}/bigvul_train.jsonl', tokenizer, cfg['max_length'])
    val_ds   = VulnDataset(f'{data_dir}/bigvul_val.jsonl',   tokenizer, cfg['max_length'])

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False, num_workers=2)

    # Model
    if is_bilstm:
        model = BiLSTMClassifier(
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.pad_token_id,
            dropout=cfg['dropout'],
        )
    else:
        model = TransformerClassifier(hf_name, dropout=cfg['dropout'], use_t5=use_t5)

    model.to(device)

    # Loss — weighted BCE for imbalance
    pos_weight = compute_pos_weight(f'{data_dir}/bigvul_train.jsonl', device)
    criterion  = nn.BCELoss(weight=pos_weight)   # logits already sigmoid'd

    optimizer = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=0.01)
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1,
                         total_iters=cfg['epochs'])

    os.makedirs(args.output_dir, exist_ok=True)
    best_f1, patience_count = 0.0, 0
    history = []

    print(f"\n[train] Starting training: model={args.model}, epochs={cfg['epochs']}")
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            ids    = batch['input_ids'].to(device)
            mask   = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits = model(ids, mask)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        val_f1, val_pre, val_rec = evaluate(model, val_loader, device)
        avg_loss = epoch_loss / len(train_loader)
        elapsed  = time.time() - t0

        print(f"  Epoch {epoch:02d} | loss={avg_loss:.4f} | "
              f"val_F1={val_f1:.4f} | val_P={val_pre:.4f} | val_R={val_rec:.4f} | "
              f"time={elapsed:.0f}s")

        history.append({'epoch': epoch, 'loss': avg_loss,
                        'val_f1': val_f1, 'val_precision': val_pre, 'val_recall': val_rec})

        # Save best checkpoint
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_count = 0
            ckpt_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ New best F1={best_f1:.4f} — checkpoint saved.")
        else:
            patience_count += 1
            if patience_count >= cfg['patience']:
                print(f"  Early stopping after {epoch} epochs (no improvement for {cfg['patience']} epochs).")
                break

    # Save training history
    with open(os.path.join(args.output_dir, 'train_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # Save config + model name for evaluate.py to use
    meta = {'model': args.model, 'hf_name': hf_name,
            'use_t5': use_t5, 'is_bilstm': is_bilstm,
            'max_length': cfg['max_length'], 'best_val_f1': best_f1}
    with open(os.path.join(args.output_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ Training complete. Best val F1: {best_f1:.4f}")
    print(f"  Outputs saved to: {args.output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      required=True,
                        choices=['codebert', 'codet5', 'graphcodebert', 'bilstm'])
    parser.add_argument('--data_dir',   default='data/processed')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--epochs',     type=int,   default=None)
    parser.add_argument('--lr',         type=float, default=None)
    parser.add_argument('--batch_size', type=int,   default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f'results/{args.model}'

    train(args)
