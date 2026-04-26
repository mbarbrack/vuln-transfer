"""
models/bilstm.py
----------------
Bidirectional LSTM baseline for vulnerability detection.
Uses the CodeBERT tokenizer vocabulary for a fair token-level comparison.
"""

import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    """
    Embedding → BiLSTM → mean pool → classification head.
    Matches the transformer classification head structure:
        Linear → ReLU → Dropout → Linear → Sigmoid
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_token_id: int = 1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_dim * 2  # bidirectional

        # Match the transformer classification head exactly
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids:      (batch, seq_len)
            attention_mask: (batch, seq_len)  — 1 for real tokens, 0 for padding

        Returns:
            logits: (batch,)  — probabilities in [0, 1]
        """
        x = self.embedding(input_ids)           # (B, L, E)
        out, _ = self.lstm(x)                   # (B, L, 2H)

        # Mean pool over non-padding positions
        mask = attention_mask.unsqueeze(-1).float()   # (B, L, 1)
        summed = (out * mask).sum(dim=1)              # (B, 2H)
        counts = mask.sum(dim=1).clamp(min=1e-9)      # (B, 1)
        pooled = summed / counts                      # (B, 2H)

        logits = self.classifier(pooled).squeeze(-1)  # (B,)
        return logits
