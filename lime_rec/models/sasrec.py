"""SASRec model components shared by training and evaluation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SASRec(nn.Module):
    """Compact self-attentive sequential recommender."""

    def __init__(self, num_items: int, hidden: int, maxlen: int,
                 num_layers: int, num_heads: int, dropout: float,
                 architecture: str = "legacy"):
        super().__init__()
        self.num_items = num_items
        self.maxlen = maxlen
        self.item_emb = nn.Embedding(num_items + 1, hidden, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden)
        self.emb_dropout = nn.Dropout(dropout)
        self.architecture = architecture
        block = PaperSASBlock if architecture == "paper" else SASBlock
        self.layers = nn.ModuleList([
            block(hidden, num_heads, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.item_emb.weight[1:])
        nn.init.xavier_uniform_(self.pos_emb.weight)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """Encode 1-indexed item sequences shaped ``[batch, length]``."""
        batch_size, length = seq.shape
        positions = torch.arange(length, device=seq.device).unsqueeze(0)
        item_embeddings = self.item_emb(seq)
        if self.architecture == "paper":
            item_embeddings = item_embeddings * math.sqrt(item_embeddings.shape[-1])
        x = self.emb_dropout(item_embeddings + self.pos_emb(positions))
        causal = torch.triu(
            torch.ones(length, length, device=seq.device), diagonal=1
        ).bool()
        pad_mask = seq == 0
        x = x * (~pad_mask).unsqueeze(-1).float()
        for layer in self.layers:
            x = layer(x, pad_mask, causal)
            x = x * (~pad_mask).unsqueeze(-1).float()
        return self.norm(x)

    def score_all(self, seq: torch.Tensor) -> torch.Tensor:
        """Return full-softmax logits for every sequence position."""
        return self.forward(seq) @ self.item_emb.weight.T

    def predict(self, seq: torch.Tensor) -> torch.Tensor:
        """Return full-catalog logits for the final sequence position."""
        return self.forward(seq)[:, -1, :] @ self.item_emb.weight.T


class SASBlock(nn.Module):
    """Pre-normalized causal attention block used by :class:`SASRec`."""

    def __init__(self, hidden: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask, causal_mask):
        hidden = self.norm1(x)
        batch_size, length, _ = hidden.shape
        combined = causal_mask.unsqueeze(0).expand(batch_size, -1, -1).clone()
        combined |= pad_mask.unsqueeze(1).expand(-1, length, -1)
        diagonal = torch.eye(
            length, device=x.device, dtype=torch.bool
        ).unsqueeze(0)
        combined &= ~diagonal
        combined = combined.unsqueeze(1).expand(
            -1, self.attn.num_heads, -1, -1
        ).reshape(batch_size * self.attn.num_heads, length, length)
        attended, _ = self.attn(hidden, hidden, hidden, attn_mask=combined)
        x = x + self.drop(attended)
        return x + self.ffn(self.norm2(x))


class PaperSASBlock(nn.Module):
    """Block used by the original ICDM'18 SASRec implementation."""

    def __init__(self, hidden: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.conv1 = nn.Conv1d(hidden, hidden, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, pad_mask, causal_mask):
        queries = self.norm1(x)
        batch_size, length, _ = queries.shape
        combined = causal_mask.unsqueeze(0).expand(batch_size, -1, -1).clone()
        combined |= pad_mask.unsqueeze(1).expand(-1, length, -1)
        diagonal = torch.eye(length, device=x.device, dtype=torch.bool).unsqueeze(0)
        combined &= ~diagonal
        combined = combined.unsqueeze(1).expand(
            -1, self.attn.num_heads, -1, -1
        ).reshape(batch_size * self.attn.num_heads, length, length)
        attended, _ = self.attn(queries, x, x, attn_mask=combined)
        x = queries + attended
        residual = x
        hidden = self.norm2(x).transpose(1, 2)
        hidden = self.dropout1(torch.relu(self.conv1(hidden)))
        hidden = self.dropout2(self.conv2(hidden)).transpose(1, 2)
        return residual + hidden
