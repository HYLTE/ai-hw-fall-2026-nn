"""Transformer encoder (Vaswani et al., 2017, "Attention Is All You Need") for MNIST.

Images become token sequences by splitting them into non-overlapping patches;
a learnable [CLS] token is prepended and its final state is classified.
"""
import math

import torch
from torch import nn


def scaled_dot_product_attention(q, k, v):
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V.  Shapes: (..., seq, d_k)."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    return scores.softmax(dim=-1) @ v


class MultiHeadAttention(nn.Module):
    """MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O,
    head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V).

    The h per-head projections (d_model -> d_k) are stored as one d_model -> d_model
    matrix each for Q, K, V. Bias-free, as in the paper's formula.
    """

    def __init__(self, d_model, h):
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by h"
        self.h, self.d_k = h, d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v):
        b = q.size(0)
        # Project, then split into heads: (b, seq, d_model) -> (b, h, seq, d_k).
        q, k, v = (w(x).view(b, -1, self.h, self.d_k).transpose(1, 2)
                   for w, x in ((self.w_q, q), (self.w_k, k), (self.w_v, v)))
        x = scaled_dot_product_attention(q, k, v)
        # Concatenate heads: (b, h, seq, d_k) -> (b, seq, d_model).
        x = x.transpose(1, 2).reshape(b, -1, self.h * self.d_k)
        return self.w_o(x)


class PositionwiseFeedForward(nn.Module):
    """FFN(x) = max(0, x W_1 + b_1) W_2 + b_2, applied to each position independently."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    """Self-attention and FFN sublayers, each as LayerNorm(x + Dropout(Sublayer(x)))."""

    def __init__(self, d_model, h, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x)))
        return self.norm2(x + self.dropout(self.ffn(x)))


def sinusoidal_positional_encoding(n_pos, d_model):
    """PE(pos, 2i) = sin(pos / 10000^(2i/d_model)),  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))."""
    pos = torch.arange(n_pos).unsqueeze(1)
    freq = torch.pow(10000.0, -torch.arange(0, d_model, 2) / d_model)
    pe = torch.zeros(n_pos, d_model)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq)
    return pe


def patchify(x, p):
    """(b, 1, H, W) -> (b, (H/p)(W/p), p*p): non-overlapping p x p patches in row-major order."""
    x = x.unfold(2, p, p).unfold(3, p, p)  # (b, 1, H/p, W/p, p, p)
    return x.reshape(x.size(0), -1, p * p)


class TransformerClassifier(nn.Module):
    """patches -> linear embedding, prepend [CLS] -> + positional encoding
    -> N encoder layers -> linear classifier on [CLS]."""

    def __init__(self, img_size=28, patch=4, d_model=64, h=4, d_ff=256, n_layers=4,
                 dropout=0.1, n_classes=10):
        super().__init__()
        self.patch = patch
        n_tokens = (img_size // patch) ** 2 + 1  # patches + [CLS]
        self.embed = nn.Linear(patch * patch, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.register_buffer("pe", sinusoidal_positional_encoding(n_tokens, d_model))
        self.dropout = nn.Dropout(dropout)  # applied to embeddings + positional encodings
        self.layers = nn.ModuleList(EncoderLayer(d_model, h, d_ff, dropout) for _ in range(n_layers))
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        x = self.embed(patchify(x, self.patch))
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], dim=1)
        x = self.dropout(x + self.pe)
        for layer in self.layers:
            x = layer(x)
        return self.head(x[:, 0])  # logits from the [CLS] position
