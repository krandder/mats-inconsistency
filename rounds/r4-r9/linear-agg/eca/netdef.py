"""The fixed small network: a pre-LN transformer over the 25 cell tokens (identical in every arm).

Token for cell i = position embedding + evidence state (none / reported miss / reported hit)
+ reveal state (hidden / revealed empty / revealed occupied; used only by arms that condition on
true cells) + query flag (used only by the cell-by-cell arm). Attention gets a learned 2-D relative
position bias. Heads (all built in every arm so the backbone init is identical per seed):
  marg  h_j -> logit P(x_j = 1 | input)                                    (single-cell readout)
  k1    relu(A h_j + B h_p) -> [parent score, logit | x_p=0, logit | x_p=1]  for every ordered (j, p)
  k2    relu(A h_j + B h_p1 + C h_p2) -> [score, 4 logits | v = x_p1 + 2 x_p2] for every j and pair p1<p2
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NC = 25
W = 5
PAIRS = torch.tensor([(a, b) for a in range(NC) for b in range(a + 1, NC)], dtype=torch.long)  # [300,2]


def rel_index() -> torch.Tensor:
    r = torch.arange(NC) // W
    c = torch.arange(NC) % W
    dy = r[:, None] - r[None, :] + (W - 1)
    dx = c[:, None] - c[None, :] + (W - 1)
    return dy * (2 * W - 1) + dx  # [25,25] in [0,81)


class Block(nn.Module):
    def __init__(self, d, h, ff, drop):
        super().__init__()
        self.h, self.drop = h, drop
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ff1, self.ff2 = nn.Linear(d, ff), nn.Linear(ff, d)

    def forward(self, x, bias):
        B, n, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, n, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=self.drop if self.training else 0.0)
        x = x + F.dropout(self.proj(a.transpose(1, 2).reshape(B, n, d)), self.drop, self.training)
        x = x + F.dropout(self.ff2(F.gelu(self.ff1(self.ln2(x)))), self.drop, self.training)
        return x


class Net(nn.Module):
    def __init__(self, d=128, layers=4, heads=4, ff=256, drop=0.1, m=64):
        super().__init__()
        self.d, self.layers, self.heads, self.ff, self.m = d, layers, heads, ff, m
        self.pos = nn.Parameter(torch.randn(NC, d) * 0.02)
        self.ev = nn.Embedding(3, d)
        self.rev = nn.Embedding(3, d)
        self.qf = nn.Embedding(2, d)
        self.relb = nn.Parameter(torch.zeros(layers, heads, (2 * W - 1) ** 2))
        self.register_buffer("relidx", rel_index(), persistent=False)
        self.register_buffer("pairs", PAIRS.clone(), persistent=False)
        self.blocks = nn.ModuleList([Block(d, heads, ff, drop) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d)
        self.marg = nn.Linear(d, 1)
        self.k1a, self.k1b, self.k1o = nn.Linear(d, m), nn.Linear(d, m, bias=False), nn.Linear(m, 3)
        self.k2a, self.k2b, self.k2c = nn.Linear(d, m), nn.Linear(d, m, bias=False), nn.Linear(d, m, bias=False)
        self.k2o = nn.Linear(m, 5)
        for e in (self.ev, self.rev, self.qf):
            nn.init.normal_(e.weight, std=0.02)

    def backbone(self, ev, rev=None, q=None):
        """ev, rev, q: [B,25] long. Returns h [B,25,d]."""
        x = self.pos[None] + self.ev(ev)
        if rev is not None:
            x = x + self.rev(rev)
        else:
            x = x + self.rev.weight[0]
        if q is not None:
            x = x + self.qf(q)
        else:
            x = x + self.qf.weight[0]
        for li, blk in enumerate(self.blocks):
            bias = self.relb[li][:, self.relidx]  # [h,25,25]
            x = blk(x, bias)
        return self.lnf(x)

    def marg_logits(self, h):
        return self.marg(h).squeeze(-1)  # [B,25]

    def k1_table(self, h):
        """[B,25(j),25(p),3]: score, logit(x_j=1 | x_p=0), logit(x_j=1 | x_p=1)."""
        a, b = self.k1a(h), self.k1b(h)
        return self.k1o(F.relu(a[:, :, None, :] + b[:, None, :, :]))

    def k2_table(self, h):
        """[B,25(j),300(pair),5]: score, logits for v = x_p1 + 2 x_p2 (pair = (p1<p2) by cell index)."""
        a, b, c = self.k2a(h), self.k2b(h), self.k2c(h)
        pb = b[:, self.pairs[:, 0]] + c[:, self.pairs[:, 1]]  # [B,300,m]
        return self.k2o(F.relu(a[:, :, None, :] + pb[:, None, :, :]))

    def flops_backbone(self) -> float:
        """Forward FLOPs per sequence (25 tokens), matmuls + attention, embeddings ignored."""
        d, L, ff = self.d, self.layers, self.ff
        per_layer = 2 * NC * (d * 3 * d + d * d + 2 * d * ff) + 2 * 2 * NC * NC * d
        return float(L * per_layer + 2 * NC * d)

    def flops_k1(self) -> float:
        return float(2 * NC * self.d * self.m * 2 + NC * NC * (self.m + 2 * self.m * 3))

    def flops_k2(self) -> float:
        P = PAIRS.shape[0]
        return float(2 * NC * self.d * self.m * 3 + NC * P * (2 * self.m + 2 * self.m * 5))
