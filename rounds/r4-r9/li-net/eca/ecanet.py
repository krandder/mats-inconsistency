"""E-CA any-order network (eurisko, 2026-10-15). Reuses smallnet's netdef.Block (pre-LN transformer block) unchanged.

Tokens = the 48 PERMUTED cell ids (learned position embeddings, no relative/2-D bias: the model is not told the
topology). Input per token: MASK / revealed 0 / revealed 1. Heads:
  marg   h_i -> logit q(x_i = 1 | x_S)                       (Bernoulli for every cell; scored on masked cells)
  stiff  h_i (detached) -> s_hat(i | S) = predicted held-out surprise -log q(x_i | x_S)   (Mechanism A)
         beta_int(i | S) = max(0, 1/(s_hat + e) - 1/(ln 2 + e))          (design v2 change 5, stiffness floor)
  key    k(i | S) = normalise(W_K [sum_{j in S} alpha_{i->j} pos_emb(j) ; h_i])   (design v2 change 4; alpha = last-layer
         attention of token i, averaged over heads, restricted to revealed sources j in S)
The stiffness and key heads are fitted after training on a held-out calibration split with the backbone frozen.
"""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

SMALLNET = os.environ.get("SMALLNET_DIR", "/home/agent/workspaces/eurisko-study/li-net/smallnet")
if SMALLNET not in sys.path:
    sys.path.insert(0, SMALLNET)
from netdef import Block  # noqa: E402  (smallnet's block, unchanged)

LN2 = math.log(2.0)
STIFF_EPS = 0.01


class Net(nn.Module):
    def __init__(self, n=48, d=128, layers=4, heads=4, ff=256, drop=0.1, kdim=16, sh=64):
        super().__init__()
        self.n, self.d, self.layers, self.heads, self.ff = n, d, layers, heads, ff
        self.pos = nn.Parameter(torch.randn(n, d) * 0.02)
        self.val = nn.Embedding(3, d)             # 0 = MASK, 1 = revealed 0, 2 = revealed 1
        nn.init.normal_(self.val.weight, std=0.02)
        self.blocks = nn.ModuleList([Block(d, heads, ff, drop) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d)
        self.marg = nn.Linear(d, 1)
        self.stiff = nn.Sequential(nn.Linear(d, sh), nn.GELU(), nn.Linear(sh, sh), nn.GELU(), nn.Linear(sh, 1))
        self.keyW = nn.Linear(2 * d, kdim)
        self.keyW2 = nn.Linear(d, kdim)          # round 2 (R2b) key: LayerNormed source-attention fingerprint only, no h_i
        self.key_mode = "v1"

    def tokens(self, rev, val):
        """rev bool [B,n], val long/uint8 [B,n] -> input ids (0 mask, 1+value revealed)."""
        return torch.where(rev, 1 + val.long(), torch.zeros_like(val, dtype=torch.long))

    def backbone(self, v, need_attn=False):
        """v [B,n] long -> h [B,n,d] (and head-averaged last-layer attention [B,n,n] if need_attn)."""
        x = self.pos[None] + self.val(v)
        attn = None
        for li, blk in enumerate(self.blocks):
            if need_attn and li == len(self.blocks) - 1:
                B, n, d = x.shape
                hd = d // blk.h
                q, k, _ = blk.qkv(blk.ln1(x)).view(B, n, 3, blk.h, hd).permute(2, 0, 3, 1, 4)
                attn = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(hd), -1).mean(1)
            x = blk(x, None)
        return self.lnf(x), attn

    def logits(self, h):
        return self.marg(h).squeeze(-1)

    def surprise(self, h):
        return F.softplus(self.stiff(h)).squeeze(-1)

    @staticmethod
    def beta_from_surprise(s):
        return torch.clamp(1.0 / (s + STIFF_EPS) - 1.0 / (LN2 + STIFF_EPS), min=0.0)

    @staticmethod
    def source_mask(rev, rev_base=None):
        """Round 2: attend over the newly materialised cells S minus E_0 when a row has any, else over all of S (per row)."""
        if rev_base is None:
            return rev
        new = rev & ~rev_base
        has = new.any(1, keepdim=True)
        return torch.where(has, new, rev)

    def attn_vec(self, attn, rev, rev_base=None):
        """Normalised attention of each token over its source set (parameter-free fingerprint), [B,n,n]."""
        m = self.source_mask(rev, rev_base)
        a = attn * m[:, None, :].to(attn.dtype)
        return a / a.sum(-1, keepdim=True).clamp_min(1e-8)

    def key(self, h, attn, rev, rev_base=None):
        if self.key_mode == "v2":                                     # R2b fix: no h_i, LayerNormed source fingerprint
            a = self.attn_vec(attn, rev, rev_base)
            ebar = F.layer_norm(a @ self.pos.detach(), (self.d,))
            return F.normalize(self.keyW2(ebar), dim=-1)
        a = attn * rev[:, None, :].to(attn.dtype)                    # only revealed sources j in S
        ebar = a @ self.pos.detach()                                  # [B,n,d]
        return F.normalize(self.keyW(torch.cat([ebar, h], -1)), dim=-1)

    def flops_forward(self) -> float:
        n, d, L, ff = self.n, self.d, self.layers, self.ff
        per_layer = 2 * n * (d * 3 * d + d * d + 2 * d * ff) + 2 * 2 * n * n * d
        return float(L * per_layer + 2 * n * d)
