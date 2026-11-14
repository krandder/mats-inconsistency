"""E19 architecture variants of the tiny-pfn PFN (PREREG-B amendment 26): one extra token-wise MLP block (pre-LN, residual),
either before the first attention ("pre": each token can transform its own reading and label before any pooling) or after the
last block ("post": the same parameters where they cannot change what is pooled).
The extra block is created after every standard module, so for a given seed all shared parameters get exactly the baseline's
initialisation, and the training data stream (pfn.train's cid depends only on the config name) is the baseline's.
Training: `python train_var.py VARIANT --seed S --out DIR` (the tiny-pfn recipe through pfn.train, with checkpoints).
"""
import torch
import torch.nn as nn
import pfn as P


class PFNVar(P.PFN):
    def __init__(self, fin, depth, d, h, attn_only=False, mlp_ratio=2, mixk=0, where='pre'):
        super().__init__(fin, depth, d, h, attn_only, mlp_ratio, mixk)
        self.where = where
        self.extra = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, mlp_ratio * d), nn.GELU(), nn.Linear(mlp_ratio * d, d))

    def forward(self, X, valid, want=False):
        B, T, _ = X.shape
        qi = T - 1
        ctx = valid.clone(); ctx[:, qi] = False
        allow = ctx[:, None, :].expand(B, T, T).clone()
        allow[:, qi, qi] = True
        eye = torch.eye(T, dtype=torch.bool, device=X.device)[None]
        allow = allow | (eye & ~valid[:, :, None])
        x = self.inp(X)
        if self.where == 'pre':
            x = x + self.extra(x)
        hs, attns = [x], []
        for blk in self.blocks:
            x, a = blk(x, allow, want)
            hs.append(x); attns.append(a)
        if self.where == 'post':
            x = x + self.extra(x)
        y = self.out(self.ln(x[:, qi]))
        if self.mixk == 0:
            head = (y[:, 0], y[:, 1].clamp(-6, 3))
        else:
            K = self.mixk
            head = (y[:, K:2 * K], y[:, 2 * K:].clamp(-6, 3), y[:, :K])
        return (*head, hs, attns) if want else head


def build(meta):
    return PFNVar(P.fin_of(meta['ds']), meta['depth'], meta['width'], meta['heads'], meta['attn_only'], mixk=meta.get('mixk', 0),
                  where=meta['variant'])
