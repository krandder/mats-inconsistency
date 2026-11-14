"""E49 architecture variant (PREREG-B amendment 57): pfn2.py's PFN with one extra token-wise MLP block BEFORE the first attention
(pre-LN, residual: x <- x + MLP(LN(x)), hidden 2d, GELU), as E19's "pre" variant (pfn_var.py, which subclasses tiny-pfn's pfn.py)
but on pfn2's PFN, so that it serves both worlds. The extra block is created after every standard module, so with
torch.manual_seed(seed) every shared parameter starts exactly as in the baseline net of that seed. Everything else (masks, blocks,
heads) is pfn2's forward unchanged.
build_any(meta, P2, fin) builds the variant when meta['variant'] == 'pre', else pfn2's PFN.
"""
import torch
import torch.nn as nn


def make_cls(P2):
    class PFNPre(P2.PFN):
        def __init__(self, fin, depth, d, h, attn_only=False, mlp_ratio=2, mixk=0):
            super().__init__(fin, depth, d, h, attn_only, mlp_ratio, mixk)
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
            x = x + self.extra(x)
            hs, attns = [x], []
            for blk in self.blocks:
                x, a = blk(x, allow, want)
                hs.append(x); attns.append(a)
            y = self.out(self.ln(x[:, qi]))
            if self.mixk == 0:
                head = (y[:, 0], y[:, 1].clamp(-6, 3))
            else:
                K = self.mixk
                head = (y[:, K:2 * K], y[:, 2 * K:].clamp(-6, 3), y[:, :K])
            return (*head, hs, attns) if want else head
    return PFNPre


def build_any(meta, P2, fin):
    if meta.get('variant') == 'pre':
        return make_cls(P2)(fin, meta['depth'], meta['width'], meta['heads'], mixk=meta.get('mixk', 0))
    return P2.PFN(fin, meta['depth'], meta['width'], meta['heads'], mixk=meta.get('mixk', 0))
