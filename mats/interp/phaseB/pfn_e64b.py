"""E64b architecture variants (PREREG-B amendment 72, addendum 1): split normalisation of block 1's input. pfn_e64's codeln adds the
identity-code part after the content's LayerNorm without normalising it, which leaves the code channel about seven times weaker than in
the baseline at initialisation (E64's seed-0 codeln and both runs sat on a plateau for over 20,000 steps). Here each part gets its own
normaliser, so neither depends on the other and both start at comparable scales:
  split    block 1's pre-attention input is h = gamma * (N(x_c) + N(x_k)) / sqrt(2) + beta, with x_c = W[:, 32:] X[:, 32:] + b (the o, y,
           missing and query-flag columns and the bias), x_k = W[:, :32] X[:, :32] (the source and question codes), N(z) = (z - mean z) /
           sqrt(mean (z - mean z)^2 + eps) (LayerNorm without affine), and gamma, beta block 1's LayerNorm parameters. The error feature's
           normaliser then contains no code, and the code part's scale contains no error; the code part is also normalised to a fixed
           norm per token (its random length is removed). The residual stream (x = W X + b) and block 2 are as in pfn2.
  splitnc  split, and block 1's summary tokens may not attend to the other sources' summary tokens (pfn_e64's nocons).
No parameter is added or removed: with torch.manual_seed(seed) every parameter starts as in the baseline net of that seed, and
train_sw.py's config name (hence the episode stream) is the baseline's.
build_any(meta, P2, fin): these variants when meta['variant'] is one of them, otherwise pfn_e64.build_any.
"""
import math
import torch
import pfn_e64 as P64
import pfn_uni as PU

VARIANTS = ('split', 'splitnc')
NCODE = P64.NCODE


def nrm(z, eps):
    zc = z - z.mean(-1, keepdim=True)
    return zc / torch.sqrt((zc ** 2).mean(-1, keepdim=True) + eps)


def split_h(model, blk, X):
    Wt = model.inp.weight
    xc = X[..., NCODE:] @ Wt[:, NCODE:].T + model.inp.bias
    xk = X[..., :NCODE] @ Wt[:, :NCODE].T
    ln = blk.ln1
    return ln.weight * (nrm(xc, ln.eps) + nrm(xk, ln.eps)) / math.sqrt(2.0) + ln.bias


def make_cls(P2, variant):
    class PFN64b(P2.PFN):
        def __init__(self, fin, depth, d, h, attn_only=False, mlp_ratio=2, mixk=0):
            super().__init__(fin, depth, d, h, attn_only, mlp_ratio, mixk)
            self.variant = variant

        def forward(self, X, valid, want=False):
            B, Tn, _ = X.shape
            assert Tn == PU.T, 'world-1 layout only'
            qi = Tn - 1
            ctx = valid.clone(); ctx[:, qi] = False
            allow = ctx[:, None, :].expand(B, Tn, Tn).clone()
            allow[:, qi, qi] = True
            eye = torch.eye(Tn, dtype=torch.bool, device=X.device)[None]
            allow = allow | (eye & ~valid[:, :, None])
            allow0 = allow.clone()
            if self.variant == 'splitnc':
                for s in range(PU.S):
                    for t in range(PU.S):
                        if t != s:
                            allow0[:, PU.SUMM[s], PU.SUMM[t]] = False
            x = self.inp(X); hs, attns = [x], []
            for i, blk in enumerate(self.blocks):
                if i == 0:
                    x, a = P64.attn_from_h(blk, x, split_h(self, blk, X), allow0)
                else:
                    x, a = PU.block_fwd(blk, x, allow, None, want)
                hs.append(x); attns.append(a if want else None)
            y = self.out(self.ln(x[:, qi]))
            if self.mixk == 0:
                head = (y[:, 0], y[:, 1].clamp(-6, 3))
            else:
                K = self.mixk
                head = (y[:, K:2 * K], y[:, 2 * K:].clamp(-6, 3), y[:, :K])
            return (*head, hs, attns) if want else head
    return PFN64b


def build_any(meta, P2, fin):
    v = meta.get('variant')
    if v in VARIANTS:
        return make_cls(P2, v)(fin, meta['depth'], meta['width'], meta['heads'], mixk=meta.get('mixk', 0))
    return P64.build_any(meta, P2, fin)
