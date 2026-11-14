"""E53 architecture variants (PREREG-B amendment 61): pfn2.py's PFN with block 1's within-source pooling fixed to uniform weights.
No parameter is added or removed (apart from blocks[0].ln1 in 'unif_noln', as pfn_noln.py), so with torch.manual_seed(seed) every
parameter starts exactly as in the baseline net of that seed (E37's ctrl and mag nets; E50's no-LayerNorm nets).
  uni        in block 1, each source's query-question (summary) token, tokens 4, 9, 14, attends with FIXED weights 1/4 to its own
             source's four labelled tokens (5s .. 5s+3) and to nothing else, in every head (its own value and other sources
             excluded); its query vector is unused there. Values, every other attention row and block 2 are as in pfn2 (learned).
  unif       uni, and the query token (15) may not attend to any labelled token in either block (it sees the three summary
             tokens and itself). Every route from a labelled token to the output then passes through the uniform pool of block 1.
  unif_noln  unif with block 1's pre-attention LayerNorm replaced by the identity (as pfn_noln.py): the pooled value of a source
             is then affine in the mean of its labelled tokens' input features, so in the mag format the summary sees exactly
             mean_q |r_q| (and in the sq format exactly SS / 4) of its source, plus terms that do not depend on the errors.
build_any(meta, P2, fin): these variants when meta['variant'] is one of them, otherwise pfn_noln.build_any (noln1, pre, baseline).
"""
import math
import torch
import torch.nn as nn
import pfn_noln as PN

VARIANTS = ('uni', 'unif', 'unif_noln')
S, Q, QL = 3, 5, 4
T = S * Q + 1
SUMM = [s * Q + QL for s in range(S)]
LAB = [s * Q + q for s in range(S) for q in range(QL)]
QI = T - 1


def fixed_rows():
    """(T,) bool mask of the fixed rows and the (T, T) fixed weights (zero outside the fixed rows)."""
    rows = torch.zeros(T, dtype=torch.bool); A = torch.zeros(T, T)
    for s in range(S):
        rows[SUMM[s]] = True
        A[SUMM[s], s * Q:s * Q + QL] = 1.0 / QL
    return rows, A


def block_fwd(blk, x, allow, fixed=None, want=False):
    """pfn2.Block.forward, with the attention rows in `fixed` (mask, weights) replaced by fixed weights after the softmax."""
    B, Tn, d = x.shape
    q, k, v = blk.qkv(blk.ln1(x)).split(blk.h * blk.dh, -1)
    sp = lambda t: t.view(B, Tn, blk.h, blk.dh).transpose(1, 2)
    q, k, v = sp(q), sp(k), sp(v)
    s = q @ k.transpose(-1, -2) / math.sqrt(blk.dh)
    s = s.masked_fill(~allow[:, None], float('-inf'))
    a = s.softmax(-1)
    if fixed is not None:
        rows, A = fixed
        a = torch.where(rows.to(a.device)[None, None, :, None], A.to(a.device, a.dtype)[None, None], a)
    x = x + blk.o((a @ v).transpose(1, 2).reshape(B, Tn, blk.h * blk.dh))
    if not blk.attn_only:
        x = x + blk.mlp(blk.ln2(x))
    return x, (a if want else None)


def make_cls(P2, variant):
    class PFNUni(P2.PFN):
        def __init__(self, fin, depth, d, h, attn_only=False, mlp_ratio=2, mixk=0):
            super().__init__(fin, depth, d, h, attn_only, mlp_ratio, mixk)
            self.variant = variant
            if variant == 'unif_noln':
                self.blocks[0].ln1 = nn.Identity()
            self._fixed = fixed_rows()

        def forward(self, X, valid, want=False):
            B, Tn, _ = X.shape
            assert Tn == T, 'world-1 layout only'
            qi = Tn - 1
            ctx = valid.clone(); ctx[:, qi] = False
            allow = ctx[:, None, :].expand(B, Tn, Tn).clone()
            allow[:, qi, qi] = True
            eye = torch.eye(Tn, dtype=torch.bool, device=X.device)[None]
            allow = allow | (eye & ~valid[:, :, None])
            if self.variant in ('unif', 'unif_noln'):
                allow[:, qi, LAB] = False                              # the query token never reads a labelled token
            x = self.inp(X); hs, attns = [x], []
            for i, blk in enumerate(self.blocks):
                x, a = block_fwd(blk, x, allow, self._fixed if i == 0 else None, want)
                hs.append(x); attns.append(a)
            y = self.out(self.ln(x[:, qi]))
            if self.mixk == 0:
                head = (y[:, 0], y[:, 1].clamp(-6, 3))
            else:
                K = self.mixk
                head = (y[:, K:2 * K], y[:, 2 * K:].clamp(-6, 3), y[:, :K])
            return (*head, hs, attns) if want else head
    return PFNUni


def build_any(meta, P2, fin):
    v = meta.get('variant')
    if v in VARIANTS:
        return make_cls(P2, v)(fin, meta['depth'], meta['width'], meta['heads'], mixk=meta.get('mixk', 0))
    return PN.build_any(meta, P2, fin)
