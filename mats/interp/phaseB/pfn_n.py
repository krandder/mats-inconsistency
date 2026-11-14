"""World-1 variants for any number of sources S (seventh agent; a tool for a later registered episode). pfn2's PFN is token-count
agnostic; these subclasses add the S-aware block-1 changes of E53 (uni), E60 (own), E64 (nocons) and E64b (split, splitnc):
  own      block 1's summary tokens (token s*5 + 4) attend only to their own source's four labelled tokens (learned softmax);
  uni      the same allowed set with fixed weights 1/4;
  nocons   block 1's summary tokens may not attend to the other sources' summary tokens;
  split    block 1's pre-attention input h = gamma * (N(x_c) + N(x_k)) / sqrt(2) + beta (pfn_e64b's split; codes columns :32);
  splitnc  split and nocons.
No parameter is added or removed: with torch.manual_seed(seed) every parameter starts as in the baseline net of that seed.
With S = 3 each variant equals its world-1 counterpart (checked in plumbing).
"""
import math
import torch
import pfn_e64 as P64
import pfn_e64b as PB
import pfn_uni as PU

Q, QL = 5, 4
VARIANTS = ('own', 'uni', 'nocons', 'split', 'splitnc')


def make_cls(P2, variant, S):
    T = S * Q + 1
    SUMM = [s * Q + QL for s in range(S)]

    class PFNn(P2.PFN):
        def __init__(self, fin, depth, d, h, attn_only=False, mlp_ratio=2, mixk=0):
            super().__init__(fin, depth, d, h, attn_only, mlp_ratio, mixk)
            self.variant = variant; self.S = S

        def forward(self, X, valid, want=False):
            B, Tn, _ = X.shape
            assert Tn == T, 'S = %d layout only' % S
            qi = Tn - 1
            ctx = valid.clone(); ctx[:, qi] = False
            allow = ctx[:, None, :].expand(B, Tn, Tn).clone(); allow[:, qi, qi] = True
            eye = torch.eye(Tn, dtype=torch.bool, device=X.device)[None]
            allow = allow | (eye & ~valid[:, :, None])
            allow0 = allow.clone(); fixed = None
            if variant in ('own', 'uni'):
                for s in range(S):
                    allow0[:, SUMM[s], :] = False; allow0[:, SUMM[s], s * Q:s * Q + QL] = True
            if variant == 'uni':
                rows = torch.zeros(Tn, dtype=torch.bool); A = torch.zeros(Tn, Tn)
                for s in range(S):
                    rows[SUMM[s]] = True; A[SUMM[s], s * Q:s * Q + QL] = 1.0 / QL
                fixed = (rows, A)
            if variant in ('nocons', 'splitnc'):
                for s in range(S):
                    for t in range(S):
                        if t != s:
                            allow0[:, SUMM[s], SUMM[t]] = False
            x = self.inp(X); hs, attns = [x], []
            for i, blk in enumerate(self.blocks):
                if i == 0 and variant in ('split', 'splitnc'):
                    x, a = P64.attn_from_h(blk, x, PB.split_h(self, blk, X), allow0)
                else:
                    x, a = PU.block_fwd(blk, x, allow0 if i == 0 else allow, fixed if i == 0 else None, want)
                hs.append(x); attns.append(a if want else None)
            y = self.out(self.ln(x[:, qi]))
            if self.mixk == 0:
                head = (y[:, 0], y[:, 1].clamp(-6, 3))
            else:
                K = self.mixk
                head = (y[:, K:2 * K], y[:, 2 * K:].clamp(-6, 3), y[:, :K])
            return (*head, hs, attns) if want else head
    return PFNn


def build_any(meta, P2, fin):
    v = meta.get('variant'); S = meta.get('S', 3)
    if v in VARIANTS:
        return make_cls(P2, v, S)(fin, meta['depth'], meta['width'], meta['heads'], mixk=meta.get('mixk', 0))
    return P2.PFN(fin, meta['depth'], meta['width'], meta['heads'], meta.get('attn_only', False), mixk=meta.get('mixk', 0))
