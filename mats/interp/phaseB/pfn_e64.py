"""E64 architecture variants (PREREG-B amendment 72, planned): two changes to block 1 of pfn2's PFN that keep content-based routing
(the summary tokens must still find their source's tokens by matching the random codes) but remove one of its side effects.
  nocons  block 1's summary tokens (4, 9, 14) may not attend to the OTHER sources' summary tokens (they keep their own source's
          tokens, the other sources' labelled tokens and themselves): the "consensus head" of E63's exploration (a head averaging
          the three query-question tokens) cannot form in block 1. Every other row and block 2 as in pfn2.
  codeln  block 1's pre-attention LayerNorm normalises only the content part of each token's embedding (the o, y, missing and
          query-flag columns and the bias); the identity-code part (W[:, :32] times the source and question codes) is added after
          the normalisation: h = LN1(W_c X_c + b) + W_code X_code. The routing codes then neither set the normaliser of the error
          feature nor scale with 1 / n_j in the logits. The residual stream (x = W X + b) and block 2 are as in pfn2.
  both    codeln and nocons together.
No parameter is added or removed, so with torch.manual_seed(seed) every parameter starts as in the baseline net of that seed, and
train_sw.py's config name (hence the episode stream) is the baseline's.
build_any(meta, P2, fin): these variants when meta['variant'] is one of them, otherwise pfn_own.build_any.
"""
import math
import torch
import pfn_own as PO
import pfn_uni as PU

VARIANTS = ('nocons', 'codeln', 'both')
NCODE = 32                                     # source code (16) + question code (16) columns in the world-1 layout


def attn_from_h(blk, x, h, allow):
    """pfn2.Block.forward with the pre-attention LayerNorm output h given."""
    B, Tn, d = x.shape
    q, k, v = blk.qkv(h).split(blk.h * blk.dh, -1)
    sp = lambda t: t.view(B, Tn, blk.h, blk.dh).transpose(1, 2)
    q, k, v = sp(q), sp(k), sp(v)
    s = (q @ k.transpose(-1, -2) / math.sqrt(blk.dh)).masked_fill(~allow[:, None], float('-inf'))
    a = s.softmax(-1)
    x = x + blk.o((a @ v).transpose(1, 2).reshape(B, Tn, blk.h * blk.dh))
    if not blk.attn_only:
        x = x + blk.mlp(blk.ln2(x))
    return x, a


def make_cls(P2, variant):
    class PFN64(P2.PFN):
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
            if self.variant in ('nocons', 'both'):
                for s in range(PU.S):
                    for t in range(PU.S):
                        if t != s:
                            allow0[:, PU.SUMM[s], PU.SUMM[t]] = False
            x = self.inp(X); hs, attns = [x], []
            for i, blk in enumerate(self.blocks):
                if i == 0 and self.variant in ('codeln', 'both'):
                    Wt = self.inp.weight
                    xc = X[..., NCODE:] @ Wt[:, NCODE:].T + self.inp.bias          # content part (o, y, missing, query flag; bias)
                    xk = X[..., :NCODE] @ Wt[:, :NCODE].T                          # identity-code part
                    x, a = attn_from_h(blk, x, blk.ln1(xc) + xk, allow0)
                else:
                    x, a = PU.block_fwd(blk, x, allow0 if i == 0 else allow, None, want)
                hs.append(x); attns.append(a if want else None)
            y = self.out(self.ln(x[:, qi]))
            if self.mixk == 0:
                head = (y[:, 0], y[:, 1].clamp(-6, 3))
            else:
                K = self.mixk
                head = (y[:, K:2 * K], y[:, 2 * K:].clamp(-6, 3), y[:, :K])
            return (*head, hs, attns) if want else head
    return PFN64


def build_any(meta, P2, fin):
    v = meta.get('variant')
    if v in VARIANTS:
        return make_cls(P2, v)(fin, meta['depth'], meta['width'], meta['heads'], mixk=meta.get('mixk', 0))
    return PO.build_any(meta, P2, fin)
