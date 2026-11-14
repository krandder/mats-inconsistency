"""E60 architecture variants (PREREG-B amendment 68): the source's pool restricted to its own labelled tokens, with LEARNED weights.
  own    in block 1, each summary token (4, 9, 14) may attend only to its own source's four labelled tokens (softmax over those four,
         weights learned from its query and their keys); every other row and block 2 as in pfn2. Same allowed set as pfn_uni's
         'uni', whose weights are fixed at 1/4.
  ownf   own, and the query token may not attend to any labelled token in either block (as pfn_uni's 'unif').
No parameter is added or removed, so with torch.manual_seed(seed) every parameter starts as in the baseline net of that seed.
build_any(meta, P2, fin): these variants when meta['variant'] is one of them, otherwise pfn_uni.build_any.
"""
import torch
import pfn_uni as PU

VARIANTS = ('own', 'ownf')


def make_cls(P2, variant):
    class PFNOwn(P2.PFN):
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
            if self.variant == 'ownf':
                allow[:, qi, PU.LAB] = False
            allow0 = allow.clone()
            for s in range(PU.S):
                allow0[:, PU.SUMM[s], :] = False
                allow0[:, PU.SUMM[s], s * PU.Q:s * PU.Q + PU.QL] = True
            x = self.inp(X); hs, attns = [x], []
            for i, blk in enumerate(self.blocks):
                x, a = PU.block_fwd(blk, x, allow0 if i == 0 else allow, None, want)
                hs.append(x); attns.append(a)
            y = self.out(self.ln(x[:, qi]))
            if self.mixk == 0:
                head = (y[:, 0], y[:, 1].clamp(-6, 3))
            else:
                K = self.mixk
                head = (y[:, K:2 * K], y[:, 2 * K:].clamp(-6, 3), y[:, :K])
            return (*head, hs, attns) if want else head
    return PFNOwn


def build_any(meta, P2, fin):
    v = meta.get('variant')
    if v in VARIANTS:
        return make_cls(P2, v)(fin, meta['depth'], meta['width'], meta['heads'], mixk=meta.get('mixk', 0))
    return PU.build_any(meta, P2, fin)
