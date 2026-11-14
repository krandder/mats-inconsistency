"""E3b (PREREG-B amendment 10): E2's procedure with every surrogate given the exact Bayes quantities b(x) as free inputs.

Same stages as qstruct.py (E2). Candidate changes: low-rank maps and the skip-merge take b into their bottleneck
(+ 24 x r parameters); named attention uses the exact reliability (labelled-posterior expected log precision of the key's
source) and the source's posterior weight; a new candidate replaces the whole last block for the query token by
sphere(x_q + MLP_16(b)) (944 parameters). b is computed from the tokens (observed readings and labels only).
usage: python qstruct_b.py RUN_DIR [--smoke] -> RUN_DIR/qstruct_b.json
"""
import math, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import nnoise as N
import world as W
import qw
import qstruct as Q
from qbayes import bayes_features

torch.set_num_threads(1)
T, QT, DS = Q.T, Q.QT, Q.DS
_cache = {}


def world_from_X(X):
    n = len(X); o = X[:, :15, 2 * DS].reshape(n, 3, 5).astype(np.float64)
    y = np.c_[X[:, 0:4, 2 * DS + 1].astype(np.float64), np.zeros(n)]
    return dict(o=o, y=y, v=np.ones((n, 3)), vidx=np.zeros((n, 3), int))


def _raw(X):
    d = world_from_X(X); F = bayes_features(d)
    lab = W.source_posterior(d)                                                     # (n, 3, 3)
    rel = (lab * (-np.log(W.VLEV))[None, None, :]).sum(2)                           # expected log precision per source
    return F, rel, F[:, 20:23]


_ref = np.random.default_rng(94_000)
_dref = W.sample_np(8192, _ref); _Xref, _ = W.tokens_np(_dref, _ref, DS)
_Fref = _raw(_Xref)[0]; MU, SD = _Fref.mean(0), _Fref.std(0) + 1e-6


def bctx(X):
    key = (X.shape, hash(np.ascontiguousarray(X).tobytes()))
    if key in _cache: return _cache[key]
    F, rel, ws = _raw(X)
    out = torch.as_tensor(np.c_[(F - MU) / SD, rel, ws], dtype=torch.float32)          # (n, 30)
    if len(_cache) > 64: _cache.clear()
    _cache[key] = out; return out


def bz(): return Q.CTX['b'][:, :24]


class EmbLRB(qw.EmbLR):
    def __init__(self, model, X, r):
        super().__init__(model, X, r); self.Wb = nn.Parameter(torch.zeros(24, r)); self.cost += 24 * r
    def forward(self, X): return N.sphere((X @ self.V + (bz() @ self.Wb)[:, None, :]) @ self.U.T + self.c)


class HidLRB(qw.HidLR):
    def __init__(self, block, A, r):
        super().__init__(block, A, r); self.Wb = nn.Parameter(torch.zeros(24, r)); self.cost += 24 * r
    def forward(self, a): return N.sphere(Fn.gelu((self.ln2(a) @ self.V + (bz() @ self.Wb)[:, None, :]) @ self.U.T + self.c))


class WriteLRB(qw.WriteLR):
    def __init__(self, block, Mh, r):
        super().__init__(block, Mh, r); self.Wb = nn.Parameter(torch.zeros(24, r)); self.cost += 24 * r
    def forward(self, a, m): return N.sphere(a + (m @ self.V + (bz() @ self.Wb)[:, None, :]) @ self.U.T + self.c)


class MergeLRB(Q.MergeLR):
    def __init__(self, block, A, r):
        super().__init__(block, A, r); self.Wb = nn.Parameter(torch.zeros(24, r)); self.cost += 24 * r
    def forward(self, a): return N.sphere(a + (self.ln2(a) @ self.V + (bz() @ self.Wb)[:, None, :]) @ self.U.T + self.c)


SRC_IDX = torch.as_tensor(Q.SRC_OF[:QT])


class AttnNamedB(Q.AttnNamed):
    """Named pattern with exact Bayes reliability (features 7-9) and the source's posterior weight (11: query row, 12: all)."""
    def __init__(self, block):
        super().__init__(block); self.theta = nn.Parameter(torch.zeros(self.h, 13)); self.cost += 2 * self.h
    def forward(self, x, mask, feats=None):
        b = Q.CTX['b']; n = len(x); F = Q.SF[None].expand(n, T, T, 11).clone()
        rk = torch.zeros(n, T); rk[:, :QT] = b[:, 24:27][:, SRC_IDX]
        wk = torch.zeros(n, T); wk[:, :QT] = b[:, 27:30][:, SRC_IDX]
        ctxk = (torch.arange(T) < QT).float()[None, None, :]; qrow = (torch.arange(T) == QT).float()[None, :, None]
        F[..., 7] = rk[:, None, :] * ctxk; F[..., 8] = F[..., 7] * qrow; F[..., 9] = F[..., 7] * Q.SF[None, :, :, 0]
        F = torch.cat([F, (wk[:, None, :] * ctxk * qrow)[..., None], (wk[:, None, :] * ctxk).expand(n, T, T)[..., None]], -1)
        v = (self.ln1(x) @ self.Wv.T + self.bv).view(n, T, self.h, self.dh).transpose(1, 2)
        lg = torch.einsum('hf,ntuf->nhtu', self.theta, F).masked_fill(~mask[None, None], float('-inf'))
        return N.sphere(x + ((lg.softmax(-1) @ v).transpose(1, 2).reshape(n, T, -1) @ self.Wo.T + self.bo))


class LastB(nn.Module):
    """Replaces the whole last block: r_L = sphere(x + MLP_16(b)) (only the query token is read)."""
    def __init__(self, d, h=16):
        super().__init__(); self.l1 = nn.Linear(24, h); self.l2 = nn.Linear(h, d)
        nn.init.zeros_(self.l2.weight); nn.init.zeros_(self.l2.bias); self.cost = sum(p.numel() for p in self.parameters())
    def forward(self, x): return N.sphere(x + self.l2(Fn.gelu(self.l1(bz())))[:, None, :])


def build_cands_E3b(c):
    model, st, nz, X, sg, Js, ntr, sw, se, sds, allow = (c[k] for k in ('model', 'st', 'nz', 'X', 'sg', 'Js', 'ntr', 'steps_w', 'steps_e', 'sds', 'allow'))
    Q.CTX['fit'] = bctx(X)
    fs = lambda mod, site, **kw: Q.fit_site(model, mod, site, st, nz, X, sg, Js[site], ntr, sw, se, sds, allow, **kw)
    cands = {'emb:LR8b': ('sur', 'emb', fs(EmbLRB(model, X, 8), 'emb'))}
    L = c['depth']
    for l in range(1, L + 1):
        b = model.blocks[l - 1]
        cands[f'a{l}:namedB'] = ('sur', f'a{l}', fs(AttnNamedB(b), f'a{l}', feats_all=c['feats_all']))
        for r in (8, 4):
            cands[f'm{l}:LR{r}b'] = ('sur', f'm{l}', fs(HidLRB(b, st[f'a{l}'], r), f'm{l}'))
            cands[f'r{l}:LR{r}b'] = ('sur', f'r{l}', fs(WriteLRB(b, st[f'm{l}'], r), f'r{l}'))
        cands[f'merge{l}:LR8b'] = ('merge', l, fs(MergeLRB(b, st[f'a{l}'], 8), f'r{l}', merge_l=l, sig_eff=c['sig_eff']))
    cands['last:B16'] = ('last', L, fs(LastB(model.d), f'r{L}', last=True))
    Q.CTX['fit'] = None
    return cands


Q.CTX['fn'] = bctx
Q.BUILD_CANDS[0] = build_cands_E3b
Q.OUT_TAG[0] = 'qstruct_b'

if __name__ == '__main__':
    Q.run(sys.argv[1], smoke='--smoke' in sys.argv)
