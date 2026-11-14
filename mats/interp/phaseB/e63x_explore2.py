"""E63 exploration, part 2 (seventh agent; NOT registered results; disclosed in amendment 71): on E43's newinit ctrl nets (excluded
from the registered E63 tests), Bayes-invariant changes of the three source codes at evaluation:
  eqn    each source code rescaled to norm 1 (direction kept);
  orth   the three codes symmetrically (Loewdin) orthogonalised, each keeping its own norm;
  orthn  orthogonalised and rescaled to norm 1.
Per net and condition: test KL on E27's first 1,000 bases, the summary tokens' block-1 attention (own labelled mass, self mass and
cross-source leakage per head), and the across-episode CV of each source's mean LayerNorm error-free norm A_s (lngeom's A).
usage: python e63x_explore2.py RUN [RUN ...] -> prints; results/e63x_explore2.json
"""
import json, os, sys
import numpy as np
import torch

import e63x_explore as X1

RU, E30, F = X1.RU, X1.E30, X1.F
S, Q, QL, DS = 3, 5, 4, 16


def transform(se, mode):
    se = se.astype(np.float64).copy()
    nrm = np.linalg.norm(se, axis=-1, keepdims=True)
    if mode == 'eqn':
        return se / nrm
    if mode in ('orth', 'orthn'):
        G = np.einsum('nsd,ntd->nst', se, se)
        w, V = np.linalg.eigh(G)
        Gm = np.einsum('nij,nj,nkj->nik', V, w ** -0.5, V)
        O = np.einsum('nst,ntd->nsd', Gm, se)                          # orthonormal rows
        return O * (nrm if mode == 'orth' else 1.0)
    return se


def main():
    z = np.load('results/e27_data.npz'); NB = 1000; sl = slice(0, NB)
    g = z['bmean'][sl, None] + RU.GRID[None]
    lpB = RU.ED.mix_logp_points(z['bmeans'][sl], z['bvars'], z['bwts'][sl], g)
    se0 = z['semb'][sl]; qe = z['qemb'][sl]
    res = {}
    for run in sys.argv[1:]:
        model, a, meta = E30.load(run)
        W = model.inp.weight.detach().numpy().astype(np.float64); b = model.inp.bias.detach().numpy().astype(np.float64)
        o = {}
        for mode in ('base', 'eqn', 'orth', 'orthn'):
            se = transform(se0, mode)
            X, v = F.make_X(z['o'][sl, 0], z['y'][sl], se, qe, a)
            h = RU.heads(model, 'clean', X, v)
            kl, _, _ = RU.set_stats([h], lpB, g)
            A = X1.attn1(model, X)
            own = np.stack([A[:, :, X1.SUMM[s], s * Q:s * Q + QL].sum(-1) for s in range(S)], 1)      # (B, S, H)
            slf = np.stack([A[:, :, X1.SUMM[s], X1.SUMM[s]] for s in range(S)], 1)
            leak = 1 - own - slf
            Xl = X[:, [s * Q + q for s in range(S) for q in range(QL)], :].astype(np.float64); X0 = Xl.copy(); X0[..., 32] = X0[..., 33]
            e = X0 @ W.T + b; e = e - e.mean(-1, keepdims=True); An = np.linalg.norm(e, axis=-1).reshape(NB, S, QL).mean(-1)
            o[mode] = dict(kl=float(kl.mean()), own_by_head=[round(float(own[:, :, hh].mean()), 4) for hh in range(own.shape[2])],
                           self_by_head=[round(float(slf[:, :, hh].mean()), 4) for hh in range(own.shape[2])],
                           leak_by_head=[round(float(leak[:, :, hh].mean()), 4) for hh in range(own.shape[2])],
                           leak_q99_by_head=[round(float(np.quantile(leak[:, :, hh], 0.99)), 4) for hh in range(own.shape[2])],
                           A_src_cv=float(An.std() / An.mean()))
        res[os.path.basename(run)] = o
        print(os.path.basename(run), json.dumps(o), flush=True)
    json.dump(res, open('results/e63x_explore2.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
