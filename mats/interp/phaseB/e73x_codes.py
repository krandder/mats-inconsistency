"""E73 exploration (eighth agent; DEV NETS ONLY, disclosed in a later amendment; not a registered test): do the random source codes
make part of the big-miss departure? On E12's matched pairs, the net's rom with the pairs' source codes as drawn (base) and
idealised at evaluation by E63's Bayes-invariant transforms (eqn: each code rescaled to norm 1; orth: Loewdin-orthogonalised, own norms;
orthn: orthonormal, norm 1); per pooling head, the between-source sd of the source-mean routing coefficients L_s, K_s on E27's first
2,000 bases under each code mode.
usage: python e73x_codes.py OUTJSON TAG=RUNDIR [...]
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import e63_run as E63  # noqa: E402
import e37_eval as EV  # noqa: E402
import e15_run as R15  # noqa: E402
import pfn_b1hw as BH  # noqa: E402

BH.install()
torch.set_num_threads(1)
S, QL = 3, 4
MODES = ('base', 'eqn', 'orth', 'orthn')


def rom_codes(model, arm, z, mode):
    src, rg = z['src'], z['range']; Wb = z['WB0'][np.arange(len(src)), src]; ws = []
    for k in (0, 1):
        ps, pq = E63.F.ids_from_X(z['X%d' % k])
        Xk, vk = E63.F.make_X(z['d%d_o' % k].astype(np.float64), z['d%d_y' % k].astype(np.float64), E63.transform(ps, mode), pq, arm)
        w, _ = R15.eff_w(model, Xk, vk, 32); ws.append(w[np.arange(len(src)), src])
    return {rr: float((ws[1] - ws[0])[rg == rr].sum() / Wb[rg == rr].sum()) for rr in ('large', 'middle')}


def coef_spread(model, arm, mode):
    z = np.load('results/e27_data.npz'); sl = slice(0, 2000)
    X, v = E63.F.make_X(z['o'][sl, 0], z['y'][sl], E63.transform(z['semb'][sl], mode), z['qemb'][sl], arm)
    A, qh, blk = E63.block1_parts(model, X); L, ratio, odd = E63.h3_logits(model, X, qh, blk)
    own = [[QL * s + q for q in range(QL)] for s in range(S)]
    P = np.stack([A[:, :, E63.SUMM[s], E63.LABS[s]].sum(-1) for s in range(S)], 1).mean((0, 1))
    Ls = np.stack([L[:, :, s, own[s]].mean(-1) for s in range(S)], 1); Ks = np.stack([odd[:, :, s, own[s]].mean(-1) for s in range(S)], 1)
    return {h: dict(P=float(P[h]), L=float(Ls[:, :, h].mean()), sdL=float(Ls[:, :, h].std()), K=float(Ks[:, :, h].mean()), sdK=float(Ks[:, :, h].std()))
            for h in range(A.shape[1])}


def main():
    outp = sys.argv[1]; res = json.load(open(outp)) if os.path.exists(outp) else {}
    z = np.load('results/e12_pairs.npz')
    for spec in sys.argv[2:]:
        tag, run = spec.split('=', 1)
        if tag in res:
            continue
        t0 = time.time(); model, arm, meta = E63.load(run, None); o = {}
        for mode in MODES:
            o[mode] = dict(rom=rom_codes(model, arm, z, mode), heads=coef_spread(model, arm, mode))
        o['seconds'] = time.time() - t0; res[tag] = o
        json.dump(res, open(outp, 'w'), indent=1)
        print(tag, json.dumps({m: o[m]['rom'] for m in MODES}), flush=True)


if __name__ == '__main__':
    main()
