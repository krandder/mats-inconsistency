"""Adversarial Goodhart slice (PREREG simplification 5; second clause of falsifier F4).

60 episodes from the 'adv' namespace (m = 6, w = 6). Each gets up to 96 candidate bits drawn from the zero-CMI part of the
type library (T0, T3a, HN1-HN4 instances), each oracle-verified CMI < 1e-12 given the episode's listed columns.
All candidates share one row draw (same x, y and listed columns); each candidate column gets its own noise stream, so the
search is over genuinely different bits. T = H + 16 + 64 rows.
Each arm (FR seed 0, tuned C4, tuned C5) picks the candidate with the largest V^preq_16. The pick's settlement gain is
log R(y | with b) - log R(y | without b) summed over rows H+16..H+79, predicted from rows < H+16 only
(FR: ctx = H+16; C4: counts and weights frozen at H+16). C5 has no predictor of its own, so its pick is valued with C4's.
Also stored: every candidate's V16 and settlement gain (the unselected baseline is their mean).

  python adv.py [--run FR_s0|none] [--n 60] [--ncand 96] [--n0 0]   -> <OUT>/eval/adv[_<n0>].npz
"""
import argparse, dataclasses, json, os, time
import numpy as np
import torch
from gen import H, make_structure, rng_for, sample_rows, ybits
from larms import library
from oracle import cmi_batch, best_device

OUT = os.environ.get('P16_OUT', 'results')
POOL = ['T0', 'T3a', 'HN1', 'HN2', 'HN3', 'HN4']
K, L = 16, 64
T = H + K + L
M, W = 6, 6


def cand_values(col, xi, y, V, rng):
    """draw the candidate column (same rules as gen.sample_rows) with its own rng -> (T,) int8, -1 before birth."""
    Tn = len(xi)
    if col.ydep is not None:
        p = col.ydep[y]
    else:
        cfg = np.zeros(Tn, np.int64)
        for i, pi in enumerate(col.par):
            cfg |= V[:, pi].astype(np.int64) << i
        p = col.P1[xi, cfg]
    v = (rng.random(Tn) < p).astype(np.int8)
    if col.markov:
        ref = rng.random(Tn) >= col.markov
        ref[0] = True; ref[H] = True
        last = np.maximum.accumulate(np.where(ref, np.arange(Tn), 0))
        v = v[last]
    v[:H] = -1
    return v


def pool(ep, ncand, rng, dev):
    lib = library(ep, POOL)
    order = rng.permutation(len(lib))
    keep, cm_keep = [], []
    for c0 in range(0, len(order), 256):
        ids = order[c0:c0 + 256]
        eps = [dataclasses.replace(ep, cols=ep.cols[:ep.m] + [lib[i][2]], ctype=lib[i][0], info=dict(ep.info)) for i in ids]
        cm = cmi_batch(eps, dev, torch.float64).cpu().numpy()
        for i, c in zip(ids, cm):
            if c < 1e-12 and len(keep) < ncand:
                keep.append(lib[i]); cm_keep.append(float(c))
        if len(keep) >= ncand:
            break
    return keep, cm_keep, len(lib)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='FR_s0')
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--n0', type=int, default=0)
    ap.add_argument('--ncand', type=int, default=96)
    a = ap.parse_args()
    dev = best_device()
    tuned = json.load(open(f'{OUT}/tuned.json'))
    c4p, c5p = tuned['C4'], tuned.get('C5', {})
    model = None
    if a.run != 'none':
        from reader import FR
        ck = torch.load(f'{OUT}/runs/{a.run}/final.pt', map_location='cpu', weights_only=False)
        model = FR(mode='joint').to(dev).eval()
        model.load_state_dict(ck['model'])
    from baselines import bma_forward, bma_reverse, c5_betting
    NC = a.ncand
    R = {k: np.full((a.n, NC), np.nan) for k in ('fr_v16', 'fr_set', 'c4_v16', 'c4_set', 'c5_v16')}
    meta = {k: np.zeros(a.n) for k in ('n_cand', 'lib_size', 'max_cmi')}
    types = np.full((a.n, NC), '', dtype=object)
    t0 = time.time()
    for e in range(a.n):
        idx = a.n0 + e
        ep = make_structure(rng_for('adv', idx, 1), 'adv', idx, 'T0', M, W)
        cands, cms, nlib = pool(ep, NC, rng_for('adv', idx, 3), dev)
        n = len(cands)
        meta['n_cand'][e], meta['lib_size'][e], meta['max_cmi'][e] = n, nlib, max(cms) if cms else np.nan
        xi, y, V0 = sample_rows(ep, T, rng_for('adv', idx, 1000))
        crng = rng_for('adv', idx, 2000)
        Vs = np.repeat(V0[None], n, 0)
        for c, (t, wt, col) in enumerate(cands):
            Vs[c, :, -1] = cand_values(col, xi, y, V0, crng)
            types[e, c] = t
        V = torch.as_tensor(Vs, dtype=torch.long, device=dev)
        Y = torch.as_tensor(np.repeat(ybits(y, W)[None], n, 0), dtype=torch.long, device=dev)
        with torch.no_grad():
            if model is not None:
                g = torch.Generator().manual_seed(idx + 17)
                order = torch.randperm(W, generator=g).to(dev)[None]
                Vn = V[:1].clone(); Vn[:, :, -1] = -1
                lo = model.row_logp(Vn, Y[:1], order)
                so = model.row_logp(Vn, Y[:1], order, ctx=H + K)
                for c0 in range(0, n, 32):
                    sl = slice(c0, min(n, c0 + 32))
                    o = order.expand(sl.stop - sl.start, -1).contiguous()
                    lw = model.row_logp(V[sl], Y[sl], o)
                    sw = model.row_logp(V[sl], Y[sl], o, ctx=H + K)
                    R['fr_v16'][e, sl] = (lw - lo)[:, H:H + K].sum(1).cpu().numpy()
                    R['fr_set'][e, sl] = (sw - so)[:, H + K:].sum(1).cpu().numpy()
            for c0 in range(0, n, 16):
                sl = slice(c0, min(n, c0 + 16))
                Vc, Yc = V[sl], Y[sl]
                f = bma_forward(Vc, Yc, alpha=c4p['alpha'], rho=c4p['rho'], flip=True)
                gf = (f['with'] - f['without'])[:, H:H + K]
                rv, phat = bma_reverse(Vc, Yc, H)
                R['c4_v16'][e, sl] = (gf.sum(1) + c4p.get('lam', 0.0) * rv[:, :K].sum(1)).cpu().numpy()
                s = bma_forward(Vc, Yc, alpha=c4p['alpha'], rho=c4p['rho'], flip=False, settle=(H + K, L))
                R['c4_set'][e, sl] = (s['set_with'] - s['set_without']).sum(1).cpu().numpy()
                gg = torch.Generator().manual_seed(idx * 1000 + c0)
                lw5 = c5_betting(f['with'][:, H:H + K], f['with_flip'][:, H:H + K], Vc[:, H:H + K, -1], phat[:, :K], gg,
                                 n_twins=8, scale=c5p.get('scale', 1.0))
                R['c5_v16'][e, sl] = lw5[:, K - 1].cpu().numpy()
        if e % 10 == 0:
            print(f'adv {e + 1}/{a.n} ncand {n} lib {nlib} {time.time() - t0:.0f}s', flush=True)
    picks = {}
    for arm, sc, st in (('FR', 'fr_v16', 'fr_set'), ('C4', 'c4_v16', 'c4_set'), ('C5', 'c5_v16', 'c4_set')):
        if np.all(np.isnan(R[sc])):
            continue
        p = np.nanargmax(np.where(np.isnan(R[sc]), -np.inf, R[sc]), 1)
        picks[arm] = p
        s = R[st][np.arange(a.n), p]
        print(f'{arm}: pick settlement gain mean {np.mean(s):+.4f} nats (all-candidate mean {np.nanmean(R[st]):+.4f})')
    out = dict(**R, **meta, types=types.astype(str), wall_s=time.time() - t0, params=json.dumps(dict(c4=c4p, c5=c5p, run=a.run)))
    for arm, p in picks.items():
        out[f'pick_{arm}'] = p
    os.makedirs(f'{OUT}/eval', exist_ok=True)
    np.savez_compressed(f'{OUT}/eval/adv{"_%d" % a.n0 if a.n0 else ""}{"_nofr" if model is None else ""}.npz', **out)
    print('done adv', a.n, f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
