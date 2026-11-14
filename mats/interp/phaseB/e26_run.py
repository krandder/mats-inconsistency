"""E26 per-net measures (PREREG-B amendment 34): the net's log-odds on every E26 variant, and per-base pattern statistics.

For each net and each of the first NB bases of results/e26_data.npz:
  L      (NB, 12, 7) log-odds L(delta) = log p(y* + delta | x) - log p(y* | x) for the design variants 0..11 (6 deltas),
                   and in the last column the score d/dy log p(y | x) at y = y*;
  tu     (NB, 7)  sample variance (ddof 1) of the log-odds over the pattern set (16 draws of the residual directions, c and
                   Z fixed); Bayes is constant there, so this is the variance of the deviation;
  tsp    (NB, 7)  the same over the sign/assignment set (9 draws: magnitudes fixed, signs and question assignment redrawn);
  kl16, kl9 (NB,) mean over the set of KL(Bayes || net) on a grid (Bayes mean +- 6, 801 points);
  pc16, pc9 (NB,) -log Zbar, Zbar = integral of exp(mean over the set of log p_net): the KL removed by averaging the net's
                   log-density over the set (>= 0 by Jensen; uncorrected for the finite number of draws);
  mass   (NB,)    smallest grid mass of any net density in either set (coverage check).
Net kinds: clean (tiny-pfn pfn.PFN), heads (pfn2.PFN), noisy (nnoise.NoisyPFN; predictive = mixture over KD = 8 noise draws
with the same generator seed for every variant), ckpt (an E16 replay checkpoint), random (plumbing only).
usage: python e26_run.py SET I0 I1 -> results/e26/<SET>/<tag>.npz   (rows I0..I1-1 of SET in results/e26_nets.json)
       (plumbing checks only: E26_DATA and E26_OUT override the data file and the output directory)
"""
import json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import world as W  # noqa: E402
import pfn as P  # noqa: E402
import nnoise as NN  # noqa: E402
import e26_data as ED  # noqa: E402

torch.set_num_threads(1)
KD, NOISE_SEED = 8, 1234
GRID = np.linspace(-6.0, 6.0, 801)
NB_OF = dict(M=6000, G=6000, L1=6000, HD=6000, NZ=4000, DY5=6000, DY6=6000, RND=500)


def load(row):
    k = row['kind']
    if k == 'noisy':
        m, meta = NN.load_run(row['run'])
    elif k == 'heads':
        import pfn2 as P2
        meta = json.load(open(os.path.join(row['run'], 'meta.json')))
        m = P2.build(meta); m.load_state_dict(torch.load(os.path.join(row['run'], 'model.pt'), map_location='cpu'))
    elif k == 'ckpt':
        meta = json.load(open(os.path.join(row['run'], 'meta.json')))
        m = P.build(meta); m.load_state_dict(torch.load(row['ckpt'], map_location='cpu'))
    elif k == 'clean':
        meta = json.load(open(os.path.join(row['run'], 'meta.json')))
        m = P.build(meta); m.load_state_dict(torch.load(os.path.join(row['run'], 'model.pt'), map_location='cpu'))
    else:                                                     # random init, plumbing only
        torch.manual_seed(0); meta = dict(ds=16, depth=2, width=32, heads=2, attn_only=False, mixk=4)
        m = P.build(meta)
    m.eval(); m.requires_grad_(False)
    return m, k


def heads(model, kind, X, valid, bs=2000):
    """Predictive as a Gaussian mixture: (mu, log-sd, normalised log-weights), each (n, C)."""
    if kind == 'noisy':
        mu, ls, lg = NN.predict_mix(model, X, valid, KD, NOISE_SEED, bs=bs)          # (n, KD, K), lg log-softmaxed
        n = len(mu)
        return mu.reshape(n, -1).astype(np.float64), ls.reshape(n, -1).astype(np.float64), \
            (lg.reshape(n, -1) - math.log(KD)).astype(np.float64)
    h = P.predict(model, X, valid, bs=bs)
    if len(h) == 2:
        mu, ls = h
        return mu[:, None].astype(np.float64), ls[:, None].astype(np.float64), np.zeros((len(mu), 1))
    mu, ls, lg = [t.astype(np.float64) for t in h]
    lg = lg - lg.max(1, keepdims=True); lg = lg - np.log(np.exp(lg).sum(1, keepdims=True))
    return mu, ls, lg


def logp(hd, pts):
    mu, ls, lw = hd
    c = lw[:, None, :] - ls[:, None, :] - 0.5 * math.log(2 * math.pi) - 0.5 * ((pts[:, :, None] - mu[:, None, :]) / np.exp(ls)[:, None, :]) ** 2
    mx = c.max(2, keepdims=True)
    return mx[:, :, 0] + np.log(np.exp(c - mx).sum(2))


def score(hd, y):
    mu, ls, lw = hd
    v = np.exp(2 * ls)
    lc = lw - ls - 0.5 * (y[:, None] - mu) ** 2 / v
    r = np.exp(lc - lc.max(1, keepdims=True)); r /= r.sum(1, keepdims=True)
    return (r * (mu - y[:, None]) / v).sum(1)


def set_stats(hds, lpB, g, chunk=250):
    """hds: list of heads (one per draw in the set); lpB (n, G) Bayes log-density on the per-base grid g (n, G)."""
    n = len(lpB); kl = np.zeros(n); pc = np.zeros(n); mass = np.ones(n)
    pB = np.exp(lpB)
    for i in range(0, n, chunk):
        sl = slice(i, i + chunk); gi = g[sl]
        L = np.stack([logp(tuple(t[sl] for t in h), gi) for h in hds], 0)          # (R, c, G)
        kl[sl] = np.trapezoid(pB[sl][None] * (lpB[sl][None] - L), gi[None], axis=2).mean(0)
        mass[sl] = np.trapezoid(np.exp(L), gi[None], axis=2).min(0)
        pc[sl] = -np.log(np.trapezoid(np.exp(L.mean(0)), gi, axis=1))
    return kl, pc, mass


def main():
    which, i0, i1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    rows = [r for r in json.load(open('results/e26_nets.json')) if r['set'] == which][i0:i1]
    z = np.load(os.environ.get('E26_DATA', 'results/e26_data.npz')); nb = min(NB_OF[which], len(z['y']))
    D = {k: z[k][:nb] for k in ('y', 'semb', 'qemb', 'vidx', 'o')}
    ystar = D['y'][:, W.QL]
    pts = ystar[:, None] + np.concatenate([[0.0], ED.DELTAS])[None]
    g = z['bmean'][:nb, None] + GRID[None]
    lpB = ED.mix_logp_points(z['bmeans'][:nb], z['bvars'], z['bwts'][:nb], g)
    out_dir = os.path.join(os.environ.get('E26_OUT', 'results/e26'), which); os.makedirs(out_dir, exist_ok=True)
    for row in rows:
        p = os.path.join(out_dir, row['tag'] + '.npz')
        if os.path.exists(p):
            continue
        t0 = time.time(); model, kind = load(row)
        Lall = np.zeros((nb, ED.NVAR, len(ED.DELTAS) + 1)); keep = {}          # last column: score at y*
        for k in range(ED.NVAR):
            X, valid = W.assemble(D['semb'], D['qemb'], D['o'][:, k], D['y'])
            hd = heads(model, kind, X, valid)
            Lall[:, k, :-1] = ED.logodds(logp(hd, pts)); Lall[:, k, -1] = score(hd, ystar)
            if k in ED.PAT or k in ED.SP:
                keep[k] = hd
        kl16, pc16, m16 = set_stats([keep[k] for k in ED.PAT], lpB, g)
        kl9, pc9, m9 = set_stats([keep[k] for k in ED.SP], lpB, g)
        out = dict(L=Lall[:, :12].astype(np.float32), tu=Lall[:, ED.PAT].var(1, ddof=1), tsp=Lall[:, ED.SP].var(1, ddof=1),
                   kl16=kl16, pc16=pc16, kl9=kl9, pc9=pc9, mass=np.minimum(m16, m9), nb=nb)
        np.savez(p + '.tmp.npz', **out); os.replace(p + '.tmp.npz', p)
        print(which, row['tag'], 'KL %.4f pc16 %.5f mass_min %.6f %.0fs' % (kl16.mean(), pc16.mean(), out['mass'].min(), time.time() - t0),
              flush=True)


if __name__ == '__main__':
    main()
