"""E72 exploration, part 2 (eighth agent; DEV NETS ONLY, disclosed in the E72 amendment; not a registered test): why the idealised
reduced pooling model (M1) misses the departure that the real-weight reader (M2) recovers.
The real pre-LN logit of own key j (summary row s, head h) is L_j g_j + K_j r_j g_j + const, with g_j = A_j / n_j and token-specific
L_j = sqrt(d / dh) q.W_K(gamma a_j) / A_j and K_j = sqrt(d / dh) q.W_K(gamma u~) / A_j (E63). M1 replaces L_j, K_j by their means and
g_j by (1 + r_j^2 / c^2)^(-1/2). On E27's first 2,000 bases, per pooling head: the mean L and K; the sd of L_j between sources (of the
source mean) and within a source (across its four own keys); the share of L_j's within-source variance explained linearly by the
label y_j; the same for K_j. Then CB readers of reduced models with noise in the logits (simulated, r only plus Gaussian noise):
  M1w   L_j = L + sd_within * eps_j (token-level, iid)
  M1bw  L_j = L + sd_between * eta + sd_within * eps_j (eta shared by the source's four keys)
  M1bwK as M1bw, with K_j = K (1 + cvK * eps'_j), cvK the within-source coefficient of variation of K_j
usage: python e72x_ladder.py OUTJSON TAG=RUNDIR [...]  (adds to OUTJSON)
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import world as W  # noqa: E402
import e63_run as E63  # noqa: E402
import e68_rpm as RPM  # noqa: E402
import lngeom as LG  # noqa: E402
import pfn_b1hw as BH  # noqa: E402

BH.install()
torch.set_num_threads(1)
S, QL = 3, 4


def measure(model, arm):
    z = np.load('results/e27_data.npz'); sl = slice(0, 2000)
    X, v = E63.F.make_X(z['o'][sl, 0], z['y'][sl], z['semb'][sl], z['qemb'][sl], arm)
    A, qh, blk = E63.block1_parts(model, X); H = A.shape[1]
    P = np.stack([A[:, :, E63.SUMM[s], E63.LABS[s]].sum(-1) for s in range(S)], 1).mean((0, 1))
    L, ratio, odd = E63.h3_logits(model, X, qh, blk)
    own_i = [[QL * s + q for q in range(QL)] for s in range(S)]
    Lown = np.stack([L[:, :, s, own_i[s]] for s in range(S)], 1); K = np.stack([odd[:, :, s, own_i[s]] for s in range(S)], 1)   # (B, S, H, 4)
    y = np.repeat(z['y'][sl, None, :QL], S, 1)                                                                                     # (B, S, 4)
    g = LG.geom(model, X); pool = [h for h in range(H) if P[h] >= 0.5]
    out = dict(pool=pool, c=g['c_median'], heads={})
    for h in pool:
        Lh, Kh = Lown[:, :, h], K[:, :, h]
        dl = Lh - Lh.mean(-1, keepdims=True); dk = Kh - Kh.mean(-1, keepdims=True); dy = y - y.mean(-1, keepdims=True)
        bl = (dl * dy).sum() / (dy ** 2).sum(); bk = (dk * dy).sum() / (dy ** 2).sum()
        out['heads'][h] = dict(L=float(Lh.mean()), K=float(Kh.mean()), sdL_between=float(Lh.mean(-1).std()), sdL_within=float(np.sqrt((dl ** 2).mean() * 4 / 3)),
                               R2L_y=float(1 - ((dl - bl * dy) ** 2).sum() / (dl ** 2).sum()), slopeL_y=float(bl),
                               cvK_within=float(np.sqrt((dk ** 2).mean() * 4 / 3) / abs(Kh.mean())), R2K_y=float(1 - ((dk - bk * dy) ** 2).sum() / (dk ** 2).sum()))
    return out


def noisy_stat(heads, c, kind, seed):
    def stat(r):
        rng = np.random.default_rng(seed)
        g = 1.0 / np.sqrt(1.0 + r ** 2 / c ** 2); out = []
        for hd in heads:
            eps = rng.standard_normal(r.shape); eta = rng.standard_normal(r.shape[:-1] + (1,)); epk = rng.standard_normal(r.shape)
            Lj = hd['L'] + hd['sdL_within'] * eps + (hd['sdL_between'] * eta if kind in ('M1bw', 'M1bwK') else 0.0)
            Kj = hd['K'] * (1 + hd['cvK_within'] * epk) if kind == 'M1bwK' else hd['K']
            zz = (Lj + Kj * r) * g; zz = zz - zz.max(-1, keepdims=True); w = np.exp(zz); w /= w.sum(-1, keepdims=True)
            out += [(w * g).sum(-1), (w * r * g).sum(-1)]
        return np.stack(out, -1)
    return stat


def main():
    outp = sys.argv[1]; res = json.load(open(outp)) if os.path.exists(outp) else {}
    z = np.load('results/e12_pairs.npz')
    for spec in sys.argv[2:]:
        tag, run = spec.split('=', 1)
        if tag in res:
            continue
        t0 = time.time(); model, arm, meta = E63.load(run, None)
        m = measure(model, arm); heads = [m['heads'][h] for h in m['pool']]
        for kind in ('M1w', 'M1bw', 'M1bwK'):
            m[kind] = RPM.rom_of(RPM.fit_reader(noisy_stat(heads, m['c'], kind, 7202)), z) if heads else None
        m['seconds'] = time.time() - t0; res[tag] = m
        json.dump(res, open(outp, 'w'), indent=1)
        print(tag, json.dumps(m), flush=True)


if __name__ == '__main__':
    main()
