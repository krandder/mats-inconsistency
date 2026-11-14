"""Pre-training audit: (1) value of reliability weighting and of modelling coupling, per world and N;
(2) epsilon-rank of the per-source (and per-pair) log-likelihood families: the message size a sum-pooling tree needs
to represent Bayes. Writes audit/*.json."""
import json, math, os, sys
import numpy as np
import sfworld as W

os.makedirs('audit', exist_ok=True)


def gls_equal(d):
    """Equal-weight Gaussian rule at the prior-mean variance (reliability-blind)."""
    vbar = W.VLEV.mean(); ost = d['o'][:, :, W.QL]
    prec = 1 + d['N'] / vbar; m = ost.sum(1) / vbar / prec
    ys = d['y'][:, W.QL]
    return 0.5 * np.log(2 * np.pi / prec) + 0.5 * (ys - m) ** 2 * prec


def value_table(n=3000):
    rows = []
    cfgs = [dict(mode='indep', N=N) for N in (3, 6, 12, 24)]
    cfgs += [dict(mode='groups', N=N, rho=r, gsize=4) for N in (4, 8, 16, 24) for r in (0.5, 0.8)]
    cfgs += [dict(mode='pairs', N=N, rho=r, a1=1.0, a2=1.0) for N in (4, 6, 8) for r in (0.5, 0.8, 0.95)]
    for c in cfgs:
        rng = np.random.default_rng(123)
        kw = {k: v for k, v in c.items() if k != 'N'}
        d = W.sample(n, c['N'], rng, **kw)
        B = W.bayes(d)
        row = dict(c, bayes_nll=float(B['nll'].mean()), eq_gap=float((gls_equal(d) - B['nll']).mean()))
        if c['mode'] != 'indep':
            I = W.indep_bayes(d)
            g = I['nll'] - B['nll']
            row.update(coupling_gap=float(g.mean()), coupling_gap_se=float(g.std() / math.sqrt(n)),
                       mean_miss=float(1 - np.corrcoef(I['mean'], B['mean'])[0, 1] ** 2))
        rows.append(row); print(json.dumps(row), flush=True)
    json.dump(rows, open('audit/value_table.json', 'w'), indent=1)


def eps_rank(n=20000, ngrid=601, span=4.0):
    """Family F = { y -> log L_s(y; o*, ss) } over sources drawn from the prior, on the window of y where the
    posterior of a single-source episode can put mass ([-span, span]). Weighted by the prior density of y*.
    Returns singular values of the centred (per-row constant removed: normalisation is free) matrix."""
    rng = np.random.default_rng(9)
    d = W.sample(n, 1, rng)
    R = d['o'][:, :, :W.QL] - d['y'][:, None, :W.QL]
    yg = np.tile(np.linspace(-span, span, ngrid), (n, 1))
    L = W._single_ll(R, d['o'][:, :, W.QL], yg)[:, 0]                       # (n, G)
    wy = np.exp(-0.5 * yg[0] ** 2); wy /= wy.sum()
    Lc = L - (L * wy).sum(1, keepdims=True)                                 # remove each row's constant
    A = Lc * np.sqrt(wy)[None]
    s = np.linalg.svd(A / math.sqrt(n), compute_uv=False)
    # tolerance: RMS (over sources and prior-weighted y) error of the rank-r approximation of a single log-lik
    tail = np.sqrt(np.maximum(0, np.cumsum((s ** 2)[::-1])[::-1]))           # tail[r] = rms error using r comps
    out = dict(sv=s[:40].tolist(), rms_err_rank=[float(tail[r]) if r < len(tail) else 0.0 for r in range(30)])
    return out


def eps_rank_pairs(rho, n=20000, ngrid=601, span=4.0):
    rng = np.random.default_rng(11)
    d = W.sample(n, 2, rng, mode='groups', rho=rho, gsize=2)
    R = d['o'][:, :, :W.QL] - d['y'][:, None, :W.QL]
    yg = np.tile(np.linspace(-span, span, ngrid), (n, 1))
    L = W._group_ll(R, d['o'][:, :, W.QL], yg, rho)
    wy = np.exp(-0.5 * yg[0] ** 2); wy /= wy.sum()
    Lc = L - (L * wy).sum(1, keepdims=True)
    s = np.linalg.svd(Lc * np.sqrt(wy)[None] / math.sqrt(n), compute_uv=False)
    tail = np.sqrt(np.maximum(0, np.cumsum((s ** 2)[::-1])[::-1]))
    return dict(sv=s[:40].tolist(), rms_err_rank=[float(tail[r]) if r < len(tail) else 0.0 for r in range(30)])


if __name__ == '__main__':
    what = sys.argv[1]
    if what == 'value':
        value_table()
    elif what == 'rank':
        out = dict(single=eps_rank())
        for r in (0.5, 0.8):
            out[f'group2_rho{r}'] = eps_rank_pairs(r)
        json.dump(out, open('audit/eps_rank.json', 'w'), indent=1)
        for k, v in out.items():
            print(k, 'rms err by rank', [round(x, 4) for x in v['rms_err_rank'][:14]])
