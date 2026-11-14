"""Corrected rerun of the PFN waste-landing diagnosis (review items 1-2, 2026-10-28 23:30Z).

Fixes relative to pfn_diag.py:
- component weights are PRESERVED through the chain (they were re-estimated from node fractions after oversampling rare
  cells to >= 256 nodes, inflating the rarest Bayes-mean cell ~31x / ~5 bits);
- equal budgets: both families start from the same 2^16 prior episodes and use the same push budget;
- per-cell covariances are shrunk to the pooled within-cell covariance with weight n_c / (n_c + 512) (dimension 512);
- censoring: a threshold beyond the deepest level a subset-simulation run reached is reported as CENSORED with the upper
  bound P <= (probability of the deepest level reached), never as a probability (the old code returned ~1e-300);
- 5 truth replicates and 3 replicates per (cut, family), with per-level acceptance rates (mixing diagnostics); every
  number is reported with its replicate range.
The Bayes-mean family is an ORACLE experiment: the indexing variable (Bayes mean) is given, not found from the weights.
usage: python pfn_diag2.py RUN_DIR -> RUN_DIR/diag2.json
"""
import json, math, os, sys, time
import numpy as np
import torch
from scipy.stats import norm, qmc
import nnoise as N
import world as W
from pfn_rare import prior
from pfn_diag import SimMean

torch.set_num_threads(1)


def sus2(score_fn, sample_fn, move_fn, rng, N=2000, p0=0.1, max_levels=13):
    """Subset simulation returning levels (gamma_k, p_k), per-level acceptance, the final particles' scores and passes."""
    st = sample_fn(N, rng); s = score_fn(st); passes = N; pk = 1.0; levels = []; acc_hist = []; rho = 0.8; last = 0.3
    first = s.copy()
    for k in range(max_levels):
        o = np.argsort(-s); nk = int(p0 * N); g = s[o[nk - 1]]
        if k > 0 and g <= levels[-1][0] + 1e-12:                       # stalled: no progress
            acc_hist.append(dict(level=k, stalled=True)); break
        seeds = tuple(a[o[:nk]] for a in st); ss = s[o[:nk]]
        pk *= p0; levels.append((float(g), pk))
        reps = N // nk; cur = tuple(np.repeat(a, reps, 0) for a in seeds); cs = np.repeat(ss, reps)
        rho = min(0.98, rho + 0.05) if last < 0.2 else (max(0.5, rho - 0.05) if last > 0.5 else rho)
        acc = []
        for step in range(8):
            prop = move_fn(cur, rho, rng); ps = score_fn(prop); passes += len(ps)
            ok = ps >= g; acc.append(float(ok.mean()))
            cur = tuple(np.where(ok.reshape((-1,) + (1,) * (a.ndim - 1)), b_, a) for a, b_ in zip(cur, prop)); cs = np.where(ok, ps, cs)
        st, s = cur, cs; last = float(np.mean(acc))
        acc_hist.append(dict(level=k, gamma=float(g), p=pk, rho=rho, accept=last, unique_frac=float(len(np.unique(s)) / len(s))))
    return dict(levels=levels, final_scores=s.tolist(), first_scores=first.tolist(), passes=passes, diag=acc_hist, p0=p0)


def tail_prob(res, t):
    """SuS estimate of P(score > t) and whether it is censored (t beyond every level reached)."""
    levels = res['levels']; first = np.array(res['first_scores']); fin = np.array(res['final_scores'])
    if not levels or t <= levels[0][0]:
        return float((first > t).mean()), False
    ks = [i for i, (g, p) in enumerate(levels) if g <= t]
    k = ks[-1]; g, pk = levels[k]
    if k == len(levels) - 1:                                            # within the last level's population
        frac = float((fin > t).mean())
        return (pk * frac, False) if frac > 0 else (pk, True)              # censored: report the upper bound P < pk
    nxt_g, nxt_p = levels[k + 1]
    # between level k and k+1: interpolate log p linearly in t (standard SuS curve)
    lp = math.log(pk) + (t - g) / (nxt_g - g) * (math.log(nxt_p) - math.log(pk))
    return math.exp(lp), False


def chain_gaussians2(sim, rng, n=2 ** 16, cells=False, n0=512):
    m = sim.m; sg = sim.sg; out = {}
    d0 = W.sample_np(n, rng); X, valid = W.tokens_np(d0, rng, sim.ds)
    with torch.no_grad():
        E = m.f_emb(torch.from_numpy(X)).reshape(n, -1).double()
    if cells == 'bayesq':
        _, _, mom, _ = W.references(d0); mb = mom['bayes'][0]
        qs = np.r_[np.linspace(1 / 16, 15 / 16, 15), 1 - 2.0 ** -np.arange(5, 14)]
        lab = np.searchsorted(np.quantile(mb, np.sort(qs)), mb)
    else:
        lab = np.zeros(n, int)
    ids = np.unique(lab); weights = np.array([(lab == c).mean() for c in ids])
    def fit(H, lab, weights):
        D = H.shape[1]; comps = []; mus = []; covs = []; ns = []
        for c in ids:
            Hc = H[torch.as_tensor(lab == c)]; mus.append(Hc.mean(0)); ns.append(len(Hc))
            covs.append(torch.cov(Hc.T) if len(Hc) > 1 else torch.zeros(D, D, dtype=torch.float64))
        pooled = sum((nn_ - 1) * C for nn_, C in zip(ns, covs)) / max(sum(ns) - len(ns), 1)
        for w, mu, C, nn_ in zip(weights, mus, covs, ns):
            a = nn_ / (nn_ + n0); Cs = a * C + (1 - a) * pooled + sg ** 2 * torch.eye(D, dtype=torch.float64)
            comps.append((float(w), mu, torch.linalg.cholesky(Cs + 1e-6 * torch.eye(D, dtype=torch.float64))))
        return comps
    out['emb'] = fit(E, lab, weights)
    allow = m.allow_mask(torch.ones(n, 16, dtype=torch.bool))
    cur = 'emb'
    for l in range(1, m.depth + 1):
        Hs, labs = [], []
        z = torch.as_tensor(norm.ppf(np.clip(qmc.Sobol(512, scramble=True, seed=l).random(n), 1e-9, 1 - 1e-9)), dtype=torch.float64)
        for ci, (w, mu, Lc) in enumerate(out[cur]):
            k = max(256, int(round(w * n)))                               # sample budget per cell (weights NOT taken from k)
            h = (mu + z[:k] @ Lc.T).float().view(k, 16, -1)
            with torch.no_grad():
                xa = m.f_a(l, h, allow[:k]) + sg * torch.randn(h.shape)
                mm = m.f_m(l, xa) + sg * torch.randn(k, 16, 2 * m.d)
                x = m.f_r(l, xa, mm)
            Hs.append(x.reshape(k, -1).double()); labs.append(np.full(k, ids[ci]))
        H = torch.cat(Hs); lab_ = np.concatenate(labs)
        cur = f'r{l}'; out[cur] = fit(H, lab_, weights)                  # weights preserved
    return out, weights


def run(run_dir, nrep_truth=5, nrep=3):
    model, meta = N.load_run(run_dir); sim = SimMean(model, meta)
    old = json.load(open(os.path.join(run_dir, 'diag.json')))
    ts = np.array(old['t'])
    out = dict(sigma=meta['sigma'], t=ts.tolist(), truth=[], desc={})
    def samp(n, g): return prior(sim, n, g)
    def score(st): return sim.score_batched(st[0], st[1])
    def move(cur, rho, g):
        u, v = cur; pu = rho * u + math.sqrt(1 - rho ** 2) * g.standard_normal(u.shape).astype(np.float32)
        pv = v.copy(); j = g.integers(0, W.S, len(v)); pv[np.arange(len(v)), j] = g.integers(0, W.NV, len(v)); return (pu, pv)
    for r in range(nrep_truth):
        res = sus2(score, samp, move, np.random.default_rng(500 + r)); est = [tail_prob(res, t) for t in ts]
        out['truth'].append(dict(rep=r, p=[e[0] for e in est], censored=[e[1] for e in est], diag=res['diag'], passes=res['passes']))
        print('truth rep', r, 'log2 p:', np.round(np.log2([e[0] for e in est]), 1), 'censored', sum(e[1] for e in est), flush=True)
        json.dump(out, open(os.path.join(run_dir, 'diag2.json'), 'w'), indent=1)
    for fam, cells in (('gauss', False), ('bayesq', 'bayesq')):
        ch, weights = chain_gaussians2(sim, np.random.default_rng(5), cells=cells)
        for cut in ['emb'] + [f'r{l}' for l in range(1, model.depth)]:
            comps = ch[cut]; ws = np.array([c[0] for c in comps]); ws /= ws.sum()
            later = [s for s in sim.sites if s[1:].isdigit() and int(s[1:]) > (0 if cut == 'emb' else int(cut[1:]))]
            def samp2(n, g, comps=comps, ws=ws, later=later):
                return (g.choice(len(ws), n, p=ws), g.standard_normal((n, 512)).astype(np.float32)) + tuple(g.standard_normal((n,) + sim.shapes[s]).astype(np.float32) for s in later)
            def score2(st, comps=comps, cut=cut, later=later):
                c, z = st[0], st[1]; H = torch.empty(len(c), 512, dtype=torch.float32)
                for ci in np.unique(c):
                    sel = torch.as_tensor(c == ci); H[sel] = (comps[ci][1] + torch.as_tensor(z[c == ci], dtype=torch.float64) @ comps[ci][2].T).float()
                o = []
                for i in range(0, len(c), 4096):
                    zt = {s: torch.as_tensor(st[2 + j][i:i + 4096]) for j, s in enumerate(later)}
                    o.append(sim.rest_from(cut, H[i:i + 4096].view(-1, 16, model.d), zt))
                return np.concatenate(o)
            def move2(cur, rho, g, ws=ws):
                c = cur[0].copy(); sw = g.random(len(c)) < 0.2; c[sw] = g.choice(len(ws), sw.sum(), p=ws)
                f = lambda a: (rho * a + math.sqrt(1 - rho ** 2) * g.standard_normal(a.shape)).astype(np.float32)
                return (c,) + tuple(f(a) for a in cur[1:])
            reps = []
            for r in range(nrep):
                res = sus2(score2, samp2, move2, np.random.default_rng(700 + 10 * r + len(cut))); est = [tail_prob(res, t) for t in ts]
                reps.append(dict(rep=r, p=[e[0] for e in est], censored=[e[1] for e in est], diag=res['diag'], passes=res['passes']))
            out['desc'][f'{cut}_{fam}'] = dict(weights=weights.tolist(), reps=reps)
            print(cut, fam, 'log2 P_l per rep (C = censored upper bound):',
                  [[('C<%.0f' % np.log2(p) if cflag else '%.1f' % np.log2(p)) for p, cflag in zip(rp['p'], rp['censored'])][::3] for rp in reps], flush=True)
            json.dump(out, open(os.path.join(run_dir, 'diag2.json'), 'w'), indent=1)


if __name__ == '__main__':
    run(sys.argv[1])
