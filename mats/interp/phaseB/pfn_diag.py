"""(1)/(2) on the noisy tiny PFN: rare output event B_t = {mu_model > t} (predictive mean of ONE noisy forward pass).
- Truth: subset simulation over all randomness (episode + every noise site).
- Description q_l at a residual cut l (full 16-token state, 512-dim): Gaussian moment-matched to the deductive chain
  (q_emb from the exact prior pushed through the embedding; q_{next} = moments of K q pushed by QMC nodes of q).
  Also a 27-cell version: one Gaussian per reliability combination v (named cells), weights 1/27.
- Diagnosis: P_l(B_t) = probability of the event when h_l ~ q_l and every later layer is the TRUE noisy network
  (subset simulation over (cell, z, later noise)). log2(P_l / p) is how much event mass the description of layer l
  creates or loses: if it is large, the waste lands on internal combinations that map to the rare event, and any
  bound 2^c q(B) is vacuous whatever c is.
- Sampling bound at equal compute.

usage: python pfn_diag.py RUN_DIR -> RUN_DIR/diag.json
"""
import json, math, os, sys, time
import numpy as np
import torch
from scipy.stats import beta, norm, qmc
import nnoise as N
import world as W
from pfn_rare import Sim, prior
from dec import sus, interp_p

torch.set_num_threads(1)


class SimMean(Sim):
    def score(self, u, vidx, want_parts=False):
        y, e, semb, qemb, Z = self.split(u)
        v = W.VLEV[vidx]; o = y[:, None, :] + e * np.sqrt(v)[:, :, None]
        X, valid = W.assemble(semb, qemb, o, y)
        with torch.no_grad():
            Xt = torch.from_numpy(X); self.allow = self.m.allow_mask(torch.from_numpy(valid))
            zt = {s: torch.as_tensor(Z[s], dtype=torch.float32) for s in self.sites}
            x = self.m.f_emb(Xt) + self.sg * zt['emb']
            return self.rest_from('emb', x, zt)

    def rest_from(self, cut, x, zt):
        """x: noisy state at residual cut ('emb' or 'r{l}'); zt: noise for the later sites. Returns predictive mean."""
        allow = self.m.allow_mask(torch.ones(x.shape[0], 16, dtype=torch.bool))
        l0 = 0 if cut == 'emb' else int(cut[1:])
        with torch.no_grad():
            for l in range(l0 + 1, self.m.depth + 1):
                xa = self.m.f_a(l, x, allow) + self.sg * zt[f'a{l}']
                mm = self.m.f_m(l, xa) + self.sg * zt[f'm{l}']
                x = self.m.f_r(l, xa, mm) + self.sg * zt[f'r{l}']
            mu, ls, lg = self.m.head(x)
            return (torch.softmax(lg, -1) * mu).sum(-1).numpy().astype(np.float64)


def chain_gaussians(sim, rng, n=8192, cells=False):
    n = 2 ** 16 if cells == 'bayesq' else n
    """Deductive Gaussian chain over residual cuts. Returns {cut: list of (weight, mean, chol)} (512-dim)."""
    m = sim.m; sg = sim.sg; out = {}
    # q_emb: from the exact prior (QMC over its Gaussian variables; v enumerated per cell or drawn)
    d0 = W.sample_np(n, rng); X, valid = W.tokens_np(d0, rng, sim.ds)
    with torch.no_grad():
        E = (m.f_emb(torch.from_numpy(X)) ).reshape(n, -1).double()
    if cells == 'bayesq':                                   # named cells: quantile bins of the Bayes mean, tail-refined
        _, _, mom, _ = W.references(d0); mb = mom['bayes'][0]
        qs = np.r_[np.linspace(1 / 16, 15 / 16, 15), 1 - 2.0 ** -np.arange(5, 14)]
        lab = np.searchsorted(np.quantile(mb, np.sort(qs)), mb)
    elif cells:
        lab = d0['vidx'] @ np.array([9, 3, 1])
    else:
        lab = np.zeros(n, int)
    def fit(H, lab):
        comps = []
        for c in np.unique(lab):
            Hc = H[torch.as_tensor(lab == c)]
            mu = Hc.mean(0); C = torch.cov(Hc.T) + sg ** 2 * torch.eye(H.shape[1], dtype=torch.float64)
            comps.append((float((lab == c).mean()), mu, torch.linalg.cholesky(C + 1e-6 * torch.eye(H.shape[1], dtype=torch.float64))))
        return comps
    out['emb'] = fit(E, lab)
    allow = m.allow_mask(torch.ones(n, 16, dtype=torch.bool))
    cur = 'emb'
    for l in range(1, m.depth + 1):
        # push QMC nodes of q_cur (per cell, keeping the cell label) through block l with its noise
        Hs, labs = [], []
        z = torch.as_tensor(norm.ppf(np.clip(qmc.Sobol(512, scramble=True, seed=l).random(n), 1e-9, 1 - 1e-9)), dtype=torch.float64)
        for ci, (w, mu, Lc) in enumerate(out[cur]):
            k = max(256, int(round(w * n)))
            h = (mu + z[:k] @ Lc.T).float().view(k, 16, -1)
            with torch.no_grad():
                xa = m.f_a(l, h, allow[:k]) + sg * torch.randn(h.shape)
                mm = m.f_m(l, xa) + sg * torch.randn(k, 16, 2 * m.d)
                x = m.f_r(l, xa, mm)
            Hs.append(x.reshape(k, -1).double()); labs.append(np.full(k, ci))
        H = torch.cat(Hs); lab = np.concatenate(labs)
        cur = f'r{l}'; out[cur] = fit(H, lab)
    return out


def run(run_dir):
    model, meta = N.load_run(run_dir); sim = SimMean(model, meta); rng = np.random.default_rng(321)
    t0 = time.time()
    # plain MC (validation) and its rate
    nmc = 2 ** 18; ss = np.concatenate([sim.score_batched(*prior(sim, 8192, rng)) for _ in range(nmc // 8192)])
    rate = nmc / (time.time() - t0)
    # truth by subset simulation (3 replicates)
    def samp(n, g): u, v = prior(sim, n, g); return (u, v)
    def score(st): return sim.score_batched(st[0], st[1])
    def move(cur, rho, g):
        u, v = cur; pu = rho * u + math.sqrt(1 - rho ** 2) * g.standard_normal(u.shape).astype(np.float32)
        pv = v.copy(); j = g.integers(0, W.S, len(v)); pv[np.arange(len(v)), j] = g.integers(0, W.NV, len(v)); return (pu, pv)
    truth = []; t1 = time.time()
    for rep in range(3):
        c, passes = sus(score, samp, move, np.random.default_rng(10 + rep), N=2000, max_levels=12); truth.append(c)
        print('truth rep', rep, 'passes', passes, flush=True)
    t_sus = (time.time() - t1) / 3
    tq = np.quantile(ss, 1 - 2.0 ** -6); tmax = min(max(t for t, p in c if p > 0) for c in truth)
    tg = np.linspace(tq, tmax, 300)
    Pg = np.exp(np.mean([np.log(np.maximum(interp_p(c, tg), 1e-300)) for c in truth], 0))
    ks = [k for k in range(6, 42, 2) if Pg.min() <= 2.0 ** -k]
    ts = np.array([tg[np.argmin(np.abs(np.log2(Pg) + k))] for k in ks])
    P = np.exp(np.mean([np.log(np.maximum(interp_p(c, ts), 1e-300)) for c in truth], 0))
    res = dict(sigma=meta['sigma'], depth=meta['depth'], t=ts.tolist(), p_true=P.tolist(), p_mc=[float((ss > t).mean()) for t in ts],
               p_reps=[interp_p(c, ts).tolist() for c in truth], rate=rate, t_sus=t_sus, nmc=nmc, diag={})
    n_eq = int(rate * t_sus)
    res['samp_bound_eq_sus'] = [float(beta.ppf(0.95, int(round(p * n_eq)) + 1, n_eq - int(round(p * n_eq)))) for p in P]
    print('truth log2 p:', np.round(np.log2(P), 1), flush=True)
    json.dump(res, open(os.path.join(run_dir, 'diag.json'), 'w'), indent=1)
    # descriptions and the per-cut diagnosis
    later = {cut: [s for s in sim.sites if s[1:].isdigit() and int(s[1:]) > (0 if cut == 'emb' else int(cut[1:]))] for cut in ['emb'] + [f'r{l}' for l in range(1, model.depth)]}
    modes = os.environ.get('DIAG_MODES', 'gauss,cells27').split(',')
    if os.path.exists(os.path.join(run_dir, 'diag.json')) and os.environ.get('DIAG_MODES'):
        old = json.load(open(os.path.join(run_dir, 'diag.json'))); res['diag'] = old.get('diag', {})
    for cells in [{'gauss': False, 'cells27': True, 'bayesq': 'bayesq'}[m_] for m_ in modes]:
        ch = chain_gaussians(sim, np.random.default_rng(5), cells=cells)
        for cut in ['emb'] + [f'r{l}' for l in range(1, model.depth)]:
            comps = ch[cut]; ws = np.array([c[0] for c in comps]); ws /= ws.sum(); rs = later[cut]
            shp = {s: sim.shapes[s] for s in rs}
            def samp2(n, g, comps=comps, ws=ws, shp=shp):
                return (g.choice(len(ws), n, p=ws), g.standard_normal((n, 512)).astype(np.float32),
                        {s: g.standard_normal((n,) + shp[s]).astype(np.float32) for s in shp})
            def score2(st, comps=comps, cut=cut):
                c, z, Z = st; H = torch.empty(len(c), 512, dtype=torch.float32)
                for ci in np.unique(c):
                    sel = torch.as_tensor(c == ci); H[sel] = (comps[ci][1] + torch.as_tensor(z[c == ci], dtype=torch.float64) @ comps[ci][2].T).float()
                out = []
                for i in range(0, len(c), 4096):
                    zt = {s: torch.as_tensor(Z[s][i:i + 4096]) for s in Z}
                    out.append(sim.rest_from(cut, H[i:i + 4096].view(-1, 16, model.d), zt))
                return np.concatenate(out)
            def move2(cur, rho, g, ws=ws):
                c, z, Z = cur; c2 = c.copy(); sw = g.random(len(c)) < 0.2; c2[sw] = g.choice(len(ws), sw.sum(), p=ws)
                f = lambda a: (rho * a + math.sqrt(1 - rho ** 2) * g.standard_normal(a.shape)).astype(np.float32)
                return (c2, f(z), {s: f(a) for s, a in Z.items()})
            # sus() handles tuples of arrays; flatten the dict into the tuple
            keys = list(shp)
            pack = lambda st: (st[0], st[1]) + tuple(st[2][k] for k in keys)
            unpack = lambda tp: (tp[0], tp[1], {k: tp[2 + i] for i, k in enumerate(keys)})
            cv, passes = sus(lambda tp: score2(unpack(tp)), lambda n, g: pack(samp2(n, g)), lambda tp, rho, g: pack(move2(unpack(tp), rho, g)),
                             np.random.default_rng(77), N=2000, max_levels=12)
            Pl = interp_p(cv, ts)
            key = f"{cut}_{'bayesq' if cells == 'bayesq' else ('cells27' if cells else 'gauss')}"
            res['diag'][key] = dict(P_l=Pl.tolist(), log2_ratio=np.log2(np.maximum(Pl, 1e-300) / P).tolist())
            print(key, 'log2(P_l/p):', np.round(np.log2(np.maximum(Pl, 1e-300) / P), 1), flush=True)
            json.dump(res, open(os.path.join(run_dir, 'diag.json'), 'w'), indent=1)


if __name__ == '__main__':
    run(sys.argv[1])
