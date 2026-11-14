"""Hölder per-event bounds on the noisy tiny PFN (2L, sigma 0.05), PREREG-B amendment 3.

Per-layer Rényi divergences of 512-dim descriptions cannot be estimated (node mixtures need ~2^I nodes), so this reports:
- truth p(A_t) for A_t = {mu_model > t} (subset-simulation ladder from pfn_diag.py + plain MC for common events);
- for descriptions q_l at cuts emb / r1 / r2 (Gaussian; Bayes-mean cells), P_l(A) = event probability when h_l ~ q_l and the
  rest is the true noisy network (subset simulation; plain sampling for common events);
- event-implied LOWER bounds D_a(p_l||q_l) >= (a/(a-1)) log2 p(A) - log2 P_l(A) (Hölder with E_q g^a' <= P_l(A));
- the binned output-level D_a(p_Y||q_Y) (Y = predictive mean): a lower bound on D_a(p_l||q_l) by data processing, hence on
  any certificate's divergence; and the best-case Hölder bound (1 - 1/a)[log2 P_l(A) + D_a(p_Y||q_Y)] it would give.
usage: python pfn_holder.py RUN_DIR -> RUN_DIR/holder.json
"""
import json, math, os, sys, time
import numpy as np
import torch
import nnoise as N
import world as W
from pfn_diag import SimMean, chain_gaussians
from pfn_rare import prior
from dec import sus, interp_p

torch.set_num_threads(1)
ALPHAS = [2.0, 4.0, 8.0]


def push_samples(sim, comps, cut, n, rng):
    ws = np.array([c[0] for c in comps]); ws /= ws.sum(); later = [s for s in sim.sites if s[1:].isdigit() and int(s[1:]) > (0 if cut == 'emb' else int(cut[1:]))]
    out = []
    for i in range(0, n, 4096):
        k = min(4096, n - i); c = rng.choice(len(ws), k, p=ws); z = torch.as_tensor(rng.standard_normal((k, 512)))
        H = torch.empty(k, 512, dtype=torch.float32)
        for ci in np.unique(c):
            sel = torch.as_tensor(c == ci); H[sel] = (comps[ci][1] + z[sel] @ comps[ci][2].T).float()
        zt = {s: torch.as_tensor(rng.standard_normal((k,) + sim.shapes[s]).astype(np.float32)) for s in later}
        out.append(sim.rest_from(cut, H.view(k, 16, sim.m.d), zt))
    return np.concatenate(out)


def sus_cut(sim, comps, cut, rng_seed):
    ws = np.array([c[0] for c in comps]); ws /= ws.sum(); model = sim.m
    later = [s for s in sim.sites if s[1:].isdigit() and int(s[1:]) > (0 if cut == 'emb' else int(cut[1:]))]
    keys = later
    def samp(n, g):
        return (g.choice(len(ws), n, p=ws), g.standard_normal((n, 512)).astype(np.float32)) + tuple(g.standard_normal((n,) + sim.shapes[s]).astype(np.float32) for s in keys)
    def score(st):
        c, z = st[0], st[1]; H = torch.empty(len(c), 512, dtype=torch.float32)
        for ci in np.unique(c):
            sel = torch.as_tensor(c == ci); H[sel] = (comps[ci][1] + torch.as_tensor(z[c == ci], dtype=torch.float64) @ comps[ci][2].T).float()
        out = []
        for i in range(0, len(c), 4096):
            zt = {s: torch.as_tensor(st[2 + j][i:i + 4096]) for j, s in enumerate(keys)}
            out.append(sim.rest_from(cut, H[i:i + 4096].view(-1, 16, model.d), zt))
        return np.concatenate(out)
    def move(cur, rho, g):
        c = cur[0].copy(); sw = g.random(len(c)) < 0.2; c[sw] = g.choice(len(ws), sw.sum(), p=ws)
        f = lambda a: (rho * a + math.sqrt(1 - rho ** 2) * g.standard_normal(a.shape)).astype(np.float32)
        return (c,) + tuple(f(a) for a in cur[1:])
    cv, _ = sus(score, samp, move, np.random.default_rng(rng_seed), N=2000, max_levels=12)
    return cv


def binned_renyi(Yp, Yq, ladder_t, Sp, Sq, a, nb=100):
    """D_a(p_Y||q_Y) (bits) on a binning: bulk = pooled-quantile bins below t_split (MC), tail = ladder bins (survival)."""
    ladder_t = np.asarray(ladder_t); Sp = np.asarray(Sp); Sq = np.asarray(Sq)
    k0 = int(np.argmin(np.abs(np.log2(np.maximum(Sp, 1e-300)) + 10)))          # split at p ~ 2^-10
    ts = ladder_t[k0:]; sp = Sp[k0:]; sq = Sq[k0:]
    pool = np.r_[Yp[Yp < ts[0]], Yq[Yq < ts[0]]]
    edges = np.unique(np.quantile(pool, np.linspace(0, 1, nb + 1)[1:-1]))
    bp = np.bincount(np.searchsorted(edges, Yp[Yp < ts[0]]), minlength=len(edges) + 1).astype(float)
    bq = np.bincount(np.searchsorted(edges, Yq[Yq < ts[0]]), minlength=len(edges) + 1).astype(float)
    bp = bp / bp.sum() * (1 - sp[0]); bq = bq / max(bq.sum(), 1) * (1 - sq[0])
    tp = np.r_[sp[:-1] - sp[1:], sp[-1]]; tq = np.r_[sq[:-1] - sq[1:], sq[-1]]
    tp = np.maximum(tp, 0); sq_left = np.r_[sq[:-1], sq[-1]]
    tq = np.where(tq > 0, tq, sq_left)                          # zero / undetermined q-bin mass -> its upper bound S_q(left edge)
    P = np.r_[bp, tp]; Q = np.r_[np.maximum(bq, 3.0 / len(Yq)), tq]          # empty q bins -> upper bound 3/n (keeps D a lower bound)
    m = P > 0
    lv = a * np.log(P[m]) + (1 - a) * np.log(Q[m])
    mx = lv.max()
    return float((mx + np.log(np.exp(lv - mx).sum())) / (a - 1) / math.log(2))


def run(run_dir):
    model, meta = N.load_run(run_dir); sim = SimMean(model, meta); rng = np.random.default_rng(11)
    Dg = json.load(open(os.path.join(run_dir, 'diag.json')))
    ts, P = np.array(Dg['t']), np.array(Dg['p_true'])
    # plain MC under p
    t0 = time.time(); Yp = np.concatenate([sim.score_batched(*prior(sim, 8192, rng)) for _ in range(32)]); rate = len(Yp) / (time.time() - t0)
    common_p = [0.5, 0.25, 0.1, 0.05]; tc = np.quantile(Yp, 1 - np.array(common_p))
    out = dict(sigma=meta['sigma'], t_ladder=ts.tolist(), p_ladder=P.tolist(), t_common=tc.tolist(), p_common=common_p, rate=rate, rows=[])
    cuts = ['emb'] + [f'r{l}' for l in range(1, model.depth + 1)]
    for fam, cells in (('gauss', False), ('bayesq', 'bayesq')):
        ch = chain_gaussians(sim, np.random.default_rng(5), cells=cells)
        for cut in cuts:
            key = f'{cut}_{fam}'
            if key in Dg['diag']:
                Pl = np.array(Dg['diag'][key]['P_l'])
            else:
                cv = sus_cut(sim, ch[cut], cut, 77); Pl = interp_p(cv, ts)
            Yq = push_samples(sim, ch[cut], cut, 2 ** 16, np.random.default_rng(13))
            Pc = np.array([(Yq > t).mean() for t in tc])
            row = dict(cut=cut, fam=fam, P_l=Pl.tolist(), P_common=Pc.tolist(),
                       q_over_p_common=(Pc / np.array(common_p)).tolist(), log2_Pl_over_p=np.log2(np.maximum(Pl, 1e-300) / P).tolist())
            for a in ALPHAS:
                lb = float(np.max((a / (a - 1)) * np.log2(P) - np.log2(np.maximum(Pl, 1e-300))))
                d1 = binned_renyi(Yp, Yq, ts, P, Pl, a)
                best = (1 - 1 / a) * (np.log2(np.maximum(Pl, 1e-300)) + d1)
                best_c = (1 - 1 / a) * (np.log2(np.maximum(Pc, 1e-300)) + d1)
                row[f'a{int(a)}'] = dict(event_implied_lb=lb, D1d=d1, best_log2_bound=best.tolist(), best_log2_bound_common=best_c.tolist(),
                                         kept_fraction=(np.minimum(0, best) / np.log2(P)).tolist())
            out['rows'].append(row)
            print(key, 'log2(P_l/p) ladder', np.round(row['log2_Pl_over_p'], 1), 'common q/p', np.round(Pc / np.array(common_p), 2),
                  {f'a{int(a)}': (round(row[f'a{int(a)}']['event_implied_lb'], 1), round(row[f'a{int(a)}']['D1d'], 2)) for a in ALPHAS}, flush=True)
            json.dump(out, open(os.path.join(run_dir, 'holder.json'), 'w'), indent=1)


if __name__ == '__main__':
    run(sys.argv[1])
