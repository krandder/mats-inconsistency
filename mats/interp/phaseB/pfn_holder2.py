"""Corrected rerun of the PFN Hölder analysis (Update 8) after review items 1-2.

Uses the corrected description chain (pfn_diag2.chain_gaussians2: preserved weights, equal budgets, shrunk covariances) and
the corrected waste-landing probabilities (diag2.json: 5 truth replicates, 3 replicates per description, censoring kept
as upper bounds). Event-implied lower bounds on D_a(p_l||q_l) use only UNCENSORED thresholds. In the binned output-level
divergence, censored description tail masses enter as their upper bounds, which keeps the binned D_a a lower bound.
usage: python pfn_holder2.py RUN_DIR -> RUN_DIR/holder2.json
"""
import json, math, os, sys
import numpy as np
import torch
import nnoise as N
from pfn_diag import SimMean
from pfn_diag2 import chain_gaussians2
from pfn_holder import push_samples, binned_renyi
from pfn_rare import prior

torch.set_num_threads(1)
ALPHAS = [2.0, 4.0, 8.0]


def run(run_dir):
    model, meta = N.load_run(run_dir); sim = SimMean(model, meta); rng = np.random.default_rng(11)
    J = json.load(open(os.path.join(run_dir, 'diag2.json')))
    t = np.array(J['t']); _, ui = np.unique(t, return_index=True); ui = np.sort(ui); ts = t[ui]
    lt = np.log2(np.array([r['p'] for r in J['truth']]))[:, ui]; P = 2.0 ** lt.mean(0)
    Yp = np.concatenate([sim.score_batched(*prior(sim, 8192, rng)) for _ in range(32)])
    common_p = [0.5, 0.25, 0.1, 0.05]; tc = np.quantile(Yp, 1 - np.array(common_p))
    out = dict(sigma=meta['sigma'], t=ts.tolist(), p=P.tolist(), p_log2_range=[lt.min(0).tolist(), lt.max(0).tolist()], rows=[])
    for fam, cells in (('gauss', False), ('bayesq', 'bayesq')):
        ch, _ = chain_gaussians2(sim, np.random.default_rng(5), cells=cells)
        for cut in ['emb'] + [f'r{l}' for l in range(1, model.depth)]:
            v = J['desc'][f'{cut}_{fam}']
            Pr = np.array([r['p'] for r in v['reps']])[:, ui]; Cr = np.array([r['censored'] for r in v['reps']])[:, ui]
            cens = Cr.any(0); Pl = np.where(cens, Pr.max(0), 2.0 ** np.log2(Pr).mean(0))
            Yq = push_samples(sim, ch[cut], cut, 2 ** 16, np.random.default_rng(13))
            Pc = np.array([(Yq > x).mean() for x in tc])
            row = dict(cut=cut, fam=fam, log2_Pl_over_p=np.log2(Pl / P).tolist(), censored=cens.tolist(), q_over_p_common=(Pc / np.array(common_p)).tolist())
            for a in ALPHAS:
                ok = ~cens
                lb = float(np.max((a / (a - 1)) * np.log2(P[ok]) - np.log2(Pl[ok]))) if ok.any() else float('nan')
                if cens.any():                                   # lump the censored tail into one bin [t_k, inf) with q-mass <= bound
                    k = int(np.argmax(cens)); tsk, Pk, Plk = ts[:k + 1], P[:k + 1], np.minimum.accumulate(Pl[:k + 1])
                else:
                    tsk, Pk, Plk = ts, P, np.minimum.accumulate(Pl)
                d1 = binned_renyi(Yp, Yq, tsk, Pk, Plk, a)
                best = (1 - 1 / a) * (np.log2(Pl) + d1)
                row[f'a{int(a)}'] = dict(event_implied_lb=max(lb, 0.0), D1d=d1, best_log2_bound=best.tolist(),
                                         best_log2_bound_common=((1 - 1 / a) * (np.log2(np.maximum(Pc, 1e-300)) + d1)).tolist())
            out['rows'].append(row)
            print(cut, fam, 'log2 Pl/p', np.round(row['log2_Pl_over_p'], 1), 'cens', cens.astype(int).tolist(), 'common q/p', np.round(Pc / np.array(common_p), 2),
                  {f'a{int(a)}': (round(row[f'a{int(a)}']['event_implied_lb'], 1), round(row[f'a{int(a)}']['D1d'], 2)) for a in ALPHAS}, flush=True)
            json.dump(out, open(os.path.join(run_dir, 'holder2.json'), 'w'), indent=1)


if __name__ == '__main__':
    run(sys.argv[1])
