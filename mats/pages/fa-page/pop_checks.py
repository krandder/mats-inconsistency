"""Population checks for the World FA explainer page.
1. History-blind level on the pilot's validation distribution (T = 0 w.p. 0.1, else log-uniform integer in [1, 128]):
   E[NLL(bayes_T0) - NLL(bayes)], so the pilot's validation-gap curve can be read against it.
2. The slider formula: expected log-loss cost of correlation-blind weights when Sigma is known,
   E KL(N(m, V) || N(m', V')) in closed form, against Monte Carlo NLL(diag_bayes) - NLL(oracle).
3. Pilot tables: diag.json and the validation curves (meta.json) for the 8 pilot arms -> pilot.json.
usage: OMP_NUM_THREADS=2 /home/agent/.venvs/aggext/bin/python pop_checks.py
"""
import json
import math
import os
import sys

WF = '/home/agent/workspaces/mats/work/world-fa'
sys.path[:0] = [WF, '/home/agent/workspaces/mats/work/worlds']
import numpy as np
import torch

import fa_core as C
import harness as H

torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__))
npy = lambda p: tuple(x.detach().cpu().numpy().astype(np.float64) for x in p)


def fam(nneg):
    return {"prior": "f", "f_templates": [{"blocks": [[2, 0.8, nneg]], "marg": 1.0}]}


def val_level(c, E=20000, seed=7):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        Te = C.sample_T(E, dict(c, T_train=(0, 128), p_T0=0.1))
        b = C.gen(E, 4, Te, c)
    y = b['v'].numpy()
    d = H.nll(npy(C.bayes(b, c, False)), y) - H.nll(npy(C.bayes(b, c, True)), y)
    return float(d.mean()), float(d.std() / math.sqrt(E)), float(Te.float().mean())


def cost_closed(rho, N=4):
    """E_r KL(oracle GLS || correlation-blind Bayes) for one pair with correlation rho, unit marginals."""
    S = np.eye(N); S[0, 1] = S[1, 0] = rho
    P = np.linalg.inv(S); a = P.sum(); V = 1 / (1 + a); w = P.sum(1) * V
    Vp = 1 / (1 + N); wp = np.full(N, Vp)
    d = w - wp
    Em = d @ (S + 1.0) @ d
    return 0.5 * (math.log(Vp / V) + V / Vp + Em / Vp - 1), w / V, V


def cost_mc(nneg, E=40000, seed=11):
    c = C.full_cfg(fam(nneg))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        b = C.gen(E, 4, 0, c)
    refs = C.references(b, c, with_bayes=False)
    y = b['v'].numpy()
    d = H.nll(npy(refs['diag_bayes']), y) - H.nll(npy(refs['oracle']), y)
    return float(d.mean()), float(d.std() / math.sqrt(E))


def main():
    out = {}
    for name, nneg in (('anti', 1), ('null', 0)):
        c = C.full_cfg(fam(nneg))
        m, se, mt = val_level(c)
        out[f'val_level_{name}'] = dict(mean=m, se=se, mean_T=mt)
        print(f'history-blind level on the pilot validation distribution, {name}: {m:.4f} +- {se:.4f} (mean T {mt:.1f})')
    for rho, nneg in ((-0.8, 1), (0.8, 0)):
        cc, w, V = cost_closed(rho)
        mc, se = cost_mc(nneg)
        out[f'cost_rho{rho}'] = dict(closed=cc, mc=mc, se=se, weights=w.tolist(), V=V)
        print(f'rho {rho}: closed form {cc:.4f}, Monte Carlo {mc:.4f} +- {se:.4f}; GLS weights {np.round(w, 4).tolist()}, V {V:.4f}')
    # pilot tables
    root = os.path.join(WF, 'runs/fa-pilot-1028b')
    arms = {}
    for arm in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and not d.startswith('es_')):
        meta = json.load(open(os.path.join(root, arm, 'meta.json')))
        diag = json.load(open(os.path.join(root, arm, 'diag.json')))
        arms[arm] = dict(curve=meta['curve'], val_bayes_nll=meta['val_bayes_nll'], best_step=meta['best_step'],
                         n_params=meta['n_params'],
                         cells={T: {k: diag['cells'][T][k] for k in ('nll_model', 'nll_bayes', 'nll_bayes_T0', 'nll_diag_bayes',
                                                                     'nll_prior', 'history_value', 'frac_history_recovered',
                                                                     'kl_relabel', 'shuf_gain_over_T0')}
                                for T in diag['cells']})
        cl = arms[arm]['cells']
        print(arm, ' '.join(f"T{T}: hv {cl[T]['history_value']:.3f} gap {cl[T]['nll_model'] - cl[T]['nll_bayes']:.3f} "
                            f"frac {cl[T]['frac_history_recovered'] if cl[T]['frac_history_recovered'] is None else round(cl[T]['frac_history_recovered'], 3)}"
                            for T in cl))
        print('   curve gap:', [round(x[2], 3) for x in meta['curve']])
    out['pilot'] = arms
    json.dump(out, open(os.path.join(HERE, 'pop_checks.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
