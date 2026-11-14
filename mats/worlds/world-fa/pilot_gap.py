"""Pilot-case check (coordinator item 6): full Bayes vs history-blind Bayes (bayes_T0) gap for isolating families.
usage: python pilot_gap.py [case-substring] [E] >> audit/pilot_gap.txt"""
import json, sys
import numpy as np, torch
sys.path.insert(0, '../worlds')
import fa_core as C, harness as H

CASES = {
    'gpt6_pair_rho.8_m1 (N4, M6)': (4, [{'blocks': [[2, 0.8, 0]], 'marg': 1.0}]),
    'pair_rho.95_m1 (N4, M6)': (4, [{'blocks': [[2, 0.95, 0]], 'marg': 1.0}]),
    'pair_rho.8_m0.25 (N4, M6)': (4, [{'blocks': [[2, 0.8, 0]], 'marg': 0.25}]),
    'pair_rho-.8_m1 (N4, M6, signed)': (4, [{'blocks': [[2, 0.8, 1]], 'marg': 1.0}]),
    'triple_rho.8_m1 (N4, M4)': (4, [{'blocks': [[3, 0.8, 0]], 'marg': 1.0}]),
    'gemini_5clique_rho.8_m1 (N8, M56)': (8, [{'blocks': [[5, 0.8, 0]], 'marg': 1.0}]),
    'gemini_5clique_rho.8_m0.25 (N8, M56)': (8, [{'blocks': [[5, 0.8, 0]], 'marg': 0.25}]),
    'gemini_5clique_rho-.8x2_m1 (N8, signed)': (8, [{'blocks': [[5, 0.8, 2]], 'marg': 1.0}]),
    'mixed_pair_or_signed_pair (N4, M6+6)': (4, [{'blocks': [[2, 0.8, 0]], 'marg': 1.0}, {'blocks': [[2, 0.8, 1]], 'marg': 1.0}]),
}
E = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
only = sys.argv[1] if len(sys.argv) > 1 else ''
for name, (N, tpl) in CASES.items():
    if only and only not in name:
        continue
    cfg = C.full_cfg(dict(prior='f', f_templates=tpl))
    M = len(C.family(N, C._tkey(cfg['f_templates']))[1])
    row = dict(case=name, M=M)
    for T in (0, 8, 32, 128):
        torch.manual_seed(T)
        b = C.gen(E, N, T, cfg)
        refs = C.references(b, cfg)
        nl = {k: C.nll_t(p, b['v']).numpy() for k, p in refs.items()}
        pb = tuple(x.numpy() for x in refs['bayes']); p0 = tuple(x.numpy() for x in refs['bayes_T0'])
        kl = H.kl_bayes_model(pb, p0)
        d = nl['bayes_T0'] - nl['bayes']
        row[f'T{T}'] = dict(gap_T0=round(float(d.mean()), 4), se=round(float(d.std() / np.sqrt(E)), 4),
                            kl_T0=round(float(kl.mean()), 4), frac_kl_ge_0p3=round(float((kl >= 0.3).mean()), 3),
                            gap_diag=round(float((nl['diag_bayes'] - nl['bayes']).mean()), 4),
                            oracle=round(float((nl['oracle'] - nl['bayes']).mean()), 4),
                            gap_prior=round(float((nl['prior'] - nl['bayes']).mean()), 4))
    print(json.dumps(row), flush=True)
