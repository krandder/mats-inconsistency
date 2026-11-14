"""History-blind references on the test sets: exact Bayes that ignores the labelled questions (each source's
likelihood of y* uses only its query reading, marginalising its variance level over the prior) - the best any net
can do without binding a source's labelled errors to its reading. Also the equal-weight rule. Prints NLL gaps to Bayes.
usage: python blind_ref.py  -> results/blind_ref.json"""
import json, math
import numpy as np
import sfworld as W

out = {}
for key, N, kw in [('indep_N3', 3, {}), ('indep_N6', 6, {}), ('indep_N12', 12, {}), ('indep_N24', 24, {}),
                   ('indep_N64', 64, {}), ('indep_N256', 256, {}),
                   ('groups_N4_r0.5_g4', 4, dict(mode='groups', rho=0.5, gsize=4)),
                   ('groups_N8_r0.5_g4', 8, dict(mode='groups', rho=0.5, gsize=4)),
                   ('groups_N16_r0.5_g4', 16, dict(mode='groups', rho=0.5, gsize=4)),
                   ('groups_N24_r0.5_g4', 24, dict(mode='groups', rho=0.5, gsize=4)),
                   ('pairs_N4_r0.9_a1-1', 4, dict(mode='pairs', rho=0.9)), ('pairs_N6_r0.9_a1-1', 6, dict(mode='pairs', rho=0.9)),
                   ('pairs_N8_r0.9_a1-1', 8, dict(mode='pairs', rho=0.9))]:
    z = np.load(f'data/{key}_test.npz')
    d = dict(o=z['o'], y=z['y'], v=z['v'], blk=z['blk'], grp=z['grp'], N=N, mode=kw.get('mode', 'indep'),
             rho=kw.get('rho', 0.0), gsize=kw.get('gsize', 1), a1=1.0, a2=1.0)
    # history-blind: zero labelled residuals carry no information if we drop their likelihood; emulate by an
    # independent-source likelihood with QL = 0 (query reading only), under the world's own block structure.
    o_hb = d['o'].copy(); y_hb = d['y'].copy()
    QL0 = W.QL
    try:
        W.QL = 0                                              # labelled questions ignored
        dd = dict(d); dd['o'] = o_hb[:, :, QL0:QL0 + 1]; dd['y'] = y_hb[:, QL0:QL0 + 1]
        W.QL = 0
        B0 = W.bayes(dd)
    finally:
        W.QL = QL0
    gap = B0['nll'] - z['nll_bayes']
    out[key] = dict(blind_gap=float(gap.mean()), blind_gap_se=float(gap.std() / math.sqrt(len(gap))))
    print(key, out[key], flush=True)
json.dump(out, open('results/blind_ref.json', 'w'), indent=1)
