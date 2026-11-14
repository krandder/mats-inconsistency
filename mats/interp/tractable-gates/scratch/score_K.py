"""PREREG scoring for grid K (P5, P6, P7; tanhb per Amendment 1) from runs/k2_summary.json (scratch/summ_k2.py).
P7's "Spearman rho <= -0.6 over the five betas" is scored as rho(1/beta, median nMSE_K2) <= -0.6, i.e. the error falls as
beta falls (the registered direction "decreases from beta = 16 to beta = 1").
usage: python3 scratch/score_K.py -> runs/K_scores.json (and stdout)"""
import json, os
import numpy as np
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, '..', 'runs', 'k2_summary.json')))
T = {(r['arm'], r['n'], r['L']): r for r in rows}
NS = sorted({r['n'] for r in rows}); LS = [2, 4, 8, 16]
SHARP = ['softplus16', 'softplus4', 'gelu', 'silu', 'gsrelu0.5', 'smoothstep1']
ALL = ['relu', 'softplus1', 'softplus2', 'softplus4', 'softplus8', 'softplus16', 'gelu', 'silu', 'tanhb', 'gsrelu0.5', 'smoothstep1']
out = {}
# P5(a): no win at L = 16, n <= 16
wins = [dict(arm=a, n=n, wins=T[(a, n, 16)]['wins']) for a in ALL for n in NS if n <= 16 and (a, n, 16) in T]
out['P5a'] = dict(win_cells=[w for w in wins if w['wins'] >= 3], refuted=any(w['wins'] >= 3 for w in wins), cells=len(wins))
# P5(b): spread of K2/MC ratio across SHARP arms < 3x in a majority of (n, L) cells
cells = []
for n in NS:
    for L in LS:
        v = [T[(a, n, L)]['ratio'] for a in SHARP if (a, n, L) in T]
        if len(v) >= 4:
            cells.append(dict(n=n, L=L, spread=float(max(v) / min(v)), arms=len(v)))
big = [c for c in cells if c['spread'] >= 3]
out['P5b'] = dict(cells=cells, n_cells=len(cells), n_spread_ge3=len(big), refuted=len(big) > len(cells) / 2)
# P6: RQMC gain
def pooled_gain(a, n):        # registered: median over nets AND depths (pooled), per final-review finding 18
    f = os.path.join(HERE, '..', 'runs', f'k2_{a}_n{n}.json')
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    return float(np.median([(1 / x[L]['N_eq']) / x[L]['nmse_rqmc_eq'] for x in d['nets'].values() for L in ('2', '4', '8', '16')]))


g8 = {a: pooled_gain(a, 8) for a in ALL if pooled_gain(a, 8) is not None}
p6a = {a: g8[a] / g8['relu'] for a in ('softplus1', 'gelu', 'silu') if a in g8}
hi = [dict(arm=a, n=n, L=L, gain=T[(a, n, L)]['rq_gain']) for a in ALL for n in NS if n >= 32 for L in LS if (a, n, L) in T]
out['P6'] = dict(gain_n8_median=g8, ratio_to_relu=p6a, part_a_holds=all(v >= 1.5 for v in p6a.values()),
                 n_ge32_gain_ge2=[h for h in hi if h['gain'] >= 2], part_b_holds=all(h['gain'] < 2 for h in hi),
                 refuted=not (all(v >= 1.5 for v in p6a.values()) and all(h['gain'] < 2 for h in hi)))
# P7: softplus ladder
betas = [1, 2, 4, 8, 16]
p7 = []
for n in NS:
    for L in LS:
        v = [T.get((f'softplus{b}', n, L), {}).get('k2') for b in betas]
        if all(x is not None for x in v):
            rho = spearmanr([1 / b for b in betas], v)[0]
            p7.append(dict(n=n, L=L, rho=float(rho), ok=bool(rho <= -0.6)))
out['P7'] = dict(cells=p7, n_ok=sum(c['ok'] for c in p7), n_cells=len(p7), refuted=sum(c['ok'] for c in p7) < 12)
json.dump(out, open(os.path.join(HERE, '..', 'runs', 'K_scores.json'), 'w'), indent=1)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ('cells',)} for k, v in out.items()}, indent=1))
