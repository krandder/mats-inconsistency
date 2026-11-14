"""POST HOC checks requested by the final GPT-6 review (review/review_gpt6_final.md). Writes logs/posthoc_review.log.
1. Finding 10: P9's literal registered statistic (median over layers of trained RMS gate input / sqrt 2), per seed.
2. Finding 6: convergence of the trained-net closures. K2 recomputed with 4x the Gauss-Hermite nodes (at least 256) and
   Mehler order 128; reported as |mu_hi - mu_reg| / |mu_reg - truth| (the change relative to the registered error norm,
   which is sqrt(nmse_k2 tr Var) from runs/R_*.json, so no new truth is needed).
3. Finding 7: the gate-input scale that grid C's spliced (front, back) pairs actually see: 8 fronts x 8 backs per arm at
   (16, 8, 8), RMS gate input at the first and last back layer per pair (2,048 Gaussian inputs per front).
usage: python3 scratch/posthoc_review.py
"""
import glob, json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import acts as A, gnet as G, k2gen as K2
R = os.path.join(HERE, '..', 'runs')
log = open(os.path.join(HERE, '..', 'logs', 'posthoc_review.log'), 'w')


def pr(s):
    print(s, flush=True); log.write(s + '\n'); log.flush()


# 1. literal P9
pr('## 1. P9 literal statistic: median over layers of trained RMS / sqrt 2 (registered), per seed; and the report\'s measured-init ratio')
for a in ('softplus1', 'softplus4', 'relu', 'gelu', 'silu', 'softplus16'):
    lit, rep = [], []
    for f in sorted(glob.glob(os.path.join(R, 'sort', f'sort_{a}_L16_s*.json'))):
        d = json.load(open(f))
        lit.append(float(np.median(np.array(d['rms_final']) / math.sqrt(2))))
        rep.append(float(np.median(np.array(d['rms_final']) / np.array(d['rms_init']))))
    pr(f'{a:11s} literal {[round(x, 3) for x in lit]} median {np.median(lit):.3f} | measured-init {[round(x, 3) for x in rep]} median {np.median(rep):.3f}')

# 2. trained-net closure convergence
pr('## 2. trained-net K2 convergence: |mu_hi - mu_reg| / registered error norm (Q x4 >= 256, Mehler 128)')
ACT = {'relu': ('relu', None), 'softplus1': ('softplus', 1.0), 'softplus2': ('softplus', 2.0), 'softplus4': ('softplus', 4.0),
       'softplus8': ('softplus', 8.0), 'softplus16': ('softplus', 16.0), 'gelu': ('gelu', None), 'silu': ('silu', None),
       'tanhb': ('tanh', None), 'smoothstep1': ('smoothstep', 1.0), 'gsrelu0.5': ('gsrelu', 0.5)}
for rf in sorted(glob.glob(os.path.join(R, 'R_*.json'))):
    Rj = json.load(open(rf)); pre = Rj.get('prefix', 'sort'); arm = Rj['arm']
    if arm not in ACT or pre == 'pn':
        continue
    name, p = ACT[arm]
    rel = []
    for seed, row in Rj['nets'].items():
        if 'nmse_k2' not in row:
            continue
        d = np.load(os.path.join(R, 'sort', f'{pre}_{arm}_L16_s{seed}.npz'))
        Ws, Vs = list(d['W']), list(d['V']); bs = list(d['b']) if 'b' in d else None
        q = K2.q_for(name, p)
        mu0, _, _ = K2.k2_residual(name, p, Ws, Vs, K2.Hermite(q, 32), bs=bs)
        mu1, _, _ = K2.k2_residual(name, p, Ws, Vs, K2.Hermite(max(4 * q, 256), 128), bs=bs)
        err = math.sqrt(row['nmse_k2'] * row['trvar'])
        rel.append(float(np.linalg.norm(mu1 - mu0) / err))
    pr(f'{pre:4s} {arm:11s} per seed {[round(x, 3) for x in rel]}')

# 3. grid C splice scales
pr('## 3. grid C (16,8,8): RMS gate input seen by spliced pairs, first / last back layer: min / median / max over 64 pairs')
n, c, Lp = 16, 8, 8
for name, p, bf in [('relu', None, 0), ('softplus', 16.0, 0), ('softplus', 4.0, 0), ('softplus', 1.0, 0), ('gelu', None, 0),
                    ('silu', None, 0), ('gsrelu', 0.5, 0), ('tanh', None, 0.13)]:
    spec = G.Spec(name, p, 'plain', bias_frac=bf)
    tag = G.tag_of('tgcut', spec.label, n, c, Lp)
    first, last = [], []
    for f in range(8):
        Wf = G.weights(spec, n, c + Lp, tag, 0, f)[:c]; bfs = G.biases(spec, n, c + Lp, tag, 0, f)
        X = G.rng(tag, 2, f, 0).standard_normal((n, 2048)).astype(np.float32)
        H = G.fwd_cols(spec, X, Wf, None if bfs is None else bfs[:c])
        for b in range(8):
            Wb = G.weights(spec, n, c + Lp, tag, 1, b)[c:]; bbs = G.biases(spec, n, c + Lp, tag, 1, b)
            bbs = None if bbs is None else bbs[c:]
            Hb = H.astype(np.float64)
            for l, (W,) in enumerate(Wb):
                Z = W.astype(np.float64) @ Hb + (0 if bbs is None else bbs[l].astype(np.float64)[:, None])
                if l == 0:
                    first.append(float(np.sqrt(np.mean(Z ** 2))))
                if l == len(Wb) - 1:
                    last.append(float(np.sqrt(np.mean(Z ** 2))))
                Hb = spec.phi(Z)
    pr(f'{spec.label:11s} first {np.min(first):.2f} / {np.median(first):.2f} / {np.max(first):.2f}   '
       f'last {np.min(last):.2f} / {np.median(last):.2f} / {np.max(last):.2f}')
pr('DONE')
