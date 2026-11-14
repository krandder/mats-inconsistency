"""Grid R (PREREG): tractability of the trained sorting nets, and the trained gates' sharpness and Hermite tails.

For each trained net runs/sort/sort_ARM_L16_s{seed}.npz:
- truth: MC with 2^21 inputs x ~ N(0, I_16) of E[output] (the residual net's h_L) and tr Var;
- K2: residual Gaussian closure (lib/k2gen.k2_residual; single-gate arms only; GLU arms are not scored: no GLU closure);
- N_eq = FLOPs(K2) / FLOPs(one sample); RQMC (scrambled Sobol, 16 replicates) at 2^floor(log2 N_eq), 2^10, 2^14;
- sharpness: per-layer trained RMS pre-activation / its init value (from the training json), median over layers;
- Hermite tail share at D = 4, 8, 16: energy-weighted over units of T_D/T_1 for x -> phi(m_u + s_u x), (m_u, s_u) the
  unit's trained pre-activation mean and std (Gaussian approximation of the unit's input law).
usage: python3 scratch/k2_trained.py PREFIX ARM [ARM ...]   (PREFIX sort, pin or pn) -> runs/R_ARM.json or runs/R_PREFIX_ARM.json
Pre-norm nets (Amendment 3) and GLU arms get MC/RQMC, sharpness and tail shares only (no K2 closure through RMSNorm/GLU).
"""
import json, math, os, sys
import numpy as np
from scipy.stats import qmc, norm
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import acts as A
import k2gen as K2

ACT = {'relu': ('relu', None), 'softplus1': ('softplus', 1.0), 'softplus2': ('softplus', 2.0), 'softplus4': ('softplus', 4.0),
       'softplus8': ('softplus', 8.0), 'softplus16': ('softplus', 16.0), 'gsrelu0.5': ('gsrelu', 0.5),
       'gelu': ('gelu', None), 'silu': ('silu', None), 'tanhb': ('tanh', None), 'smoothstep1': ('smoothstep', 1.0),
       'swiglu': ('silu', None), 'bilinear': ('linear', None)}


def tail_shares(name, p, means, stds, Ds=(4, 8, 16)):
    num = {D: 0.0 for D in Ds}; den = 0.0
    phi, _ = A.act(name, p)
    h = 0.005; xg = np.arange(-12, 12 + h / 2, h); wg = A.npdf(xg) * h       # trapezoid: fine enough for kinks
    for m_, s_ in zip(means, stds):
        c = A.fine_coef(lambda z, m_=m_: phi(m_ + z), 40, max(s_, 1e-12), L=12.0, h=h)
        e2 = float(wg @ phi(m_ + s_ * xg) ** 2)
        e = c ** 2
        T1 = max(e2 - e[0] - e[1], 0.0)
        den += T1
        for D in Ds:
            num[D] += max(e2 - e[:D + 1].sum(), 0.0)
    return {str(D): (num[D] / den if den > 0 else None) for D in Ds}


PREFIX = sys.argv[1]
for arm in sys.argv[2:]:
    name, p = ACT[arm]
    glu = arm in ('swiglu', 'bilinear')
    out = dict(arm=arm, prefix=PREFIX, nets={})
    for seed in (0, 1, 2):
        base = os.path.join(HERE, '..', 'runs', 'sort', f'{PREFIX}_{arm}_L16_s{seed}')
        if not os.path.exists(base + '.npz'):
            continue
        d = np.load(base + '.npz'); meta = json.load(open(base + '.json'))
        W = d['W']; V = d['V']; W2 = d['W2'] if glu else None; b = d['b'] if 'b' in d else None
        L, n = W.shape[0], W.shape[2]
        phi, _ = A.act(name, p)

        PN = bool(meta.get('prenorm', False))

        def fwd(X):
            H = X
            for l in range(L):
                Hin = H * (np.sqrt(n) / np.sqrt((H * H).sum(1, keepdims=True) + 1e-12)) if PN else H
                U = Hin @ W[l].T + (0 if b is None else b[l][None, :])
                Aa = phi(U) * (Hin @ W2[l].T) if glu else phi(U)
                H = H + Aa @ V[l].T
            return H

        g = np.random.default_rng(1000 + seed)
        S1 = np.zeros(n); S2 = np.zeros(n); SX = np.zeros((n, n)); Nt = 1 << 21
        for i in range(Nt >> 15):
            X = g.standard_normal((1 << 15, n))
            H = fwd(X)
            S1 += H.sum(0); S2 += (H * H).sum(0); SX += X.T @ H
        truth = S1 / Nt; trvar = float(np.sum(S2 / Nt - truth ** 2))
        lin_frac = float(np.sum((SX / Nt) ** 2) / trvar)       # post hoc: share of output variance in the linear (degree-1) chaos
        row = dict(nmse_task=meta['nmse'], trvar=trvar, params=meta['params'], lin_frac=lin_frac)
        mcf = K2.mc_flops_res(name, n, L) + (L * K2.mm(1, n, n) if glu else 0)
        if not glu and not PN:
            herm = K2.Hermite(K2.q_for(name, p), 32)
            mu, C, fl = K2.k2_residual(name, p, list(W), list(V), herm, bs=None if b is None else list(b))
            row.update(nmse_k2=float(np.sum((mu - truth) ** 2) / trvar), k2_flops=fl, N_eq=fl / mcf, nmse_mc=mcf / fl)
            Neq = fl / mcf
        else:
            Neq = 1000.0
        for N in sorted({1 << int(math.floor(math.log2(max(Neq, 2)))), 1 << 10, 1 << 14}):
            errs = []
            for r in range(16):
                U = qmc.Sobol(d=n, scramble=True, seed=7919 * seed + 104729 * r + N).random(N)
                est = fwd(norm.ppf(np.clip(U, 1e-12, 1 - 1e-12))).mean(0)
                errs.append(float(np.sum((est - truth) ** 2) / trvar))
            row[f'nmse_rqmc_{N}'] = float(np.mean(errs))
            if not glu and not PN and N == 1 << int(math.floor(math.log2(max(Neq, 2)))):
                row['nmse_rqmc_eq'] = float(np.mean(errs)); row['N_rqmc_eq'] = N
        r0 = np.array(meta['rms_init']); r1 = np.array(meta['rms_final'])
        row['sharp_ratio_median'] = float(np.median(r1 / r0))
        row['rms_final_median'] = float(np.median(r1))
        um = np.array(meta['unit_mean']).reshape(-1); us = np.array(meta['unit_std']).reshape(-1)
        row['tail_share'] = tail_shares(name, p, um, us)
        if not glu and not PN:
            row['k2_win'] = bool(row['nmse_k2'] < row['nmse_mc'] and row['nmse_k2'] < row['nmse_rqmc_eq'])
        out['nets'][str(seed)] = row
        print(arm, seed, json.dumps({k: v for k, v in row.items() if k not in ('tail_share',)}), row['tail_share'], flush=True)
    json.dump(out, open(os.path.join(HERE, '..', 'runs', f'R_{arm}.json' if PREFIX == 'sort' else f'R_{PREFIX}_{arm}.json'), 'w'), indent=1)
