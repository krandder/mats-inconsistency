"""All tables for the arc-noisy report: python analyze.py > results/tables.md

nMSE = mean_j err_j^2 / mean_j Var(a_L,j), debiased by the truth's own MC variance (1/N_truth). Per net; then median and
mean over the evaluation nets 0-23 with bootstrap 95% CIs (10,000 resamples over nets).
Sample-free: kprop variants (ARC counter FLOPs); at budget B a variant counts only if its cost <= B.
Sampling: MC (expected nMSE 1/N exactly, N = floor(B / c_mc)); MC with the last noise integrated (expected nMSE
(mean Var_RB / mean Var) / N, from the truth's Rao-Blackwell sums); scrambled Sobol and Kerdock-MUB (mean over replicates).
"Best entry of a type at B" = lowest median nMSE among that type's entries affordable at B.
Candidates (ours): H1 Kerdock + layer-m control with a kprop mean, H2 kprop/Kerdock combination (combine.py), C6
antithetic noise on Kerdock. Win = per-net ratio best-competitor / candidate with bootstrap CI (median and mean) above 1.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import anet, truthlib

EVAL = list(range(24))
SIGMAS = (0.001, 0.003, 0.01, 0.03, 0.1)
BUDGETS = np.array([1e9, 2e9, 5e9, 1e10, 2e10, 5e10, 6.8e10, 1e11, 2e11, 2.72e11, 3e11])
KPROP = ('k1', 'k2', 'k2a', 'k3f', 'k3fa')
RNG = np.random.default_rng(12345)
_T = {}


def boot(x, stat, B=10000):
    x = np.asarray(x, dtype=float)
    idx = RNG.integers(0, len(x), (B, len(x)))
    v = stat(x[idx], axis=1)
    return float(stat(x)), float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))


def fmt(t, f='.2e'):
    return f"{t[0]:{f}} [{t[1]:{f}}, {t[2]:{f}}]"


_CHK = {}
_RT = anet.load_rtable()


def truth(seed, sigma):
    """Truth for (seed, sigma); verifies the stored weight checksum and noise scales against the canonical net
    (the GPU truth ran float32-rounded weights; kprop runs float64 ones; max weight difference ~1.5e-8)."""
    k = (seed, sigma)
    if k not in _T:
        p = os.path.join(HERE, 'truth', f'truth_256_32_s{sigma:g}_seed{seed}.npz')
        T = truthlib.load(seed, sigma) if os.path.exists(p) else None
        if T is not None:
            if seed not in _CHK:
                _CHK[seed] = anet.weights_checksum(anet.make_weights(seed))
            assert abs(T['chk'] - _CHK[seed]) < 1e-6 * max(1.0, abs(_CHK[seed])), f'weights checksum mismatch {seed}'
            assert np.allclose(T['r'], _RT[f'256_32_{seed}_{sigma:g}']), f'noise scale mismatch {seed} {sigma}'
        _T[k] = T
    return _T[k]


def kprop(seed, sigma, L, v):
    p = os.path.join(HERE, 'results', f'kprop_{seed}_s{sigma:g}_L{L}_{v}.json')
    T = truth(seed, sigma)
    if not os.path.exists(p) or T is None:
        return None
    d = json.load(open(p))
    return truthlib.nmse(np.array(d['means'][L - 1]), T, L), float(d['flops']), np.array(d['means'])


def mc_cost(L, sigma):
    """flopscope cost of one plain noisy path (same formula as anet.Net.flops_path)."""
    return 256 * anet.F_NORMAL + L * (anet.mm_flops(1, 256, 256) + 256) + (L - 1) * 256 * (anet.F_NORMAL + 2)


def rb_factor(seed, sigma, L):
    T = truth(seed, sigma)
    if T is None or f'R1_{L}' not in T['d'].files:
        return None
    d = T['d']; N = T['N']; m = d[f'R1_{L}'] / N; v = d[f'R2_{L}'] / N - m * m
    return float(v.mean() / T['var'][L - 1].mean())


def sampler(seed, sigma, L, method):
    """nMSE (mean over replicates) at each budget index; plus per-replicate arrays for pairing."""
    if method == 'mc':
        return {bi: 1.0 / max(1, int(B / mc_cost(L, sigma))) for bi, B in enumerate(BUDGETS)}
    if method == 'mcrb':
        f = rb_factor(seed, sigma, L)
        if f is None:
            return None
        c = mc_cost(L, sigma) + 256 * (anet.F_CDF + anet.F_PDF + 6) - 256 * (anet.F_NORMAL + 2)
        return {bi: f / max(1, int(B / c)) for bi, B in enumerate(BUDGETS)}
    key = method
    T = truth(seed, sigma)
    if T is None:
        return None
    d = None
    p2 = os.path.join(HERE, 'samp_v2', f'samp_{seed}_s{sigma:g}_L{L}.npz')
    if os.path.exists(p2):
        d2 = np.load(p2)
        if f'N_{key}' in d2.files:
            d = d2; N = d[f'N_{key}']; C = d[f'costs_{key}']
    if d is None:                                         # version-1 files: sparse checkpoints, costs corrected
        sub = 'sampanti' if method == 'mubanti' else 'samp'
        p1 = os.path.join(HERE, sub, f'samp_{seed}_s{sigma:g}_L{L}.npz')
        if not os.path.exists(p1):
            return None
        d = np.load(p1)
        if f'N_{key}' not in d.files:
            return None
        N = d[f'N_{key}']; c1 = float(d[f'cost_{key}'])
        setup = 6 * 256 ** 3 + 256 * 256 * anet.F_NORMAL
        if key == 'mub':
            C = N * (c1 - 1500) + np.ceil(N / 66048) * setup
        elif key == 'mubanti':
            C = N * (c1 - 3000) + np.ceil(N / 66048) * setup
        elif key == 'sobol':
            C = N * (c1 - 256 * anet.F_NORMAL + 256 * 100)
        else:
            C = N * c1
    S = d[f'sum_{key}_{L}']
    out = {}
    for bi, B in enumerate(BUDGETS):
        j = int(np.searchsorted(C, B, side='right') - 1)
        if j >= 0:
            out[bi] = float(np.mean([truthlib.nmse(S[r, j] / N[j], T, L) for r in range(S.shape[0])]))
    return out


def collect(L, sigma):
    """per-net dict: entries -> {budget_index: nMSE} (sample-free entries: constant where affordable)."""
    nets = {}
    for s in EVAL:
        if truth(s, sigma) is None:
            continue
        e = {}
        for v in KPROP:
            k = kprop(s, sigma, L, v)
            if k is not None:
                e[f'kprop-{v}'] = {bi: k[0] for bi, B in enumerate(BUDGETS) if k[1] <= B}
                e[f'kprop-{v}-flops'] = k[1]
        for m in ('mc', 'mcrb', 'sobol', 'mub', 'mubanti'):
            c = sampler(s, sigma, L, m)
            if c:
                e[m] = c
        nets[s] = e
    return nets


def best_of(nets, names, bi):
    """entry with the lowest median nMSE at budget index bi among `names` (each must cover >= 3 nets)."""
    best = None
    for nm in names:
        vals = [e[nm][bi] for e in nets.values() if nm in e and bi in e[nm]]
        if len(vals) >= 3:
            med = np.median(vals)
            if best is None or med < best[1]:
                best = (nm, med)
    return best


def main():
    print('# arc-noisy tables (regenerated by analyze.py)\n')
    print(__doc__)
    for L in (8, 32):
        for sigma in SIGMAS:
            nets = collect(L, sigma)
            if len(nets) < 3:
                continue
            print(f'\n## L = {L}, sigma = {sigma:g} ({len(nets)} evaluation nets with truth)\n')
            print('Sample-free (kprop, ARC counter):\n')
            print('| variant | nets | FLOPs | median nMSE [95% CI] | mean nMSE [95% CI] |')
            print('|---|---|---|---|---|')
            for v in KPROP:
                vals = [e[f'kprop-{v}'][len(BUDGETS) - 1] for e in nets.values() if f'kprop-{v}' in e and len(BUDGETS) - 1 in e[f'kprop-{v}']]
                fl = [e[f'kprop-{v}-flops'] for e in nets.values() if f'kprop-{v}-flops' in e]
                if len(vals) >= 3:
                    print(f'| {v} | {len(vals)} | {np.median(fl):.2e} | {fmt(boot(vals, np.median))} | {fmt(boot(vals, np.mean))} |')
            print('\nSamplers, median nMSE over nets at each budget (FLOPs):\n')
            print('| method | ' + ' | '.join(f'{b:.2g}' for b in BUDGETS) + ' |')
            print('|---|' + '---|' * len(BUDGETS))
            for m in ('mc', 'mcrb', 'sobol', 'mub', 'mubanti'):
                cells = []
                for bi in range(len(BUDGETS)):
                    vals = [e[m][bi] for e in nets.values() if m in e and bi in e[m]]
                    cells.append(f'{np.median(vals):.2e}' if len(vals) >= 3 else '-')
                if any(c != '-' for c in cells):
                    print(f'| {m} | ' + ' | '.join(cells) + ' |')
            print('\nBest entry per type (by median):\n')
            print('| budget | best sample-free | median | best sampler | median |')
            print('|---|---|---|---|---|')
            for bi, B in enumerate(BUDGETS):
                sf = best_of(nets, [f'kprop-{v}' for v in KPROP], bi)
                sp = best_of(nets, ['mc', 'mcrb', 'sobol', 'mub'], bi)
                print(f"| {B:.2g} | {sf[0] if sf else '-'} | {sf[1]:.2e} | {sp[0] if sp else '-'} | {sp[1]:.2e} |" if sf and sp else
                      f"| {B:.2g} | {sf[0] if sf else '-'} | {sf[1] if sf else float('nan'):.2e} | {sp[0] if sp else '-'} | {sp[1] if sp else float('nan'):.2e} |")
            # candidate C6 vs best sampler, per net
            if any('mubanti' in e for e in nets.values()):
                print('\nC6 antithetic noise vs the best plain sampler (per-net ratio best/C6):\n')
                print('| budget | nets | median ratio [95% CI] | mean ratio [95% CI] |')
                print('|---|---|---|---|')
                for bi, B in enumerate(BUDGETS):
                    sp = best_of(nets, ['mc', 'mcrb', 'sobol', 'mub'], bi)
                    r = [e[sp[0]][bi] / e['mubanti'][bi] for e in nets.values()
                         if sp and 'mubanti' in e and bi in e['mubanti'] and sp[0] in e and bi in e[sp[0]]]
                    if len(r) >= 3:
                        print(f'| {B:.2g} | {len(r)} | {fmt(boot(r, np.median), ".2f")} | {fmt(boot(r, np.mean), ".2f")} |')


if __name__ == '__main__':
    main()
