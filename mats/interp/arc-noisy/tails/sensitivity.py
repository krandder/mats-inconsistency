"""Claims-review follow-ups (no new runs): (1) U3 win ratios under two median conventions (lower median, as in
analyze_tails.py, and the usual midpoint median, where a midpoint involving an unreached event is infinite);
(2) fixed-budget (2^20 paths only) vs budget-selected fractions within 50% for U3; (3) bootstrap CIs over nets for the
reached fractions of U3, C-K3f and U2 (post hoc). python tails/sensitivity.py"""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deduce as D
from common import c_path, c_move, run_evals, load_truth

TR, MC, KP = 'runs/truth_eval', 'runs/arcn-tdev1/mc', 'runs/kp'
n = 256; NN = 8; INF = np.inf; rng = np.random.default_rng(7); BOOT = 4000
MOM_ACC = 2 * 4 * n


def sp4(t, kk):
    v = D.saddle(t, kk, 4)
    if not np.isfinite(v):
        v = D.saddle(t, kk, 3)
    if not np.isfinite(v):
        v = D.gauss(t, kk)
    return v


def med_low(x):
    return float(np.quantile(x, 0.5, method='inverted_cdf'))


def med_mid(x):
    x = np.sort(np.asarray(x, float)); m = len(x)
    if m % 2:
        return float(x[m // 2])
    a, b = x[m // 2 - 1], x[m // 2]
    return float('inf') if (np.isinf(a) or np.isinf(b)) else float((a + b) / 2)


def boot(v, f):
    v = np.asarray(v, float); m = len(v)
    meds = np.array([f(v[rng.integers(0, m, m)]) for _ in range(BOOT)])
    return f(v), float(np.quantile(meds, 0.025, method='inverted_cdf')), float(np.quantile(meds, 0.975, method='inverted_cdf'))


for sigma in (0.001, 0.003, 0.01):
    for L in (4, 8):
        cm, cp = c_move(L), c_path(L)
        u3 = []; u3f = []; gs = []; mcb = []; reach = {'U3': [], 'U3 fixed 2^20': [], 'C-K3f': [], 'U2': []}
        for s in range(20):
            T = load_truth(os.path.join(TR, f'truth_{s}_s{sigma:g}.json'))[L]; p = np.exp(T['lp'])
            ok_any = np.zeros((NN, 5), bool); b = np.full((NN, 5), INF); okf = None
            for tag in ('m20', 'm22'):
                m = json.load(open(os.path.join(MC, f'mc_{tag}_{s}_s{sigma:g}.json')))
                E = np.array([[[sp4(T['thr'][i, k], D.cumulants_from_moments(np.array(rep['moments'][str(L)])[:, i]))
                                for k in range(5)] for i in range(NN)] for rep in m['reps']])
                rm = ((E / p - 1) ** 2).mean(0); ok = rm <= 0.25
                cost = m['N'] * (cp + MOM_ACC) / (5 * n) + D.F_SADDLE + D.F_CUMULANTS
                b = np.where(ok & ~ok_any, cost, b); ok_any |= ok
                if tag == 'm20':
                    okf = ok
            u3.append(b); reach['U3'].append(ok_any.mean(0)); reach['U3 fixed 2^20'].append(okf.mean(0))
            nl, rv1 = T['nlev'], T['rv1']
            gs.append(run_evals(500, nl) * cm + run_evals(2000, nl) * cm * rv1 / 0.25)
            kp = json.load(open(os.path.join(KP, f'kp_{s}_L{L}_s{sigma:g}_k3f.json')))
            reach['C-K3f'].append((np.abs(np.array(kp['heavi']) / p - 1) <= 0.5).mean(0))
            M = np.array(kp['moments'])
            e2 = np.array([[sp4(T['thr'][i, k], D.cumulants_from_moments(M[:, i])) for k in range(5)] for i in range(NN)])
            reach['U2'].append((np.abs(e2 / p - 1) <= 0.5).mean(0))
        U3 = np.array(u3); GS = np.array(gs)
        print(f'\n=== sigma {sigma}, L = {L}')
        for nm, R in reach.items():
            R = np.array(R)
            row = []
            for k in range(5):
                m_, lo, hi = boot(R[:, k], lambda v: float(np.mean(v)))
                row.append(f'{m_:.2f} [{lo:.2f}, {hi:.2f}]')
            print(f'  fraction within 50% {nm:>14}: ' + ' | '.join(row))
        for conv, f in (('lower median', med_low), ('midpoint median', med_mid)):
            row = []
            for k in range(5):
                bo = np.array([f(U3[j, :, k]) for j in range(20)]); bc = np.array([f(GS[j, :, k]) for j in range(20)])
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = np.where(np.isinf(bo) & np.isinf(bc), 1.0, bc / bo)
                m_, lo, hi = boot(ratio, f)
                row.append(f'{m_:.3g} [{lo:.3g}, {hi:.3g}]' + (' WIN' if lo > 1 else ''))
            print(f'  U3 / splitting B50 ratio ({conv}): ' + ' | '.join(row))
