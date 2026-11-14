"""Score PREREG Part G (exact Gram chain vs L0 effective theory vs the stationary lane) -> results/partG_scores.json."""
import json, math, os, sys
import numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/stationary-numerics/scratch')

D = '/home/agent/workspaces/mats/work/stationary-numerics'
L0 = {int(k): v for k, v in json.load(open(f'{D}/results/l0_theory.json')).items()}


def kappa(c):
    c = min(max(c, -1.0), 1.0)
    return (math.sqrt(max(1 - c * c, 0.0)) + (math.pi - math.acos(c)) * c) / math.pi


def delta_star(Q):
    rho = 1 - 2.23 * Q ** (-2 / 3)
    for _ in range(400000):
        r2 = Q * kappa(rho) / (Q + 1)
        if abs(r2 - rho) < 1e-15:
            break
        rho = r2
    return 1 - rho


W = {}
for n in (8, 16, 32, 64, 128, 256, 512, 1024):
    f = f'{D}/runs/gram_window_n{n}.json'
    if os.path.exists(f):
        W[n] = json.load(open(f))
PR = {}
for n in (16, 32, 64, 128, 256, 512):
    f = f'{D}/runs/gram_pr_n{n}.json'
    if os.path.exists(f):
        PR[n] = json.load(open(f))


def medbin(rows):
    for r in rows:
        if r['lo'] == 0.5:
            return r
    return None


S = {}
rows = []
g2 = {}
for n, w in W.items():
    b = medbin(w['D_by_Qbin'])
    Q = b['Qmed_over_n'] * n
    ds = delta_star(Q)
    cv2 = b['sR2'] / 2
    law = 2.23 * Q ** (1 / 3) / n + 1 / (2 * n)
    g2[n] = dict(delta_exact=b['delta_mean'], delta_L0=ds, ratio=b['delta_mean'] / ds, cv2=cv2, cv2_law=law,
                 cv2_ratio=cv2 / law, Du_bin=b['D']['u'], Du_L0=L0[n]['D_ang_u'], bin_count=b['count'])
ns = sorted(g2)
big = [n for n in ns if n >= 64]
S['G2'] = dict(hit=bool(all(0.8 <= g2[n]['ratio'] <= 1.2 for n in big) and
                        abs(g2[max(big)]['ratio'] - 1) < abs(g2[64]['ratio'] - 1)) if 1024 in g2 else None,
               rows={str(n): {k: g2[n][k] for k in ('delta_exact', 'delta_L0', 'ratio')} for n in ns})
S['G3'] = dict(hit=bool(all(1 / 1.5 <= g2[n]['cv2_ratio'] <= 1.5 for n in big)),
               rows={str(n): {k: g2[n][k] for k in ('cv2', 'cv2_law', 'cv2_ratio')} for n in ns})
G4a = all(0.7 <= g2[n]['Du_bin'] / g2[n]['Du_L0'] <= 1.3 for n in ns if n >= 32)
Dh = {n: W[n]['D']['h'] for n in ns}
Du = {n: W[n]['D']['u'] for n in ns}
Dg = {n: W[n]['D']['g'] for n in ns}
G4b = (1024 in Dh) and Dh[1024] >= 1.3 * Dh[64] and Du[1024] >= 1.3 * Du[64]
sl = [n for n in ns if 64 <= n <= 1024]
slope = float(np.polyfit(np.log(sl), np.log([Dg[n] for n in sl]), 1)[0])
G4c = 0.40 <= slope <= 0.75
SLmed = {16: dict(h=2.49, u=2.745, a=2.715), 64: dict(h=2.135, u=2.54, a=2.235), 256: dict(h=3.055, u=3.015, a=3.145)}
G4d_rows = {}
G4d = True
for n in (16, 64, 256):
    rr = {k: W[n]['D'][k] / SLmed[n][k] for k in ('h', 'u', 'a')}
    G4d_rows[str(n)] = rr
    G4d &= all(1 / 1.5 <= v <= 1.5 for v in rr.values())
S['G4'] = dict(hit_a=bool(G4a), hit_b=bool(G4b), hit_c=bool(G4c), hit_d=bool(G4d),
               D={str(n): W[n]['D'] for n in ns}, D_se={str(n): W[n]['D_se'] for n in ns},
               Du_bin_vs_L0={str(n): (g2[n]['Du_bin'], g2[n]['Du_L0']) for n in ns}, Dg_slope_64_1024=slope,
               exact_over_stationary_lane=G4d_rows)
G5b = {n: W[n]['births_over_sqrt_n'] / L0[n]['births_over_sqrt_n'] for n in ns}
G5c = {n: W[n]['cascade_2n_total_over_n'] / L0[n]['cascade_over_n'] for n in ns}
SLc = {64: 2.47, 256: 4.15}
G5s = {n: W[n]['cascade_2n_total_over_n'] / SLc[n] for n in (64, 256)}
S['G5'] = dict(hit=bool(all(abs(G5b[n] - 1) <= 0.05 for n in ns if n >= 16) and
                        all(abs(G5c[n] - 1) <= 0.25 for n in ns if n >= 32) and all(abs(v - 1) <= 0.3 for v in G5s.values())),
               births_exact_over_L0={str(n): G5b[n] for n in ns}, cascade_exact_over_L0={str(n): G5c[n] for n in ns},
               cascade_exact_over_stationary_lane={str(n): v for n, v in G5s.items()},
               cascade_exact_over_n={str(n): W[n]['cascade_2n_total_over_n'] for n in ns},
               births_over_sqrt_n={str(n): W[n]['births_over_sqrt_n'] for n in ns})
g6 = {}
for n, p in PR.items():
    b = medbin(p['by_Qbin'])
    H = sum(1.0 / k for k in range(1, n + 1))
    harm = (H - 1) ** 2 / (math.pi ** 2 / 6 - 1)
    g6[n] = dict(PR_u=b['PR_u'], harmonic=harm, ratio=b['PR_u'] / harm, PR_h=b['PR_h'])
S['G6'] = dict(hit=bool(all(0.5 <= g6[n]['ratio'] <= 2 for n in g6) and len(g6) == 6), rows={str(n): g6[n] for n in g6})
json.dump(S, open(f'{D}/results/partG_scores.json', 'w'), indent=1)
for k, v in S.items():
    print(k, {kk: vv for kk, vv in v.items() if kk.startswith('hit')})
print('n | delta exact/L0 | CV2/law | D_u(bin)/L0 | D_h D_u D_a D_g (all) | births/sqrt n | cascade/n (L0) | PR_u/harm')
for n in ns:
    print(n, '| %.3f | %.2f | %.2f | %.2f %.2f %.2f %.1f | %.3f | %.2f (%.2f) | %s' % (
        g2[n]['ratio'], g2[n]['cv2_ratio'], g2[n]['Du_bin'] / g2[n]['Du_L0'], Dh[n], Du[n], W[n]['D']['a'], Dg[n],
        W[n]['births_over_sqrt_n'], W[n]['cascade_2n_total_over_n'], L0[n]['cascade_over_n'],
        '%.2f' % g6[n]['ratio'] if n in g6 else '-'))
