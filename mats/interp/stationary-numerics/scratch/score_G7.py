"""Score Amendment G1 / G7: the cloud excess over the coordinator's Lyapunov formula, predicted from the finite-angle
fixed point, on the stationary lane's fresh nets (runs/cloudA3_*) and, post hoc, on the Part A nets (runs/cloud_*)."""
import json, glob, math
import numpy as np

SL = '/home/agent/workspaces/mats/work/stationary/runs'
D = '/home/agent/workspaces/mats/work/stationary-numerics'


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


def rows_for(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f))
        n = d['n']
        Q = d['X_quantiles'][len(d['X_quantiles']) // 2] if isinstance(d['X_quantiles'], list) else d['X_quantiles']['0.5']
        ev = np.array(d['evC']); cp = np.array(d['coord_pred'])
        r = ev[1:8] / cp[1:8]
        H = sum(1.0 / k for k in range(1, n + 1))
        pred = 2.1 * Q * delta_star(Q) / (H - 1)
        pred_rad = 2.5 * Q * (2.23 * Q ** (1 / 3) / n + 1 / (2 * n))
        out.append(dict(file=f.split('/')[-1], n=n, Q=Q, ratio_meas=float(np.median(r)), ratio_pred=pred,
                        mp=float(np.median(r) / pred), rad_meas=float(ev[0] / cp[0]), rad_pred=pred_rad,
                        mp_rad=float(ev[0] / cp[0] / pred_rad)))
    return out


fresh = rows_for(f'{SL}/cloudA3_n*_s*.json')
old = rows_for(f'{SL}/cloud_n*_s100[0-3].json')
g8 = rows_for(f'{D}/runs/cloudG8_n*_s*.json')
G7 = all(0.6 <= r['mp'] <= 1.6 and 0.4 <= r['mp_rad'] <= 1.5 for r in fresh if r['n'] >= 64)
G8 = len(g8) == 12 and all(0.6 <= r['mp'] <= 1.6 and 0.4 <= r['mp_rad'] <= 1.5 for r in g8)
G8_ang = len(g8) == 12 and all(0.6 <= r['mp'] <= 1.6 for r in g8)
json.dump(dict(hit=bool(G7), fresh=fresh, partA_posthoc=old, G8_hit=bool(G8), G8_angular_clause=bool(G8_ang), G8=g8),
          open(f'{D}/results/partG7_scores.json', 'w'), indent=1)
print('G7 hit:', G7, ' G8 hit:', G8, '(angular clause:', G8_ang, ')')
for lab, rows in (('G8 prospective', g8), ('fresh (registered)', fresh), ('Part A nets (post hoc)', old)):
    print(lab)
    for r in sorted(rows, key=lambda r: (r['n'], r['file'])):
        print('  n=%3d Q=%7.1f  k2-8: meas %.2f pred %.2f (m/p %.2f)   radial: meas %.1f pred %.1f (m/p %.2f)' % (
            r['n'], r['Q'], r['ratio_meas'], r['ratio_pred'], r['mp'], r['rad_meas'], r['rad_pred'], r['mp_rad']))
