"""Fixed point of the replica four-point vertex for the radial asymmetry (post hoc, HEURISTIC adiabatic closure).
eta = log|h1| - log|h2| for two noise replicas (same weights, independent noise). One exact Gram-chain step from a
state at radius Q, angle delta*(Q) (L0 fixed point) and asymmetry eta gives E[eta' | eta] ~ b eta, b = Q/(Q+1), and
Var(eta' | eta = 0) = V1 (estimated by sampling the exact one-step law). AR(1) fixed point: E eta^2 = V1 / (1 - b^2).
Compared with the exact chain's E eta^2 in the median-radius bin (results/partG_scores.json) and astra's closure."""
import sys, json, math
sys.path.insert(0, '/home/agent/workspaces/mats/work/stationary-numerics/lib')
import numpy as np
import gram_chain as gc

D = '/home/agent/workspaces/mats/work/stationary-numerics'
G = json.load(open(f'{D}/results/partG_scores.json'))
W = {n: json.load(open(f'{D}/runs/gram_window_n{n}.json')) for n in (16, 32, 64, 128, 256, 512)}


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


rng = np.random.default_rng(7)
out = {}
for n, w in W.items():
    b_ = [r for r in w['D_by_Qbin'] if r['lo'] == 0.5][0]
    Q = b_['Qmed_over_n'] * n
    ds = delta_star(Q)
    P = 40000 if n <= 128 else 12000
    res = {}
    for eta in (0.0, 0.05):
        q1, q2 = Q * math.exp(eta), Q * math.exp(-eta)
        p = (1 - ds) * math.sqrt(q1 * q2)
        G0 = np.tile(np.array([[q1, p], [p, q2]]), (P, 1, 1))
        G1 = gc.step(G0, np.array([[1, 0], [0, 1]]), n, rng)
        e1 = 0.5 * (np.log(G1[:, 0, 0]) - np.log(G1[:, 1, 1]))
        res[eta] = (e1.mean(), e1.var(), e1.var() / math.sqrt(P / 2))
    b_sampled = res[0.05][0] / 0.05  # too noisy at large n (1 - b ~ 1/Q); kept for reference
    b = Q / (Q + 1.0)  # deterministic restoring of eta by the additive noise: d log(AQe^eta + B)/d eta = AQ/(AQ + B)
    V1 = res[0.0][1]
    fp = V1 / (1 - b * b)
    exact = b_['sR2']  # E[(log|h1| - log|h2|)^2] in the bin
    closure = 2 * (2.23 * Q ** (1 / 3) / n + 1 / (2 * n))  # astra's CV^2 law times 2 (E eta^2 = 2 CV^2)
    out[n] = dict(Q=Q, delta_star=ds, b=b, b_sampled=b_sampled, V1_times_n=V1 * n, fixed_point=fp, exact=exact, closure=closure,
                  fp_over_exact=fp / exact, closure_over_exact=closure / exact, one_minus_bsampled_times_Q=(1 - b_sampled) * Q)
    print(n, ' '.join('%s=%.4g' % kv for kv in out[n].items()), flush=True)
json.dump(out, open(f'{D}/results/vertex_fixed_point.json', 'w'), indent=1)
