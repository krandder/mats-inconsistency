"""Level-0 effective theory (infinite-width noisy kernel recursion, radius held at Q; HEURISTIC adiabatic model).
Two replicas, preactivation correlation rho; one layer:  rho' = (Q kappa(rho) + s) / (Q + 1),
s = 1 if the replicas share that layer's noise, 0 if not (exact conditional mean of the Gram chain at q1 = q2 = Q,
ignoring the +1 growth of Q, which finite-width multiplicative fluctuations balance on average).
Outputs, for Q = 0.5771 n and a few Q:
  delta* = 1 - rho* (independent noise fixed point; compare 2.2309 Q^-2/3);
  single-refresh decay delta_t (replicas share all noise except one layer) and the angular interaction order
  D_ang = sum_t delta_t / delta* over t = 0..J-1 (J = 4n window), also D for a = relu(W h) and output gates g;
  births per layer n arccos(Q/(Q+1))/pi and the gate cascade sum_t n arccos(rho_t)/pi over 2n layers."""
import math, json, sys
import numpy as np


def kappa(c):
    c = min(max(c, -1.0), 1.0)
    return (math.sqrt(max(1 - c * c, 0.0)) + (math.pi - math.acos(c)) * c) / math.pi


def fixed_point(Q):
    rho = 1 - 2.23 * Q ** (-2 / 3)
    for _ in range(200000):
        r2 = Q * kappa(rho) / (Q + 1)
        if abs(r2 - rho) < 1e-15:
            break
        rho = r2
    return rho


def f_a(rho):  # E|a1-a2|^2 / (n Q) for equal radii
    return 2 * (1 - kappa(rho))


def f_g(rho):  # gate disagreement fraction
    return math.acos(min(rho, 1.0)) / math.pi


out = {}
for n in [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
    Q = 0.5770780 * n
    J = 4 * n
    rs = fixed_point(Q)
    ds = 1 - rs
    rho = Q / (Q + 1)
    tot = dict(u=0.0, a=0.0, g=0.0)
    casc = 0.0
    for t in range(J):
        tot['u'] += 1 - rho
        tot['a'] += f_a(rho)
        tot['g'] += f_g(rho)
        if t < 2 * n:
            casc += n * f_g(rho)
        rho = (Q * kappa(rho) + 1) / (Q + 1)
    D = dict(u=tot['u'] / ds, a=tot['a'] / f_a(rs), g=tot['g'] / f_g(rs))
    births = n * math.acos(Q / (Q + 1)) / math.pi
    out[n] = dict(Q=Q, delta_star=ds, closure=2.2309 * Q ** (-2 / 3), D_ang_u=D['u'], D_a=D['a'], D_g=D['g'],
                  births=births, births_over_sqrt_n=births / math.sqrt(n), cascade_2n=casc, cascade_over_n=casc / n)
    print(n, ' '.join('%s=%.4g' % (k, v) for k, v in out[n].items()), flush=True)
json.dump(out, open('/home/agent/workspaces/mats/work/stationary-numerics/results/l0_theory.json', 'w'), indent=1)
