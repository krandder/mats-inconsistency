"""A2: closed forms checked against numerics.
(1) Large-L' edge law of Fuss-Catalan: n * FC_N(mass above delta^2 x_+) -> n G(delta) / (N + 1), with
    G(delta) = int_0^{theta_delta} (t/pi) |d log f/dt| dt, f(t) = (sin t / t) exp(t cot t), log f(theta_delta) = 1 + 2 ln delta.
(2) Single-path ReLU counts: N_surv(delta) ~ (n/2) * FC_{L'} mass (half width); compare the measured counts.
(3) Two-replica (annealed) participation ratio of M = E[J^T J], from the measured per-layer pairwise both-active
    fractions q_l and active fractions p_l: exact Wick recursion for F = ||Y||_F^2, T = tr Y, Q = tr Y^2 of the cross
    Gram Y_l = A_l ... A_1 A'_1^T ... A'_l^T (A = D W), started at Y_0 = D_c D'_c.
python a2_theory.py runs/n2_256_64_c32_s113_sig0.001.json ..."""
import json, math, os, sys
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import fc


def logf(t):
    return math.log(math.sin(t) / t) + t / math.tan(t)


def dlogf(t):
    return 2 / math.tan(t) - 1 / t - t / math.sin(t) ** 2


def G(delta):
    target = 1 + 2 * math.log(delta)
    th = brentq(lambda t: logf(t) - target, 1e-9, math.pi - 1e-12)
    val, _ = quad(lambda t: (t / math.pi) * abs(dlogf(t)), 1e-9, th, limit=400)
    return val, th


print("(1) edge law: G(delta) vs (N+1) * FC_N mass above delta^2 x_+ ; Lyapunov 2 ln(1/delta)")
for d in (math.exp(-1), math.exp(-2), 1e-2, 1e-3):
    g, th = G(d)
    fin = [(N + 1) * fc.tail_mass(d * d * fc.edge(N), N) for N in (8, 16, 32, 64, 128)]
    print(f"  delta {d:.3g}: G = {g:.3f} (theta {th:.3f}); (N+1)*mass at N=8,16,32,64,128: "
          + " ".join(f"{v:.3f}" for v in fin) + f"; 2 ln(1/delta) = {2*math.log(1/d):.3f}")


def two_replica(q, p, n, Lp, start_q, start_p):
    """returns (E||J||_F^2 / n, ||J J'^T||_F^2 / n) for one pair with overlaps q[l], actives p[l] (layers 1..Lp),
    and at the cut start_q (both active), start_p (active) -- all as fractions of n. Same-path: q = p."""
    v = 2.0 / n
    o0 = start_q * n
    F, T, Q = o0, o0, o0          # Y_0 = D D' : ||Y||_F^2 = o0, tr Y = o0, tr Y^2 = o0
    T2 = o0 * o0
    for l in range(Lp):
        P1 = p[l] * n; P2 = p[l] * n; o = q[l] * n
        F_new = v * v * (P1 * P2 * F + o * (T2 + Q))
        T2_new = v * v * (o * o * T2 + o * (F + Q))
        Q_new = v * v * (o * o * Q + o * (T2 + F))
        F, T2, Q = F_new, T2_new, Q_new
    return F / n


for fn in sys.argv[1:]:
    r = json.load(open(fn))
    n = r['n']
    qm = r['overlap']['q_meas']; pa = r['overlap']['p_active']
    print(f"\n{fn.split('/')[-1]}")
    for Lp in r['Lps']:
        R = r['Lp'][str(Lp)]
        # same-path: q = p
        f_same = two_replica(pa[1:], pa[1:], n, Lp, pa[0], pa[0])
        f_pair = two_replica(qm[1:], pa[1:], n, Lp, qm[0], pa[0])
        # tr M = E ||J||_F^2 = E[T(x,x)] ; in the annealed model T(x,x) = prod (2 p_l) * m_c ... use measured
        trM = R['trM']; trM2 = R['trM2']
        pr_pred = (trM ** 2) / (n * f_pair * (trM / (n * 0.5)) ** 2 * 0.25) if False else None
        # annealed normalisation: E||J||^2 = n p_c prod_l (2 p_l)
        T_same = n * pa[0] * np.prod([2 * x for x in pa[1:Lp + 1]])
        pr_same_pred = T_same ** 2 / (n * f_same)
        pr_M_pred = T_same ** 2 / (n * f_pair)
        print(f"  L' {Lp}: PR single path measured {R['pr_single_mean']:.2f}, annealed {pr_same_pred:.2f}, "
              f"n/(1+2L') {n/(1+2*Lp):.2f} | PR(M) measured {R['pr_M']:.2f}, two-replica {pr_M_pred:.2f} | "
              f"tr M/n measured {trM/n:.3f}, annealed {T_same/n:.3f}")
    # half-width FC counts vs measured single-path counts
    for Lp in r['Lps']:
        R = r['Lp'][str(Lp)]
        meas = {d: np.mean([x[d] for x in R['nsurv']['single']]) for d in R['nsurv']['single'][0]}
        pred = {f'{d:.3g}': (n / 2) * fc.tail_mass(d * d * fc.edge(Lp), Lp) for d in (math.exp(-1), math.exp(-2), 1e-2, 1e-3)}
        lyap_half = {f'{d:.3g}': 1 + (n / Lp) * math.log(1 / d) for d in (math.exp(-1), math.exp(-2), 1e-2, 1e-3)}
        coord = {f'{d:.3g}': 1 + (2 * n / Lp) * math.log(1 / d) for d in (math.exp(-1), math.exp(-2), 1e-2, 1e-3)}
        print(f"  L' {Lp} N_surv: " + "; ".join(f"d={d}: meas {meas[d]:.1f} FC(n/2) {pred[d]:.1f} Lyap(n/2) {lyap_half[d]:.1f} coordinator {coord[d]:.1f}" for d in meas))
