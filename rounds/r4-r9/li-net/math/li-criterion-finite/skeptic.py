"""Skeptic checks for li-criterion-finite.md (numpy/scipy only)."""
import itertools, numpy as np
from scipy.optimize import linprog, minimize
rng = np.random.default_rng(1)
np.set_printoptions(precision=4, suppress=True)
def val(a, b): return np.array([a, b, a & b, a | b, 1 - a], float)
ALL = [val(a, b) for a in (0, 1) for b in (0, 1)]
PC = np.array([val(a, b) for a in (0, 1) for b in (0, 1) if (not a) or b])
relu = lambda x: max(0.0, x)
def in_K(p):
    W = PC; k = len(W); n = len(p)
    c = np.r_[np.zeros(k), np.ones(2 * n)]
    A = np.c_[W.T, np.eye(n), -np.eye(n)]; A = np.r_[A, np.r_[np.ones(k), np.zeros(2 * n)][None]]
    return linprog(c, A_eq=A, b_eq=np.r_[p, 1], bounds=[(0, None)] * (k + 2 * n)).fun
def fp(traders, p0, eta=0.1, iters=4000):
    p = p0.copy()
    for _ in range(iters): p = np.clip(p + eta * sum(T(p) for T in traders), 0, 1)
    return p
e = np.eye(5)
def eqT(a, b):  # equality a.p=b, two-sided
    return lambda p: -(a @ p - b) * a
def ineqT(a, b):  # a.p<=b, one-sided relu
    return lambda p: -relu(a @ p - b) * a
A_, B_, AB, AoB, nA = range(5)
T_neg = eqT(e[A_] + e[nA], 1); T_ie = eqT(e[AoB] + e[AB] - e[A_] - e[B_], 0); T_ax = eqT(e[A_] - e[AB], 0)
print("=== S1: purely LOCAL Frechet traders (each connective vs its immediate subformulas) + axiom ===")
frechet = [ineqT(e[AB] - e[A_], 0), ineqT(e[AB] - e[B_], 0), ineqT(e[A_] + e[B_] - e[AB], 1),
           ineqT(e[A_] - e[AoB], 0), ineqT(e[B_] - e[AoB], 0), ineqT(e[AoB] - e[A_] - e[B_], 0)]
loc = [T_neg, T_ax] + frechet
inc = sum(in_K(fp(loc, rng.random(5))) > 1e-6 for _ in range(60))
print(f" local Frechet + negation + axiom (no IE, no hand-picked facet A<=B): incoherent fixed points {inc}/60")
inc = sum(in_K(fp(loc + [T_ie], rng.random(5))) > 1e-6 for _ in range(60))
print(f" ... plus inclusion-exclusion: incoherent {inc}/60")

print("=== S2: Thm 4 gap: zero of convex F_k on the boundary need not have grad 0 ===")
F = lambda p: p; T = lambda p: -1.0
print(f" F(p)=p on [0,1]: zero at p=0, T(0)={T(0)} != 0, yet clip(0+T(0))={np.clip(0+T(0),0,1)} (still a fixed point)")

print("=== S3: finite-weight relu^2 relation traders do NOT give argmin over K_Gamma ===")
q = np.array([0.9, 0.2, 0.5, 0.5, 0.5])  # incoherent claims (A > B)
claim = lambda p: q - p
for w in (1.0, 10.0, 100.0):
    trs = [claim] + [lambda p, T=T: w * T(p) for T in [T_neg, T_ie, T_ax, ineqT(e[A_] - e[B_], 0)]]
    p = fp(trs, np.full(5, .5), eta=0.5 / (1 + 4 * w), iters=200000)
    print(f" relation weight {w:6.1f}: fixed point {p}, L1 dist to K_Gamma {in_K(p):.4f}")
# true argmin over K of 1/2|p-q|^2
cons = [{"type": "eq", "fun": lambda m: m.sum() - 1}]
r = minimize(lambda m: 0.5 * np.sum((PC.T @ m - q) ** 2), np.ones(3) / 3, constraints=cons, bounds=[(0, 1)] * 3, method="SLSQP", options={"ftol": 1e-14})
print(f" exact argmin over K_Gamma: {PC.T @ r.x}")

print("=== S4: KL direction. PDG Inc (Richardson 2022 eq.1, Prop 9) = sum beta D(mu||claim) = KL(p||q) ===")
qq = np.array([0.95, 0.10, 0.60, 0.40, 0.20]); c = np.ones(5)
kl = lambda a, b: a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))
def argmin_K(f):
    r = minimize(lambda m: f(np.clip(PC.T @ m, 1e-9, 1 - 1e-9)), np.ones(3) / 3, constraints=cons, bounds=[(1e-9, 1)] * 3, method="SLSQP", options={"ftol": 1e-15, "maxiter": 1000})
    return PC.T @ r.x
pq = argmin_K(lambda p: np.sum(c * kl(qq, p)))   # the file's trader (q-p)/(p(1-p)) = -grad KL(q||p)
pp = argmin_K(lambda p: np.sum(c * kl(p, qq)))   # PDG inconsistency
print(f" argmin_K sum KL(q||p) [file's trader]: {pq}")
print(f" argmin_K sum KL(p||q) [PDG Inc]     : {pp}")
print(f" difference L1 = {np.abs(pq-pp).sum():.4f}")
pt = 0.3; qt = 0.8
g_num = (kl(qt, pt + 1e-6) - kl(qt, pt - 1e-6)) / 2e-6
print(f" check d/dp KL(q||p) = (p-q)/(p(1-p)): numeric {g_num:.5f} vs formula {(pt-qt)/(pt*(1-pt)):.5f}")
g2 = (kl(pt + 1e-6, qt) - kl(pt - 1e-6, qt)) / 2e-6
lg = lambda x: np.log(x / (1 - x))
print(f" check d/dp KL(p||q) = logit p - logit q: numeric {g2:.5f} vs formula {lg(pt)-lg(qt):.5f}")

print("=== S5: Thm 7 relies on no block-index input: a trader using (t, p_t) exploits the a/b oscillation ===")
a = PC.T @ np.array([.5, .3, .2]); b = PC.T @ np.array([.2, .3, .5])
seq = []; L = 1
for k in range(9): seq += [a if k % 2 == 0 else b] * L; L *= 4
hold = np.zeros(5); cash = 0.0; vals = []
for t, p in enumerate(seq):
    nxt_change = t + 1 < len(seq) and not np.allclose(seq[t + 1], p)
    # trader sees only t and p_t; schedule known from t. buy B at the last a-day, sell it at the last b-day
    x = np.zeros(5)
    if nxt_change and np.allclose(p, a): x[1] = 1
    if nxt_change and np.allclose(p, b) and hold[1] > 0: x[1] = -1
    hold += x; cash -= x @ p
    vals.append([hold @ W + cash for W in PC])
vals = np.array(vals)
print(f" min value over time/worlds {vals.min():.3f}, final value {vals[-1]} (grows by 0.3 per a->b cycle; unbounded as cycles -> inf)")

print("=== S6: Theorem 3 tightness: (1+|x|inf) vs max(1,|x|inf) ===")
worst = 0; worst2 = 0
for _ in range(100000):
    p = rng.random(5); x = rng.normal(size=5) * rng.exponential(size=5)
    g = np.sum(np.where(x > 0, x * (1 - p), -x * p)); r = np.abs(np.clip(p + x, 0, 1) - p).sum()
    worst = max(worst, g - max(1, np.abs(x).max()) * r)
print(f" max(gain - max(1,|x|inf)|r|1) = {worst:.2e}  (the sharper constant also holds)")
# but the residual is not scale invariant: x -> eta x
p = np.array([0.5]); x = np.array([0.4])
for eta in (1, 0.1, 0.01):
    r = abs(np.clip(p + eta * x, 0, 1) - p).sum(); print(f" eta={eta}: residual of clip(p+eta x)-p = {r:.4f}, gain of x = {x[0]*(1-p[0]):.3f}; bound with eta-residual needs factor max(1,|x|)/eta")

print("=== S7: minimax explanation. LP dual of Check 2 = the certificate; minimax HOLDS ===")
grid = np.linspace(0, 1, 1001); T1 = lambda p: 0.3 - p; T2 = lambda p: 0.7 - p
G = np.array([[T(g) * (W - g) for g in grid] for T in (T1, T2) for W in (0, 1)])  # 4 x grid
# max_lambda min_p lambda.G : LP
res = linprog(np.r_[np.zeros(4), -1], A_ub=np.c_[-G.T, np.ones(len(grid))], b_ub=np.zeros(len(grid)),
              A_eq=np.r_[np.ones(4), 0][None], b_eq=[1], bounds=[(0, None)] * 4 + [(None, None)])
res2 = linprog(np.r_[np.zeros(len(grid)), 1], A_ub=np.c_[G, -np.ones(4)], b_ub=np.zeros(4),
               A_eq=np.r_[np.ones(len(grid)), 0][None], b_eq=[1], bounds=[(0, None)] * len(grid) + [(None, None)])
print(f" max_lambda min_p = {-res.fun:.4f} with lambda = {res.x[:4]}; min_pi max_(k,W) = {res2.fun:.4f}  (equal: von Neumann minimax holds)")

print("=== S8: Thm 4 + Idea: convex but not strictly convex F -> fixed points not unique ===")
Fg = lambda p: -np.array([relu(p[0] - p[1]), -relu(p[0] - p[1])])  # T = -grad 1/2 relu(pA-pB)^2
pts = [fp([Fg], rng.random(2)) for _ in range(5)]
print(f" distinct fixed points: {np.array(pts)}")

print("=== S9: Theorem 6(a) needs sentences whose truth does not depend on prices ===")
p0 = 0.5; p = 0.5
print(f" paradox chi <-> (p_chi < {p0}) at constant p={p}: truth=0, seller of 1 share gains {p} per day: constant market at an 'interior' point exploited")
