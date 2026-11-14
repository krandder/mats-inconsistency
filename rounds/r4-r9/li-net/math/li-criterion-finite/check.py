"""Numerical checks for li-criterion-finite.md (numpy/scipy only)."""
import itertools, numpy as np
from scipy.optimize import linprog
rng = np.random.default_rng(0)
np.set_printoptions(precision=4, suppress=True)

# ---------- toy: atoms A,B; Gamma = {A -> B}; S = [A, B, A&B, AvB, ~A]
names = ["A", "B", "A&B", "AvB", "~A"]
def val(a, b): return np.array([a, b, a & b, a | b, 1 - a], float)
ALL = [val(a, b) for a in (0, 1) for b in (0, 1)]
PC = [val(a, b) for a in (0, 1) for b in (0, 1) if (not a) or b]   # A->B
PC = np.array(PC)
relu = lambda x: max(0.0, x)

# relation traders: each sells (or buys) a combination proportional to its violation
def T_neg(p):  r = p[0] + p[4] - 1;            return -r * np.array([1, 0, 0, 0, 1.])
def T_ie(p):   r = p[3] + p[2] - p[0] - p[1];  return -r * np.array([-1, -1, 1, 1, 0.])
def T_ax(p):   r = p[0] - p[2];                return -r * np.array([1, 0, -1, 0, 0.])
def T_facet(p): r = relu(p[0] - p[1]);         return -r * np.array([1, -1, 0, 0, 0.])  # A<=B facet
REL3 = [T_neg, T_ie, T_ax]

def gain(T, p, W): return float(T(p) @ (W - p))

def fixed_point(traders, p0, eta=0.1, iters=3000):
    p = p0.copy()
    for _ in range(iters):
        p = np.clip(p + eta * sum(T(p) for T in traders), 0, 1)
    return p

def in_K(p, worlds):
    """LP: is p a convex combination of worlds? returns residual of best fit."""
    W = np.array(worlds); k = len(W)
    # min sum|s| st W^T mu + s+ - s- = p, mu in simplex
    n = len(p)
    c = np.r_[np.zeros(k), np.ones(2 * n)]
    A = np.c_[W.T, np.eye(n), -np.eye(n)]; A = np.r_[A, np.r_[np.ones(k), np.zeros(2 * n)][None]]
    b = np.r_[p, 1]
    r = linprog(c, A_eq=A, b_eq=b, bounds=[(0, None)] * (k + 2 * n))
    return r.fun

print("=== Check 1: 5-sentence toy, 3 relation traders, common fixed point ===")
worst = -1; incoh = 0
for trial in range(60):
    p = fixed_point(REL3, rng.random(5))
    res = max(np.abs(np.clip(p + T(p), 0, 1) - p).max() for T in REL3)  # each T's own clip map
    g = max(gain(T, p, W) for T in REL3 for W in ALL)                     # all 4 worlds, even inconsistent
    worst = max(worst, g)
    if in_K(p, PC) > 1e-7: incoh += 1
    if trial < 3:
        print(f" p*={p}  max own-clip residual={res:.1e}  max gain over all T, all W={g:.2e}  dist-to-K_Gamma(L1)={in_K(p, PC):.3f}")
print(f" 60 random starts: worst gain of any relation trader in any world = {worst:.2e}")
print(f" fixed points that are INCOHERENT (P(A)>P(B) though A->B): {incoh}/60")
p = fixed_point(REL3, np.array([0.9, 0.2, 0.5, 0.5, 0.5]))
x = np.array([-1, 1, 0, 0, 0.])  # buy B, sell A
print(f" example p*={p}; Dutch book 'buy B sell A' gains in each consistent world: {[float(x @ (W - p)) for W in PC]}")

print("=== Check 1b: add the facet trader A<=B -> common fixed points = K_Gamma ===")
worst = -1; incoh = 0
for trial in range(60):
    p = fixed_point(REL3 + [T_facet], rng.random(5))
    worst = max(worst, max(gain(T, p, W) for T in REL3 + [T_facet] for W in ALL))
    if in_K(p, PC) > 1e-6: incoh += 1
print(f" worst gain={worst:.2e}; incoherent fixed points: {incoh}/60")

print("=== Check 1c: projection trader Pi_K(p)-p is a strict Dutch book off K ===")
from scipy.optimize import minimize
def proj(p):
    k = len(PC)
    f = lambda mu: np.sum((PC.T @ mu - p) ** 2)
    r = minimize(f, np.ones(k) / k, constraints=[{"type": "eq", "fun": lambda m: m.sum() - 1}], bounds=[(0, 1)] * k, method="SLSQP", options={"ftol": 1e-14})
    return PC.T @ r.x
mins = []
for _ in range(100):
    p = rng.random(5); q = proj(p); x = q - p
    mins.append((min(x @ (W - p) for W in PC), np.sum(x * x)))
mins = np.array(mins)
print(f" min over 100 random p of [min_W gain - |Pi(p)-p|^2] = {np.min(mins[:,0]-mins[:,1]):.2e} (>= 0 up to solver tol); min gain = {mins[:,0].min():.3f}")

print("=== Check 2: two traders with no common fixed point; no randomized market defends each (T,W) pair ===")
T1 = lambda p: 0.3 - p; T2 = lambda p: 0.7 - p
grid = np.linspace(0, 1, 1001)
# LP over distributions pi on grid: E_pi[T_k(p)(W-p)] <= 0 for k=1,2, W=0,1
A = np.array([[T(g) * (W - g) for g in grid] for T in (T1, T2) for W in (0, 1)])
r = linprog(np.zeros(len(grid)), A_ub=A, b_ub=np.zeros(4), A_eq=np.ones((1, len(grid))), b_eq=[1], bounds=[(0, None)] * len(grid))
print(f" deterministic: min_p max_(k,W) gain = {min(max(T(g)*(W-g) for T in (T1,T2) for W in (0,1)) for g in grid):.4f}")
print(f" randomized LP feasible? {r.status == 0} (status {r.status}; 2=infeasible)")
g = grid ** 2 - grid + 0.35
print(f" certificate lambda=1/2(T1,W=0)+1/2(T2,W=1): min_p g(p) = {g.min():.4f} at p={grid[g.argmin()]:.2f}")
agg = lambda p: T1(p) + T2(p)
pf = 0.5
print(f" aggregate firm T1+T2 fixed point p=0.5: firm gain W=0,1 = {agg(pf)*(0-pf):.2f},{agg(pf)*(1-pf):.2f}; T1 gain W=0 = {T1(pf)*(0-pf):.2f}, T2 gain W=1 = {T2(pf)*(1-pf):.2f}")

print("=== Check 3: self-forecasting trader T(p)=Phi(p)-p; value >= -C + 1/2 sum|dp|^2 ===")
M = np.array([[0, -1], [1, 0]]) * 0.9
Phi = lambda p: 0.5 + M @ (p - 0.5) * 0.4 / max(0.4, np.linalg.norm(p - 0.5)) if False else 0.5 + 0.4 * np.array([np.cos(np.arctan2(*(p-0.5)[::-1]) + 0.7), np.sin(np.arctan2(*(p-0.5)[::-1]) + 0.7)])
p = np.array([0.9, 0.5]); hold = np.zeros(2); cash = 0.0; qv = 0.0; p0 = p.copy()
for t in range(2000):
    x = Phi(p) - p; hold += x; cash -= x @ p; pn = Phi(p); qv += np.sum((pn - p) ** 2); p = pn
vals = [hold @ np.array(W) + cash for W in itertools.product((0, 1), repeat=2)]
ident = 0.5 * qv - 0.5 * (p @ p - p0 @ p0) - cash
print(f" limit cycle (rotation by 0.7 rad on a circle): value in 4 worlds after 2000 steps = {np.round(vals,2)}; |holdings|={np.linalg.norm(hold):.3f}")
print(f" identity check cash = 1/2 QV - 1/2(|p_t|^2-|p_0|^2): residual {ident:.2e}")

print("=== Check 4: memoryless traders cannot force convergence (two coherent points, growing runs) ===")
a = PC.T @ np.array([.5, .3, .2]); b = PC.T @ np.array([.2, .3, .5])
isa = []; L = 1
for k in range(9):
    isa += [k % 2 == 0] * L; L *= 4
isa = np.array(isa, float)
na = np.cumsum(isa); nb = np.arange(1, len(isa) + 1) - na
worst_exploit = 0
for trial in range(300):
    Wt = rng.normal(size=(5, 5)); c = rng.normal(size=5)
    f = lambda p: np.tanh(Wt @ p + c)          # arbitrary continuous memoryless trader
    fa, fb = f(a), f(b)
    V = np.array([na * (fa @ (W - a)) + nb * (fb @ (W - b)) for W in PC])   # value path in each consistent world
    lower = V.min(); upper = V.max()
    # exploitation needs lower bounded & upper unbounded: report ratio upper/|lower|
    if lower > -1e-9 and upper > 1: worst_exploit += 1
print(f" a={a}, b={b} (both in relint K_Gamma); random tanh traders with value bounded below yet growing: {worst_exploit}/300")

print("=== Check 5: constant market in relint K is not exploited by ANY (history-dependent) trader ===")
mu = np.array([.5, .3, .2]); p = PC.T @ mu
worst_ratio = 0
for trial in range(2000):
    xs = rng.normal(size=(50, 5)) * rng.random((50, 1))
    V = np.cumsum(xs @ (PC - p).T, axis=0)       # value paths, worlds in columns
    lo = -V.min(); hi = V.max()
    if lo > 0: worst_ratio = max(worst_ratio, hi / lo)
print(f" max over random trade histories of (max value)/(max loss) = {worst_ratio:.3f} <= (1-mu_min)/mu_min = {(1-mu.min())/mu.min():.3f}")
pb = PC.T @ np.array([.5, .5, 0.])   # boundary point: world (1,1) has prob 0
x = np.array([0, 0, 1, 0, 0.])      # buy A&B at price 0
print(f" boundary p={pb}: buying A&B at price {pb[2]} gains per day {[float(x@(W-pb)) for W in PC]} (weak Dutch book)")

print("=== Check 6: paradox chi <-> (p_chi < p0), ramp trader + weight-w constant seller ===")
p0 = 0.8
for w in (0.0, 0.05, 0.2):
    T = lambda q: (p0 - q) - w
    q = 0.3
    for _ in range(5000): q = min(1, max(0, q + 0.1 * T(q)))
    truth = 1.0 if q < p0 else 0.0
    print(f" w={w}: fixed point p_chi={q:.4f}, truth={truth}, ramp trader gain={(p0-q)*(truth-q):+.4f}, seller gain={-w*(truth-q):+.4f}, firm gain={T(q)*(truth-q):+.4f}")

print("=== Check 7: enumeration-free bound  max_{W in cube} x.(W-p) <= (1+|x|_inf) * |clip(p+x)-p|_1 ===")
worst = 0
for _ in range(100000):
    n = 5; p = rng.random(n) * (rng.random(n) > 0.2) ; p = np.where(rng.random(n) < 0.1, 1.0, p)
    x = rng.normal(size=n) * rng.exponential(size=n)
    gain_cube = np.sum(np.where(x > 0, x * (1 - p), -x * p))       # exact max over the cube's vertices
    r = np.abs(np.clip(p + x, 0, 1) - p).sum()
    bound = (1 + np.abs(x).max()) * r
    worst = max(worst, gain_cube - bound)
print(f" max(gain - bound) over 1e5 random (p,x) = {worst:.2e} (<= 0 confirms)")
