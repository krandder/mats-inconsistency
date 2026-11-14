"""frequency-and-prior: numerical checks (numpy/scipy only).
Run: python3 check.py > output.txt
"""
import itertools, numpy as np
from scipy.special import xlogy, expit, logit
from scipy.optimize import minimize

rng = np.random.default_rng(0)
K = 6
W = np.array(list(itertools.product([0, 1], repeat=K)))          # 64 x 6 worlds

def bkl(x, p):
    """KL(Bern x || Bern p), elementwise, 0log0=0."""
    x = np.clip(x, 0.0, 1.0); p = np.clip(p, 0.0, 1.0)
    return xlogy(x, x) - xlogy(x, p) + xlogy(1 - x, 1 - x) - xlogy(1 - x, 1 - p)

def kl(m, q):
    return float(np.sum(xlogy(m, m) - xlogy(m, q)))

# ---------------- generator ----------------
def clause_val(cl):
    return np.any([W[:, v] == s for v, s in cl], axis=0)

def rand_clause(width):
    vs = rng.choice(K, width, replace=False)
    return [(int(v), int(rng.integers(0, 2))) for v in vs]

def gen_theory(m=5):
    while True:
        cls = [rand_clause(int(rng.choice([2, 3]))) for _ in range(m)]
        mod = np.all([clause_val(c) for c in cls], axis=0)
        if mod.sum() >= 3:
            return cls, mod

def gen_queries(cls, nq=12):
    """random formulas + one entailed (weakened axiom) + one refuted (its negation)."""
    qs, names = [], []
    for _ in range(nq - 2):
        t = rng.integers(0, 4)
        a, b = rng.choice(K, 2, replace=False); sa, sb = rng.integers(0, 2, 2)
        la, lb = W[:, a] == sa, W[:, b] == sb
        if t == 0: qs.append(la); names.append(f"x{a}={sa}")
        elif t == 1: qs.append(la | lb); names.append(f"x{a}={sa} or x{b}={sb}")
        elif t == 2: qs.append(la & lb); names.append(f"x{a}={sa} and x{b}={sb}")
        else: qs.append(la ^ lb); names.append(f"x{a}={sa} xor x{b}={sb}")
    c = cls[0]
    extra = [v for v in range(K) if v not in [u for u, _ in c]][0]
    ent = clause_val(c + [(extra, 1)])
    qs.append(ent); names.append("weakened axiom 0 (entailed)")
    qs.append(~ent); names.append("negated weakened axiom 0 (refuted)")
    return np.array(qs, dtype=float), names       # Q x 64

# ---------------- solvers ----------------
def project(A, P, base=None, b0=0.0, bP=1.0, iters=None, mu0=None):
    """min over mu in simplex(Mod) of b0*KL(mu||base) + bP*sum_j KL(A mu || P).
    A: Q x n (rows = query indicators restricted to models). L-BFGS on softmax logits."""
    n = A.shape[1]
    lP = logit(P)
    def F(z):
        z = z - z.max(); e = np.exp(z); mu = e / e.sum()
        x = np.clip(A @ mu, 1e-15, 1 - 1e-15)
        v = bP * bkl(x, P).sum()
        gm = bP * (A.T @ (logit(x) - lP))
        if b0 > 0:
            v += b0 * kl(mu, base)
            gm = gm + b0 * (np.log(np.maximum(mu, 1e-300)) - np.log(base) + 1)
        return v, mu * (gm - mu @ gm)
    z0 = np.zeros(n) if mu0 is None else np.log(np.maximum(mu0, 1e-300))
    r = minimize(F, z0, jac=True, method="L-BFGS-B", options={"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-12})
    z = r.x - r.x.max(); mu = np.exp(z) / np.exp(z).sum()
    x = A @ mu
    val = bP * bkl(x, P).sum() + (b0 * kl(mu, base) if b0 > 0 else 0.0)
    return mu, x, val

def project_slsqp(A, P):
    n = A.shape[1]
    f = lambda m: bkl(np.clip(A @ m, 1e-12, 1 - 1e-12), P).sum()
    r = minimize(f, np.full(n, 1 / n), method="SLSQP", bounds=[(0, 1)] * n,
                 constraints=[{"type": "eq", "fun": lambda m: m.sum() - 1}], options={"maxiter": 2000, "ftol": 1e-14})
    return r.fun

out = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s)

say("=== Part 4: generator K=6, exact model frequencies, coherent projection ===")
cls, mod = gen_theory()
Q, names = gen_queries(cls)
theta = np.array([0.8, 0.3, 0.6, 0.5, 0.2, 0.7])               # 'learned' product prior over atoms
pi = np.prod(np.where(W == 1, theta, 1 - theta), axis=1)
piT = pi * mod / (pi * mod).sum()
A = Q[:, mod]                                                   # restrict to models
base = piT[mod]
f = Q @ piT                                                     # model frequencies under pi
f_unif = Q @ (mod / mod.sum())
say("theory clauses (var,sign):", cls)
say("#models =", int(mod.sum()), " pi(Mod T) =", round(float((pi * mod).sum()), 5),
    " -log pi(T) =", round(float(-np.log((pi * mod).sum())), 5))
for nm, a, b in zip(names, f, f_unif):
    say(f"  f_pi={a:.4f}  f_unif={b:.4f}  {nm}")

# R2: <<pi(b0), T(inf)>> = -b0 log pi(T), minimiser pi_T
b0 = 2.0
mu_r2, _, v_r2 = project(A, np.full(len(f), .5), base=pi[mod] / pi[mod].sum(), b0=b0, bP=0.0, iters=3000)
# with bP=0 the problem is min b0 KL(mu||pi) over mu on Mod: compute directly against unnormalised pi
val_r2 = b0 * kl(mu_r2, pi[mod])
say("\n[R2] min_{mu on Mod} b0*KL(mu||pi) =", round(val_r2, 6), " vs -b0 log pi(T) =",
    round(float(-b0 * np.log(pi[mod].sum())), 6), " max|mu-pi_T| =", float(np.abs(mu_r2 - base).max()))

# noisy price table
sig = 1.0
P = expit(logit(np.clip(f, 1e-3, 1 - 1e-3)) + sig * rng.standard_normal(len(f)))
say("\nnoisy prices P:", np.round(P, 3))

# Coherence inconsistency (beta0 = 0)
mu_c, x_c, coh = project(A, P, iters=40000)
say("[R4] Coh(P) = <<T(inf),P(1)>> via L-BFGS on softmax =", round(coh, 6), "; via SLSQP =", round(project_slsqp(A, P), 6))
say("     projected prices x* =", np.round(x_c, 3))
ce_ex = bkl(f, P).sum()
say("[R6] Coh(P) <= sum KL(f||P):", round(coh, 5), "<=", round(ce_ex, 5), coh <= ce_ex + 1e-9)
rhs = coh + bkl(f, x_c).sum()
say("[R4] Pythagorean: sum KL(f||P) =", round(ce_ex, 5), ">= Coh + sum KL(f||x*) =", round(rhs, 5), ce_ex >= rhs - 1e-7)
margins = []
for _ in range(200):
    y = A @ rng.dirichlet(np.full(A.shape[1], 0.3))
    y = np.where(np.abs(y - np.round(y)) < 1e-12, np.round(y), y)   # forced (entailed/refuted) rows exactly 0/1
    margins.append(bkl(y, P).sum() - bkl(y, x_c).sum() - coh)
margins = np.array(margins)
say("     Pythagorean margin D(y,P)-D(y,x*)-Coh on 200 random coherent y: min =", f"{margins.min():.2e}",
    " #below -1e-5:", int((margins < -1e-5).sum()), " (nan count:", int(np.isnan(margins).sum()), ")")
gv = logit(np.clip(x_c, 1e-12, 1 - 1e-12)) - logit(P)
say("     KKT of the Bregman projection: min over models v of <grad h(x*) - grad h(P), v - x*> =", f"{((A.T - x_c) @ gv).min():.2e}", "(should be >= 0 up to solver tol)")
say("     log-loss excess vs f before / after projection:", round(ce_ex, 5), "->", round(bkl(f, x_c).sum(), 5))
say("     coherent but wrong prices (f_unif, i.e. wrong prior): Coh =", round(project(A, np.clip(f_unif, 1e-9, 1 - 1e-9), iters=20000)[2], 6),
    " sum KL(f||f_unif) =", round(bkl(f, f_unif).sum(), 5), " -> coherence alone is blind to the prior")

# R3 sandwich with base
say("\n[R3] <<pi(b0), T(inf), P(bP)>> sandwich, bP=1:")
c0 = -np.log(pi[mod].sum())
for b0 in [0.01, 0.1, 1.0, 10.0, 100.0]:
    mu, x, v = project(A, P, base=base, b0=b0, bP=1.0, iters=40000)
    tot = b0 * c0 + v
    lo, hi = b0 * c0 + coh, b0 * c0 + ce_ex
    # tilt check: mu ∝ base * exp(-(bP/b0) sum_j A_j (logit x_j - logit P_j))
    lt = np.log(base) - (1.0 / b0) * (A.T @ (logit(np.clip(x, 1e-15, 1 - 1e-15)) - logit(P)))
    tilt = np.exp(lt - lt.max()); tilt /= tilt.sum()
    say(f"  b0={b0:7.2f}: lo={lo:.5f}  value={tot:.5f}  hi={hi:.5f}  ok={lo - 1e-6 <= tot <= hi + 1e-6}"
        f"  max|x-f|={np.abs(x - f).max():.4f}  tilt-form err={np.abs(tilt - mu).max():.2e}")

# R7 iterated projection with base: P_{t+1} = marg mu*(P_t)
say("\n[R7] iterated projection P_{t+1} = marginals of argmin b0 KL(mu||pi_T) + bP sum KL(A mu||P_t), b0=0.1, bP=1:")
Pt = P.copy(); mu = None; hist = []
for t in range(1, 201):
    mu, Pt, _ = project(A, np.clip(Pt, 1e-12, 1 - 1e-12), base=base, b0=0.1, bP=1.0, iters=3000, mu0=mu)
    if t == 1:
        D1 = bkl(f, Pt).sum()
    hist.append((t, kl(mu, base), np.abs(Pt - f).max()))
for t, k_, e in hist:
    if t in (1, 2, 5, 10, 20, 50, 100, 200):
        bound = (1.0 / 0.1) * D1 / (t - 1) if t > 1 else float("inf")
        say(f"  t={t:4d}  KL(mu_t||pi_T)={k_:.3e}  proximal-point bound (bP/b0)D(f,P_1)/(t-1)={bound:.3e}  max|P_t - f|={e:.3e}")
mono = all(hist[i + 1][1] <= hist[i][1] + 1e-9 for i in range(len(hist) - 1))
say("  KL(mu_t||pi_T) monotone non-increasing:", mono)

# R1/R5: labels from sampled worlds; CE decomposition
say("\n[R1/R5] labels y_j = phi_j(w), w ~ pi_T; E[CE(y,P)] = H(f) + KL(f||P):")
ws = rng.choice(len(W), size=200000, p=piT)
Y = Q[:, ws]
ce_emp = -(xlogy(Y, P[:, None]) + xlogy(1 - Y, 1 - P[:, None])).mean(axis=1).sum()
H = -(xlogy(f, f) + xlogy(1 - f, 1 - f)).sum()
say("  empirical CE =", round(ce_emp, 4), "  H(f)+sum KL(f||P) =", round(H + ce_ex, 4))
say("  empirical label freq vs f (max abs diff):", round(float(np.abs(Y.mean(1) - f).max()), 4))

# R5: learning the prior: conditional NLL (= <<M+D>> - <<pi,T>>) recovers theta; unconditional NLL is biased
say("\n[R5] learning the base prior from (theory, world) pairs across random theories:")
true_theta = theta
data = []
for _ in range(1500):
    c_, m_ = gen_theory()
    pT = pi * m_ / (pi * m_).sum()
    w = rng.choice(len(W), p=pT)
    data.append((m_, w))
M_ = np.array([d[0] for d in data], float); wi = np.array([d[1] for d in data])
def nll(z, cond):
    th = expit(z)
    lp = np.sum(np.where(W == 1, np.log(th), np.log(1 - th)), axis=1)
    v = -lp[wi].mean()
    if cond:
        v += np.mean(np.log((M_ * np.exp(lp)).sum(1)))
    return v
zc = minimize(nll, np.zeros(K), args=(True,)).x
zu = minimize(nll, np.zeros(K), args=(False,)).x
say("  true theta        :", np.round(true_theta, 3))
say("  conditional ML    :", np.round(expit(zc), 3), " (=argmin <<pi,T,D(inf)>> - <<pi,T>>)")
say("  unconditional ML  :", np.round(expit(zu), 3), " (ignores that the world was conditioned on T: biased)")

# ---------------- Part 3: provability induction without resolutions ----------------
say("\n=== Part 3: budget-r provability induction, toy unit-propagation block model ===")
def chain_instance(n, family):
    """atoms x_0..x_n; anchor x_0=b; constraints x_i xor x_{i+1} = c_i.
    family A: b=1, all c=0 (equivalence chain, pattern visible). family B: b,c random (parity)."""
    if family == "A":
        b, c = 1, np.zeros(n, int)
    else:
        b, c = int(rng.integers(0, 2)), rng.integers(0, 2, n)
    d = int(rng.integers(1, n + 1))                      # query atom x_d (positive literal)
    val = (b + c[:d].sum()) % 2                          # its entailed value
    return b, c, d, val

def view(b, c, d, r, feature):
    """what an r-block reasoner sees: unit propagation reaches x_d iff d <= r;
    plus an O(1)-depth syntactic feature (all constants zero AND anchor positive)."""
    if d <= r:
        return ("derived", int((b + c[:d].sum()) % 2))
    return ("unknown", (int(b == 1 and c.sum() == 0),) if feature else ())

def bayes_prices(mix, r, n=60, N=40000, feature=True):
    tab = {}
    samples = []
    for _ in range(N):
        fam = "A" if rng.random() < mix else "B"
        b, c, d, val = chain_instance(n, fam)
        v = view(b, c, d, r, feature)
        tab.setdefault(v, []).append(val); samples.append((fam, d, val, v))
    price = {k: np.mean(v) for k, v in tab.items()}
    return price, samples

r = 20
for mix, feat, label in [(1.0, True, "family A only"), (0.0, True, "family B only"),
                          (0.5, True, "50/50 mix, template feature visible"),
                          (0.5, False, "50/50 mix, template feature hidden")]:
    price, samples = bayes_prices(mix, r, feature=feat)
    ent_long = [price[v] for fam, d, val, v in samples if val == 1 and d > r]
    ent_A = [price[v] for fam, d, val, v in samples if val == 1 and d > r and fam == "A"]
    ent_B = [price[v] for fam, d, val, v in samples if val == 1 and d > r and fam == "B"]
    ent_short = [price[v] for fam, d, val, v in samples if val == 1 and d <= r]
    say(f"  {label:40s}: Bayes budget-{r} price on entailed, derivation>{r}: "
        f"all={np.mean(ent_long):.3f} A={np.mean(ent_A) if ent_A else float('nan'):.3f} "
        f"B={np.mean(ent_B) if ent_B else float('nan'):.3f}; derivation<={r}: {np.mean(ent_short):.3f}")

# martingale / no expected net update across budgets (R9)
say("\n[R9] E[P_r2 | view_r1] = P_r1 (no expected net update), family B+A mix 0.5, r1=10, r2=30:")
N = 60000; recs = []
for _ in range(N):
    fam = "A" if rng.random() < 0.5 else "B"
    b, c, d, val = chain_instance(60, fam)
    recs.append((view(b, c, d, 10, True), view(b, c, d, 30, True), val))
t1, t2 = {}, {}
for v1, v2, y in recs:
    t1.setdefault(v1, []).append(y); t2.setdefault(v2, []).append(y)
p1 = {k: np.mean(v) for k, v in t1.items()}; p2 = {k: np.mean(v) for k, v in t2.items()}
g = {}
for v1, v2, y in recs:
    g.setdefault(v1, []).append(p2[v2])
for k in sorted(g, key=str):
    say(f"  view10={k}: P_10={p1[k]:.4f}  E[P_30|view10]={np.mean(g[k]):.4f}  n={len(g[k])}")

# R1b: world-first generator (w ~ pi, then m clauses drawn uniformly from the 3-clauses w satisfies)
say("\n[R1b] world-first generator gives the same posterior pi(.|Mod T) (each world satisfies the same number of k-clauses):")
allc = [list(zip(vs, sg)) for vs in itertools.combinations(range(K), 3) for sg in itertools.product([0, 1], repeat=3)]
cv = np.array([clause_val([(int(v), int(s)) for v, s in c]) for c in allc])      # 160 x 64
say("  #3-clauses satisfied per world: min", int(cv.sum(0).min()), "max", int(cv.sum(0).max()))
idx = rng.choice(len(allc), 5, replace=False)
modT = np.all(cv[idx], axis=0)
lik = np.prod(cv[idx] / cv.sum(0), axis=0)                                       # p(T | w), clauses iid uniform among satisfied
post = pi * lik / (pi * lik).sum()
say("  max |p(w|T) - pi(w|Mod T)| =", float(np.abs(post - pi * modT / (pi * modT).sum()).max()))
