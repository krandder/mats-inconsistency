"""T2 exact checks (no learning noise): Theorem 1 (stop-gradient fixed point vs counterfactual oracle)
and Theorem 2 (fused stationary point: desire-aligned error = (lam/2)|grad U|^2 >= 0).
Usage: python3 t2_exact.py OUT.json      (numpy only, seed 0, ~10 s)
"""
import json, sys, numpy as np
sig = lambda z: 1 / (1 + np.exp(-z))
rng = np.random.default_rng(0)
P0 = 0.5
G = np.linspace(1e-6, 1 - 1e-6, 20001)          # dense grid on (0,1)

def rand_reader():
    """Random smooth reader f:(0,1)->(0,1): P(action | report p). Mixture of 1-3 logistic terms (allows
    non-monotone, steep, self-fulfilling and diagonalising readers)."""
    k = rng.integers(1, 4)
    a = rng.normal(0, 1.5, k); w = rng.normal(0, 6, k); m = rng.dirichlet(np.ones(k))
    return lambda p: (m[None, :] * sig(a[None, :] + w[None, :] * (2 * np.asarray(p)[..., None] - 1))).sum(-1)

def fixed_points(f):
    g = f(G) - G; s = np.sign(g); idx = np.nonzero(s[:-1] * s[1:] < 0)[0]; out = []
    for i in idx:
        lo, hi = G[i], G[i + 1]
        for _ in range(60):
            mid = (lo + hi) / 2
            if (f(np.array([mid]))[0] - mid) * (f(np.array([lo]))[0] - lo) > 0: lo = mid
            else: hi = mid
        out.append((lo + hi) / 2)
    return out

def deriv(f, x, h=1e-5): return (f(x + h) - f(x - h)) / (2 * h)

# ---------------- Part A: Theorem 1, 1-D ----------------
N = 4000
A = dict(n_readers=N, n_fixed_points=0, identity_max_resid=0.0, bound_seg_violations=0,
         Lglobal_lt1=0, Lglobal_lt1_unique_violations=0, Lglobal_lt1_bound_violations=0,
         decreasing=0, decreasing_unique_violations=0, decreasing_between_violations=0,
         multi_fp_readers=0, multi_fp_max_ratio_dist_over_h_minus_p0=0.0,
         Lseg_ge1_bound_ratio_max=0.0)
ratios_amp = []
for _ in range(N):
    f = rand_reader(); fps = fixed_points(f); h = f(np.array([P0]))[0]
    d = deriv(f, G[1:-1]); L = np.abs(d).max(); dec = (d <= 0).all()
    A["n_fixed_points"] += len(fps)
    if len(fps) > 1:
        A["multi_fp_readers"] += 1
        r = max(abs(p - h) for p in fps) / max(abs(h - P0), 1e-9)
        A["multi_fp_max_ratio_dist_over_h_minus_p0"] = max(A["multi_fp_max_ratio_dist_over_h_minus_p0"], min(r, 1e6))
    for p in fps:
        if abs(p - P0) > 1e-6:
            kap = (f(np.array([p]))[0] - h) / (p - P0)
            if abs(1 - kap) > 1e-6:
                A["identity_max_resid"] = max(A["identity_max_resid"], abs((p - P0) - (h - P0) / (1 - kap)))
            seg = np.linspace(min(p, P0), max(p, P0), 2001); Ls = np.abs(deriv(f, seg)).max()
            if abs(p - h) > Ls * abs(p - P0) + 1e-7: A["bound_seg_violations"] += 1
            if Ls >= 1: A["Lseg_ge1_bound_ratio_max"] = max(A["Lseg_ge1_bound_ratio_max"], abs(p - h) / max(abs(h - P0), 1e-9))
    if L < 1:
        A["Lglobal_lt1"] += 1
        if len(fps) != 1: A["Lglobal_lt1_unique_violations"] += 1
        for p in fps:
            if abs(p - h) > L / (1 - L) * abs(h - P0) + 1e-7: A["Lglobal_lt1_bound_violations"] += 1
            ratios_amp.append(abs(p - h) / max(L / (1 - L) * abs(h - P0), 1e-12))
    if dec:
        A["decreasing"] += 1
        if len(fps) != 1: A["decreasing_unique_violations"] += 1
        for p in fps:
            if not (min(P0, h) - 1e-7 <= p <= max(P0, h) + 1e-7): A["decreasing_between_violations"] += 1
A["Lglobal_lt1_bound_tightness_max"] = float(max(ratios_amp)) if ratios_amp else None
A["Lglobal_lt1_bound_tightness_median"] = float(np.median(ratios_amp)) if ratios_amp else None

# Liar sentence with finite precision sigma (LI Thm 4.11.2 analogue): chi true iff report < theta.
liar = {}
for th in [0.2, 0.5, 0.8]:
    for s in [0.1, 0.01, 0.001]:
        from math import erf, sqrt
        f = lambda p, th=th, s=s: 0.5 * (1 - np.vectorize(erf)((np.asarray(p) - th) / (s * sqrt(2))))
        fps = fixed_points(f); h = float(f(np.array([P0]))[0])
        liar[f"theta{th}_sigma{s}"] = dict(fixed_points=[float(x) for x in fps], h=h,
                                           between_p0_and_h=bool(all(min(P0, h) - 1e-9 <= x <= max(P0, h) + 1e-9 for x in fps)))

# Part A2: Theorem 1 on the 3-simplex, contraction readers, stop-gradient iteration
def rand_simplex_reader(k=3):
    W = rng.normal(0, 1, (k, k)); W *= rng.uniform(0.2, 3.0) / np.linalg.norm(W, 2); c = rng.normal(0, 1, k)
    def f(p):
        z = c + W @ p; e = np.exp(z - z.max()); return e / e.sum()
    return f
def jac(f, p, h=1e-6):
    k = len(p); J = np.zeros((k, k))
    for j in range(k):
        e = np.zeros(k); e[j] = h; J[:, j] = (f(p + e) - f(p - e)) / (2 * h)
    # restrict to tangent of the simplex
    T = np.eye(k) - 1.0 / k; return J @ T
A2 = dict(n=0, Lseg_bound_violations=0, Llt1=0, Llt1_bound_violations=0, max_resid_fixed_point=0.0)
p0 = np.ones(3) / 3
for _ in range(1000):
    f = rand_simplex_reader(); p = rng.dirichlet(np.ones(3))
    for _ in range(4000): p = 0.5 * p + 0.5 * f(p)            # stop-gradient Brier step, eta = 1/2
    A2["max_resid_fixed_point"] = max(A2["max_resid_fixed_point"], float(np.abs(f(p) - p).max()))
    if np.abs(f(p) - p).max() > 1e-8: continue
    A2["n"] += 1; h = f(p0)
    seg = [p0 + t * (p - p0) for t in np.linspace(0, 1, 101)]
    Ls = max(np.linalg.norm(jac(f, x), 2) for x in seg)
    if np.linalg.norm(p - h) > Ls * np.linalg.norm(p - p0) + 1e-7: A2["Lseg_bound_violations"] += 1
    Lg = max(np.linalg.norm(jac(f, rng.dirichlet(np.ones(3))), 2) for _ in range(200))
    if Lg < 1:
        A2["Llt1"] += 1
        if np.linalg.norm(p - h) > Lg / (1 - Lg) * np.linalg.norm(h - p0) + 1e-7: A2["Llt1_bound_violations"] += 1

# ---------------- Part B: Theorem 2 ----------------
def fused_stationary(f, U, lam, loss, p_init):
    """Gradient flow in logit space for loss (Brier or log) minus lam*U, with f held fixed inside the loss gradient
    (stop-gradient on the truth) but U differentiated through the reader (reward gradient)."""
    l = np.log(p_init / (1 - p_init))
    for _ in range(40000):
        p = sig(l); fp = f(np.array([p]))[0]; dU = deriv(U, np.array([p]))[0]
        if loss == "brier": gl = (-2 * (p - fp) + lam * dU) * p * (1 - p)
        else: gl = -(p - fp) + lam * dU * p * (1 - p)
        l += 0.5 * gl
    p = sig(l); return p, f(np.array([p]))[0], deriv(U, np.array([p]))[0], abs(gl)
B = dict(n=0, nonconverged=0, brier_formula_max_resid=0.0, log_formula_max_resid=0.0, D_min=np.inf,
         raw_err_negative=0, raw_err_sign_eq_sign_dU=0, exogenous_n=0, exogenous_formula_max_resid=0.0, cases=[])
for it in range(300):
    f = rand_reader(); Delta = rng.choice([-1, 1]) * rng.uniform(0.2, 1.0); r0 = rng.uniform(0, 0.5)
    lam = rng.choice([0.1, 0.3, 1.0]); loss = ["brier", "log"][it % 2]
    exo = it % 3 == 0
    if exo:   # situation-report / complete-plan-forecast case: truth exogenous (f constant), reward still depends on report
        y = rng.uniform(0.1, 0.9); fr = f; f = lambda p, y=y: np.full(np.shape(p), y)
        U = lambda p, fr=fr: r0 + Delta * fr(p)
    else:
        U = lambda p, f=f: r0 + Delta * f(p)
    p, fp, dU, res = fused_stationary(f, U, lam, loss, rng.uniform(0.2, 0.8))
    if res > 1e-7 or not (1e-4 < p < 1 - 1e-4): B["nonconverged"] += 1; continue
    B["n"] += 1
    pred = lam / 2 * dU if loss == "brier" else lam * p * (1 - p) * dU
    r = abs((p - fp) - pred)
    B["brier_formula_max_resid" if loss == "brier" else "log_formula_max_resid"] = max(B["brier_formula_max_resid" if loss == "brier" else "log_formula_max_resid"], r)
    if exo: B["exogenous_n"] += 1; B["exogenous_formula_max_resid"] = max(B["exogenous_formula_max_resid"], r)
    D = np.sign(dU) * (p - fp); B["D_min"] = min(B["D_min"], float(D))
    if p - fp < -1e-9: B["raw_err_negative"] += 1
    if np.sign(p - fp) == np.sign(dU) or abs(dU) < 1e-9: B["raw_err_sign_eq_sign_dU"] += 1
    if len(B["cases"]) < 12: B["cases"].append(dict(loss=loss, exogenous=bool(exo), lam=float(lam), p=float(p), truth=float(fp), dU=float(dU), err=float(p - fp), pred=float(pred)))
B["D_min"] = float(B["D_min"])

# Part B2: multidim fused (k=3), Brier and log, exogenous truth, U linear in a reader of the report
B2 = dict(n=0, brier_max_resid=0.0, log_max_resid=0.0, D_min=np.inf)
for it in range(200):
    k = 3; y = rng.dirichlet(np.ones(k)); W = rng.normal(0, 2, (k, k)); r = rng.normal(0, 1, k); lam = rng.choice([0.1, 0.5])
    def gU(p):   # U(p) = r . softmax(W p): reward of the reader's action distribution
        z = W @ p; e = np.exp(z - z.max()); s = e / e.sum(); return W.T @ ((np.diag(s) - np.outer(s, s)) @ r)
    loss = ["brier", "log"][it % 2]; th = np.zeros(k)
    for _ in range(30000):
        p = np.exp(th - th.max()); p /= p.sum(); S = np.diag(p) - np.outer(p, p); g = gU(p)
        if loss == "brier": gp = -2 * (p - y) + lam * g; th += 0.2 * S @ gp
        else: th += 0.2 * ((y - p) + lam * S @ g)
    p = np.exp(th - th.max()); p /= p.sum(); S = np.diag(p) - np.outer(p, p); g = gU(p)
    T = np.eye(k) - 1.0 / k
    if loss == "brier":
        pred = lam / 2 * T @ g; res = np.abs(T @ (p - y) - pred).max()   # tangent component (p - y sums to 0)
        if np.abs(S @ (-2 * (p - y) + lam * g)).max() > 1e-8: continue
        B2["brier_max_resid"] = max(B2["brier_max_resid"], float(res))
    else:
        pred = lam * S @ g; res = np.abs((p - y) - pred).max()
        if np.abs((y - p) + lam * S @ g).max() > 1e-8: continue
        B2["log_max_resid"] = max(B2["log_max_resid"], float(res))
    B2["n"] += 1; B2["D_min"] = min(B2["D_min"], float((p - y) @ g))
B2["D_min"] = float(B2["D_min"])

out = dict(partA_thm1_1d=A, partA_liar_finite_precision=liar, partA2_thm1_simplex=A2, partB_thm2_1d=B, partB2_thm2_simplex=B2)
json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)
for k, v in out.items():
    print(k, json.dumps({a: b for a, b in v.items() if a != "cases"}, default=float) if isinstance(v, dict) else v)
