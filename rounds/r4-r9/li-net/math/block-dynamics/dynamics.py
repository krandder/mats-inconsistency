"""block-dynamics numerical check (eurisko li-net, 2026-10-02).
Random 6-atom theory, 2 axioms (fixed), Q query sentences, 30 relation constraints.
Dynamics: (a) projected gradient on PDG inconsistency over worlds (anchored claims q0),
(a2) self-referential: mirror (dual) flow and primal gradient flow on q,
(b) exponentiated gradient over worlds (= damped simultaneous IPF),
(c) row-action Bregman/Hildreth on the 30 relations: Euclidean + Bernoulli-KL, cyclic and parallel.
numpy/scipy only (+ matplotlib for the plot if importable)."""
import itertools, sys, json
import numpy as np
from scipy.optimize import minimize, brentq
from scipy.special import expit, logit

rng = np.random.default_rng(7)
N_ATOMS = 6
OMEGA = np.array(list(itertools.product([0, 1], repeat=N_ATOMS)), dtype=bool)  # 64 x 6

# ---------------- theory -----------------
sents, struct = [], []  # truth vectors over OMEGA; (op, i, j)
for k in range(N_ATOMS):
    sents.append(OMEGA[:, k].copy()); struct.append(("atom", k, None))
def lit(v, neg): return ~v if neg else v
axioms = []
while True:
    axioms = []
    for _ in range(2):
        idx = rng.choice(N_ATOMS, 3, replace=False); neg = rng.random(3) < 0.5
        axioms.append(lit(OMEGA[:, idx[0]], neg[0]) | lit(OMEGA[:, idx[1]], neg[1]) | lit(OMEGA[:, idx[2]], neg[2]))
    Wmask = np.logical_and.reduce(axioms)
    if 0 < Wmask.sum() < 64: break
for ax in axioms:
    sents.append(ax); struct.append(("axiom", None, None))
ops = ["and", "or", "not", "imp"]
while len(sents) < N_ATOMS + 2 + 16:
    op = ops[rng.integers(4)]
    i, j = rng.choice(len(sents), 2, replace=False)
    if op == "and": v = sents[i] & sents[j]
    elif op == "or": v = sents[i] | sents[j]
    elif op == "not": v = ~sents[i]
    else: v = (~sents[i]) | sents[j]
    vw = v[Wmask]
    if vw.all() or (~vw).all():  # skip sentences decided by the axioms (they are fixed, not queried)
        continue
    if any((v == s).all() for s in sents): continue
    sents.append(v); struct.append((op, i, j))
Q = len(sents)
T = np.array(sents).T.astype(float)  # 64 x Q truth table
A = T[Wmask]                          # |W| x Q (coherent set C = conv rows of A)
nW = A.shape[0]
fixed = np.array([s[0] == "axiom" for s in struct])  # price 1 forced
free = ~fixed

# relation constraints, rows (a, b, kind) meaning a.p <= b ('le') or a.p == b ('eq'), valid on all of W
cands = []
def row(coeffs, b, kind):
    a = np.zeros(Q)
    for idx, c in coeffs: a[idx] += c
    return (a, b, kind)
for s, (op, i, j) in enumerate(struct):
    if op == "and":
        cands += [row([(s, 1), (i, -1)], 0, "le"), row([(s, 1), (j, -1)], 0, "le"), row([(i, 1), (j, 1), (s, -1)], 1, "le")]
    elif op == "or":
        cands += [row([(i, 1), (s, -1)], 0, "le"), row([(j, 1), (s, -1)], 0, "le"), row([(s, 1), (i, -1), (j, -1)], 0, "le")]
    elif op == "not":
        cands += [row([(s, 1), (i, 1)], 1, "eq")]
    elif op == "imp":
        cands += [row([(i, -1), (s, -1)], -1, "le"), row([(j, 1), (s, -1)], 0, "le"), row([(s, 1), (i, 1), (j, -1)], 1, "le")]
# axiom-forced relations: entailments under Gamma that are not tautologies: P(i) <= P(j)
for i in range(Q):
    for j in range(Q):
        if i != j and free[i] and free[j]:
            if (A[:, i] <= A[:, j]).all() and not (T[:, i] <= T[:, j]).all():
                cands.append(row([(i, 1), (j, -1)], 0, "le"))
# drop rows that become trivial after fixing axioms, then dedupe, then pick 30
def reduce_row(r):
    a, b, kind = r
    b2 = b - a[fixed].sum()  # fixed prices are 1
    a2 = a.copy(); a2[fixed] = 0
    return a2, b2, kind
red = []
for r in cands:
    a, b, kind = reduce_row(r)
    if not np.any(a): continue
    # trivial if satisfied for every p in [0,1]^Q
    if kind == "le" and np.clip(a, 0, None).sum() <= b + 1e-12: continue
    if any(np.allclose(a, x[0]) and abs(b - x[1]) < 1e-12 for x in red): continue
    red.append((a, b, kind))
assert len(red) >= 30, len(red)
pick = rng.choice(len(red), 30, replace=False)
R = [red[k] for k in sorted(pick)]
Rm = np.array([r[0] for r in R]); Rb = np.array([r[1] for r in R]); Req = np.array([r[2] == "eq" for r in R])
# sanity: relations hold on every admissible world
assert all((A @ a <= b + 1e-12).all() if k == "le" else np.allclose(A @ a, b) for a, b, k in R)

q0 = np.where(fixed, 1.0, rng.uniform(0.05, 0.95, Q))  # incoherent initial prices (claims)
beta = np.ones(Q)

info = dict(Q=Q, n_worlds_admissible=int(nW), n_free=int(free.sum()), n_relation_candidates=len(red),
            n_eq_relations=int(Req.sum()), n_relations=len(R))

# ---------------- objectives -----------------
def bkl(p, q):  # Bernoulli KL, elementwise, with 0log0=0
    p = np.clip(p, 0, 1); out = np.zeros_like(p)
    m = p > 0; out[m] += p[m] * np.log(p[m] / q[m])
    m = p < 1; out[m] += (1 - p[m]) * np.log((1 - p[m]) / (1 - q[m]))
    return out
def inc(p, q=q0):   # PDG inconsistency of claims q at price p (axioms at infinite confidence enforced by p in C)
    return float((beta * bkl(p, q))[free].sum())
def grad_world(mu, q=q0):  # gradient of Inc(A^T mu) wrt mu
    m = np.clip(A.T @ mu, 1e-300, 1 - 1e-16)
    g = np.where(free, beta * (np.log(m / q) - np.log((1 - m) / (1 - q))), 0.0)
    return A @ g  # constant part dropped (cancels on simplex)

def simplex_proj(v):
    u = np.sort(v)[::-1]; css = np.cumsum(u); k = np.arange(1, len(v) + 1)
    rho = np.nonzero(u * k > (css - 1))[0][-1]; th = (css[rho] - 1) / (rho + 1)
    return np.maximum(v - th, 0)

# ---- ground truth: KL projection onto C (long EG + SLSQP polish), Euclidean projection onto C ----
def eg(q, iters, eta, mu=None, record=None):
    mu = np.full(nW, 1 / nW) if mu is None else mu.copy()
    for t in range(iters):
        if record is not None: record.append(A.T @ mu)
        g = grad_world(mu, q); z = np.log(mu) - eta * g; z -= z.max(); mu = np.exp(z); mu /= mu.sum()
    return mu
L_rel = beta[free].sum()
mu_star = eg(q0, 200000, 1.0 / L_rel * 4)
p_kl_C = A.T @ mu_star
def fista_euc(q, iters=60000):
    Lc = np.linalg.norm(A, 2) ** 2; mu = np.full(nW, 1 / nW); y = mu.copy(); t = 1
    for _ in range(iters):
        g = A @ ((A.T @ y - q) * free); mn = simplex_proj(y - g / Lc)
        tn = (1 + np.sqrt(1 + 4 * t * t)) / 2; y = mn + (t - 1) / tn * (mn - mu); mu, t = mn, tn
    return mu
p_euc_C = A.T @ fista_euc(q0)
# KKT/optimality check for the KL projection: variational inequality <grad, a_w - p*> >= 0 for all worlds
gstar = np.where(free, logit(np.clip(p_kl_C, 1e-15, 1 - 1e-15)) - logit(q0), 0)
vi_kl = float((A @ gstar - gstar @ p_kl_C).min())
gE = (p_euc_C - q0) * free
vi_euc = float((A @ gE - gE @ p_euc_C).min())

# ---- projections onto the relaxation L (30 relations + box), via SLSQP ----
def proj_L(q, kind):
    x0 = np.clip(q[free], 0.01, 0.99)
    Af, bf = Rm[:, free], Rb
    cons = [{"type": "ineq", "fun": lambda x, i=i: bf[i] - Af[i] @ x, "jac": lambda x, i=i: -Af[i]} if not Req[i]
            else {"type": "eq", "fun": lambda x, i=i: bf[i] - Af[i] @ x, "jac": lambda x, i=i: -Af[i]} for i in range(len(bf))]
    qf = q[free]
    if kind == "euc":
        f = lambda x: 0.5 * ((x - qf) ** 2).sum(); g = lambda x: x - qf; bnds = [(0, 1)] * len(x0)
    else:
        f = lambda x: float(bkl(x, qf).sum()); g = lambda x: logit(np.clip(x, 1e-12, 1 - 1e-12)) - logit(qf); bnds = [(1e-9, 1 - 1e-9)] * len(x0)
    r = minimize(f, x0, jac=g, constraints=cons, bounds=bnds, method="SLSQP", options=dict(ftol=1e-15, maxiter=2000))
    p = q.copy(); p[free] = r.x; return p
p_euc_L = proj_L(q0, "euc"); p_kl_L = proj_L(q0, "kl")
_R, _Rm, _Rb, _Req = R, Rm, Rb, Req
Rm = np.array([r[0] for r in red]); Rb = np.array([r[1] for r in red]); Req = np.array([r[2] == "eq" for r in red])
p_euc_Lall = proj_L(q0, "euc")
Rm, Rb, Req = _Rm, _Rb, _Req

# ---------------- (a) anchored projected gradient over worlds ----------------
def run_a(iters, eta):
    mu = np.full(nW, 1 / nW); P = []
    for t in range(iters):
        P.append(A.T @ mu); mu = simplex_proj(mu - eta * grad_world(mu))
    return np.array(P)
# step: inc is smooth on the region visited; use a conservative step from the Hessian at mu_star
Hs = A[:, free] @ np.diag(1 / (p_kl_C[free] * (1 - p_kl_C[free]))) @ A[:, free].T
eta_a = 1.0 / np.linalg.eigvalsh(Hs).max()
Pa = run_a(4000, eta_a)
# ---------------- (b) EG / damped simultaneous IPF over worlds ----------------
Pb_theory, Pb_fast = [], []
eg(q0, 4000, 1.0 / L_rel, record=Pb_theory)
eg(q0, 4000, 4.0 / L_rel, record=Pb_fast)
Pb_theory, Pb_fast = np.array(Pb_theory), np.array(Pb_fast)
# EG == damped simultaneous IPF: check one step identity
mu0 = np.full(nW, 1 / nW); m0 = A.T @ mu0; eta = 1.0 / L_rel
ipf = mu0 * np.prod([np.where(A[:, i] > 0, (q0[i] / m0[i]) ** (eta * beta[i]), ((1 - q0[i]) / (1 - m0[i])) ** (eta * beta[i]))
                     for i in range(Q) if free[i]], axis=0)
ipf /= ipf.sum(); egstep = eg(q0, 1, eta)
ipf_identity_err = float(np.abs(ipf - egstep).max())
# implicit bias: EG from uniform converges to max-entropy mu among minimisers
Hmu = lambda mu: float(-(mu[mu > 0] * np.log(mu[mu > 0])).sum())
# max-entropy mu with moments p_kl_C via the convex dual: min_theta logsumexp(A theta) - theta.p
from scipy.special import logsumexp
Af_ = A[:, free]; pf_ = p_kl_C[free]
def dual(th):
    z = Af_ @ th; lse = logsumexp(z); w = np.exp(z - lse); return lse - th @ pf_, Af_.T @ w - pf_
r = minimize(dual, np.zeros(free.sum()), jac=True, method="L-BFGS-B", options=dict(maxiter=20000, gtol=1e-12, ftol=1e-16))
z = Af_ @ r.x; maxent_mu = np.exp(z - logsumexp(z)); maxent_moment_err = float(np.abs(Af_.T @ maxent_mu - pf_).max())
maxent_gap = float(np.abs(maxent_mu - mu_star).max()); maxent_H = (Hmu(np.clip(maxent_mu, 0, 1)), Hmu(mu_star))

# ---------------- (c) row-action methods on the 30 relations ----------------
# KKT: grad F(p) = grad F(q) - Rm^T lam, lam_j>=0 for 'le'. Dual coordinate ascent = Hildreth (euc) / Bregman row action (kl)
def rows_euc():  # 30 relations + box rows (p<=1, -p<=0) on free coords
    M = [Rm]; b = [Rb]; eq = [Req]
    I = np.eye(Q)[free]
    M += [I, -I]; b += [np.ones(free.sum()), np.zeros(free.sum())]; eq += [np.zeros(free.sum(), bool)] * 2
    return np.vstack(M), np.concatenate(b), np.concatenate(eq)
def run_c(kind, sweeps, parallel=False):
    M, b, eq = rows_euc() if kind == "euc" else (Rm, Rb, Req)
    lam = np.zeros(len(b)); P = []; V = []
    if kind == "euc":
        p = q0.copy()
        if parallel:
            tau = 1.0 / np.linalg.norm(M[:, free], 2) ** 2
        for s in range(sweeps):
            P.append(p.copy()); V.append(viol(p))
            if parallel:
                lam_new = lam + tau * (M @ p - b); lam_new = np.where(eq, lam_new, np.maximum(lam_new, 0))
                p = p - M.T @ (lam_new - lam); lam = lam_new
            else:
                for j in range(len(b)):
                    a = M[j]; d = (a @ p - b[j]) / (a @ a)
                    if not eq[j]: d = max(d, -lam[j])
                    lam[j] += d; p = p - d * a
        return np.array(P), np.array(V)
    th = np.where(free, logit(np.clip(q0, 1e-15, 1 - 1e-15)), 0.0)
    def price(th): return np.where(free, expit(th), 1.0)
    for s in range(sweeps):
        p = price(th); P.append(p); V.append(viol(p))
        if parallel:
            # damped simultaneous version: each row computes its own exact step, average with weight 1/k (k = max rows per coord)
            k = int((M != 0).sum(0).max()); dth = np.zeros(Q); dl = np.zeros(len(b))
            for j in range(len(b)):
                d = kl_step(th, M[j], b[j], lam[j], eq[j]); dl[j] = d / k; dth -= d / k * M[j] / beta
            th = th + dth; lam = lam + dl
        else:
            for j in range(len(b)):
                d = kl_step(th, M[j], b[j], lam[j], eq[j]); lam[j] += d; th = th - d * M[j] / beta
    return np.array(P), np.array(V)
def kl_step(th, a, b, lam, eq):
    g = lambda d: a @ np.where(free, expit(th - d * a / beta), 1.0) - b  # decreasing in d
    lo = -np.inf if eq else -lam
    if not eq and g(lo) <= 0: return lo
    # bracket root
    a_, b_ = (lo if np.isfinite(lo) else -1.0), 1.0
    while g(a_) < 0: a_ = a_ * 2 - 1
    while g(b_) > 0: b_ = b_ * 2 + 1
    return brentq(g, a_, b_, xtol=1e-14)
def viol(p):
    r = Rm @ p - Rb; return float(np.max(np.where(Req, np.abs(r), np.maximum(r, 0))))
Pc_euc, Vc_euc = run_c("euc", 400)
Pc_kl, Vc_kl = run_c("kl", 400)
Pc_eucpar, Vc_eucpar = run_c("euc", 20000, parallel=True)
Pc_klpar, Vc_klpar = run_c("kl", 3000, parallel=True)

# ---------------- (a2) self-referential dynamics on q ----------------
def kl_proj_C(q, mu=None, iters=3000):
    mu = eg(q, iters, 4.0 / L_rel, mu); return A.T @ mu, mu
# mirror (dual) flow: theta <- theta + eta (logit Pi(q) - theta): projection invariant along path
qm = q0.copy(); mu_w = None; mirror_drift = []
for t in range(15):
    pp, mu_w = kl_proj_C(qm, mu_w, 3000 if t == 0 else 800); mirror_drift.append(float(np.abs(pp - p_kl_C).max()))
    th = logit(np.clip(qm, 1e-15, 1 - 1e-15)); thn = th + 0.5 * (logit(np.clip(pp, 1e-15, 1 - 1e-15)) - th)
    qm = np.where(free, expit(thn), 1.0)
# primal Euclidean gradient flow of d(q)=min_p KL(p||q): dq/dt = H(q)(p*(q)-q), H = beta/(q(1-q))
qp = q0.copy(); mu_w = None; h = 0.01
for t in range(400):
    pp, mu_w = kl_proj_C(qp, mu_w, 3000 if t == 0 else 300)
    step = np.zeros(Q); step[free] = (beta[free] / (qp[free] * (1 - qp[free]))) * (pp[free] - qp[free])
    qp = qp + h * step / max(1.0, np.abs(step).max() * h / 0.02); qp[free] = np.clip(qp[free], 1e-6, 1 - 1e-6)
pp_final, _ = kl_proj_C(qp, mu_w, 20000)
primal_endpoint_gap = float(np.abs(pp_final - p_kl_C).max()); primal_dist_to_C = float(np.abs(pp_final - qp).max())

# ---------------- (d) extrapolation: contraction bound and a clock counterexample ----------------
# linear learned update e_{t+1}=M e_t with ||M^20|| small => ||M^500|| <= ||M^20||^25
Mlin = rng.normal(size=(Q, Q)); Mlin = 0.7 * Mlin / np.abs(np.linalg.eigvals(Mlin)).max()
lin = dict(norm20=float(np.linalg.norm(np.linalg.matrix_power(Mlin, 20), 2)), norm500=float(np.linalg.norm(np.linalg.matrix_power(Mlin, 500), 2)))
# clock counterexample: state (p, c). Exact at block 20, diverges after.
def clock_run(Tn):
    p = q0.copy(); c = 0.0; out = []
    for t in range(Tn):
        k = 1.0 if 19 <= c < 20 else 0.0           # jump to the target on the 20th block (relu-implementable bump)
        drift = 0.05 if c >= 20 else 0.0            # afterwards: wander away
        p = p + k * (p_kl_C - p) + drift * (0.5 - p_kl_C) * free; p = np.clip(p, 0, 1); c += 1
        out.append(float(np.abs(p - p_kl_C).max()))
    return out
clk = clock_run(500)

M_, b_, eq_ = rows_euc(); tau0 = 1.0 / np.linalg.norm(M_[:, free], 2) ** 2
qs = [np.where(fixed, 1.0, rng.uniform(0.05, 0.95, Q)) for _ in range(8)]
targets = [proj_L(qq, "euc") for qq in qs]
def hb(qq, tau, mom, Tn):
    lam = np.zeros(len(b_)); prev = lam.copy(); p = qq.copy(); errs_ = {}
    for t in range(1, Tn + 1):
        y = lam + mom * (lam - prev); ln = y + tau * tau0 * (M_ @ (qq - M_.T @ y) - b_); ln = np.where(eq_, ln, np.maximum(ln, 0))
        prev, lam = lam, ln
        if t in (20, 100, 500): errs_[t] = float(np.abs((qq - M_.T @ lam) - targ).max())
    return errs_
grid = []
for tau in [0.5, 1, 1.5, 1.9, 2.0, 2.2, 2.6, 3.0]:
    for mom in [0, 0.5, 0.8, 0.9, 0.95, 0.99]:
        e = {20: 0, 100: 0, 500: 0}
        for qq, targ in zip(qs, targets):
            r_ = hb(qq, tau, mom, 500)
            for k in e: e[k] = max(e[k], r_[k])
        grid.append((tau, mom, e[20], e[100], e[500]))
best20 = min(grid, key=lambda g: g[2]); best500 = min(grid, key=lambda g: g[4])
learned_opt = dict(rows_tau_ge_2=[g for g in grid if g[0] >= 2.0 and g[1] in (0, 0.8)], best_at_20=best20, best_at_500=best500, note="(tau multiple of 1/||M||^2, momentum, maxerr@20, @100, @500)")
# ---------------- reporting -----------------
def first_below(err, tol=1e-3):
    idx = np.nonzero(np.asarray(err) < tol)[0]
    # require it stays below afterwards
    for i in idx:
        if (np.asarray(err)[i:] < tol).all(): return int(i)
    return None
def errs(P, target): return np.abs(P - target).max(1)
res = {}
res["(a) projected grad on Inc, worlds"] = dict(target="Pi_KL_C", it_err=first_below(errs(Pa, p_kl_C)),
    it_incgap=first_below([inc(p) - inc(p_kl_C) for p in Pa]), eta=eta_a)
res["(b) EG eta=1/L (theory)"] = dict(target="Pi_KL_C", it_err=first_below(errs(Pb_theory, p_kl_C)), it_incgap=first_below([inc(p) - inc(p_kl_C) for p in Pb_theory]))
res["(b) EG eta=4/L"] = dict(target="Pi_KL_C", it_err=first_below(errs(Pb_fast, p_kl_C)), it_incgap=first_below([inc(p) - inc(p_kl_C) for p in Pb_fast]))
res["(c) Hildreth cyclic, Euclidean (sweeps)"] = dict(target="Pi_E_L", it_err=first_below(errs(Pc_euc, p_euc_L)), it_viol=first_below(Vc_euc))
res["(c) Bregman row-action cyclic, KL (sweeps)"] = dict(target="Pi_KL_L", it_err=first_below(errs(Pc_kl, p_kl_L)), it_viol=first_below(Vc_kl))
res["(c) parallel Hildreth = 1 ReLU layer/iter"] = dict(target="Pi_E_L", it_err=first_below(errs(Pc_eucpar, p_euc_L)), it_viol=first_below(Vc_eucpar))
res["(c) parallel KL row-action (averaged)"] = dict(target="Pi_KL_L", it_err=first_below(errs(Pc_klpar, p_kl_L)), it_viol=first_below(Vc_klpar))
def rate(err):
    err = np.asarray(err); m = (err > 1e-11) & (err < 1e-2)
    if m.sum() < 5: return None
    x = np.nonzero(m)[0]; return float(np.exp(np.polyfit(x, np.log(err[m]), 1)[0]))
for k, P, tgt in [("(a) projected grad on Inc, worlds", Pa, p_kl_C), ("(b) EG eta=4/L", Pb_fast, p_kl_C), ("(b) EG eta=1/L (theory)", Pb_theory, p_kl_C),
                  ("(c) Hildreth cyclic, Euclidean (sweeps)", Pc_euc, p_euc_L), ("(c) Bregman row-action cyclic, KL (sweeps)", Pc_kl, p_kl_L),
                  ("(c) parallel Hildreth = 1 ReLU layer/iter", Pc_eucpar, p_euc_L), ("(c) parallel KL row-action (averaged)", Pc_klpar, p_kl_L)]:
    res[k]["linear_rate_per_iter"] = rate(errs(P, tgt))
# theory bound check for EG eta=1/L: Inc gap <= L log|W| / T
gaps = np.array([inc(p) - inc(p_kl_C) for p in Pb_theory]); Ts = np.arange(1, len(gaps))
bound_ok = bool((gaps[1:] <= L_rel * np.log(nW) / Ts + 1e-12).all())
# plain cyclic projections (no dual correction) end at a point of L that is NOT the projection
def pocs_euc(sweeps=2000):
    M, b, eq = rows_euc(); p = q0.copy()
    for s in range(sweeps):
        for j in range(len(b)):
            a = M[j]; d = (a @ p - b[j]) / (a @ a)
            if not eq[j]: d = max(d, 0)
            p = p - d * a
    return p
p_pocs = pocs_euc()
summary = dict(info=info, results=res, vi_kl_C=vi_kl, vi_euc_C=vi_euc,
    gap_L_vs_C=dict(euc=float(np.abs(p_euc_L - p_euc_C).max()), euc_all_candidates=float(np.abs(p_euc_Lall - p_euc_C).max()), kl=float(np.abs(p_kl_L - p_kl_C).max()),
                    dist_q0_C_euc=float(np.abs(q0 - p_euc_C).max()), viol_C_proj_in_L=viol(p_kl_C),
                    max_C_violation_of_p_kl_L=None),
    pocs_vs_proj=dict(pocs_viol=viol(p_pocs), pocs_err_to_Pi_E_L=float(np.abs(p_pocs - p_euc_L).max())),
    eg_bound_holds=bound_ok, eg_ipf_identity_err=ipf_identity_err,
    eg_implicit_bias=dict(maxent_moment_err=maxent_moment_err, max_abs_mu_gap_to_maxent=maxent_gap, H_maxent_vs_H_EG=maxent_H),
    self_ref=dict(mirror_flow_proj_drift_max=max(mirror_drift), primal_flow_endpoint_vs_Pi_q0=primal_endpoint_gap, primal_flow_q_dist_to_C=primal_dist_to_C),
    extrapolation=dict(learned_step_toy=learned_opt, linear=lin, clock_err_at=dict(t20=clk[19], t100=clk[99], t500=clk[499])),
    inc_values=dict(inc_q0_proj=inc(p_kl_C), inc_at_kl_L=inc(p_kl_L)))
# is the KL projection onto L coherent? LP feasibility of p in C via min ||A^T mu - p||
mu_chk = fista_euc(p_kl_L, 30000); summary["gap_L_vs_C"]["max_C_violation_of_p_kl_L"] = float(np.abs(A.T @ mu_chk - p_kl_L)[free].max())
print(json.dumps(summary, indent=1))
np.savez("traj.npz", Pa=Pa, Pb=Pb_fast, Pb1=Pb_theory, Pc_euc=Pc_euc, Pc_kl=Pc_kl, Pc_eucpar=Pc_eucpar, Pc_klpar=Pc_klpar)
try:
    sys.path.insert(0, "/tmp/bd_pylib"); import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].semilogy([max(inc(p) - inc(p_kl_C), 1e-16) for p in Pa], label="(a) proj. gradient on Inc (worlds)")
    ax[0].semilogy([max(g, 1e-16) for g in gaps], label="(b) EG eta=1/L")
    ax[0].semilogy([max(inc(p) - inc(p_kl_C), 1e-16) for p in Pb_fast], label="(b) EG eta=4/L")
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel("PDG inconsistency gap Inc(p_t) - <<M>>"); ax[0].axhline(1e-3, ls=":", c="gray"); ax[0].legend(); ax[0].set_xlim(0, 2000)
    ax[1].loglog(np.arange(1, len(np.maximum(Vc_euc, 1e-16)) + 1), np.maximum(Vc_euc, 1e-16), label="(c) Hildreth cyclic Euclid (per sweep)")
    ax[1].loglog(np.arange(1, len(np.maximum(Vc_kl, 1e-16)) + 1), np.maximum(Vc_kl, 1e-16), label="(c) Bregman KL cyclic (per sweep)")
    ax[1].loglog(np.arange(1, len(np.maximum(Vc_eucpar[:3000], 1e-16)) + 1), np.maximum(Vc_eucpar[:3000], 1e-16), label="(c) parallel Hildreth (per layer)")
    ax[1].loglog(np.arange(1, len(np.maximum(Vc_klpar[:3000], 1e-16)) + 1), np.maximum(Vc_klpar[:3000], 1e-16), label="(c) parallel KL (per layer)")
    ax[1].set_xlabel("sweep / layer"); ax[1].set_ylabel("max relation violation"); ax[1].axhline(1e-3, ls=":", c="gray"); ax[1].legend(); ax[1].set_ylim(1e-16, 2)
    plt.tight_layout(); plt.savefig("inconsistency.png", dpi=110); print("plot saved")
except Exception as e:
    print("no plot:", e)
