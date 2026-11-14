"""Skeptic checks for block-dynamics.md (2026-10-02). numpy/scipy only.
Reuses the theory/ground-truth prefix of ../dynamics.py (same seed), then runs independent checks."""
import numpy as np, warnings, json
from scipy.integrate import solve_ivp
from scipy.optimize import linprog
from scipy.special import expit, logit
warnings.simplefilter("ignore")
src = open("../dynamics.py").read()
cut = src.index("# ---------------- (a) anchored projected gradient")
G = {}; exec(compile(src[:cut], "dyn", "exec"), G)
A, free, fixed, q0, Q, nW = G["A"], G["free"], G["fixed"], G["q0"], G["Q"], G["nW"]
out = {}

# S1. R4(iii) on a 2-sentence closed-form example: C = {0<=p1<=p2<=1} (phi1 -> phi2), q0=(0.9,0.2).
def projC(q):  # KL projection onto {p1<=p2}: active => logit p = mean logit q
    if q[0] <= q[1]: return q.copy()
    p = expit(logit(q).mean()); return np.array([p, p])
q = np.array([0.9, 0.2]); P0 = projC(q)
def primal(t, q):
    q = np.clip(q, 1e-12, 1 - 1e-12); p = projC(q); return (p - q) / (q * (1 - q))
ev = lambda t, q: q[1] - q[0] - 1e-10; ev.terminal = True
sol = solve_ivp(primal, [0, 50], q, rtol=1e-11, atol=1e-13, events=ev)
qend = sol.y[:, -1]
out["S1_primal_flow_2d"] = dict(Pi_q0=P0.tolist(), endpoint=qend.tolist(), Pi_endpoint=projC(qend).tolist(),
                                drift=float(np.abs(projC(qend) - P0).max()))
# mirror flow in logit coords: d theta/dt = logit Pi(q) - theta  -> invariant
def mirror(t, th): return logit(projC(expit(th))) - th
solm = solve_ivp(mirror, [0, 30], logit(q), rtol=1e-11, atol=1e-13)
out["S1_mirror_flow_2d_drift"] = float(np.abs(projC(expit(solm.y[:, -1])) - P0).max())

# S2. R7: exact l_inf distance from Pi^KL_L(q0) to C via LP (min t s.t. |A^T mu - p|<=t on free coords)
src2 = src[:src.index("# ---------------- (a) anchored projected gradient")]
p_kl_L, p_kl_C, p_euc_L = G["p_kl_L"], G["p_kl_C"], G["p_euc_L"]
def linf_dist_to_C(p):
    Af = A[:, free].T; pf = p[free]; k = Af.shape[0]
    c = np.r_[np.zeros(nW), 1.0]
    Aub = np.block([[Af, -np.ones((k, 1))], [-Af, -np.ones((k, 1))]]); bub = np.r_[pf, -pf]
    Aeq = np.r_[np.ones(nW), 0.0][None]; r = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=[1], bounds=[(0, None)] * (nW + 1))
    return float(r.fun)
out["S2_linf_dist_to_C"] = dict(p_kl_L=linf_dist_to_C(p_kl_L), p_euc_L=linf_dist_to_C(p_euc_L), p_kl_C=linf_dist_to_C(p_kl_C))

# S3. R5(iv) "model frequency" consequence: unclaimed psi gets the max-ent conditional, not |W_psi|/|W|,
# unless the claims are themselves the uniform-world prices. Atoms a,b; Gamma empty; claim P(a)=0.9; psi=a&b.
W2 = np.array([[0,0],[0,1],[1,0],[1,1]]); mu = np.array([0.05, 0.05, 0.45, 0.45])  # maxent with P(a)=0.9
out["S3_psi_price_maxent_vs_model_freq"] = dict(maxent=float(mu[3]), model_freq=0.25)

# S4. R5(ii) relative smoothness: D_f(mu,nu) <= sum(beta) KL(mu||nu) on random pairs, random beta
rng = np.random.default_rng(1); worst = -np.inf
for _ in range(2000):
    beta = rng.uniform(0.1, 3, Q); mu1 = rng.dirichlet(np.ones(nW) * 0.3); mu2 = rng.dirichlet(np.ones(nW) * 0.3)
    m1, m2 = A.T @ mu1, A.T @ mu2; f_ = free & (m1 > 0) & (m1 < 1) & (m2 > 0) & (m2 < 1)
    bk = m1 * np.log(m1 / m2) + (1 - m1) * np.log((1 - m1) / (1 - m2))
    Df = float((beta[f_] * bk[f_]).sum()); KL = float((mu1 * np.log(mu1 / mu2)).sum())
    worst = max(worst, Df / (beta[free].sum() * KL))
out["S4_max_ratio_Df_over_LKL"] = worst

# S5. R3 region assumption: projected gradient with the Hessian-at-p* step on 20 claim vectors with extreme entries
def eg(q, iters, eta):
    mu = np.full(nW, 1 / nW)
    for _ in range(iters):
        m = np.clip(A.T @ mu, 1e-300, 1 - 1e-16)
        g = A @ np.where(free, logit(m) - logit(np.clip(q, 1e-15, 1 - 1e-15)), 0.0)
        z = np.log(np.maximum(mu, 1e-300)) - eta * g; z -= z.max(); mu = np.exp(z); mu /= mu.sum()
    return mu
sp = G["simplex_proj"]; res5 = []; eg4 = []
for k in range(20):
    qq = np.where(fixed, 1.0, rng.choice([0.002, 0.998], Q) * (rng.random(Q) < 0.5) + rng.uniform(0.002, 0.998, Q) * (rng.random(Q) >= 0.5))
    qq = np.where(fixed, 1.0, np.clip(qq, 0.002, 0.998))
    ps = A.T @ eg(qq, 300000, 1.0 / free.sum())          # proved-step EG as reference
    ps4 = A.T @ eg(qq, 30000, 4.0 / free.sum())           # the doc's fast step eta=4/L
    gq = lambda p: np.where(free, logit(np.clip(p, 1e-15, 1 - 1e-15)) - logit(qq), 0)
    vi = lambda p: float((A @ gq(p) - gq(p) @ p).min() / max(1.0, np.abs(gq(p)).max()))
    eg4.append((float(np.abs(ps4 - ps).max()), vi(ps), vi(ps4)))
    H = A[:, free] @ np.diag(1 / (ps[free] * (1 - ps[free]))) @ A[:, free].T; eta = 1 / np.linalg.eigvalsh(H).max()
    mu = np.full(nW, 1 / nW); hit = 0; it = None
    for t in range(20000):
        m = A.T @ mu
        if (m[free] < 1e-9).any() or (m[free] > 1 - 1e-9).any(): hit += 1
        if np.abs(m - ps).max() < 1e-3 and it is None: it = t
        mc = np.clip(m, 1e-300, 1 - 1e-16)
        mu = sp(mu - eta * (A @ np.where(free, logit(mc) - logit(qq), 0.0)))
    res5.append((it, hit, float(np.abs(A.T @ mu - ps).max())))
out["S5_projgrad_extreme_q"] = dict(iters_to_1e3=[r[0] for r in res5], boundary_hits=[r[1] for r in res5],
                                    final_err=[r[2] for r in res5])
out["S5b_EG_4overL_vs_1overL"] = dict(err_4L_vs_ref=[e[0] for e in eg4], VI_ref_normalised=[e[1] for e in eg4], VI_4L_normalised=[e[2] for e in eg4])

# S6. R9(v) conjecture probe: heavy-ball Hildreth, fine grid; look for small err@20 but larger err@500
Rm, Rb, Req = G["Rm"], G["Rb"], G["Req"]; I_ = np.eye(Q)[free]
M_ = np.vstack([Rm, I_, -I_]); b_ = np.r_[Rb, np.ones(free.sum()), np.zeros(free.sum())]; eq_ = np.r_[Req, np.zeros(2 * free.sum(), bool)]
tau0 = 1.0 / np.linalg.norm(M_[:, free], 2) ** 2
qs = [np.where(fixed, 1.0, rng.uniform(0.05, 0.95, Q)) for _ in range(8)]
targets = [G["proj_L"](qq, "euc") for qq in qs]
def hb(qq, targ, tau, mom):
    lam = np.zeros(len(b_)); prev = lam.copy(); e = {}
    for t in range(1, 501):
        y = lam + mom * (lam - prev); ln = y + tau * tau0 * (M_ @ (qq - M_.T @ y) - b_); ln = np.where(eq_, ln, np.maximum(ln, 0))
        prev, lam = lam, ln
        if t in (20, 500): e[t] = float(np.abs((qq - M_.T @ lam) - targ).max())
    return e
bad = []
for tau in np.linspace(1.5, 4.0, 26):
    for mom in [0, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]:
        e20 = e500 = 0
        for qq, tg in zip(qs, targets):
            r = hb(qq, tg, tau, mom); e20 = max(e20, r[20]); e500 = max(e500, r[500])
        if e20 < 2e-2 and e500 > e20: bad.append((round(float(tau), 2), mom, e20, e500))
out["S6_small_at_20_worse_at_500"] = bad
def hb_curve(qq, targ, tau, mom, Ts=(10, 20, 40, 60, 100, 200, 500)):
    lam = np.zeros(len(b_)); prev = lam.copy(); e = {}
    for t in range(1, 501):
        y = lam + mom * (lam - prev); ln = y + tau * tau0 * (M_ @ (qq - M_.T @ y) - b_); ln = np.where(eq_, ln, np.maximum(ln, 0))
        prev, lam = lam, ln
        if t in Ts: e[t] = float(np.abs((qq - M_.T @ lam) - targ).max())
    return e
out["S6_per_input_tau2.8_mom0.3"] = [hb_curve(qq, tg, 2.8, 0.3) for qq, tg in zip(qs, targets)]
out["S6_per_input_tau2.0_mom0.8_docs_pick"] = [hb_curve(qq, tg, 2.0, 0.8)[500] for qq, tg in zip(qs, targets)]
print(json.dumps(out, indent=1))
