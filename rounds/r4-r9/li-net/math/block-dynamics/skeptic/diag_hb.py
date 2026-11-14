"""Error curve of heavy-ball parallel Hildreth at (tau=2.8/||M||^2, mom=0.3): small at 20, explodes later."""
import numpy as np, warnings
warnings.simplefilter("ignore")
src = open("../dynamics.py").read(); G = {}
exec(compile(src[:src.index("# ---------------- (a) anchored projected gradient")], "dyn", "exec"), G)
free, fixed, Q, Rm, Rb, Req = G["free"], G["fixed"], G["Q"], G["Rm"], G["Rb"], G["Req"]
I_ = np.eye(Q)[free]; M_ = np.vstack([Rm, I_, -I_]); b_ = np.r_[Rb, np.ones(free.sum()), np.zeros(free.sum())]; eq_ = np.r_[Req, np.zeros(2*free.sum(), bool)]
tau0 = 1/np.linalg.norm(M_[:, free], 2)**2
rng = np.random.default_rng(5); qs = [np.where(fixed, 1.0, rng.uniform(0.05, 0.95, Q)) for _ in range(200)]
tg = [G["proj_L"](q, "euc") for q in qs]
def run1(q, t_, tau, mom):
    lam = np.zeros(len(b_)); prev = lam.copy(); e = {}
    for t in range(1, 501):
        y = lam + mom*(lam-prev); ln = y + tau*tau0*(M_ @ (q - M_.T @ y) - b_); ln = np.where(eq_, ln, np.maximum(ln, 0)); prev, lam = lam, ln
        if t in (20, 500): e[t] = np.abs(q - M_.T @ lam - t_).max()
    return e
for tau, mom in [(2.8, 0.3), (2.0, 0.8), (1.9, 0.0)]:
    E = np.zeros(501)
    for q, t_ in zip(qs, tg):
        lam = np.zeros(len(b_)); prev = lam.copy()
        for t in range(1, 501):
            y = lam + mom*(lam-prev); ln = y + tau*tau0*(M_ @ (q - M_.T @ y) - b_); ln = np.where(eq_, ln, np.maximum(ln, 0)); prev, lam = lam, ln
            E[t] = max(E[t], np.abs(q - M_.T @ lam - t_).max())
    print("n_inputs=%d" % len(qs), "exploding@500 (err>1):", int(sum(1 for q, t_ in zip(qs, tg) if run1(q, t_, tau, mom)[500] > 1)), "min/max err@20 among them: %s" % (lambda v: "%.1e/%.1e" % (min(v), max(v)) if v else "-")([run1(q, t_, tau, mom)[20] for q, t_ in zip(qs, tg) if run1(q, t_, tau, mom)[500] > 1]))
    print("tau=%.1f mom=%.1f maxerr @10 %.1e @20 %.1e @40 %.1e @100 %.1e @500 %.1e" % (tau, mom, E[10], E[20], E[40], E[100], E[500]))
