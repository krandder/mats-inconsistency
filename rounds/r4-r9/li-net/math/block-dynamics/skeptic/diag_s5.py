"""Diagnose S5: is projected gradient (step 1/lambda_max H(p*)) slow, oscillating, or is the EG reference wrong?"""
import numpy as np, warnings
from scipy.special import logit
warnings.simplefilter("ignore")
src = open("../dynamics.py").read(); G = {}
exec(compile(src[:src.index("# ---------------- (a) anchored projected gradient")], "dyn", "exec"), G)
A, free, fixed, Q, nW, sp = G["A"], G["free"], G["fixed"], G["Q"], G["nW"], G["simplex_proj"]
def inc(p, q):
    p = p[free]; q = q[free]; p = np.clip(p, 1e-300, 1-1e-16)
    return float((p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))).sum())
def vi(p, q):
    g = np.where(free, logit(np.clip(p, 1e-15, 1-1e-15)) - logit(np.clip(q,1e-15,1-1e-15)), 0); return float((A @ g - g @ p).min())
def eg(q, iters, eta):
    mu = np.full(nW, 1/nW)
    for _ in range(iters):
        m = np.clip(A.T @ mu, 1e-300, 1-1e-16); g = A @ np.where(free, logit(m) - logit(q), 0.0)
        z = np.log(np.maximum(mu, 1e-300)) - eta*g; z -= z.max(); mu = np.exp(z); mu /= mu.sum()
    return mu
rng = np.random.default_rng(1)
# replay the S5 rng stream: S4 consumed draws first, so regenerate identically
for _ in range(2000):
    rng.uniform(0.1, 3, Q); rng.dirichlet(np.ones(nW)*0.3); rng.dirichlet(np.ones(nW)*0.3)
for k in range(20):
    qq = np.where(fixed, 1.0, rng.choice([0.002, 0.998], Q) * (rng.random(Q) < 0.5) + rng.uniform(0.002, 0.998, Q) * (rng.random(Q) >= 0.5))
    qq = np.where(fixed, 1.0, np.clip(qq, 0.002, 0.998))
    if k not in (0, 1, 7, 8, 13): continue
    ps = A.T @ eg(qq, 30000, 4.0/free.sum()); ps2 = A.T @ eg(qq, 300000, 1.0/free.sum())
    H = A[:, free] @ np.diag(1/(ps[free]*(1-ps[free]))) @ A[:, free].T; eta = 1/np.linalg.eigvalsh(H).max()
    mu = np.full(nW, 1/nW); incs = []
    for t in range(20000):
        m = A.T @ mu; incs.append(inc(m, qq))
        mu = sp(mu - eta*(A @ np.where(free, logit(np.clip(m,1e-300,1-1e-16)) - logit(qq), 0.0)))
    incs = np.array(incs); nonmono = int((np.diff(incs) > 1e-12).sum())
    print(k, "eta=%.2e" % eta, "EG30k vs EG300k(1/L) %.1e" % np.abs(ps-ps2).max(), "VI(ps)=%.1e" % vi(ps, qq),
          "Inc ps=%.4f pg_end=%.4f pg@0=%.4f" % (inc(ps, qq), incs[-1], incs[0]), "nonmonotone steps", nonmono,
          "min p*(1-p*) %.1e" % (ps[free]*(1-ps[free])).min())
