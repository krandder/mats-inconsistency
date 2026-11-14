"""What does EG with eta=c/L do on an extreme-claim instance where c=4 fails? (case 0 of skeptic S5)"""
import numpy as np, warnings
from scipy.special import logit
warnings.simplefilter("ignore")
src = open("../dynamics.py").read(); G = {}
exec(compile(src[:src.index("# ---------------- (a) anchored projected gradient")], "dyn", "exec"), G)
A, free, fixed, Q, nW = G["A"], G["free"], G["fixed"], G["Q"], G["nW"]
rng = np.random.default_rng(1)
for _ in range(2000):
    rng.uniform(0.1, 3, Q); rng.dirichlet(np.ones(nW)*0.3); rng.dirichlet(np.ones(nW)*0.3)
qq = np.where(fixed, 1.0, rng.choice([0.002, 0.998], Q) * (rng.random(Q) < 0.5) + rng.uniform(0.002, 0.998, Q) * (rng.random(Q) >= 0.5))
qq = np.where(fixed, 1.0, np.clip(qq, 0.002, 0.998)); L = free.sum()
def inc(p):
    p = np.clip(p[free], 1e-300, 1-1e-16); q = qq[free]; return float((p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))).sum())
def run(c, iters=30000):
    mu = np.full(nW, 1/nW); P = []
    for _ in range(iters):
        m = np.clip(A.T @ mu, 1e-300, 1-1e-16); P.append(m)
        g = A @ np.where(free, logit(m) - logit(qq), 0.0)
        z = np.log(np.maximum(mu, 1e-300)) - c/L*g; z -= z.max(); mu = np.exp(z); mu /= mu.sum()
    return np.array(P)
ref = run(1, 300000)[-1]
for c in [1, 1.5, 2, 2.5, 3, 4]:
    P = run(c); last = P[-6:]
    print("c=%.1f err=%.2e Inc_end=%.4f Inc*=%.4f  |p_T - p_{T-1}|=%.1e |p_T - p_{T-2}|=%.1e" % (c, np.abs(P[-1]-ref).max(), inc(P[-1]), inc(ref),
          np.abs(P[-1]-P[-2]).max(), np.abs(P[-1]-P[-3]).max()))
