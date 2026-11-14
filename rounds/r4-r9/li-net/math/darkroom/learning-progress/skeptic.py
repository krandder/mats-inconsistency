"""Skeptic checks for learning-progress.md (2026-10-03).

1. Hidden parameter: the per-context Beta(alpha, alpha) prior. toy.py fixes alpha = 1/2 (KT / Jeffreys).
   The dark room is a BOUNDARY parameter (theta = 0); Bayesian regret there is alpha*log2 t, while an interior
   parameter (noise, the noisy XOR chain) costs 1/2*log2 t per parameter whatever alpha is. So the dark room's
   learning progress is set by the prior's edge behaviour. Sweep alpha and see whether the ordering D vs N vs S flips.
2. Fairness of the 'learned room S* is below the dark room' claim: toy.py compares a pre-trained S* with a FRESH D.
   Compare S* with D* pre-trained for the same N0.
Learner: Bayesian mixture over Markov orders 0..4, uniform order prior (D=4 is a setting where toy/sens report S>N).
"""
import numpy as np
from math import log2
from scipy.special import digamma
LN2 = np.log(2.0); EPS = 0.05; DMAX = 4

def gen(room, n, rng):
    if room == "D": return np.zeros(n, dtype=np.int8)
    if room == "N": return rng.integers(0, 2, n).astype(np.int8)
    x = np.zeros(n, dtype=np.int8); x[:2] = rng.integers(0, 2, 2); f = rng.random(n) < EPS
    for t in range(2, n): x[t] = (x[t-1] ^ x[t-2]) ^ f[t]
    return x

def true_lp(room, x):
    if room == "D": return np.zeros(len(x))
    if room == "N": return np.full(len(x), -1.0)
    lp = np.full(len(x), -1.0); pred = x[1:-1] ^ x[:-2]
    lp[2:] = np.where(x[2:] == pred, log2(1-EPS), log2(EPS)); return lp

def h2(p):
    p = np.clip(p, 1e-300, 1-1e-16); return -(p*np.log2(p) + (1-p)*np.log2(1-p))

def run(x, alpha, skip=0):
    ks = np.arange(DMAX+1); masks = (1 << ks) - 1
    c = np.zeros((DMAX+1, 2**DMAX, 2)); logml = np.zeros(DMAX+1); hist = 0
    loss = np.empty(len(x)); eig = np.empty(len(x))
    for t, v in enumerate(x):
        v = int(v); ctx = hist & masks
        n0 = c[ks, ctx, 0]; n1 = c[ks, ctx, 1]; a, b = n1+alpha, n0+alpha; s = a+b; q = a/s
        post = np.exp2(logml-logml.max()); post /= post.sum(); qm = float(post @ q)
        eh = (digamma(s+1) - a/s*digamma(a+1) - b/s*digamma(b+1))/LN2
        eig[t] = h2(np.array(qm)) - post @ eh
        loss[t] = -log2(qm if v else 1-qm)
        c[ks, ctx, v] += 1; logml += np.log2(np.where(v == 1, q, 1-q)); hist = ((hist << 1) | v) & ((1 << DMAX)-1)
    return loss[skip:], eig[skip:]

T = 8192; SEEDS = 4; cps = [256, 1024, 8192]
print(f"== 1. Prior edge parameter alpha (Beta(alpha,alpha) per context), order mixture 0..{DMAX}, T={T}, seeds={SEEDS}")
print("   C3 = realized info about the truth (cum bits), C2 = agent-side expected info gain (cum bits);")
print("   rate = per-step C2 x t over the last 512 steps (-> k_eff_alpha/(2 ln2)); D's C3 should be alpha*log2 t")
for alpha in (0.05, 0.5, 1.0, 3.0):
    res = {}
    for room in ("D", "N", "S"):
        c3s, c2s = [], []
        for s in range(1 if room == "D" else SEEDS):
            x = gen(room, T, np.random.default_rng(100+s)); l, e = run(x, alpha)
            c3s.append(np.cumsum(l + true_lp(room, x))); c2s.append(np.cumsum(e))
        res[room] = (np.mean(c3s, 0), np.mean(c2s, 0))
    for room in ("D", "N", "S"):
        c3, c2 = res[room]
        rate = (c2[-1]-c2[-513]) / 512 * T
        print(f"  alpha={alpha:4.2f} {room}  C3 " + " ".join(f"{c3[t-1]:7.2f}" for t in cps)
              + "   C2 " + " ".join(f"{c2[t-1]:7.2f}" for t in cps) + f"   rate {rate:5.2f}")
    print(f"           alpha*log2(T) = {alpha*log2(T):.2f}   ordering by C3 at T: "
          + " > ".join(sorted("DNS", key=lambda r: -res[r][0][-1]))
          + "   by C2 at T: " + " > ".join(sorted("DNS", key=lambda r: -res[r][1][-1])))

print("\n== 1b. Dark room alone, longer horizon (deterministic, one run): C3 / (log2 t) -> alpha")
for alpha in (0.05, 0.5, 1.0, 3.0):
    x = np.zeros(65536, dtype=np.int8); l, e = run(x, alpha); c3 = np.cumsum(l)
    print(f"  alpha={alpha:4.2f}  C3(2^10)={c3[1023]:7.2f}  C3(2^16)={c3[-1]:7.2f}  slope per log2 t = {(c3[-1]-c3[1023])/6:.3f}")

print("\n== 2. 'Learned structure is worth less than the dark room': fair comparison, both pre-trained N0 (alpha=1/2)")
for N0 in (0, 2000, 20000):
    out = {}
    for room in ("D", "S"):
        c2s = []
        for s in range(1 if room == "D" else SEEDS):
            x = gen(room, N0+T, np.random.default_rng(300+s)); _, e = run(x, 0.5, skip=N0); c2s.append(e.sum())
        out[room] = np.mean(c2s)
    print(f"  N0={N0:6d}: info over next {T} steps  D {out['D']:6.3f}   S {out['S']:6.3f}   S/D = {out['S']/out['D']:.2f}")
