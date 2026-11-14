"""Sensitivity of the agent-side expected information gain (Bayesian order mixture) to the hidden
choices: maximum order D and the prior over orders. Question: does the agent prefer S over N, and from
which horizon on? Rooms D, N, S as in toy.py."""
import numpy as np
from math import log2
from scipy.special import digamma
LN2 = np.log(2.0); EPS = 0.05; T = 8192; SEEDS = 8

def gen(room, n, rng):
    if room == "D": return np.zeros(n, dtype=np.int8)
    if room == "N": return rng.integers(0, 2, n).astype(np.int8)
    x = np.zeros(n, dtype=np.int8); x[:2] = rng.integers(0, 2, 2); f = rng.random(n) < EPS
    for t in range(2, n): x[t] = (x[t-1] ^ x[t-2]) ^ f[t]
    return x

def h2(p):
    p = np.clip(p, 1e-300, 1 - 1e-16); return -(p*np.log2(p) + (1-p)*np.log2(1-p))

def eig_run(x, D, logprior):
    ks = np.arange(D+1); masks = (1 << ks) - 1
    c = np.zeros((D+1, 2**D, 2)); logml = logprior.copy(); hist = 0; out = np.empty(len(x))
    for t, v in enumerate(x):
        v = int(v); ctx = hist & masks
        n0 = c[ks, ctx, 0]; n1 = c[ks, ctx, 1]; q = (n1+.5)/(n0+n1+1)
        post = np.exp2(logml - logml.max()); post /= post.sum()
        a, b = n1+.5, n0+.5; s = a+b
        eh = (digamma(s+1) - a/s*digamma(a+1) - b/s*digamma(b+1))/LN2
        out[t] = h2(np.array(post @ q)) - post @ eh
        c[ks, ctx, v] += 1; logml += np.log2(np.where(v == 1, q, 1-q))
        hist = ((hist << 1) | v) & ((1 << D) - 1)
    return out

priors = {
    "uniform":     lambda D: np.zeros(D+1),
    "2^-(k+1)":    lambda D: -(np.arange(D+1)+1.0),
    "2^-(2^k)":    lambda D: -(2.0**np.arange(D+1)),
}
print(f"cumulative expected information gain (bits) at t=T={T}; rate = per-step gain x t over the last 512 steps")
print("crossover = horizon where S overtakes N, extrapolating cum(t) = cum(T) + rate*log2(t/T)*ln2 (inf if never)")
for pname, pf in priors.items():
    for D in (2, 4, 8, 12):
        r = {}
        for room in ("D", "N", "S"):
            runs = [eig_run(gen(room, T, np.random.default_rng(31*s+room.__hash__() % 97)), D, pf(D)) for s in range(SEEDS)]
            m = np.mean(runs, axis=0)
            r[room] = (m.sum(), m[-512:].mean()*T)
        cN, rN = r["N"]; cS, rS = r["S"]
        if cS >= cN: cross = "already S>N"
        elif rS > rN: cross = f"t ~ 2^{np.log2(T) + (cN-cS)/((rS-rN)*np.log(2)):.1f}"
        else: cross = "never"
        print(f"  prior={pname:9s} D={D:2d}  cum D {r['D'][0]:6.2f}  N {cN:7.2f}  S {cS:6.2f}   "
              f"rate D {r['D'][1]:.2f} N {rN:.2f} S {rS:.2f}   S beats N: {cross}")
