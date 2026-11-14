"""Skeptic check 3: parameter counting is a property of the model class, not of the room.
Class K2 = {order-0 Bernoulli(theta)} U {noisy XOR chain x_t = x_{t-1} ^ x_{t-2} ^ b, b ~ Bernoulli(eps)},
uniform prior over the two models, KT (Beta(1/2,1/2)) on theta and on eps. In K2 the structured room S
has ONE parameter (eps), like the noise room N (theta). The asymptotic advantage 'S has k_eff=4 > 1'
seen with the Markov-order mixture disappears."""
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
    p = np.clip(p, 1e-300, 1-1e-16); return -(p*np.log2(p) + (1-p)*np.log2(1-p))
def ebh(a, b):
    s = a+b; return (digamma(s+1) - a/s*digamma(a+1) - b/s*digamma(b+1))/LN2

def run_k2(x):
    c0 = np.zeros(2); cb = np.zeros(2); logml = np.zeros(2); eig = np.empty(len(x)); p1, p2 = 0, 0
    for t, v in enumerate(x):
        v = int(v); pred = p1 ^ p2
        q0 = (c0[1]+.5)/(c0.sum()+1); qb = (cb[1]+.5)/(cb.sum()+1); q1 = qb if pred == 0 else 1-qb
        q = np.array([q0, q1]); post = np.exp2(logml-logml.max()); post /= post.sum()
        eh = np.array([ebh(c0[1]+.5, c0[0]+.5), ebh(cb[1]+.5, cb[0]+.5)])
        eig[t] = h2(np.array(post @ q)) - post @ eh
        logml += np.log2(np.where(v == 1, q, 1-q)); c0[v] += 1; cb[v ^ pred] += 1; p2, p1 = p1, v
    return eig

print("== 3. Class K2 = {Bernoulli} U {XOR chain with flip rate}: agent-side expected info gain C2 (cum bits)")
cps = [256, 1024, 8192]
for room in ("D", "N", "S"):
    es = [run_k2(gen(room, T, np.random.default_rng(500+s))) for s in range(1 if room == "D" else SEEDS)]
    e = np.mean(es, 0); c = np.cumsum(e); rate = (c[-1]-c[-513])/512*T
    print(f"  {room}  C2 " + " ".join(f"{c[t-1]:7.2f}" for t in cps) + f"   rate x t {rate:.2f}  (k=1 interior -> {1/(2*LN2):.2f})")
print("  -> in K2 noise is fit by BOTH models (theta=1/2 and eps=1/2), so the agent keeps learning two parameters and\n     an unresolved model choice; N stays above S at every checkpoint and its rate x t is higher: the order flips vs the Markov mixture")
