"""Fourth room P: a pseudo-random stream (LFSR, x_t = x_{t-28} XOR x_{t-31}, period 2^31-1, unknown
31-bit seed). Its true excess entropy is 31 bits (more than S's 1.43), its entropy rate is 0. Whether it is
'white noise' or 'structure' depends only on the learner's model class:
  - context-tree / order-mixture learner (D=8): never sees a repeated 31-bit state, so it predicts 1 bit/step
  - Berlekamp-Massey learner (class = linear recurrences over GF(2)): identifies it after 2*31 = 62 bits."""
import numpy as np
from math import log2
from scipy.special import digamma
LN2 = np.log(2.0); T = 8192

def lfsr(n, rng):
    x = np.zeros(n + 31, dtype=np.int8)
    x[:31] = rng.integers(0, 2, 31)
    if not x[:31].any(): x[0] = 1
    for t in range(31, n + 31): x[t] = x[t - 28] ^ x[t - 31]
    return x[31:]

def h2(p):
    p = np.clip(p, 1e-300, 1 - 1e-16); return -(p*np.log2(p) + (1-p)*np.log2(1-p))

def mix_run(x, D=8):
    ks = np.arange(D+1); masks = (1 << ks) - 1
    c = np.zeros((D+1, 2**D, 2)); logml = np.zeros(D+1); hist = 0; loss = np.empty(len(x)); eig = np.empty(len(x))
    for t, v in enumerate(x):
        v = int(v); ctx = hist & masks
        n0 = c[ks, ctx, 0]; n1 = c[ks, ctx, 1]; q = (n1+.5)/(n0+n1+1)
        post = np.exp2(logml - logml.max()); post /= post.sum(); qm = post @ q
        a, b = n1+.5, n0+.5; s = a+b
        eig[t] = h2(np.array(qm)) - post @ ((digamma(s+1) - a/s*digamma(a+1) - b/s*digamma(b+1))/LN2)
        loss[t] = -log2(qm if v else 1-qm)
        c[ks, ctx, v] += 1; logml += np.log2(np.where(v == 1, q, 1-q)); hist = ((hist << 1) | v) & ((1 << D) - 1)
    return loss, eig

def bm_errors(x):
    """Berlekamp-Massey run online; predict x_t from the current shortest LFSR (predict 0 if L=0).
    Returns the step after which all predictions are correct, and the final linear complexity."""
    n = len(x); C = np.zeros(n+1, dtype=np.int8); B = np.zeros(n+1, dtype=np.int8); C[0] = B[0] = 1
    L, m = 0, 1; last_err = -1
    for t in range(n):
        d = int(x[t])
        for i in range(1, L+1): d ^= int(C[i]) & int(x[t-i])
        if d:  # prediction (= x_t xor d) was wrong
            last_err = t
            Tm = C.copy(); C[m:] ^= B[:n+1-m]
            if 2*L <= t: L, B, m = t+1-L, Tm, 1
            else: m += 1
        else: m += 1
    return last_err + 1, L

rng = np.random.default_rng(7)
losses, eigs, bms = [], [], []
for s in range(8):
    x = lfsr(T, rng)
    l, e = mix_run(x); losses.append(l); eigs.append(e); bms.append(bm_errors(x))
l = np.mean(losses, axis=0); e = np.mean(eigs, axis=0)
print("Room P (LFSR, 31-bit unknown seed). True source: entropy rate 0, excess entropy 31 bits (S has 1.43).")
print(f"  order-mixture D=8: mean loss steps 4096..8192 = {l[4096:].mean():.4f} bits/step (noise: 1.0)")
print(f"  order-mixture D=8: cumulative expected info gain at T = {e.sum():.2f} bits (N had ~44, S ~24 in sens.py)")
print(f"  Berlekamp-Massey: last error at step {[b[0] for b in bms]} , linear complexity {[b[1] for b in bms]}")
print("  -> BM's information gain about the source = 31 bits within 62 steps, then 0 forever (a learned room).")
