"""Part 2 (PREREG Part 2), noise regime: Hermite shares of the stationary-lane estimand in the RQMC coordinates.
Setting exactly as work/stationary/scratch/partB.py (rqmc mode): Model S, sigma = 1, weights st.W_layer(seed, n, l) for
l = -B+1..1, burn-in B = 8n from h = 0, a = relu(W_1 h_0); RQMC there put scrambled Sobol' on the noise of the most
recent d = min(B, 21201 // n) layers (age 0 first) and MC on older layers.
usage: part2_noise.py n seed [--P 16384] [--out file]
Measures (two independent halves, unbiased inner products):
  W1_win   : Hermite degree-1 share over the d-layer window coordinates  sum_{j in win} sum_c E[a_c xi_j]^2 / V
  W1_all   : the same over the whole history (all B layers)
  U2,U3,U4 : univariate degree-2..4 shares over the window: sum_j sum_c E[a_c He_k(xi_j)]^2 / k! / V
  S1_win   : W1_win + U2 + U3 + U4 (first-order Sobol share of the window coordinates, truncated at degree 4)
  s_even   : Var((a(xi) + a(-xi))/2) / V with the whole history negated (the stationary lane's antithetic pair)
  V        : tr Var(a) from the sample; the per-age profile of the degree-1 share.
Noise of sample batch b at layer l is drawn from rng [7201, seed, n, b, l + 10^7] so it can be regenerated."""
import argparse, json, math, sys, time
import numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/stationary/lib')
import st

ap = argparse.ArgumentParser(); ap.add_argument('n', type=int); ap.add_argument('seed', type=int)
ap.add_argument('--P', type=int, default=16384); ap.add_argument('--out')
a = ap.parse_args()
n, seed, P = a.n, a.seed, a.P
t0 = time.time()
B = 8 * n
dwin = min(B, 21201 // n)
Wc = st.WCache(seed, n, max_bytes=1.5e9)
layers = list(range(-B + 1, 1))            # age = -l
CHK = 1024
nb = P // CHK
half_of = lambda b: 0 if b < nb // 2 else 1


def noise(b, l, sign=1.0):
    return sign * np.random.default_rng([7201, seed, n, b, l + st.OFF]).standard_normal((CHK, n))


def fwd(b, sign=1.0):
    h = np.zeros((CHK, n))
    for l in layers:
        h = np.maximum(h @ Wc(l).T, 0.0) + noise(b, l, sign)
    return np.maximum(h @ Wc(1).T, 0.0)


Aout = np.concatenate([fwd(b) for b in range(nb)])
Aneg = np.concatenate([fwd(b, -1.0) for b in range(nb)])
print(f'forward done {time.time() - t0:.0f}s', flush=True)
V = float(np.sum(Aout.var(0, ddof=1)))
Ev = 0.5 * (Aout + Aneg)
Vpair = float(np.sum(np.concatenate([Aout, Aneg]).var(0, ddof=1)))
s_even = float(np.sum(Ev.var(0, ddof=1))) / Vpair
hP = (nb // 2) * CHK
means = [Aout[:hP].mean(0), Aout[hP:2 * hP].mean(0)]
# accumulators: degree 1 for every layer; degrees 2-4 for window layers
G1 = np.zeros((2, B, n, n))
GU = np.zeros((2, 3, dwin, n, n))
for b in range(nb):
    hf = half_of(b)
    Ac = Aout[b * CHK:(b + 1) * CHK] - means[hf]
    for l in layers:
        age = -l
        xi = noise(b, l)
        G1[hf, age] += Ac.T @ xi
        if age < dwin:
            x2 = xi * xi
            GU[hf, 0, age] += Ac.T @ (x2 - 1)
            GU[hf, 1, age] += Ac.T @ (x2 * xi - 3 * xi)
            GU[hf, 2, age] += Ac.T @ (x2 * x2 - 6 * x2 + 3)
    if b % 4 == 3:
        print(f'batch {b + 1}/{nb} {time.time() - t0:.0f}s', flush=True)
G1 /= hP; GU /= hP
prof = np.sum(G1[0] * G1[1], (1, 2)) / V          # degree-1 share per age
W1_win = float(prof[:dwin].sum()); W1_all = float(prof.sum())
U = [float(np.sum(GU[0, k] * GU[1, k])) / math.factorial(k + 2) / V for k in range(3)]
res = dict(n=n, seed=seed, P=P, B=B, d_layers=dwin, V=V, W1_win=W1_win, W1_all=W1_all, U2=U[0], U3=U[1], U4=U[2],
           S1_win=W1_win + sum(U), s_even=s_even, anti_pair_ratio=2 * s_even, W1_profile=prof.tolist(),
           sec=time.time() - t0)
print({k: v for k, v in res.items() if k != 'W1_profile'}, flush=True)
if a.out:
    json.dump(res, open(a.out, 'w'))
