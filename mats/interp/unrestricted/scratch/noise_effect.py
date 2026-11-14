"""M5: noise effect at sigma (default 1e-3): nMSE of E[a_L noisy] - E[a_L noiseless], paired MC with 2^m Gaussian
inputs. Usage: noise_effect.py n L seed [--sigma 1e-3] [--m 20]. Appends to runs/noise_effect.json."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from nets import Net
ap = argparse.ArgumentParser(); ap.add_argument('n', type=int); ap.add_argument('L', type=int); ap.add_argument('seed', type=int)
ap.add_argument('--sigma', type=float, default=1e-3); ap.add_argument('--m', type=int, default=20)
a = ap.parse_args()
net = Net(a.seed, a.n, a.L, sigma=a.sigma)
rng = np.random.default_rng([555, a.seed, a.n, a.L])
N = 1 << a.m
s0 = np.zeros(a.n); s1 = np.zeros(a.n); sd = np.zeros(a.n); sd2 = np.zeros(a.n); q0 = np.zeros(a.n)
for c in range(0, N, 1 << 15):
    X = rng.standard_normal((1 << 15, a.n))
    h0, h1 = net.forward_pair(X, rng)
    d = h1 - h0
    s0 += h0.sum(0); q0 += (h0 * h0).sum(0); sd += d.sum(0); sd2 += (d * d).sum(0)
V = float(np.sum(q0 / N - (s0 / N) ** 2))
D = sd / N
se2 = float(np.sum(sd2 / N - D * D) / N)
fn = os.path.join(os.path.dirname(__file__), '..', 'runs', 'noise_effect.json')
res = json.load(open(fn)) if os.path.exists(fn) else []
res.append(dict(n=a.n, L=a.L, seed=a.seed, sigma=a.sigma, V=V, effect_nmse=float(np.sum(D * D) / V),
                se2_nmse=se2 / V))
json.dump(res, open(fn, 'w'))
print(f'n={a.n} L={a.L} seed={a.seed} sigma={a.sigma}: noise effect nMSE {np.sum(D * D) / V:.2e} (SE^2 {se2 / V:.1e})', flush=True)
