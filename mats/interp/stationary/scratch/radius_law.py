"""PA1/PA2: the exact scalar radius chain of FORMAL item 3 against long chains of evaluation nets (Model S, sigma = 1).
usage: radius_law.py n seeds(comma) paths layers --out file
Scalar chain: n X' = (sqrt(g n X) + eta)^2 + C, g ~ (2/n) chi2_{Bin(n,1/2)}, eta ~ N(0,1), C ~ chi2_{n-1}."""
import argparse, json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
import st

ap = argparse.ArgumentParser(); ap.add_argument('n', type=int); ap.add_argument('seeds'); ap.add_argument('paths', type=int)
ap.add_argument('layers', type=int); ap.add_argument('--out')
a = ap.parse_args(); n = a.n
t0 = time.time()
QS = (0.1, 0.5, 0.9, 0.99)
# scalar chain
rng = np.random.default_rng([123, n])
M = 4096; burn = 30 * n; rec = max(1000, 4_000_000 // M)
X = np.zeros(M); samp = []
for t in range(burn + rec):
    N = rng.binomial(n, 0.5, M)
    g = (2.0 / n) * rng.chisquare(np.maximum(N, 1e-9)) * (N > 0)
    eta = rng.standard_normal(M); C = rng.chisquare(n - 1, M)
    X = ((np.sqrt(g * n * X) + eta) ** 2 + C) / n
    if t >= burn:
        samp.append(X.copy())
S = np.concatenate(samp)
chain_q = [float(np.quantile(S, q)) for q in QS]
# Dufresne approximation
duf_q = [0.4 * n / math.log(1 / q) for q in QS]
# nets
netX = []
per_net = {}
for seed in [int(s) for s in a.seeds.split(',')]:
    r2 = np.random.default_rng([2, seed, n])
    h = np.zeros((a.paths, n)); xs = []
    for t in range(a.layers):
        W = st.W_layer(seed, n, -a.layers + t)
        h = np.maximum(h @ W.T, 0.0) + r2.standard_normal((a.paths, n))
        if t >= 20 * n:
            xs.append(np.sum(h * h, 1) / n)
    xs = np.concatenate(xs); netX.append(xs)
    per_net[seed] = [float(np.quantile(xs, q)) for q in QS]
NX = np.concatenate(netX)
net_q = [float(np.quantile(NX, q)) for q in QS]
x = np.sort(NX)[::-1]; k = max(10, len(x) // 100)
hill = float(1.0 / np.mean(np.log(x[:k] / x[k])))
xs_ = np.sort(S)[::-1]; ks = len(xs_) // 100
hill_chain = float(1.0 / np.mean(np.log(xs_[:ks] / xs_[ks])))
res = dict(n=n, seeds=a.seeds, paths=a.paths, layers=a.layers, quantiles=QS, chain_q=chain_q, net_q=net_q,
           dufresne_q=duf_q, ratio_net_chain=[b / c for b, c in zip(net_q, chain_q)], per_net=per_net,
           chain_median_over_n=chain_q[1] / n, dufresne_median_over_n=2 / (5 * math.log(2)), hill_net_1pct=hill,
           hill_chain_1pct=hill_chain, chain_samples=len(S), net_samples=len(NX), sec=time.time() - t0)
open(a.out, 'w').write(json.dumps(res))
print(json.dumps(res), flush=True)
