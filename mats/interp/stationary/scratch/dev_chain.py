"""DEV ONLY (dev seeds, before PREREG): long chains of Model S (sigma = 1) to see the radius law over layers.
Tracks X_l = |h_l|^2 / n for a few paths, the per-layer log-gain g_l = |relu(W_l h_{l-1})|^2 / |h_{l-1}|^2, and
reports drift/variance of log g, quantiles of X over layers, and a Hill tail index of X over layers."""
import sys, os, math, time, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
import st

n = int(sys.argv[1]); T = int(sys.argv[2]); P = int(sys.argv[3]); seed = int(sys.argv[4])
rng = np.random.default_rng([2, seed, n])
h = np.zeros((P, n))
X = np.empty((T, P)); LG = np.empty((T, P)); act = np.empty((T, P))
t0 = time.time()
for t in range(T):
    W = st.W_layer(seed, n, -T + t)
    z = h @ W.T
    a = np.maximum(z, 0.0)
    nh = np.sum(h * h, 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        LG[t] = np.log(np.sum(a * a, 1) / nh)
    act[t] = np.mean(z > 0, 1)
    h = a + rng.standard_normal(a.shape)
    X[t] = np.sum(h * h, 1) / n
burn = 20 * n
Xs = X[burn:]; lg = LG[burn:]
def hill(x, k):
    x = np.sort(x.ravel())[::-1]
    return 1.0 / np.mean(np.log(x[:k] / x[k]))
res = dict(n=n, T=T, P=P, seed=seed, logg_mean=float(np.mean(lg)), logg_var=float(np.var(lg)),
           pred_mean=-2.5 / n, pred_var=5.0 / n,
           X_q=[float(np.quantile(Xs, q)) for q in (0.1, 0.5, 0.9, 0.99, 0.999)],
           X_mean=float(np.mean(Xs)), dufresne_median=0.4 * n / math.log(2),
           hill_1pct=float(hill(Xs, max(10, Xs.size // 100))), hill_01pct=float(hill(Xs, max(10, Xs.size // 1000))),
           act_mean=float(np.mean(act[burn:])),
           acf_logX=[float(np.corrcoef(np.log(Xs[:-k, 0]), np.log(Xs[k:, 0]))[0, 1]) for k in (1, n // 4, n, 4 * n)],
           sec=time.time() - t0)
print(json.dumps(res), flush=True)
