"""DEV ONLY (dev seeds 1-9, before PREREG): calibrate the basic scales of Model S's stationary regime.
For each n: P paths from h = 0 at layer -B+1 to 0, sigma = 1 (Model S is scale-free; checked separately).
Prints per-net: median and mean of |h_0|^2/(n sigma^2), tau_det/s ratio, flip fraction under resampling the last
layer's noise, and the coupling distance of two starts under common noise and weights."""
import sys, os, math, time, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
import st

n = int(sys.argv[1]); B = int(sys.argv[2]); P = int(sys.argv[3]); seeds = [int(x) for x in sys.argv[4].split(',')]
sigma = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
out = []
for seed in seeds:
    t0 = time.time()
    Wf = st.WCache(seed, n, max_bytes=50e6)
    rng = np.random.default_rng([1, seed, n])
    # two starts under common noise: start A = 0, start B = a large random point; same noise stream via same rng state
    hA = np.zeros((P, n)); hB = rng.standard_normal((P, n)) * sigma * math.sqrt(n) * 3
    dist = []
    for l in range(-B + 1, 1):
        W = Wf(l)
        xi = rng.standard_normal((P, n)) * sigma
        aA = np.maximum(hA @ W.T, 0.0); aB = np.maximum(hB @ W.T, 0.0)
        hA_prev_a = aA
        hA = aA + xi; hB = aB + xi
        if (l + B - 1) % max(1, B // 16) == 0 or l == 0:
            dist.append((l, float(np.median(np.linalg.norm(hA - hB, axis=1) / np.linalg.norm(hA, axis=1)))))
    r2 = np.sum(hA * hA, 1) / (n * sigma ** 2)
    # last layer's noise resampling: z0 from a_{-1} + xi vs a_{-1} + xi'
    Wn = Wf(1)
    a_prev = hA_prev_a
    z1 = (a_prev + sigma * rng.standard_normal(a_prev.shape)) @ Wn.T
    z2 = (a_prev + sigma * rng.standard_normal(a_prev.shape)) @ Wn.T
    flip = float(np.mean((z1 > 0) != (z2 > 0)))
    tau_det2 = float(np.mean((a_prev @ Wn.T) ** 2)); s2 = float(np.mean(np.sum(Wn ** 2, 1))) * sigma ** 2
    rr = tau_det2 / (tau_det2 + s2)
    pred_flip = math.acos(rr) / math.pi
    res = dict(seed=seed, n=n, B=B, P=P, sigma=sigma, r2_median=float(np.median(r2)), r2_mean=float(np.mean(r2)),
               r2_q90=float(np.quantile(r2, 0.9)), r2_q99=float(np.quantile(r2, 0.99)),
               tau_over_s=math.sqrt(tau_det2 / s2), flip=flip, pred_flip_gauss=pred_flip,
               coupling=dist[-4:], sec=time.time() - t0)
    out.append(res)
    print(json.dumps(res), flush=True)
