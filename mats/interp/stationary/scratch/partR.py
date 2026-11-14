"""PR1/PR2: Model R (relative noise s_l = sigma r_l, sibling-lane convention) with ONE fixed input per net.
usage: partR.py n seed sigma --P 1024 --out file
Input x fixed (seeded), layers 1..Lmax = 16n, pilot r_l from 2^12 noisy paths of x ~ N(0, I). At L in {4n, 16n}:
the noise cloud of h_L around its mean (covariance spectrum, angular variance relative to |h|^2 measured against the
noiseless trajectory of x), the spread of log|h_L|, and births: flips under resampling one layer's noise over the 32
layers before L, against the Gaussian-unit formula with the per-path ratio."""
import argparse, json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
import st

ap = argparse.ArgumentParser(); ap.add_argument('n', type=int); ap.add_argument('seed', type=int); ap.add_argument('sigma', type=float)
ap.add_argument('--P', type=int, default=1024); ap.add_argument('--out')
a = ap.parse_args(); n, seed, sig, P = a.n, a.seed, a.sigma, a.P
t0 = time.time()
Wf = st.WCache(seed, n, max_bytes=300e6)
Lmax = 16 * n
r = st.pilot_rms(Wf, n, 1, Lmax, sig, m=1 << 12, seed=seed)
x = np.random.default_rng([31, seed, n]).standard_normal(n)
# noiseless trajectory of x
h0 = x[None, :].copy(); traj = {}
for l in range(1, Lmax + 1):
    h0 = np.maximum(h0 @ Wf(l).T, 0.0)
    traj[l] = h0[0].copy()
rng = np.random.default_rng([77, seed, n, int(round(sig * 1e6))])
h = np.repeat(x[None, :], P, 0)
out = {}
Lset = {4 * n, 16 * n}
births = {L: [] for L in Lset}
for l in range(1, Lmax + 1):
    W = Wf(l)
    z = h @ W.T; al = np.maximum(z, 0.0)
    xi = rng.standard_normal(al.shape)
    hn = al + sig * r[l] * xi
    for L in Lset:
        if L - 32 <= l < L:
            Wn = Wf(l + 1)
            xi2 = rng.standard_normal(al.shape)
            z1 = hn @ Wn.T; z2 = (al + sig * r[l] * xi2) @ Wn.T
            fl = np.sum((z1 > 0) != (z2 > 0), 1)
            tau2 = np.mean((al @ Wn.T) ** 2, 1)          # per-path deterministic pre-activation spread
            s2 = (sig * r[l]) ** 2 * np.mean(np.sum(Wn ** 2, 1))
            pred = n * np.arccos(tau2 / (tau2 + s2)) / math.pi
            births[L].append((float(fl.mean()), float(pred.mean())))
    h = hn
    if l in Lset:
        m = h.mean(0); C = np.cov(h.T)
        ev = np.sort(np.linalg.eigvalsh(C))[::-1]
        u = traj[l] / max(np.linalg.norm(traj[l]), 1e-300)
        R2 = np.sum(h * h, 1)
        proj = h @ u
        ang = 1 - proj ** 2 / R2
        kk = np.arange(1, n + 1); sel = (kk >= 2) & (kk <= max(4, n // 8)) & (ev > 0)
        slope = float(np.polyfit(np.log(kk[sel]), np.log(ev[sel]), 1)[0])
        b = np.array(births[l])
        out[str(l)] = dict(ang_var_rel_mean=float(np.mean(ang)), ang_var_rel_median=float(np.median(ang)),
                           evC_rel=(ev[:16] / np.mean(R2)).tolist(), evC_slope=slope,
                           logR_sd=float(np.std(0.5 * np.log(R2))), noiseless_norm=float(np.linalg.norm(traj[l])),
                           mean_norm=float(np.linalg.norm(m)), births_mean=float(b[:, 0].mean()),
                           births_pred=float(b[:, 1].mean()), coord_births=0.8 * n * sig)
res = dict(n=n, seed=seed, sigma=sig, P=P, Lmax=Lmax, at=out, sec=time.time() - t0)
open(a.out, 'w').write(json.dumps(res))
print(json.dumps(res)[:2000], flush=True)
