"""Forward-chain scorecard by family (best member of each family fitted to the real p_t; a capacity test).

Families for h_t, all with Gaussian residuals:
  G    one Gaussian;
  AC   one Gaussian per algorithmic cell (a_t, b_t);
  CL   per cell, linear in named variables u = (M, S, x_last, relu(M), relu(S), relu(x_last));
  CL3  CL plus the 3rd max T and relu(T) and its position one-hot (T is the next value the task may need);
  CLX  per cell, linear in all inputs seen so far and their ReLUs (the input-exact limit of a linear-Gaussian family).
Per step and family: R^2 of h_t; 'unexplained bits' = 1/2 log2 det(I + Sigma_res / (sigma^2 diag r_t^2)) averaged over
cells (the information a description must add to reach the noise precision, Gaussian-channel formula); parameter count.
At t = n the family's predicted error rate: sample u from its exact law (fresh inputs), h ~ N(mean(u), Sigma_res),
error = argmax W_oh h != b.  The deductive ADF (adf.py) numbers are reported separately.
usage: python fam.py MODEL.npz OUT.json
"""
import sys, json, time
import numpy as np
import az, alg

t0 = time.time()
path, out = sys.argv[1], sys.argv[2]
W, r, sigma = az.load_npz(path)
W_hi, W_hh, W_oh = W; d = W_hi.shape[0]; n = W_oh.shape[0]
rng = np.random.default_rng(5150)
N = 2 ** 18
x = rng.standard_normal((N, n)); hs, _, lo = alg.sample_chain(W, r, sigma, x, rng)
A = alg.alg_states(x)
xe = rng.standard_normal((N, n)); hse, _, loe = alg.sample_chain(W, r, sigma, xe, rng); Ae = alg.alg_states(xe)


def feats(kind, t, xx, AA):
    B = xx.shape[0]; one = np.ones((B, 1))
    if kind in ('G', 'AC'): return one
    M = AA['M'][:, t]; S = np.where(np.isfinite(AA['S'][:, t]), AA['S'][:, t], 0.0); xl = xx[:, t - 1]
    F = [one, M[:, None], S[:, None], xl[:, None], np.maximum(M, 0)[:, None], np.maximum(S, 0)[:, None], np.maximum(xl, 0)[:, None]]
    if kind == 'CL3':
        srt = np.sort(xx[:, :t], 1)
        T = srt[:, -3] if t >= 3 else np.zeros(B)
        pos = np.argsort(xx[:, :t], 1)[:, -3] if t >= 3 else np.zeros(B, int)
        F += [T[:, None], np.maximum(T, 0)[:, None], np.eye(n)[pos][:, :t]]
    if kind == 'CLX':
        F = [one, xx[:, :t], np.maximum(xx[:, :t], 0)]
    return np.concatenate(F, 1)


def cell_key(kind, t, AA):
    if kind == 'G': return np.zeros(AA['a'].shape[0], int)
    return (AA['a'][:, t] + 1) * 11 + (AA['b'][:, t] + 1)


res = dict(model=path, sigma=sigma, N=N, steps={}, error_pred={})
noise_var = (sigma * r) ** 2
for kind in ('G', 'AC', 'CL', 'CL3', 'CLX'):
    rows = []
    for t in range(1, n + 1):
        H = hs[:, t]; key = cell_key(kind, t, A); F = feats(kind, t, x, A)
        resid = np.zeros_like(H); bits = 0.0; npar = 0; models = {}
        for k in np.unique(key):
            s = key == k
            beta, *_ = np.linalg.lstsq(F[s], H[s], rcond=None)
            R_ = H[s] - F[s] @ beta; resid[s] = R_
            C = np.cov(R_.T) if s.sum() > d + 1 else np.diag(R_.var(0) + 1e-12)
            sd = np.sqrt(noise_var[t - 1])
            Cn = C / np.outer(sd, sd) + np.eye(d)
            sign, ld = np.linalg.slogdet(Cn); bits += s.mean() * 0.5 * ld / np.log(2)
            npar += beta.size + d * (d + 1) // 2
            models[k] = (beta, C)
        tot = ((H - H.mean(0)) ** 2).sum(0)
        r2 = 1 - (resid ** 2).sum(0) / tot
        rows.append(dict(t=t, R2_mean=float(r2.mean()), R2_min=float(r2.min()), R2=r2.tolist(), unexplained_bits=float(bits), params=int(npar)))
        if t == n:
            # predicted error rate from the family: fresh inputs, h ~ N(F beta, C)
            key_e = cell_key(kind, t, Ae); Fe = feats(kind, t, xe, Ae); errs = np.zeros(N, bool); ok = np.ones(N, bool)
            for k in np.unique(key_e):
                s = key_e == k
                if k not in models: ok[s] = False; continue
                beta, C = models[k]
                L = np.linalg.cholesky(C + 1e-9 * np.eye(d))
                hh = Fe[s] @ beta + rng.standard_normal((s.sum(), d)) @ L.T
                errs[s] = (hh @ W_oh.T).argmax(1) != Ae['b'][s, n]
            res['error_pred'][kind] = dict(p_family=float(errs[ok].mean()), p_net=float((loe.argmax(1) != Ae['b'][:, n]).mean()))
    res['steps'][kind] = rows
    print(kind, [(rr['t'], round(rr['R2_mean'], 3), round(rr['unexplained_bits'], 1)) for rr in rows], res['error_pred'].get(kind), flush=True)
res['seconds'] = time.time() - t0
json.dump(res, open(out, 'w'), indent=1)
