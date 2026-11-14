"""Meeting in the middle with the deductive description (adf.py) on either side, as a diagnostic.

Pr[E] = E_{p_m}[v_m] for every m (v_m = Pr[error | state at m]). With a forward description q_m and a backward
description psi_m:  Pr[E] - <q_m, psi_m> = <p_m - q_m, v_m> + <q_m, v_m - psi_m>.
 (a) forward error at m:  <q_m, v_m> - Pr[E], q_m = the ADF mixture after m steps; v_m by real continuations from
     states sampled from q_m (h, M, S, cell);
 (b) backward error at m: <p_m, psi_m> - Pr[E], psi_m(z) = the ADF run from the point state z (zero covariance) to the
     end; z sampled from the real network at step m;
 (c) both: <q_m, psi_m> (ADF forward to m, then ADF from points sampled from q_m).
usage: python meet.py MODEL.npz OUT.json
"""
import sys, json, time
import numpy as np
import az, adf, alg

t0 = time.time()
path, out = sys.argv[1], sys.argv[2]
W, r, sigma = az.load_npz(path)
W_hi, W_hh, W_oh = W; d = W_hi.shape[0]; n = W_oh.shape[0]
rng = np.random.default_rng(606)
NT = 2 ** 20
xt = rng.standard_normal((NT, n)); _, _, lot = alg.sample_chain(W, r, sigma, xt, rng)
p_true = float((lot.argmax(1) != np.argsort(xt, -1)[:, -2]).mean())


def continue_from(t, h, a, b, M, S):
    B = h.shape[0]
    for s in range(t, n):
        xs = rng.standard_normal(B)
        h = np.maximum(h @ W_hh.T + xs[:, None] * W_hi, 0) + sigma * r[s] * rng.standard_normal((B, d))
        up1 = xs > M; up2 = (~up1) & (xs > S)
        nb = np.where(up1, a, np.where(up2, s, b)); na = np.where(up1, s, a)
        nS = np.where(up1, M, np.where(up2, xs, S)); nM = np.where(up1, xs, M)
        a, b, M, S = na, nb, nM, nS
    return (h @ W_oh.T).argmax(1) != b


def sample_q(comps, S_):
    keys = list(comps.keys()); wts = np.array([comps[k][0][0] for k in keys]); wts = wts / wts.sum()
    idx = rng.choice(len(keys), size=S_, p=wts)
    H = np.zeros((S_, d)); a = np.zeros(S_, int); b = np.zeros(S_, int); M = np.zeros(S_); Sv = np.zeros(S_)
    for c in np.unique(idx):
        sel = idx == c; k = keys[c]; _, mu, Sig = comps[k][0]
        L = np.linalg.cholesky(Sig + 1e-9 * np.eye(len(mu)))
        v = mu + rng.standard_normal((sel.sum(), len(mu))) @ L.T
        H[sel] = v[:, :d]; M[sel] = v[:, d]; Sv[sel] = v[:, d + 1] if k[1] >= 0 else -np.inf; a[sel] = k[0]; b[sel] = k[1]
    return H, a, b, M, Sv


def psi_adf(t, h, a, b, M, S, k_split=2):
    A = adf.ADF(W, r, sigma, k_split=k_split)
    D = d + 2; mu = np.concatenate([h, [M, S if np.isfinite(S) else 0.0]])
    comps = {(int(a), int(b), ()): [(1.0, mu, np.zeros((D, D)))]}
    for s in range(t, n):
        comps = A.step(comps, s)
    e, _ = A.error_rate(comps, 'seq')
    return e, A.fl.n


res = dict(model=path, sigma=sigma, p_true=p_true, rows=[])
Afw = adf.ADF(W, r, sigma, k_split=2)
comps = {(-1, -1, ()): [(1.0, np.zeros(d + 2), np.zeros((d + 2, d + 2)))]}
xs_real = rng.standard_normal((4096, n)); hs_real, _, _ = alg.sample_chain(W, r, sigma, xs_real, rng); Ar = alg.alg_states(xs_real)
for m in range(1, n):
    comps = Afw.step(comps, m - 1)
    H, a, b, M, Sv = sample_q(comps, 2 ** 16)
    fwd = float(continue_from(m, H, a, b, M, Sv).mean())
    row = dict(m=m, forward_q_v=fwd, forward_err=fwd - p_true, adf_flops_to_m=Afw.fl.n)
    if m >= 4:
        NB = 256 if m >= 6 else 128
        ps = []; fl = []
        for i in range(NB):
            e, f_ = psi_adf(m, hs_real[i, m], Ar['a'][i, m], Ar['b'][i, m], Ar['M'][i, m], Ar['S'][i, m]); ps.append(e); fl.append(f_)
        row['backward_p_psi'] = float(np.mean(ps)); row['backward_err'] = float(np.mean(ps) - p_true)
        row['backward_err_se'] = float(np.std(ps) / np.sqrt(NB)); row['psi_flops_per_point'] = float(np.mean(fl))
        # both sides: points from q_m
        ps2 = []
        for i in range(NB):
            e, _ = psi_adf(m, H[i], a[i], b[i], M[i], Sv[i]); ps2.append(e)
        row['both_q_psi'] = float(np.mean(ps2)); row['both_err'] = float(np.mean(ps2) - p_true); row['both_err_se'] = float(np.std(ps2) / np.sqrt(NB))
    res['rows'].append(row); print(json.dumps(row), flush=True)
    json.dump(res, open(out, 'w'), indent=1)
res['seconds'] = time.time() - t0
json.dump(res, open(out, 'w'), indent=1)
