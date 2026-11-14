"""Backward functions v_t(z) = Pr[error | state at step t = z], state z = (h_t, algorithmic state), by nested simulation
(K continuations with fresh future inputs and noise). Reports:
 - the information the state carries about the eventual error, I(E; Z_t) = H(E) - E H_b(v_t), per t;
 - the part carried by the algorithmic state alone (bins of (a, b, M, S)): I(E; A_t) <= I(E; Z_t);
 - the step at which each error is decided (v_t first exceeds 0.5 and stays) and the algorithmic event at that step.
 - the meeting identity Pr[E] = E_{p_t}[v_t] at every t (a consistency check).
usage: python backward.py MODEL.npz OUT.json [NS_LOG2] [K]
"""
import sys, json, time, math
import numpy as np
import az, alg

t0 = time.time()
path, out = sys.argv[1], sys.argv[2]
NS = 2 ** int(sys.argv[3]) if len(sys.argv) > 3 else 2 ** 13
K = int(sys.argv[4]) if len(sys.argv) > 4 else 64
W, r, sigma = az.load_npz(path)
W_hi, W_hh, W_oh = W; d = W_hi.shape[0]; n = W_oh.shape[0]
rng = np.random.default_rng(99)
x = rng.standard_normal((NS, n))
hs, zs, lo = alg.sample_chain(W, r, sigma, x, rng)
A = alg.alg_states(x)
y = A['b'][:, n]; err = lo.argmax(1) != y


def continue_from(t, h, a, b, M, S, K, rng):
    """K continuations from states at step t (inputs x_0..x_{t-1} seen). Returns error frequency per state."""
    B = h.shape[0]
    h = np.repeat(h, K, 0); a = np.repeat(a, K); b = np.repeat(b, K); M = np.repeat(M, K); S = np.repeat(S, K)
    for s in range(t, n):
        xs = rng.standard_normal(B * K)
        z = h @ W_hh.T + xs[:, None] * W_hi
        h = np.maximum(z, 0) + sigma * r[s] * rng.standard_normal((B * K, d))
        up1 = xs > M; up2 = (~up1) & (xs > S)
        nb = np.where(up1, a, np.where(up2, s, b)); na = np.where(up1, s, a)
        nS = np.where(up1, M, np.where(up2, xs, S)); nM = np.where(up1, xs, M)
        a, b, M, S = na, nb, nM, nS
    e = (h @ W_oh.T).argmax(1) != b
    return e.reshape(B, K).mean(1)


def Hb(p):
    p = np.clip(p, 1e-12, 1 - 1e-12); return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


V = np.zeros((NS, n + 1)); V[:, n] = err.astype(float)
for t in range(1, n):
    for i in range(0, NS, 1024):
        sl = slice(i, i + 1024)
        V[sl, t] = continue_from(t, hs[sl, t], A['a'][sl, t], A['b'][sl, t], A['M'][sl, t], A['S'][sl, t], K, rng)
V[:, 0] = err.mean()
p = err.mean()
res = dict(model=path, sigma=sigma, NS=NS, K=K, p=float(p), H_E=float(Hb(p)))
# Information about the eventual error E carried at step t. v_hat_t comes from independent continuations, so binning
# states by v_hat_t (a function of Z_t) and measuring the plug-in MI with the realised error E of the original run gives
# a lower bound on I(E; Z_t) (data processing) with small plug-in bias (~ bins / (2 N ln 2) bits).
def mi_binned(e, v, nb=20):
    qs = np.unique(np.quantile(v, np.linspace(0, 1, nb + 1)[1:-1]))
    bins = np.digitize(v, qs) + 2 * (v <= 0) * 0
    bins = np.where(v <= 0, -1, np.where(v >= 1, nb + 1, bins))
    I = 0.0; pe = e.mean()
    for bb in np.unique(bins):
        s = bins == bb; pb = s.mean(); q = e[s].mean()
        for val, pv in ((1, q), (0, 1 - q)):
            pj = pb * pv; pm = pe if val == 1 else 1 - pe
            if pj > 0: I += pj * np.log2(pj / (pb * pm))
    return float(I)


info = []; info_alg = []; meet = []
ef = err.astype(float)
for t in range(n + 1):
    meet.append(float(V[:, t].mean()))
    if t == 0: info.append(0.0); info_alg.append(0.0); continue
    info.append(mi_binned(ef, V[:, t]) if t < n else float(Hb(p)))
    Mq = np.digitize(A['M'][:, t], np.quantile(A['M'][:, t], [0.25, 0.5, 0.75]))
    Sv = np.where(np.isfinite(A['S'][:, t]), A['S'][:, t], -9.0)
    Sq = np.digitize(Sv, np.quantile(Sv, [0.25, 0.5, 0.75]))
    key = ((A['a'][:, t] + 1) * 11 + (A['b'][:, t] + 1)) * 16 + Mq * 4 + Sq
    # cross-fitted: bin averages of v from one half, MI scored on the other half (and vice versa); at t = n, v is the
    # realised error itself, so in-sample averaging would overfit (review 1, finding 1)
    half = np.arange(NS) % 2 == 0
    Ivals = []
    for fit, sc in ((half, ~half), (~half, half)):
        vb = np.full(NS, np.nan); glob = V[fit, t].mean()
        for kk in np.unique(key):
            s_fit = (key == kk) & fit; s_sc = (key == kk) & sc
            vb[s_sc] = V[s_fit, t].mean() if s_fit.any() else glob
        Ivals.append(mi_binned(ef[sc], vb[sc]))
    info_alg.append(float(np.mean(Ivals)))
# shuffle control: MI of E with a random permutation of v (plug-in bias level)
res['I_bias_control'] = mi_binned(ef, rng.permutation(V[:, n // 2]))
res['I_E_state'] = info; res['I_E_alg'] = info_alg; res['meet_E_v'] = meet
# when is each error decided: first t after which v_t >= 0.5 at all later steps (for erroneous runs)
dec = np.full(NS, -1)
for i in np.where(err)[0]:
    above = V[i, 1:] >= 0.5
    for t in range(1, n + 1):
        if above[t - 1:].all(): dec[i] = t; break
ev = np.zeros((NS, n + 1), int)   # event at step t (input x_{t-1}): 1 new max, 2 new 2nd max, 0 none
for t in range(1, n + 1):
    ev[:, t] = np.where(A['a'][:, t] == t - 1, 1, np.where(A['b'][:, t] == t - 1, 2, 0))
dd = dec[err]
res['decided_step_hist'] = np.bincount(dd[dd >= 1], minlength=n + 1).tolist()
evd = np.array([ev[i, dec[i]] for i in np.where(err)[0] if dec[i] >= 1])
res['decided_event_hist'] = dict(none=int((evd == 0).sum()), new_max=int((evd == 1).sum()), new_2nd=int((evd == 2).sum()))
base_ev = ev[:, 1:].ravel()
res['event_base_rates'] = dict(none=float((base_ev == 0).mean()), new_max=float((base_ev == 1).mean()), new_2nd=float((base_ev == 2).mean()))
# step-matched expectation (review 1): at step t (input x_{t-1}, t >= 2) P(new max) = P(new 2nd max) = 1/t
hist = res['decided_step_hist']; tot = sum(hist)
exp_none = sum(hist[t] * (1 - 2 / t) for t in range(2, n + 1)) / max(tot, 1)
exp_max = sum(hist[t] * (1 / t) for t in range(2, n + 1)) / max(tot, 1)
res['event_rates_matched_to_decision_steps'] = dict(none=exp_none, new_max=exp_max, new_2nd=exp_max)
res['seconds'] = time.time() - t0
json.dump(res, open(out, 'w'), indent=1)
print(json.dumps(res, indent=0)[:3000])
