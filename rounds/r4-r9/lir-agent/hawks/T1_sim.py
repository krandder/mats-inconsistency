"""T1 hawk: toy Monte Carlo of the optimizer's curse under the assumptions the policy may violate.
Pure numpy, single process, ~seconds. Output: T1_sim.json next to this file."""
import json, os, numpy as np
rng = np.random.default_rng(0)
T = 20000; NS = [1, 2, 4, 8, 16]; SIG = 0.15
out = {"T": T, "sigma": SIG, "ns": NS}

def trueV(shape):
    return rng.beta(2, 5, size=shape)          # candidate success probabilities, mean ~0.29

def run(eps_fn, label):
    V = trueV((T, 16)); E = eps_fn(); F = V + E
    res = {}
    for n in NS:
        i = F[:, :n].argmax(1); r = rng.integers(0, n, T); idx = np.arange(T)
        res[n] = dict(gap_argmax=float(E[idx, i].mean()), gap_random=float(E[idx, r].mean()),
                      trueV_argmax=float(V[idx, i].mean()), oracle_best=float(V[:, :n].max(1).mean()),
                      apparent_F=float(F[idx, i].mean()))
    out[label] = res

# A independent errors
run(lambda: rng.normal(0, SIG, (T, 16)), "A_independent")
# B equicorrelated errors rho=0.8 (shared misbelief about the situation)
def corr(rho=0.8):
    c = rng.normal(0, 1, (T, 1)); e = rng.normal(0, 1, (T, 16))
    return SIG * (np.sqrt(rho) * c + np.sqrt(1 - rho) * e)
run(corr, "B_correlated_rho0.8")
# C biased forecaster: errors mean +0.05 (overconfident) -> random-pick control isolates bias from curse
run(lambda: rng.normal(0.05, SIG, (T, 16)), "C_biased_plus0.05")
# D heteroscedastic: true error sd s_i in [0.03,0.3]; forecaster's stated uncertainty u_i has corr q with s_i
def hetero_stats(q):
    s = rng.uniform(0.03, 0.30, (T, 16)); z = (s - s.mean()) / s.std()
    u = q * z + np.sqrt(1 - q * q) * rng.normal(0, 1, (T, 16))
    V = trueV((T, 16)); E = rng.normal(0, 1, (T, 16)) * s; F = V + E
    i = F.argmax(1); idx = np.arange(T)
    sel_err = E[idx, i]; sel_u = u[idx, i]
    rho_u_abs = float(np.corrcoef(u.ravel(), np.abs(E).ravel())[0, 1])
    # does stated uncertainty of the chosen candidate predict its optimism?
    rho_u_sel = float(np.corrcoef(sel_u, sel_err)[0, 1])
    return dict(q=q, corr_u_abs_err_all=rho_u_abs, corr_u_selected_err=rho_u_sel,
                gap_argmax_n16=float(sel_err.mean()))
out["D_hetero"] = [hetero_stats(q) for q in (0.0, 0.3, 1.0)]
# E adaptive proposer: plans x in R^4; V(x)=sigmoid(w.x); error field = SIG * smooth random function g(x).
d, K = 4, 64
W = rng.normal(0, 1, (K, d)) * 1.5; b = rng.uniform(0, 2 * np.pi, K); a = rng.normal(0, 1, K)
def g(x):  # smooth random field with sd ~1
    return (np.cos(x @ W.T + b) @ a) * np.sqrt(2.0 / K)
w = np.ones(d) / np.sqrt(d) * 0.5
Vf = lambda x: 1 / (1 + np.exp(-(x @ w) + 1.0))
Te = 4000
def adaptive(mode):
    res = {}
    X0 = rng.normal(0, 1, (Te, d))
    best = X0.copy(); Fb = Vf(best) + SIG * g(best)
    hist = {1: (best.copy(), Fb.copy())}
    for n in range(2, 17):
        if mode == "iid":
            xn = rng.normal(0, 1, (Te, d))
        else:  # propose near current forecast-best (policy reads forecasts, proposes more)
            xn = best + 0.4 * rng.normal(0, 1, (Te, d))
        Fn = Vf(xn) + SIG * g(xn); upd = Fn > Fb
        best[upd] = xn[upd]; Fb[upd] = Fn[upd]
        if n in NS: hist[n] = (best.copy(), Fb.copy())
    for n in NS:
        x, f = hist[n]; v = Vf(x)
        res[n] = dict(gap=float((f - v).mean()), trueV=float(v.mean()))
    return res
gsd = float(g(rng.normal(0, 1, (20000, d))).std())
out["E_error_field_sd_units_of_sigma"] = gsd
out["E_iid_proposals"] = adaptive("iid"); out["E_adaptive_proposals"] = adaptive("adaptive")
# F optional stopping: policy stops at first candidate with F > tau (max 16) -> endogenous n
V = trueV((T, 16)); E = rng.normal(0, SIG, (T, 16)); F = V + E; tau = 0.55
hit = F > tau; first = np.where(hit.any(1), hit.argmax(1), 15); n_end = first + 1
idx = np.arange(T)
sel = np.where(hit.any(1), first, F.argmax(1))
gap = E[idx, sel]
out["F_optional_stopping"] = {str(k): dict(count=int((n_end == k).sum()), gap=float(gap[n_end == k].mean()))
                              for k in (1, 2, 4, 8, 16) if (n_end == k).sum() > 50}
# bounds
out["bounds"] = {n: dict(max_bound_sigma_sqrt2logn=SIG * np.sqrt(2 * np.log(n)) if n > 1 else 0.0,
                         regret_bound_2sigma_sqrt2log2n=2 * SIG * np.sqrt(2 * np.log(2 * n))) for n in NS}
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "T1_sim.json")
json.dump(out, open(p, "w"), indent=1, default=float); print(p)
