"""T1 theory check (numpy toy). Verifies the propositions in ../T1.md:
 P1  within-pool cancellation: curse(n) := E[eps_sel] - E[eps_rand] depends only on the within-pool part of the error;
     bound sigma_w*sqrt(2 ln n) under any dependence; exact Gaussian formula sigma_w^2/sqrt(sV_w^2+sigma_w^2)*e_n.
 P2  joint calibration => zero gap for ANY selection measurable on the forecaster's information (argmax, adaptive
     proposal, optional stopping).
 P3  iid candidates: E[V_sel(n)] = E_{u~Beta(n,1)} r(u), r = E[V | forecast quantile u]; non-decreasing in n whenever
     r is non-decreasing; rise-then-fall needs a decreasing tail of r (hacked forecasts).
 P4  stopping with cost c per forecast: compare rules at equal forecaster calls, scored by V_sel - c*calls.
Single process, numpy + scipy. Output: t1_toy.json next to this file."""
import json, os, sys, time
import numpy as np
from scipy import integrate, stats, optimize

HERE = os.path.dirname(os.path.abspath(__file__))
NS = [1, 2, 4, 8, 16]
out = {"numpy": np.__version__, "ns": NS}
t0 = time.time()


def e_n(n):
    """E[max of n iid N(0,1)]."""
    if n == 1:
        return 0.0
    f = lambda x: x * n * stats.norm.pdf(x) * stats.norm.cdf(x) ** (n - 1)
    return integrate.quad(f, -12, 12)[0]


EN = {n: e_n(n) for n in NS}
out["e_n"] = EN

# ---------------- P1: Gaussian curse formula, equicorrelated errors, any-dependence bound ----------------
rng = np.random.default_rng(0)
T = 200_000; sV = 0.16; sig = 0.15
p1 = {}
for rho in (0.0, 0.5, 0.8, 0.95):
    V = rng.normal(0, sV, (T, 16))
    c = rng.normal(0, 1, (T, 1)); e = rng.normal(0, 1, (T, 16))
    E = sig * (np.sqrt(rho) * c + np.sqrt(1 - rho) * e); F = V + E
    sw = sig * np.sqrt(1 - rho)
    row = {}
    for n in NS:
        i = F[:, :n].argmax(1); idx = np.arange(T)
        gap = E[idx, i]; gap_r = E[:, :n].mean(1)            # random pick = pool mean in expectation
        curse = gap - gap_r
        row[n] = dict(curse_mc=float(curse.mean()), curse_se=float(curse.std() / np.sqrt(T)),
                      gap_mc=float(gap.mean()),
                      curse_formula=float(sw ** 2 / np.sqrt(sV ** 2 + sw ** 2) * EN[n]),
                      trueV_mc=float(V[idx, i].mean()),
                      trueV_formula=float(sV ** 2 / np.sqrt(sV ** 2 + sw ** 2) * EN[n]),
                      bound_total_sigma=float(sig * np.sqrt(2 * np.log(n))),
                      bound_within_sigma=float(sw * np.sqrt(2 * np.log(n))))
    p1[f"rho{rho}"] = row
# any-dependence: adversarial dependence (errors comonotone with V rank) still under sigma*sqrt(2 ln n)
V = rng.normal(0, sV, (T, 16)); z = rng.normal(0, 1, (T, 16))
rk = V.argsort(1).argsort(1); zs = np.sort(z, 1); E = sig * np.take_along_axis(zs, rk, 1)  # error rank = V rank
F = V + E; i = F.argmax(1)
p1["adversarial_comonotone"] = {"gap_n16": float(E[np.arange(T), i].mean()),
                                "bound_sigma_sqrt2ln16": float(sig * np.sqrt(2 * np.log(16))),
                                "E_max_eps_n16": float(sig * EN[16])}
# P1b: compressed / over-dispersed forecaster F = alpha + beta*V + eta. curse = [(beta-1)beta sV^2 + tau^2]/s * e_n,
# s = sqrt(beta^2 sV^2 + tau^2): NEGATIVE when tau^2 < (1-beta) beta sV^2 (weak, compressed forecasters).
for beta in (0.3, 1.0, 1.5):
    for tau in (0.02, 0.10):
        V = rng.normal(0.6, sV, (T, 16)); F = 0.2 + beta * V + rng.normal(0, tau, (T, 16)); E = F - V
        s_ = np.sqrt(beta ** 2 * sV ** 2 + tau ** 2); row = {}
        for n in (2, 8, 16):
            i = F[:, :n].argmax(1); idx = np.arange(T)
            row[n] = dict(curse_mc=float((E[idx, i] - E[:, :n].mean(1)).mean()),
                          curse_formula=float(((beta - 1) * beta * sV ** 2 + tau ** 2) / s_ * EN[n]))
        p1[f"linear_beta{beta}_tau{tau}"] = row
out["P1"] = p1

# ---------------- P2: joint calibration => zero gap under argmax / adaptive / optional stopping ----------------
# V_i = theta + beta*z_i + a_i. Forecaster sees y = theta + N(0,sy^2) (shared belief evidence), plan feature z_i,
# x_i = a_i + N(0,sx^2) (plan-specific signal). Calibrated F_i = E[V_i | y, z_i, x_i]; raw F_i = y + beta z_i + x_i.
rng = np.random.default_rng(1)
T = 100_000; sth, sy, sa, sx, beta = 1.0, 0.8, 0.5, 0.7, 0.4
kth = sth ** 2 / (sth ** 2 + sy ** 2); ka = sa ** 2 / (sa ** 2 + sx ** 2)
p2 = {}
for mode in ("iid_argmax", "adaptive_argmax", "optional_stop"):
    res = {}
    for fc in ("calibrated", "raw"):
        th = rng.normal(0, sth, T); y = th + rng.normal(0, sy, T)
        best_F = np.full(T, -np.inf); best_V = np.zeros(T); zbest = np.zeros(T)
        stopped = np.zeros(T, bool); n_used = np.zeros(T, int)
        for k in range(16):
            if mode == "adaptive_argmax" and k > 0:
                zk = zbest + 0.5 * rng.normal(0, 1, T)          # hill-climb in plan space on the forecast
            else:
                zk = rng.normal(0, 1, T)
            a = rng.normal(0, sa, T); x = a + rng.normal(0, sx, T); Vk = th + beta * zk + a
            Fk = (kth * y + beta * zk + ka * x) if fc == "calibrated" else (y + beta * zk + x)
            act = ~stopped
            upd = act & (Fk > best_F)
            best_F[upd] = Fk[upd]; best_V[upd] = Vk[upd]; zbest[upd] = zk[upd]; n_used[act] += 1
            if mode == "optional_stop":
                stopped |= best_F > 1.0
        res[fc] = dict(gap=float((best_F - best_V).mean()), gap_se=float((best_F - best_V).std() / np.sqrt(T)),
                       trueV=float(best_V.mean()), mean_n=float(n_used.mean()))
    p2[mode] = res
out["P2"] = p2

# ---------------- P3: monotone r => monotone value; hacked tail => rise-then-fall (HedgeTune regime) ----------------
rng = np.random.default_rng(2)
T = 200_000; p3 = {}
for name, h in (("honest", 0.0), ("hacked_h0.03", 0.03), ("hacked_h0.10", 0.10)):
    V = rng.beta(4, 3, (T, 64)); F = V + rng.normal(0, 0.10, (T, 64))
    hack = rng.random((T, 64)) < h                  # exploit plans: forecaster fooled (F ~ 1.0), true value 0.15
    F = np.where(hack, 1.0 + rng.normal(0, 0.05, (T, 64)), F)
    V = np.where(hack, 0.15, V)
    row = {}
    for n in (1, 2, 4, 8, 16, 32, 64):
        i = F[:, :n].argmax(1); row[n] = float(V[np.arange(T), i].mean())
    p3[name] = row
out["P3"] = p3

# ---------------- P4: stopping rules with a cost per forecast, at equal forecaster calls ----------------
def sim_states(rng, S, t, q, shared_frac=0.5):
    """S states x 16 iid candidates. V ~ state-level mean + spread; heteroscedastic per-plan error sd s in [0.2t,1.8t];
    stated sd u tracks s with corr q; two forecaster seeds share shared_frac of the error variance."""
    mu = np.clip(rng.normal(0.6, 0.12, (S, 1)), 0.1, 0.9)
    V = np.clip(mu + rng.normal(0, 0.12, (S, 16)), 0, 1)
    s = rng.uniform(0.2 * t, 1.8 * t, (S, 16))
    zc = rng.normal(0, 1, (S, 16))
    Ea = s * (np.sqrt(shared_frac) * zc + np.sqrt(1 - shared_frac) * rng.normal(0, 1, (S, 16)))
    Eb = s * (np.sqrt(shared_frac) * zc + np.sqrt(1 - shared_frac) * rng.normal(0, 1, (S, 16)))
    sz = (s - s.mean()) / s.std()
    u = s.mean() + s.std() * (q * sz + np.sqrt(1 - q * q) * rng.normal(0, 1, (S, 16)))
    u = np.maximum(u, 0.005)
    return dict(V=V, Fa=V + Ea, Fb=V + Eb, u=u, R=(rng.random((S, 16)) < V).astype(float))


def nested_value(score, V, n):
    i = score[:, :n].argmax(1)
    return V[np.arange(len(V)), i]


def best_fixed_n(score, V, c, calls_per=1):
    vals = {n: nested_value(score, V, n).mean() - c * calls_per * n for n in range(1, 17)}
    return max(vals, key=vals.get)


def feats(F, u):
    return np.stack([np.ones_like(F), F, F * F, u, F * u, u * u], -1)


def fit_gap_model(F, u, Y):
    X = feats(F, u).reshape(-1, 6); w, *_ = np.linalg.lstsq(X, Y.reshape(-1), rcond=None)
    return lambda F2, u2: feats(F2, u2) @ w


def reservation(vhat_cal, c):
    """Weitzman reservation value z: E[(v - z)^+] = c over the calibration distribution of calibrated values."""
    v = vhat_cal.reshape(-1)
    g = lambda z: np.maximum(v - z, 0).mean() - c
    return optimize.brentq(g, v.min() - 1, v.max() + 1)


def pandora(vhat, V, z):
    """Sequential: forecast candidates in order; stop at first k with max vhat >= z (cap 16); commit argmax vhat."""
    S = len(V); out_v = np.zeros(S); n = np.zeros(S)
    for s in range(S):
        b = -np.inf; bi = 0
        for k in range(16):
            if vhat[s, k] > b:
                b, bi = vhat[s, k], k
            if b >= z or k == 15:
                out_v[s] = V[s, bi]; n[s] = k + 1; break
    return out_v, n


def reservation_rel(vhat_cal, c):
    """Reservation on within-state deviations d = v - state mean (state level unknown to the searcher)."""
    d = vhat_cal - vhat_cal.mean(1, keepdims=True)
    return reservation(d, c)


def pandora_rel(vhat, V, zd, kmin=2):
    """State-adaptive Weitzman: stop at first k >= kmin with best - mean(seen) >= zd (cap 16); commit argmax vhat."""
    S = len(V); out_v = np.zeros(S); n = np.zeros(S)
    for s in range(S):
        for k in range(16):
            seen = vhat[s, :k + 1]; bi = int(seen.argmax())
            if (k + 1 >= kmin and seen[bi] - seen.mean() >= zd) or k == 15:
                out_v[s] = V[s, bi]; n[s] = k + 1; break
    return out_v, n


def hedgetune_n(Fcal, Vcal, c):
    """HedgeTune for BoN (Khalaf et al. 2025, Thm 3 / Alg. 4) with a cost term: per state map forecasts to empirical
    quantiles u, r(u) = true value; solve d/dn E_{u~Beta(n,1)}[r(u)] = c, i.e. mean_t sum_k r_tk*(1/n+ln u_tk)*n*u^(n-1)
    /K = c (Monte Carlo over the K candidates as quantile draws)."""
    K = Fcal.shape[1]
    U = (Fcal.argsort(1).argsort(1) + 0.5) / K
    def resid(n):
        return float(np.mean(np.sum(Vcal * (1 / n + np.log(U)) * n * U ** (n - 1), 1) / K)) - c
    lo, hi = 1.0, 16.0
    if resid(lo) <= 0: return 1
    if resid(hi) >= 0: return 16
    return int(round(optimize.brentq(resid, lo, hi)))


p4 = {}
for t in (0.05, 0.15):
    for q in (0.0, 0.3, 1.0):
        for c in (0.005, 0.01, 0.02):
            rows = {}
            for seed in range(5):
                rng = np.random.default_rng(100 + seed)
                cal = sim_states(rng, 3000, t, q); te = sim_states(rng, 3000, t, q)
                V, Fa, u = te["V"], te["Fa"], te["u"]
                r = {}
                # oracle (true-value argmax with tuned n): reference
                n_or = best_fixed_n(cal["V"], cal["V"], c)
                r["oracle_fixed_n"] = nested_value(V, V, n_or).mean() - c * n_or
                # 1 fixed tuned n on raw forecasts
                n_f = best_fixed_n(cal["Fa"], cal["V"], c)
                r["fixed_n"] = nested_value(Fa, V, n_f).mean() - c * n_f
                # 2 HedgeTune-style n (cost-aware root of the BoN gradient on calibration ranks)
                n_h = hedgetune_n(cal["Fa"], cal["V"], c)
                r["hedgetune_n"] = nested_value(Fa, V, n_h).mean() - c * n_h
                # 3 learned gap model on realised 0/1 outcomes (GOAL signal 4) + Weitzman reservation stopping
                m = fit_gap_model(cal["Fa"], cal["u"], cal["R"])
                z = reservation(m(cal["Fa"], cal["u"]), c)
                v, n = pandora(m(Fa, u), V, z)
                r["gap_model_pandora"] = (v - c * n).mean(); r["gap_model_pandora_mean_n"] = n.mean()
                # 3c state-adaptive reservation (within-state deviations; kmin tuned on calibration in {2,3,4})
                zd = reservation_rel(m(cal["Fa"], cal["u"]), c)
                km = max((2, 3, 4), key=lambda kk: (lambda vv, nn: (vv - c * nn).mean())(*pandora_rel(m(cal["Fa"][:600], cal["u"][:600]), cal["V"][:600], zd, kk)))
                v, n = pandora_rel(m(Fa, u), V, zd, km)
                r["gap_model_pandora_rel"] = (v - c * n).mean(); r["gap_model_pandora_rel_mean_n"] = n.mean()
                # 3d control: the same state-adaptive stopping on RAW forecasts (no gap model)
                zr = reservation_rel(cal["Fa"], c)
                kr = max((2, 3, 4), key=lambda kk: (lambda vv, nn: (vv - c * nn).mean())(*pandora_rel(cal["Fa"][:600], cal["V"][:600], zr, kk)))
                v, n = pandora_rel(Fa, V, zr, kr)
                r["raw_pandora_rel"] = (v - c * n).mean()
                # 3b learned gap model, fixed tuned n (separates the shrinkage from the stopping rule)
                sc_cal = m(cal["Fa"], cal["u"]); n_g = best_fixed_n(sc_cal, cal["V"], c)
                r["gap_model_fixed_n"] = nested_value(m(Fa, u), V, n_g).mean() - c * n_g
                # 4 Smith-Winkler shrinkage with stated sigma (prior from calibration forecasts), + reservation stop
                mu0 = cal["Fa"].mean(); s0sq = max(cal["Fa"].var() - (cal["u"] ** 2).mean(), 1e-4)
                sw = lambda F2, u2: mu0 + s0sq / (s0sq + u2 ** 2) * (F2 - mu0)
                z_sw = reservation(sw(cal["Fa"], cal["u"]), c)
                v, n = pandora(sw(Fa, u), V, z_sw)
                r["smith_winkler_stated_pandora"] = (v - c * n).mean()
                n_s = best_fixed_n(sw(cal["Fa"], cal["u"]), cal["V"], c)
                r["smith_winkler_stated_fixed_n"] = nested_value(sw(Fa, u), V, n_s).mean() - c * n_s
                # 5 LCB pessimism F - kappa*u, kappa and n tuned on calibration
                best = (-1e9, 0, 1)
                for kap in (0, 0.5, 1, 2, 3):
                    sc = cal["Fa"] - kap * cal["u"]; nn = best_fixed_n(sc, cal["V"], c)
                    val = nested_value(sc, cal["V"], nn).mean() - c * nn
                    if val > best[0]: best = (val, kap, nn)
                _, kap, nn = best
                r["lcb"] = nested_value(Fa - kap * u, V, nn).mean() - c * nn
                # 6 ensemble WCO (Coste et al. 2023): min over two forecaster seeds, 2 calls per candidate
                n_w = best_fixed_n(np.minimum(cal["Fa"], cal["Fb"]), cal["V"], c, calls_per=2)
                r["wco_2seeds"] = nested_value(np.minimum(te["Fa"], te["Fb"]), V, n_w).mean() - 2 * c * n_w
                r["n_fixed"], r["n_hedgetune"] = n_f, n_h
                # Spearman(stated u, |eps|) overall and at the argmax (n=16)
                eps = Fa - V
                r["spearman_u_abs_eps"] = stats.spearmanr(u.ravel(), np.abs(eps).ravel()).statistic
                i = Fa.argmax(1); idx = np.arange(len(V))
                r["spearman_u_eps_at_argmax"] = stats.spearmanr(u[idx, i], eps[idx, i]).statistic
                r["curse_n16"] = float((eps[idx, i] - eps.mean(1)).mean())
                for k, vv in r.items():
                    rows.setdefault(k, []).append(float(vv))
            p4[f"t{t}_q{q}_c{c}"] = {k: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)), per_seed=v) for k, v in rows.items()}
            for a in ("gap_model_pandora_rel", "raw_pandora_rel", "gap_model_fixed_n", "smith_winkler_stated_fixed_n", "smith_winkler_stated_pandora", "lcb", "hedgetune_n", "wco_2seeds"):
                d = np.array(rows[a]) - np.array(rows["fixed_n"])
                p4[f"t{t}_q{q}_c{c}"][f"paired_{a}_minus_fixed_n"] = dict(mean=float(d.mean()), sd=float(d.std(ddof=1)))
out["P4"] = p4
out["seconds"] = time.time() - t0
json.dump(out, open(os.path.join(HERE, "t1_toy.json"), "w"), indent=1)

# ---------------- printout ----------------
print("e_n", {n: round(v, 3) for n, v in EN.items()})
for k, row in p1.items():
    if k.startswith("rho"):
        print("P1", k, " ".join(f"n{n}: curse {row[n]['curse_mc']:.4f} (formula {row[n]['curse_formula']:.4f}) "
                                f"V {row[n]['trueV_mc']:.3f}/{row[n]['trueV_formula']:.3f} bnd_w {row[n]['bound_within_sigma']:.3f}"
                                for n in (2, 16)))
print("P1 adversarial", p1["adversarial_comonotone"])
for k, row in p1.items():
    if k.startswith("linear"):
        print("P1b", k, " ".join(f"n{n}: {row[n]['curse_mc']:+.4f} (formula {row[n]['curse_formula']:+.4f})" for n in row))
for k, v in p2.items():
    print("P2", k, {fc: {kk: round(x, 4) for kk, x in d.items()} for fc, d in v.items()})
for k, v in p3.items():
    print("P3", k, {n: round(x, 4) for n, x in v.items()})
for k, v in p4.items():
    print("P4", k, " ".join(f"{kk} {d['mean']:.4f}±{d['sd']:.4f}" for kk, d in v.items()))
print("seconds", round(out["seconds"], 1))
