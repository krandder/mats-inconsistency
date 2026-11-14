"""T2b (hawks/T2.md section 5) as the learning check of theory/T2.md Theorems 1-2.
Two-step bistable chain per state s:  self-model p (P(I act carefully)) -> forecaster q = sig(c + v(2x-1)) of
P(success) given the self-report x -> policy commits careful w.p. sig(b + w(2q_r - 1)) reading the emitted forecast
q_r -> success Y ~ Bern(rho_c or rho_n); reward = Y. Emitted reports carry logit noise sigma (all arms; it is the
exploration the fused score-function gradient needs). Easy states: b learnable; hard states: b frozen.
Arms for the self-model AND forecaster: P protected (log loss on realised a / Y, stop-gradient), F fused (P + lam *
REINFORCE on the emitted report, i.e. a score-function estimate of lam * dU/dlogit), CF counterfactual oracle for the
self-model (self-report replaced by a neutral noisy 0.5 on 50% of episodes, trained only there; forecaster P).
Policy modes: learn (co-adapted), replay (policy params replayed step by step from a driver run = matched
trajectory), frozen (policy fixed at the driver's final params = lag-free reference).
Report intervention: on 10% of episodes the self-report (5%) or the forecast (5%) is set to clip(value +- 0.2); these
episodes are probes only (excluded from all training). Exact metrics by Gauss-Hermite quadrature over the report noise.
Usage: python3 t2b_sim.py OUT.json    (numpy only; seeds 0-4)
"""
import json, sys, time, numpy as np
sig = lambda z: 1 / (1 + np.exp(-z))
SEEDS, S, T = 10, 8, 30000
EASY = np.array([1, 1, 1, 1, 0, 0, 0, 0], bool)
RHO_C = np.array([0.9, 0.8, 0.9, 0.3, 0.9, 0.8, 0.9, 0.3]); RHO_N = np.array([0.3, 0.5, 0.4, 0.7, 0.3, 0.5, 0.4, 0.7])
B_HARD = np.array([0., 0., 0., 0., -1., -1., 0.5, -1.])
SIG, LR_POL, LR_SELF, P_INT, P_HID = 0.3, 0.05, 0.02, 0.10, 0.5
ZN, ZW = np.polynomial.hermite_e.hermegauss(9); ZW = ZW / ZW.sum()

def exact(lp, c, v, b, w, lp_override=None):
    """Exact (noise-integrated) quantities for arrays of shape (n, S)."""
    lpp = lp if lp_override is None else lp_override
    x = sig(lpp[..., None] + SIG * ZN)                                  # (n,S,k) emitted self-report
    ql = c[..., None] + v[..., None] * (2 * x - 1); q = sig(ql)          # forecaster output at input x_k
    qr = sig(ql[..., None] + SIG * ZN)                                  # (n,S,k,j) emitted forecast
    pi = sig(b[..., None, None] + w[..., None, None] * (2 * qr - 1))
    pij = (pi * ZW).sum(-1)                                             # P(careful | x_k)
    f = (pij * ZW).sum(-1)                                              # self-model truth, noise integrated
    dpi_dlq = pi * (1 - pi) * 2 * w[..., None, None] * qr * (1 - qr)
    gq = ((RHO_C - RHO_N)[..., None] * (dpi_dlq * ZW).sum(-1) * ZW).sum(-1)   # E dU/dlogit(q)
    yk = RHO_N[..., None] + (RHO_C - RHO_N)[..., None] * pij
    eq = ((q - yk) * ZW).sum(-1)                                        # forecaster signed error (c-component)
    return f, eq, gq

def self_metrics(lp, c, v, b, w):
    p = sig(lp); f, eq, gq = exact(lp, c, v, b, w); h = 1e-4
    fu, _, _ = exact(lp + h, c, v, b, w); fd, _, _ = exact(lp - h, c, v, b, w)
    dU_dl = (RHO_C - RHO_N) * (fu - fd) / (2 * h); dU_dp = dU_dl / (p * (1 - p))
    hcf, _, _ = exact(lp, c, v, b, w, lp_override=np.zeros_like(lp))    # counterfactual-oracle answer f(p0 = 0.5)
    # sup |df/dp| on the segment [0.5, p]
    Ls = np.zeros_like(p)
    for t in np.linspace(0, 1, 11):
        pt = 0.5 + t * (p - 0.5); lt = np.log(pt / (1 - pt))
        a, _, _ = exact(lt + h, c, v, b, w); d, _, _ = exact(lt - h, c, v, b, w)
        Ls = np.maximum(Ls, np.abs((a - d) / (2 * h)) / (pt * (1 - pt)))
    U = RHO_N + (RHO_C - RHO_N) * f
    return dict(p=p, f=f, e=p - f, dU_dl=dU_dl, dU_dp=dU_dp, D_sign=np.sign(dU_dp) * (p - f), D_w=dU_dp * (p - f),
                eq=eq, gq=gq, Dq_sign=np.sign(gq) * eq, h=hcf, Lseg=Ls, U=U, b=b, w=w)

def run(arm, mode, seed_base, lam=0.0, driver=None, lp0=None):
    rng = np.random.default_rng(1000 + seed_base)                         # common random numbers across arms
    n = SEEDS
    lp = lp0.copy(); c = np.zeros((n, S)); v = np.full((n, S), 2.0)
    b = np.where(EASY, 0.0, B_HARD)[None, :].repeat(n, 0).astype(float); w = np.full((n, S), 3.0)
    if mode == "frozen": b, w = driver["b"][-1].copy(), driver["w"][-1].copy()
    base = np.full((n, S), 0.5); traj_b, traj_w = [], []
    ev_start = int(0.75 * T); acc = {}; nacc = 0; acc_all, acc_q1, nall, nq1 = {}, {}, 0, 0; ints = {k: [] for k in ["sd", "sY", "fd", "fY"]}
    for t in range(T):
        if mode == "replay": b, w = driver["b"][t], driver["w"][t]
        z1, z2 = rng.standard_normal((n, S)), rng.standard_normal((n, S))
        u_int, u_kind, u_sgn = rng.random((n, S)), rng.random((n, S)), np.where(rng.random((n, S)) < 0.5, -1.0, 1.0)
        hid = (rng.random((n, S)) < P_HID) if arm == "CF" else np.zeros((n, S), bool)
        intv = u_int < P_INT; int_self = intv & (u_kind < 0.5); int_fc = intv & ~int_self
        p = sig(lp)
        x = np.where(hid, sig(SIG * z1), sig(lp + SIG * z1))
        x = np.where(int_self, np.clip(p + 0.2 * u_sgn, 0.01, 0.99), x)
        ql = c + v * (2 * x - 1); q = sig(ql)
        qr = np.where(int_fc, np.clip(q + 0.2 * u_sgn, 0.01, 0.99), sig(ql + SIG * z2))
        pi = sig(b + w * (2 * qr - 1)); a = (rng.random((n, S)) < pi).astype(float)
        Y = (rng.random((n, S)) < np.where(a > 0, RHO_C, RHO_N)).astype(float)
        tr = ~intv; adv = Y - base
        # self-model
        m_self = tr & (hid if arm == "CF" else np.ones_like(tr))
        g = (a - p)
        if arm == "F": g = g + lam * adv * z1 / SIG
        lp = lp + LR_SELF * np.where(m_self, g, 0.0)
        # forecaster (protected in P and CF; fused in F)
        gc = (Y - q)
        if arm == "F": gc = gc + lam * adv * z2 / SIG
        c = c + LR_SELF * np.where(tr, gc, 0.0); v = v + LR_SELF * np.where(tr, gc * (2 * x - 1), 0.0)
        if mode == "learn":
            gp = np.where(tr, adv * (a - pi), 0.0)
            b = b + LR_POL * gp * EASY; w = w + LR_POL * gp * (2 * qr - 1)
            traj_b.append(b.copy()); traj_w.append(w.copy())
        base = base + 0.01 * np.where(tr, Y - base, 0.0)
        if t % 250 == 0:
            m = self_metrics(lp, c, v, b, w); nall += 1; q1 = t < T // 4
            for k in ["e", "D_sign", "D_w", "eq", "Dq_sign", "U"]:
                acc_all[k] = acc_all.get(k, 0) + m[k]
                if q1: acc_q1[k] = acc_q1.get(k, 0) + m[k]
            nq1 += q1
        if t >= ev_start:
            if t % 100 == 0:
                m = self_metrics(lp, c, v, b, w); nacc += 1
                for k, val in m.items(): acc[k] = acc.get(k, 0) + val
            ints["sd"].append(np.where(int_self, u_sgn, 0.0)); ints["sY"].append(np.where(int_self, Y, np.nan))
            ints["fd"].append(np.where(int_fc, u_sgn, 0.0)); ints["fY"].append(np.where(int_fc, Y, np.nan))
    res = {k: val / nacc for k, val in acc.items()}
    for k in acc_all: res["run_" + k] = acc_all[k] / nall; res["q1_" + k] = acc_q1[k] / nq1
    for pre in ["s", "f"]:
        d, Yv = np.array(ints[pre + "d"]), np.array(ints[pre + "Y"])
        up = np.nanmean(np.where(d > 0, Yv, np.nan), 0); dn = np.nanmean(np.where(d < 0, Yv, np.nan), 0)
        res["int_dU_d" + ("p" if pre == "s" else "q")] = (up - dn) / 0.4               # report-intervention estimate
    res["final"] = self_metrics(lp, c, v, b, w)
    if mode == "learn": res["traj"] = dict(b=np.array(traj_b), w=np.array(traj_w))
    return res

def ms(x): x = np.asarray(x, float); return dict(mean=float(np.mean(x)), sd=float(np.std(x, ddof=1)))
def strat(res, key, mask):            # mean over states in mask per seed, then mean +- sd over seeds
    return ms(np.array([res[key][s][mask[s]].mean() if mask[s].any() else np.nan for s in range(SEEDS)]))

t0 = time.time()
lp0 = np.random.default_rng(7).normal(0, 1.5, (SEEDS, S))
R = {}
R["P/learn"] = run("P", "learn", 0, lp0=lp0)
R["F0.3/learn"] = run("F", "learn", 0, lam=0.3, lp0=lp0)
for drv in ["P/learn", "F0.3/learn"]:
    D = R[drv]["traj"]; tag = drv.split("/")[0]
    for arm, lam in [("P", 0.0), ("F", 0.3), ("F", 1.0), ("CF", 0.0)]:
        name = arm + (str(lam) if arm == "F" else "")
        R[f"{name}/replay[{tag}]"] = run(arm, "replay", 0, lam=lam, driver=D, lp0=lp0)
        R[f"{name}/frozen[{tag}]"] = run(arm, "frozen", 0, lam=lam, driver=D, lp0=lp0)
elapsed = time.time() - t0

out = dict(config=dict(SEEDS=SEEDS, S=S, T=T, EASY=EASY.tolist(), RHO_C=RHO_C.tolist(), RHO_N=RHO_N.tolist(),
                       B_HARD=B_HARD.tolist(), SIG=SIG, LR_POL=LR_POL, LR_SELF=LR_SELF, P_INT=P_INT, P_HID=P_HID,
                       eval_window="last 25% of T, exact metrics every 100 steps, then averaged", elapsed_s=elapsed),
           arms={})
hard = np.repeat(~EASY[None, :], SEEDS, 0); easy = ~hard
# sensitivity strata defined once, from the protected frozen reference on the P driver (same policy params for all frozen arms)
ref = R["P/frozen[P]"]; sens = np.abs(ref["dU_dp"]); hi = sens >= np.median(sens); lo = ~hi
for name, r in R.items():
    lam = 0.3 if "F0.3" in name else (1.0 if "F1.0" in name else 0.0)
    d = {}
    for sname, mask in [("hard", hard), ("easy", easy), ("hiSens", hi), ("loSens", lo), ("all", np.ones_like(hard))]:
        d[sname] = {k: strat(r, k, mask) for k in ["e", "D_sign", "D_w", "eq", "Dq_sign", "dU_dp", "U", "w", "p"]
                    + [pre + k for pre in ["run_", "q1_"] for k in ["e", "D_sign", "D_w", "eq", "Dq_sign", "U"]]}
        d[sname]["careful_basin_frac"] = strat(dict(x=(r["p"] > 0.5).astype(float)), "x", mask)
    # Theorem 2: fused self-model signed error = lam * dU/dlogit (log score, logit parameterisation)
    pred = lam * r["dU_dl"]; d["thm2_pred_e_minus_lam_dUdl"] = dict(max_abs=float(np.abs(r["e"] - pred).max()),
        mean_abs=float(np.abs(r["e"] - pred).mean()), mean_abs_e=float(np.abs(r["e"]).mean()),
        slope_e_on_pred=float((r["e"] * pred).sum() / max((pred ** 2).sum(), 1e-12)) if lam > 0 else None)
    d["thm2_forecaster_eq_minus_lam_gq"] = dict(mean_abs=float(np.abs(r["eq"] - lam * r["gq"]).mean()), mean_abs_eq=float(np.abs(r["eq"]).mean()))
    # Theorem 1: protected p vs counterfactual-oracle answer h; bound |p - h| <= Lseg |p - 0.5|
    d["thm1"] = dict(mean_abs_p_minus_h=float(np.abs(r["p"] - r["h"]).mean()), max_abs_p_minus_h=float(np.abs(r["p"] - r["h"]).max()),
                     bound_violations=int((np.abs(r["p"] - r["h"]) > r["Lseg"] * np.abs(r["p"] - 0.5) + 0.02).sum()),
                     n=int(r["p"].size), mean_Lseg=float(r["Lseg"].mean()), frac_Lseg_ge1=float((r["Lseg"] >= 1).mean()))
    # report intervention: estimated vs exact dU/dp
    est, ex = r["int_dU_dp"], r["dU_dp"]; big = np.abs(ex) > 0.1
    d["intervention"] = dict(corr=float(np.corrcoef(est.ravel(), ex.ravel())[0, 1]),
                             sign_agree_all=float((np.sign(est) == np.sign(ex)).mean()),
                             sign_agree_where_abs_dUdp_gt_0p1=float((np.sign(est) == np.sign(ex))[big].mean()) if big.any() else None,
                             n_big=int(big.sum()),
                             D_sign_with_estimated_sign_hard=strat(dict(x=np.sign(est) * r["e"]), "x", hard),
                             D_sign_with_estimated_sign_hiSens=strat(dict(x=np.sign(est) * r["e"]), "x", hi))
    d["basin_careful_frac"] = float((r["p"] > 0.5).mean())
    d["per_state_mean_over_seeds"] = {k: r[k].mean(0).round(4).tolist() for k in ["p", "f", "e", "dU_dp", "h", "w", "b", "U"]}
    out["arms"][name] = d
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(f"elapsed {elapsed:.1f}s")
for name, d in out["arms"].items():
    print(f"{name:22s} hard D={d['hard']['D_sign']['mean']:+.4f}±{d['hard']['D_sign']['sd']:.4f} "
          f"e={d['hard']['e']['mean']:+.4f} | easy D={d['easy']['D_sign']['mean']:+.4f}±{d['easy']['D_sign']['sd']:.4f} "
          f"e={d['easy']['e']['mean']:+.4f} | hiS D={d['hiSens']['D_sign']['mean']:+.4f}±{d['hiSens']['D_sign']['sd']:.4f} "
          f"loS D={d['loSens']['D_sign']['mean']:+.4f}±{d['loSens']['D_sign']['sd']:.4f} | Dq hard {d['hard']['Dq_sign']['mean']:+.4f} "
          f"| thm2 res {d['thm2_pred_e_minus_lam_dUdl']['mean_abs']:.4f}/{d['thm2_pred_e_minus_lam_dUdl']['mean_abs_e']:.4f} "
          f"| |p-h| {d['thm1']['mean_abs_p_minus_h']:.3f} viol {d['thm1']['bound_violations']} | int sign {d['intervention']['sign_agree_where_abs_dUdp_gt_0p1']} "
          f"| U hard {d['hard']['U']['mean']:.3f} basin {d['basin_careful_frac']:.2f}")
    print(f"{'':22s} whole-run hard D={d['hard']['run_D_sign']['mean']:+.4f}±{d['hard']['run_D_sign']['sd']:.4f} Dq={d['hard']['run_Dq_sign']['mean']:+.4f}±{d['hard']['run_Dq_sign']['sd']:.4f} "
          f"q1 D={d['hard']['q1_D_sign']['mean']:+.4f}±{d['hard']['q1_D_sign']['sd']:.4f} | easy run D={d['easy']['run_D_sign']['mean']:+.4f}±{d['easy']['run_D_sign']['sd']:.4f} "
          f"| hard basin {d['hard']['careful_basin_frac']['mean']:.2f} U run {d['hard']['run_U']['mean']:.3f}")
