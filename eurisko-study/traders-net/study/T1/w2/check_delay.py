"""T1 lemma 4 checks: delayed settlement (see PREREG.md, check 4a-4e).

Toy market = per-question online logistic regression on the S0 streams (PRNG instead of pi).
Forecast at t uses bits revealed before t; y_t is revealed at t+D and the cached trade-time
gradient is applied then (stale-gradient SGD). Books per contract (question q, time t):
  tangent  : l_t (y - p_t)                       (sum of unit shares z_j (y - p_t))
  exact    : log p_t(y) - log p0_t(y)            (LMSR from the prior l0 = running base rate)
  hindsight: log p'(y) - log p0_t(y), p' = reveal-time model on the cached trade input
  marks    : current model re-priced on the cached input every MARK steps (mark-to-market)
Usage: python3 check_delay.py [--seeds 1 2 3] [--T 8000]
"""
import argparse, hashlib, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SUB = os.path.normpath(os.path.join(HERE, "../../S0/substrate"))
sys.path.insert(0, SUB)
import streams_ext  # noqa: E402

FAMS = ["periodic", "lfsr16", "ca30", "ca110", "prime", "prng", "and", "or"]
W, ETA, MARK = 16, 0.05, 16


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def logp(l, y):
    # log sigma(l) if y==1 else log sigma(-l), stable
    s = np.where(y > 0.5, l, -l)
    return -np.logaddexp(0.0, -s)


def run(seed, D, T, shuffle=False, eta_scale=False):
    eta = ETA * 16.0 / (16.0 + D) if eta_scale else ETA
    bits, fam, _ = streams_ext.make_questions(Q=64, T=T, seed=seed)
    bits = bits.astype(np.float64)
    Q = bits.shape[0]
    rng = np.random.default_rng(1000 + seed)
    if shuffle:
        for q in range(Q):
            bits[q] = bits[q, rng.permutation(T)]
    xs = 2.0 * bits - 1.0
    th = np.zeros((Q, W + 1))            # weights + bias (last column)
    n1 = np.zeros(Q); n = np.zeros(Q)    # revealed-label counts for the prior
    R = D + 1
    buf_x = np.zeros((R, Q, W + 1)); buf_th = np.zeros((R, Q, W + 1))
    buf_l = np.zeros((R, Q)); buf_l0 = np.zeros((R, Q)); buf_ys = np.zeros((R, Q))
    buf_mark = np.zeros((R, Q)); buf_mtm = np.zeros((R, Q)); buf_mid = np.full((R, Q), np.nan)
    buf_t = np.full(R, -1)
    th_start = None
    rec = {k: [] for k in ["tan", "ex", "hind", "gap", "tel_tan", "tel_ex", "mid_ex",
                           "self_tan", "self_ex", "kl"]}
    W_trade = 0.0; S_stale = 0.0; G2 = 0.0
    t0 = D + W
    for t in range(t0, T):
        # --- trade at t
        x = np.concatenate([xs[:, t - D - W:t - D], np.ones((Q, 1))], 1)
        l = (th * x).sum(1)
        l0 = np.log((n1 + 1) / (n - n1 + 1))
        k = t % R
        buf_x[k] = x; buf_th[k] = th; buf_l[k] = l; buf_l0[k] = l0; buf_t[k] = t
        p = sig(l)
        buf_ys[k] = (rng.random(Q) < p).astype(np.float64)   # self-resolved outcome (4d)
        buf_mark[k] = p; buf_mtm[k] = 0.0; buf_mid[k] = np.nan
        # --- mark-to-market outstanding contracts every MARK steps (and at the half-way point)
        age = t - buf_t
        sel = np.where((buf_t >= 0) & (age > 0) & ((age % MARK == 0) | (age == D // 2)))[0]
        if sel.size:
            m = sig(np.einsum("qj,kqj->kq", th, buf_x[sel]))
            buf_mtm[sel] += buf_l[sel] * (m - buf_mark[sel])      # tangent increments
            buf_mark[sel] = m
            mid = sel[age[sel] == D // 2]
            if mid.size:
                mm = buf_mark[mid]; lp, l0p = buf_l[mid], buf_l0[mid]
                buf_mid[mid] = mm * (logp(lp, 1) - logp(l0p, 1)) + (1 - mm) * (logp(lp, 0) - logp(l0p, 0))
        # --- reveal and settle contract s = t - D (end of step t)
        s = t - D
        ks = s % R
        if buf_t[ks] == s and s >= t0:
            y = bits[:, s]
            ls, l0s, xsv, ths = buf_l[ks], buf_l0[ks], buf_x[ks], buf_th[ks]
            ps = sig(ls)
            tan = ls * (y - ps)
            ex = logp(ls, y) - logp(l0s, y)
            lh = (th * xsv).sum(1)
            hind = logp(lh, y) - logp(l0s, y)
            # telescoping: accumulated marks + final increment vs settlement at reveal
            mtm_tot = buf_mtm[ks] + ls * (y - buf_mark[ks])
            tel_tan = np.abs(mtm_tot - tan)
            # exact marks: value at mark m of the move l0->l; at m=y equals ex
            tel_ex = np.abs((y * (logp(ls, 1) - logp(l0s, 1)) + (1 - y) * (logp(ls, 0) - logp(l0s, 0))) - ex)
            ysf = buf_ys[ks]
            p0 = sig(l0s)
            rec["tan"].append(tan); rec["ex"].append(ex); rec["hind"].append(hind); rec["gap"].append(hind - ex)
            rec["tel_tan"].append(tel_tan); rec["tel_ex"].append(tel_ex)
            rec["mid_ex"].append(buf_mid[ks] if D >= 2 else np.full(Q, np.nan))
            rec["self_tan"].append(ls * (ysf - ps)); rec["self_ex"].append(logp(ls, ysf) - logp(l0s, ysf))
            rec["kl"].append(ps * (np.log(ps + 1e-300) - np.log(p0)) + (1 - ps) * (np.log(1 - ps + 1e-300) - np.log(1 - p0)))
            # stale-gradient update with the cached trade-time gradient
            g = (ps - y)[:, None] * xsv
            if th_start is None:
                th_start = th.copy()
            W_trade += float((-(ths * g).sum()))
            S_stale += float(((th - ths) * g).sum())
            G2 += float((g * g).sum())
            th = th - eta * g
            n1 += y; n += 1
            buf_t[ks] = -1
    lhs = float(((th ** 2).sum() - (th_start ** 2).sum()) / (2 * eta))
    rhs = W_trade - S_stale + eta / 2 * G2
    A = {k: np.array(v) for k, v in rec.items()}           # [n_contracts, Q]
    out = {"seed": seed, "D": D, "T": T, "shuffle": shuffle, "n_contracts": int(A["ex"].shape[0]),
           "tel_tan_max": float(A["tel_tan"].max()), "tel_ex_max": float(A["tel_ex"].max()),
           "norm_lhs": lhs, "norm_rhs": rhs, "norm_rel_err": abs(lhs - rhs) / max(1e-12, abs(lhs)),
           "eta": eta, "W_trade": W_trade, "S_stale": S_stale, "eta_half_G2": eta / 2 * G2,
           "stale_ratio": S_stale / max(1e-12, abs(W_trade)), "fam": {}}
    for f, name in enumerate(FAMS):
        qs = np.where(fam == f)[0]
        d = {}
        for key in ["tan", "ex", "hind", "gap", "self_tan", "self_ex", "kl"]:
            v = A[key][:, qs]
            d[key + "_mean"] = float(v.mean())
            d[key + "_se"] = float(v.mean(1).std() / np.sqrt(v.shape[0]))  # SE over contracts (per-step mean over qs)
        mid = A["mid_ex"][:, qs]
        if np.isfinite(mid).any():
            ok = np.isfinite(mid)
            d["mid_ex_mean"] = float(mid[ok].mean())
            d["mtm_bias_ex"] = float((A["ex"][:, qs][ok] - mid[ok]).mean())
            d["corr_mid_final"] = float(np.corrcoef(mid[ok], A["ex"][:, qs][ok])[0, 1])
        out["fam"][name] = d
    return out


def mrp_check(seed, S=20, H=50, gamma=0.9, n_ep=400):
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(S) * 0.3, size=S); r = rng.normal(size=S)
    # true finite-horizon values V_h(s) for remaining steps h
    Vt = np.zeros((H + 1, S))
    for h in range(1, H + 1):
        Vt[h] = r + gamma * P @ Vt[h - 1]
    err = rng.normal(0.5, 1.0, size=S)          # wrong critic: biased + noisy
    tel, bias0, idres = [], [], []
    for _ in range(n_ep):
        s = [int(rng.integers(S))]
        for _t in range(H - 1):
            s.append(int(rng.choice(S, p=P[s[-1]])))
        s = np.array(s); rr = r[s]
        hrem = H - np.arange(H)                  # steps remaining incl. current
        V = Vt[hrem, s] + err[s]
        Vn = np.append(V[1:], 0.0)
        Vtn = np.append(Vt[hrem[1:], s[1:]], 0.0)
        delta = rr + gamma * Vn - V
        G = np.array([(rr[t:] * gamma ** np.arange(H - t)).sum() for t in range(H)])
        lam1 = np.array([(delta[t:] * gamma ** np.arange(H - t)).sum() for t in range(H)])
        tel.append(np.abs(lam1 - (G - V)).max())
        b = delta - (rr + gamma * Vtn - V)
        bias0.append(b.mean())
        en = np.append(err[s[1:]], 0.0)
        idres.append(np.abs(b - gamma * en).max())
    return {"seed": seed, "tel_max": float(max(tel)), "td0_bias_mean": float(np.mean(bias0)),
            "td0_bias_identity_max_resid": float(max(idres)),
            "gamma_times_mean_critic_err": float(gamma * err.mean())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--T", type=int, default=8000)
    ap.add_argument("--Ds", type=int, nargs="+", default=[0, 64, 512])
    ap.add_argument("--eta-scale", action="store_true", help="exploratory: eta = 0.05*16/(16+D)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sha = hashlib.sha256(open(os.path.join(SUB, "streams_ext.py"), "rb").read()).hexdigest()[:16]
    res = {"streams_ext_sha256_16": sha, "runs": [], "mrp": []}
    for sd in a.seeds:
        for D in a.Ds:
            for sh in (False, True):
                t1 = time.time()
                o = run(sd, D, a.T, shuffle=sh, eta_scale=a.eta_scale)
                o["wall_s"] = round(time.time() - t1, 1)
                res["runs"].append(o)
                print(f"seed={sd} D={D} shuf={sh} tel={o['tel_tan_max']:.1e}/{o['tel_ex_max']:.1e} "
                      f"normrel={o['norm_rel_err']:.1e} stale={o['stale_ratio']:.3f} "
                      f"gap(periodic)={o['fam']['periodic']['gap_mean']:.4f} gap(prng)={o['fam']['prng']['gap_mean']:.4f} "
                      f"{o['wall_s']}s", flush=True)
        res["mrp"].append(mrp_check(sd))
        print("mrp", res["mrp"][-1], flush=True)
    with open(a.out or os.path.join(HERE, "results", "delay.json"), "w") as fh:
        json.dump(res, fh, indent=1)
