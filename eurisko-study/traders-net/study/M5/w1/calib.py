"""M5 worker 1: calibration-bin shadow traders against one market. usage: calib.py SOURCE SEED
SOURCE: P4-net | P4-logistic | P16-net | P16-logistic | wn-ckpt<step>
Writes results/<source>-s<seed>.json and results/<source>-s<seed>-curve.csv (PREREG.md)."""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import exploit_core as ec

NB, DELTAS, EPS, LAG, NULLS = 10, (0.1, 0.3, 1.0), 1e-3, 16, 3
CENT = 0.05 + 0.1 * np.arange(NB)


def kernels(p):
    """Hat kernels on bin centres, edge bins clamped: a partition of unity on [0,1]."""
    K = np.clip(1.0 - np.abs(p[None] - CENT.reshape(-1, *([1] * p.ndim))) / 0.1, 0.0, 1.0)
    K[0][p < CENT[0]] = 1.0
    K[-1][p > CENT[-1]] = 1.0
    return K


def traders():
    return [(b, s, dl) for b in range(NB) for s in (1, -1) for dl in DELTAS]


def cal_online(z, y, fam, nf, per_fam):
    """Prequential per-bin (per family x bin) logit correction; uses labels of blocks < current."""
    T, Q = z.shape
    G = fam if per_fam else np.zeros(Q, int)
    ng = nf if per_fam else 1
    my, mp, n = np.zeros((ng, NB)), np.zeros((ng, NB)), np.zeros((ng, NB))
    d = np.zeros_like(z)
    lg = lambda x: np.log(x) - np.log1p(-x)
    for lo in range(0, T, LAG):
        zz, yy = z[lo:lo + LAG], y[lo:lo + LAG]
        p = 1.0 / (1.0 + np.exp(-zz))
        K = kernels(p)                                        # [NB, L, Q]
        c = np.clip(lg((my + 0.5) / (n + 1)) - lg((mp + 0.5) / (n + 1)), -2, 2)  # [ng, NB]
        d[lo:lo + LAG] = np.einsum("blq,qb->lq", K, c[G])
        for g in range(ng):
            cols = G == g
            Kg = K[:, :, cols]
            my[g] += (Kg * yy[:, cols]).sum((1, 2)); mp[g] += (Kg * p[:, cols]).sum((1, 2)); n[g] += Kg.sum((1, 2))
    return d


def ece(z, y, nb=10):
    p = 1.0 / (1.0 + np.exp(-z)).ravel(); yy = y.ravel()
    b = np.minimum((p * nb).astype(int), nb - 1)
    tot = 0.0
    for i in range(nb):
        m = b == i
        if m.any():
            tot += m.sum() * abs(yy[m].mean() - p[m].mean())
    return float(tot / len(p))


def analyse(M, y, rng_null=None):
    """Everything for one label matrix y (real or null). Returns dict + sampled curves."""
    z, fam = M["z"], M["fam"]
    T, Q = z.shape; nf = int(fam.max()) + 1; h = T // 2
    p = 1.0 / (1.0 + np.exp(-z))
    K = kernels(p)
    idx = np.unique(np.linspace(0, T - 1, 400).astype(int))
    out, curves = {}, {}
    E_all, fam_final, first_half, fam_first = [], [], [], []
    for (b, s, dl) in traders():
        d = s * dl * K[b]
        g = ec.gain_matrix(z, d, y)
        pos = g.sum(1)
        E = np.cumsum(pos)
        E_all.append(E[idx]); first_half.append(E[h - 1])
        Gf = np.stack([g[:, fam == f].sum(1) for f in range(nf)], 1)  # [T, nf]
        Ef = np.cumsum(Gf, 0)
        fam_final.append(Ef[-1]); fam_first.append(Ef[h - 1])
        curves.setdefault("fam", []).append(Ef[idx])
    E_all = np.array(E_all)                     # [60, n_idx]
    fam_final = np.array(fam_final); fam_first = np.array(fam_first)
    Fc = np.array(curves["fam"])                # [60, n_idx, nf]
    EC = E_all.max(0); ECf = Fc.reshape(len(E_all), len(idx), nf).max(axis=(0, 2))
    out["C_cal"] = dict(ec_stats(idx, EC), argmax=list(traders()[int(E_all[:, -1].argmax())]))
    out["C_cal_fam"] = ec_stats(idx, ECf)
    # held-out selection: best on the first half, invasion fitness on the second half
    tb = traders()[int(np.argmax(first_half))]
    d_best = tb[1] * tb[2] * K[tb[0]]
    out["best_first_half"] = dict(trader=list(tb), phi_eps_2nd_half=ec.invasion(z, d_best, y, EPS, h),
                                  E_2nd_half=float(ec.gain_matrix(z[h:], d_best[h:], y[h:]).sum()))
    ti, fi = np.unravel_index(np.argmax(fam_first), fam_first.shape)
    tb = traders()[ti]
    d_f = np.where(fam[None, :] == fi, tb[1] * tb[2] * K[tb[0]], 0.0)
    out["best_fam_first_half"] = dict(trader=list(tb), family=M["fam_names"][fi],
                                      phi_eps_2nd_half=ec.invasion(z, d_f, y, EPS, h),
                                      E_2nd_half=float(ec.gain_matrix(z[h:], d_f[h:], y[h:]).sum()))
    ib = int(E_all[:, -1].argmax())
    out["best_final_by_family"] = dict(zip(M["fam_names"], [float(x) for x in fam_final[ib]]))
    for pf, key in ((False, "cal_online"), (True, "cal_online_fam")):
        d = cal_online(z, y, fam, nf, pf)
        g = ec.gain_matrix(z, d, y)
        E = np.cumsum(g.sum(1))
        out[key] = dict(ec_stats(idx, E[idx]), phi_eps_2nd_half=ec.invasion(z, d, y, EPS, h),
                        by_family={M["fam_names"][f]: float(g[:, fam == f].sum()) for f in range(nf)})
        curves[key] = E[idx]
    # soup-comparable c50 one-share trader
    d50 = np.sign(0.5 - p)
    pnl = ec.one_share_pnl(z, np.where(d50 == 0, np.nan, d50), y)
    out["c50_one_share_per_qtrade_2nd_half"] = float(pnl[h:].sum() / max(1, (d50[h:] != 0).sum()))
    out["c50_one_share_per_qtrade_all"] = float(pnl.sum() / max(1, (d50 != 0).sum()))
    curves["idx"] = idx; curves["EC"] = EC; curves["ECf"] = ECf
    return out, curves


def ec_stats(idx, E):
    from panel import ec_slopes
    r = ec_slopes(idx + 1, E)
    r["final"] = float(E[-1])
    return r


def main():
    src, seed = sys.argv[1], int(sys.argv[2])
    t0 = time.time()
    if src.startswith("wn-ckpt"):
        M = ec.load_wealthnet_ckpt(seed, int(src[len("wn-ckpt"):]))
    else:
        arm, kind = src.split("-")
        M = ec.load_prequential(arm, seed, "z" if kind == "net" else "zb")
    z, y = M["z"], M["y"]
    T, Q = z.shape
    res = dict(source=src, seed=seed, T=int(T), Q=int(Q), families=M["fam_names"],
               bce=float(-ec.log_sigmoid(np.where(y > 0, 1, -1) * z).mean()),
               bce_2nd_half=float(-ec.log_sigmoid(np.where(y[T // 2:] > 0, 1, -1) * z[T // 2:]).mean()),
               ece_2nd_half=ece(z[T // 2:], y[T // 2:]))
    res["real"], cv = analyse(M, y)
    rng = np.random.default_rng(1000 + seed)
    res["null"] = []
    ncurves = []
    for k in range(NULLS):
        o, c = analyse(M, ec.null_labels(z, rng))
        res["null"].append(dict(C_cal_final=o["C_cal"]["final"], C_cal_fam_final=o["C_cal_fam"]["final"],
                                C_cal_loglog=o["C_cal"]["loglog"], cal_online_final=o["cal_online"]["final"],
                                cal_online_fam_final=o["cal_online_fam"]["final"],
                                cal_online_phi=o["cal_online"]["phi_eps_2nd_half"],
                                best_first_half_E2=o["best_first_half"]["E_2nd_half"]))
        ncurves.append(c["EC"])
    res["wall_s"] = time.time() - t0
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", f"{src}-s{seed}.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(HERE, "results", f"{src}-s{seed}-curve.csv"), "w") as f:
        f.write("t,EC_cal,EC_cal_fam,cal_online,cal_online_fam," + ",".join(f"EC_cal_null{k}" for k in range(NULLS)) + "\n")
        for i, t in enumerate(cv["idx"]):
            f.write(f"{t + 1},{cv['EC'][i]:.4f},{cv['ECf'][i]:.4f},{cv['cal_online'][i]:.4f},{cv['cal_online_fam'][i]:.4f},"
                    + ",".join(f"{c[i]:.4f}" for c in ncurves) + "\n")
    print(src, seed, "done", round(res["wall_s"], 1), "s")


if __name__ == "__main__":
    main()
