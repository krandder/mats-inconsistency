"""P9b evaluation.

  p9b_eval.py prep                      build every eval set into --cache (run once, before the eval jobs)
  p9b_eval.py gate --arm A --seed s     dev 9x9 (T=20) per-cell acc + exact match; dev 25x25 exact match to T=100
  p9b_eval.py test --arm A --seed s     test 9/15/21/33 (500/1000/1000/1000 mazes) exact-match curve to T=150;
                                        200 of the 33x33 mazes to T=300; d(S,G) terciles; Wilson CIs
  p9b_eval.py imp  --arm A --seed s     (binary-state arms) importance estimators BS/LW/FCI/AtP/H0 vs BFS
                                        'c walled' truth at 9x9 (200 mazes, T=20) and 15x15 (60 mazes, T=60);
                                        IIA; mediation; channel allocation at 15x15/33x33 (200 mazes, T=150)

Definitions (spec P9b):
  truth for an open cell c (not S or G): wall c and recompute the onpath mask; must-pass = d(S,G) grows or S,G
    disconnect; graded = Hamming change of the mask (c itself included; graded_x excludes c).
  BS(c, t_c)   = max_k mean_c' |p(y_c' | do z_k[c]=1 from t_c) - p(y_c' | do z_k[c]=0 from t_c)|
  LW(c)        = mean_c' |p(. | all 8 channels at c held from t=1 at the maze's modal wall-cell code) - p_nat|
                 (modal 8-bit code of z_T over the maze's wall cells)
  FCI(c)       = mean over 4 source mazes of mean_c' |p(. | z[:, c] := source z_t[:, c] each step) - p_nat|
  AtP(c, t_c)  = max_k |sum_{t>=t_c} dm/dz_t[k,c]| / n^2, m = sum_c' sign(p_nat - .5) p_c' (linearised do(1)-do(0)
                 through the straight-through surrogate; a weak baseline)
  H0(c)        = number of open 4-connected runs in the 8-cell ring around c
  IIA          = intervened prediction vs the BFS 'c walled' mask, on onpath cells: changed-cell accuracy and
                 whole-maze match; interventions LW, best (channel, value) held from t=1 chosen on the other
                 half of the mazes (2-fold cross-fit), a random (channel, value), and no intervention.
  mediation    = direct effect (only z_T[k,c] flipped under the readout; every other site natural) / total
                 effect (BS_k at t_c=1), per channel.
  allocation   = rank channels by mean BS_k (9x9 onpath cells), by mean AtP_k, and at random (5 draws); hold the
                 top-2 / bottom-2 channels on the whole grid at their natural mode from t=1; exact-match drop.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import p9util as U
from p9b_data import make_onpath
from p9b_train import build
from p9models import BinNet

HERE = os.path.dirname(os.path.abspath(__file__))
SETS = {"dev9": (500, 9, 50_009), "dev25": (500, 25, 50_025),
        "test9": (500, 9, 60_009), "test15": (1000, 15, 60_015), "test21": (1000, 21, 60_021),
        "test33": (1000, 33, 60_033), "imp9": (200, 9, 70_009), "imp15": (60, 15, 70_015)}
if os.environ.get("P9_SMOKE"):  # tiny sets for smoke tests only (separate cache dir)
    SETS = {k: (min(v[0], 12), v[1], v[2]) for k, v in SETS.items()}


def get_set(name, cache):
    p = os.path.join(cache, f"p9b_{name}.pt")
    if os.path.exists(p):
        return torch.load(p, weights_only=False)
    num, n, seed = SETS[name]
    ds = make_onpath(num, n, seed)
    os.makedirs(cache, exist_ok=True)
    tmp = f"{p}.{os.getpid()}.tmp"
    torch.save(ds, tmp)
    os.replace(tmp, p)
    return ds


def load_net(out, arm, seed, dev):
    net = build(arm)
    net.load_state_dict(torch.load(os.path.join(out, "train", f"{arm}_s{seed}.pt"), map_location="cpu"))
    return net.to(dev).eval()


# ------------------------------------------------------------------ rollouts
def rollout(net, x, T, hook=None, cb=None, rec=False):
    """Run T steps. hook(t, z) -> z applies clamps (binary nets). cb(t, logits) is called every step if given.
    Returns final logits and (if rec) the trajectory list of z_t (binary nets)."""
    traj = []
    if isinstance(net, BinNet):
        e, z = net.init(x)
        for t in range(1, T + 1):
            z = net.step(z, e)
            if hook is not None:
                z = hook(t, z)
            if rec:
                traj.append(z)
            if cb is not None:
                cb(t, net.readout(z))
        return net.readout(z), traj
    h = net.init(x)
    for t in range(1, T + 1):
        h = net.step(h, x)
        if cb is not None:
            cb(t, net.readout(h))
    return net.readout(h), traj


def em_curve(net, ds, T, dev, idx=None, chunk=200):
    """Per-maze exact match at every t (N, T) bool and per-cell accuracy at every t (N, T) float."""
    X, Y = ds["x"], ds["y"]
    if idx is not None:
        X, Y = X[idx], Y[idx]
    ems, accs = [], []
    with torch.no_grad():
        for a in range(0, X.shape[0], chunk):
            x, y = X[a:a + chunk].to(dev), Y[a:a + chunk].to(dev) > 0.5
            em = torch.zeros(x.shape[0], T, dtype=torch.bool, device=dev)
            acc = torch.zeros(x.shape[0], T, device=dev)

            def cb(t, lg):
                ok = (lg > 0) == y
                em[:, t - 1] = ok.all((1, 2))
                acc[:, t - 1] = ok.float().mean((1, 2))

            rollout(net, x, T, cb=cb)
            ems.append(em.cpu()); accs.append(acc.cpu())
    return torch.cat(ems).numpy(), torch.cat(accs).numpy()


def summarize(em, acc, dsg, Ts):
    """Headline numbers at the given Ts, overall and by d(S,G) tercile, with Wilson CIs."""
    q1, q2 = np.quantile(dsg, [1 / 3, 2 / 3])
    terc = np.where(dsg <= q1, 0, np.where(dsg <= q2, 1, 2))
    out = {"n": int(len(dsg)), "tercile_edges": [float(q1), float(q2)],
           "tercile_n": [int((terc == k).sum()) for k in range(3)],
           "em_curve": [round(float(v), 4) for v in em.mean(0)],
           "em_curve_tercile": [[round(float(v), 4) for v in em[terc == k].mean(0)] for k in range(3)]}
    for T in Ts:
        e = em[:, T - 1]
        ci = U.binom_ci(int(e.sum()), len(e))
        out[f"T{T}"] = {"em": float(e.mean()), "em_ci95": [ci["lo"], ci["hi"]], "cell_acc": float(acc[:, T - 1].mean()),
                        "em_tercile": [float(e[terc == k].mean()) for k in range(3)]}
    return out


def mode_gate(args, dev):
    net = load_net(args.out, args.arm, args.seed, dev)
    res = {"arm": args.arm, "seed": args.seed, "mode": "gate"}
    d9 = get_set("dev9", args.cache)
    em, acc = em_curve(net, d9, 40, dev)
    res["dev9"] = summarize(em, acc, d9["dsg"], [20, 40])
    d25 = get_set("dev25", args.cache)
    em, acc = em_curve(net, d25, 100, dev, chunk=100)
    res["dev25"] = summarize(em, acc, d25["dsg"], [50, 100])
    return res


def mode_test(args, dev):
    net = load_net(args.out, args.arm, args.seed, dev)
    res = {"arm": args.arm, "seed": args.seed, "mode": "test"}
    for n in (9, 15, 21, 33):
        ds = get_set(f"test{n}", args.cache)
        em, acc = em_curve(net, ds, 150, dev, chunk=250 if n < 33 else 100)
        res[f"test{n}"] = summarize(em, acc, ds["dsg"], [20, 60, 100, 150] if n < 33 else [60, 100, 150])
    ds = get_set("test33", args.cache)
    idx = np.arange(200)
    em, acc = em_curve(net, ds, 300, dev, idx=idx, chunk=100)
    res["test33_T300_200mazes"] = summarize(em, acc, ds["dsg"][idx], [150, 200, 300])
    return res


# ------------------------------------------------------------------ importance
def ring_components(walls):
    """H0 per cell: number of open 4-connected runs in the 8-cell ring (out-of-grid = wall). walls (n,n) bool."""
    n = walls.shape[0]
    pad = np.ones((n + 2, n + 2), bool)
    pad[1:-1, 1:-1] = walls
    ring = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    H = np.zeros((n, n), int)
    for i in range(n):
        for j in range(n):
            o = [not pad[i + 1 + di, j + 1 + dj] for di, dj in ring]
            if all(o):
                H[i, j] = 1
                continue
            # runs of open cells in the cyclic sequence (ring neighbours in cyclic order are 4-adjacent)
            H[i, j] = sum(1 for k in range(8) if o[k] and not o[k - 1])
    return H


def clamp_hook(M, V, tc):
    """M (R,K,n,n) bool; V (R,K,n,n) float, or (T,R,K,n,n) time-varying; tc int or (R,) long."""
    def hook(t, z):
        v = V[t - 1] if V.dim() == 5 else V
        if isinstance(tc, int):
            return torch.where(M, v, z) if t >= tc else z
        m = M & (t >= tc)[:, None, None, None]
        return torch.where(m, v, z)
    return hook


def imp_set(net, ds, T, dev, rng, max_rows=4096):
    """Per-cell records for one eval set."""
    X, Y = ds["x"], ds["y"]
    N, n = X.shape[0], ds["n"]
    K = net.K
    tcs = [1, max(1, T // 4), max(1, T // 2)]
    rec = {k: [] for k in ("maze", "ci", "cj", "onpath", "must", "graded", "graded_x", "H0", "LW", "FCI",
                           "BS", "AtP", "DE", "iia_kv_wm", "iia_kv_cca", "iia_lw_wm", "iia_lw_cca",
                           "iia_nat_wm", "iia_nat_cca")}
    zT_all, nat = [], {}
    with torch.no_grad():  # natural trajectories for all mazes (FCI sources)
        for a in range(0, N, 100):
            _, tr = rollout(net, X[a:a + 100].to(dev), T, rec=True)
            nat[a] = torch.stack(tr).cpu()  # (T,B,K,n,n)
    natZ = torch.cat([nat[a] for a in sorted(nat)], 1)  # (T,N,K,n,n)
    for i in range(N):
        x = X[i:i + 1].to(dev)
        walls = X[i, 0] > 0.5
        S, G = X[i, 1] > 0.5, X[i, 2] > 0.5
        open_ = ~walls
        y = Y[i] > 0.5
        mask, _, _, dsg = U.onpath_mask(open_[None], S[None], G[None])
        assert torch.equal(mask[0], y)
        onp = (y & ~S & ~G).nonzero().tolist()
        offc = (open_ & ~y & ~S & ~G).nonzero().tolist()
        if len(offc) > 10:
            offc = [offc[j] for j in rng.choice(len(offc), 10, replace=False)]
        cells = onp + offc
        C = len(cells)
        if C == 0:
            continue
        ci = torch.tensor([c[0] for c in cells]); cj = torch.tensor([c[1] for c in cells])
        # truth
        oc = open_[None].repeat(C, 1, 1)
        oc[torch.arange(C), ci, cj] = False
        mc, _, _, dsgc = U.onpath_mask(oc, S[None].repeat(C, 1, 1), G[None].repeat(C, 1, 1))
        diff = mc != y[None]
        graded = diff.sum((1, 2))
        dx = diff.clone(); dx[torch.arange(C), ci, cj] = False
        must = dsgc > dsg
        H0 = ring_components(walls.numpy())
        Z = natZ[:, i].to(dev)  # (T,K,n,n)
        with torch.no_grad():
            p_nat = torch.sigmoid(net.readout(Z[-1:]))[0]
        e1, _ = net.init(x)
        s = torch.sign(p_nat - 0.5)
        # --- BS: rows (c, k, v, tc)
        R = C * K * 2 * len(tcs)
        M = torch.zeros(R, K, n, n, dtype=torch.bool, device=dev)
        V = torch.zeros(R, K, n, n, device=dev)
        TC = torch.zeros(R, dtype=torch.long, device=dev)
        r = 0
        for c in range(C):
            for k in range(K):
                for v in (0, 1):
                    for tci in tcs:
                        M[r, k, ci[c], cj[c]] = True; V[r, k, ci[c], cj[c]] = v; TC[r] = tci; r += 1
        P = []
        with torch.no_grad():
            for a in range(0, R, max_rows):
                b = slice(a, min(R, a + max_rows))
                lg, _ = rollout(net, x.expand(b.stop - a, -1, -1, -1), T, hook=clamp_hook(M[b], V[b], TC[b]))
                P.append(torch.sigmoid(lg))
        P = torch.cat(P).view(C, K, 2, len(tcs), n, n)
        BS = (P[:, :, 1] - P[:, :, 0]).abs().mean((-1, -2))  # (C,K,3)
        pred_kv = P[:, :, :, 0] > 0.5  # t_c = 1: (C,K,2,n,n)
        mcd = mc.to(dev)
        chg = diff.to(dev)
        wm_kv = (pred_kv == mcd[:, None, None]).all(-1).all(-1)
        cca_kv = ((pred_kv == mcd[:, None, None]) & chg[:, None, None]).sum((-1, -2)).float() / \
            chg.sum((1, 2)).clamp(min=1)[:, None, None].float()
        # --- LW: modal wall-cell code of z_T
        zw = Z[-1][:, walls.to(dev)]  # (K, nwall)
        if zw.shape[1] > 0:
            codes = ((zw > 0.5).long().T * (2 ** torch.arange(K, device=dev))).sum(1)  # no long matmul on CUDA
            code = int(torch.mode(codes).values)
        else:
            code = 0
        bits = torch.tensor([(code >> k) & 1 for k in range(K)], dtype=torch.float32, device=dev)
        M2 = torch.zeros(C, K, n, n, dtype=torch.bool, device=dev)
        V2 = torch.zeros(C, K, n, n, device=dev)
        for c in range(C):
            M2[c, :, ci[c], cj[c]] = True; V2[c, :, ci[c], cj[c]] = bits
        with torch.no_grad():
            lg, _ = rollout(net, x.expand(C, -1, -1, -1), T, hook=clamp_hook(M2, V2, 1))
        p_lw = torch.sigmoid(lg)
        LW = (p_lw - p_nat).abs().mean((1, 2))
        pl = p_lw > 0.5
        lw_wm = (pl == mcd).all(-1).all(-1)
        lw_cca = ((pl == mcd) & chg).sum((1, 2)).float() / chg.sum((1, 2)).clamp(min=1).float()
        pn = (p_nat > 0.5)[None]
        nat_wm = (pn == mcd).all(-1).all(-1)
        nat_cca = ((pn == mcd) & chg).sum((1, 2)).float() / chg.sum((1, 2)).clamp(min=1).float()
        # --- FCI: 4 sources (the next 4 mazes of the set), time-varying full-cell patch
        src = [(i + d) % N for d in range(1, 5)]
        M3 = torch.zeros(4 * C, K, n, n, dtype=torch.bool, device=dev)
        V3 = torch.zeros(T, 4 * C, K, n, n, device=dev)
        for a, j in enumerate(src):
            Zs = natZ[:, j].to(dev)
            for c in range(C):
                rr = a * C + c
                M3[rr, :, ci[c], cj[c]] = True
                V3[:, rr, :, ci[c], cj[c]] = Zs[:, :, ci[c], cj[c]]
        with torch.no_grad():
            lg, _ = rollout(net, x.expand(4 * C, -1, -1, -1), T, hook=clamp_hook(M3, V3, 1))
        FCI = (torch.sigmoid(lg) - p_nat).abs().mean((1, 2)).view(4, C).mean(0)
        # --- AtP through the ST surrogate
        with torch.enable_grad():
            e, z = net.init(x)
            zs = []
            for t in range(1, T + 1):
                z = net.step(z, e)
                z.retain_grad()
                zs.append(z)
            m = (s * torch.sigmoid(net.readout(z))[0]).sum()
            m.backward()
        g = torch.stack([zz.grad[0] for zz in zs])  # (T,K,n,n)
        gc = g[:, :, ci.to(dev), cj.to(dev)]  # (T,K,C)
        AtP = torch.stack([gc[tci - 1:].sum(0).abs().T / (n * n) for tci in tcs], -1)  # (C,K,3)
        # --- mediation: direct effect of flipping only z_T[k,c] under the readout
        zT = Z[-1].detach()
        R4 = C * K * 2
        Zm = zT[None].repeat(R4, 1, 1, 1)
        r = 0
        for c in range(C):
            for k in range(K):
                for v in (0, 1):
                    Zm[r, k, ci[c], cj[c]] = v; r += 1
        with torch.no_grad():
            pm = torch.sigmoid(net.readout(Zm)).view(C, K, 2, n, n)
        DE = (pm[:, :, 1] - pm[:, :, 0]).abs().mean((-1, -2))  # (C,K)
        del e1
        for c in range(C):
            rec["maze"].append(i); rec["ci"].append(int(ci[c])); rec["cj"].append(int(cj[c]))
            rec["onpath"].append(c < len(onp)); rec["must"].append(bool(must[c]))
            rec["graded"].append(int(graded[c])); rec["graded_x"].append(int(dx[c].sum()))
            rec["H0"].append(int(H0[ci[c], cj[c]]))
        rec["LW"].append(LW.cpu()); rec["FCI"].append(FCI.cpu()); rec["BS"].append(BS.cpu())
        rec["AtP"].append(AtP.detach().cpu()); rec["DE"].append(DE.cpu())
        rec["iia_kv_wm"].append(wm_kv.cpu()); rec["iia_kv_cca"].append(cca_kv.cpu())
        rec["iia_lw_wm"].append(lw_wm.cpu()); rec["iia_lw_cca"].append(lw_cca.cpu())
        rec["iia_nat_wm"].append(nat_wm.expand(C).cpu()); rec["iia_nat_cca"].append(nat_cca.cpu())
    out = {}
    for k, v in rec.items():
        if v and torch.is_tensor(v[0]):
            out[k] = torch.cat(v).numpy()
        else:
            out[k] = np.array(v)
    # channel modes over natural runs (for allocation)
    out["channel_mode"] = (natZ.float().mean((0, 1, 3, 4)) > 0.5).float().numpy()
    out["channel_on_rate"] = natZ.float().mean((0, 1, 3, 4)).numpy()
    return out


def imp_summary(r, rng):
    on = r["onpath"].astype(bool)
    res = {"n_cells": int(len(on)), "n_onpath": int(on.sum()), "n_offpath": int((~on).sum()),
           "must_rate_onpath": float(r["must"][on].mean()),
           "offpath_truth_zero_rate": float((r["graded"][~on] == 0).mean()) if (~on).any() else None,
           "offpath_must_rate": float(r["must"][~on].mean()) if (~on).any() else None}
    est = {"BS": r["BS"][:, :, 0].max(1), "BS_T4": r["BS"][:, :, 1].max(1), "BS_T2": r["BS"][:, :, 2].max(1),
           "LW": r["LW"], "FCI": r["FCI"], "AtP": r["AtP"][:, :, 0].max(1), "AtP_T4": r["AtP"][:, :, 1].max(1),
           "AtP_T2": r["AtP"][:, :, 2].max(1), "H0": r["H0"].astype(float)}
    # per-onpath-cell scores at full precision for the maze-then-seed bootstrap (the npz is float16)
    res["cells_onpath"] = {"maze": r["maze"][on].astype(int).tolist(), "must": r["must"][on].astype(int).tolist(),
                           **{k: [float("%.6g" % x) for x in np.asarray(est[k])[on]] for k in ("BS", "AtP", "FCI", "H0", "LW")}}
    res["estimators"] = {}
    for k, v in est.items():
        res["estimators"][k] = {
            "auroc_must_onpath": U.auroc(v[on], r["must"][on]),
            "spearman_graded_onpath": U.spearman(v[on], r["graded"][on]),
            "mean_onpath": float(np.mean(v[on])), "mean_offpath": float(np.mean(v[~on])) if (~on).any() else None,
            "offpath_zero_rate": float(np.mean(v[~on] < 1e-6)) if (~on).any() else None}
    # IIA on onpath cells (every onpath cell is an effective intervention: walling it changes the mask)
    eff = on & (r["graded"] > 0)
    mz = r["maze"]
    halves = [np.isin(mz, np.unique(mz)[0::2]), np.isin(mz, np.unique(mz)[1::2])]
    K = r["iia_kv_wm"].shape[1]
    best_wm, best_cca, choice = np.zeros(len(on)), np.zeros(len(on)), []
    for h in range(2):
        fit, app = eff & halves[1 - h], eff & halves[h]
        sc = r["iia_kv_cca"][fit].mean(0)  # (K,2)
        k, v = np.unravel_index(int(np.argmax(sc)), sc.shape)
        choice.append([int(k), int(v)])
        best_wm[app] = r["iia_kv_wm"][app, k, v]; best_cca[app] = r["iia_kv_cca"][app, k, v]
    rk = rng.integers(0, K, len(on)); rv = rng.integers(0, 2, len(on))
    rnd_wm = r["iia_kv_wm"][np.arange(len(on)), rk, rv]; rnd_cca = r["iia_kv_cca"][np.arange(len(on)), rk, rv]
    res["iia"] = {"n_effective": int(eff.sum()), "best_choice_per_half": choice}
    for name, wm, cca in (("LW", r["iia_lw_wm"], r["iia_lw_cca"]), ("best_kv_crossfit", best_wm, best_cca),
                          ("random_kv", rnd_wm, rnd_cca), ("none", r["iia_nat_wm"], r["iia_nat_cca"])):
        res["iia"][name] = {"changed_cell_acc": float(np.mean(cca[eff])), "whole_maze": float(np.mean(wm[eff]))}
    # mediation per channel (onpath cells)
    te = r["BS"][on, :, 0].mean(0); de = r["DE"][on].mean(0)
    res["mediation"] = {"total_effect": te.round(5).tolist(), "direct_effect": de.round(5).tolist(),
                        "direct_over_total": (de / np.maximum(te, 1e-9)).round(4).tolist()}
    res["channel_BS_mean_onpath"] = r["BS"][on, :, 0].mean(0).round(6).tolist()
    res["channel_AtP_mean_onpath"] = r["AtP"][on, :, 0].mean(0).round(6).tolist()
    res["channel_on_rate"] = np.round(r["channel_on_rate"], 4).tolist()
    return res


def allocation(net, rank_bs, rank_atp, modes, args, dev, rng):
    K = net.K
    rand_pairs = [sorted(rng.choice(K, 2, replace=False).tolist()) for _ in range(5)]
    groups = {"pay_top2": rank_bs[:2], "pay_bottom2": rank_bs[-2:], "atp_top2": rank_atp[:2],
              "atp_bottom2": rank_atp[-2:]}
    for j, pr in enumerate(rand_pairs):
        groups[f"random2_{j}"] = pr
    out = {"groups": {k: [int(a) for a in v] for k, v in groups.items()}, "modes": [int(m) for m in modes]}
    for n in (15, 33):
        ds = get_set(f"test{n}", args.cache)
        X, Y = ds["x"][:200], ds["y"][:200]
        res = {}
        for gname, chans in [("none", [])] + list(groups.items()):
            ems = []
            with torch.no_grad():
                for a in range(0, 200, 100):
                    x, y = X[a:a + 100].to(dev), Y[a:a + 100].to(dev) > 0.5
                    B = x.shape[0]
                    M = torch.zeros(B, K, n, n, dtype=torch.bool, device=dev)
                    V = torch.zeros(B, K, n, n, device=dev)
                    for k in chans:
                        M[:, k] = True; V[:, k] = float(modes[k])
                    lg, _ = rollout(net, x, 150, hook=clamp_hook(M, V, 1) if chans else None)
                    ems.append(((lg > 0) == y).all(-1).all(-1).float().cpu())
            res[gname] = torch.cat(ems).numpy()
        base = res["none"]
        out[f"test{n}_T150_200mazes"] = {
            "em_none": float(base.mean()),
            "drop": {g: float(base.mean() - v.mean()) for g, v in res.items() if g != "none"},
            "drop_random2_mean": float(np.mean([base.mean() - res[f"random2_{j}"].mean() for j in range(5)]))}
    return out


def mode_imp(args, dev):
    net = load_net(args.out, args.arm, args.seed, dev)
    assert isinstance(net, BinNet), "importance estimators need the binary-state backbone"
    rng = np.random.default_rng(7000 + args.seed)
    res = {"arm": args.arm, "seed": args.seed, "mode": "imp"}
    npz = {}
    for name, T in (("imp9", 20), ("imp15", 60)):
        ds = get_set(name, args.cache)
        if args.limit:
            ds = {"x": ds["x"][:args.limit], "y": ds["y"][:args.limit], "dsg": ds["dsg"][:args.limit], "n": ds["n"]}
        t0 = time.time()
        r = imp_set(net, ds, T, dev, rng)
        res[name] = imp_summary(r, rng)
        res[name]["T"] = T
        res[name]["wall_s"] = round(time.time() - t0, 1)
        for k, v in r.items():
            npz[f"{name}_{k}"] = v.astype(np.float16) if v.dtype.kind == "f" else v
        print(name, json.dumps({k: round(v["auroc_must_onpath"], 3) for k, v in res[name]["estimators"].items()}),
              flush=True)
    bs = np.array(res["imp9"]["channel_BS_mean_onpath"]); atp = np.array(res["imp9"]["channel_AtP_mean_onpath"])
    modes = npz["imp9_channel_mode"]
    res["allocation"] = allocation(net, list(np.argsort(-bs)), list(np.argsort(-atp)), modes, args, dev, rng)
    os.makedirs(os.path.join(args.out, "eval"), exist_ok=True)
    np.savez_compressed(os.path.join(args.out, "eval", f"imp_{args.arm}_s{args.seed}.npz"), **npz)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "gate", "test", "imp"])
    ap.add_argument("--arm", default="N1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--cache", default="/tmp/p9/cache")
    ap.add_argument("--limit", type=int, default=0, help="imp: only the first N mazes (smoke tests)")
    ap.add_argument("--sets", default="", help="prep: comma list (default all)")
    args = ap.parse_args()
    torch.set_num_threads(1)
    dev = U.device()
    t0 = time.time()
    if args.mode == "prep":
        for name in (args.sets.split(",") if args.sets else SETS):
            t1 = time.time(); ds = get_set(name, args.cache)
            print(name, ds["x"].shape, "dsg median", float(np.median(ds["dsg"])), round(time.time() - t1, 1), "s",
                  flush=True)
        os.makedirs(os.path.join(args.out, "eval"), exist_ok=True)
        U.dump(os.path.join(args.out, "eval", "prep.json"), {"sets": {k: list(v) for k, v in SETS.items()}})
        return
    res = {"gate": mode_gate, "test": mode_test, "imp": mode_imp}[args.mode](args, dev)
    res["wall_s"] = round(time.time() - t0, 1)
    res["device"] = str(dev)
    os.makedirs(os.path.join(args.out, "eval"), exist_ok=True)
    U.dump(os.path.join(args.out, "eval", f"{args.mode}_{args.arm}_s{args.seed}.json"), res)
    print("DONE", args.mode, args.arm, args.seed, res["wall_s"], flush=True)


if __name__ == "__main__":
    main()
