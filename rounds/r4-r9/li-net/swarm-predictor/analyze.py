#!/usr/bin/env python3
"""Every metric of PROPOSAL.md section 5 from the synced run logs, and the pre-registered decision rule.

Inputs (runs/ next to this file, written by run_swarm.py and synced by the driver):
  runs/<ARM>-s<SEED>/eval/eval_chains.jsonl   claims of every evaluation chain (ARM in S0..S3)
  runs/<ARM>-s<SEED>/train/{steps,small_eval}.jsonl   training curves (S1..S3)
  runs/gate0-*/gate0.json                      Gate 0
Boards, truth and exact Bayes are rebuilt from the board seeds (refocus.make_board, the harness
oracle). Pools are recomputed here, never read from the runs. Learned stakes (kappa, alpha) are
cross-fitted: fit on the even boards of a cell, applied to the odd ones, and vice versa.
Writes analysis.json and analysis_tables.md next to this file.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pool import Board, Claims, fit_stakes, logit, price  # noqa: E402
from settle import settle  # noqa: E402
from refocus import make_board, open_mask  # noqa: E402
from env.shiphunt import PRESETS  # noqa: E402

TEST0 = 900_000
METHODS = ("prior", "mean", "vote", "LL-fit", "W-fit", "D-avg", "D-fit")
PRIMARY = ("main", 256, 16, 2)
RNG = np.random.default_rng(12345)


def boot_mean(x, n=2000):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return [float(x.mean()) if len(x) else float("nan"), float("nan"), float("nan")]
    m = x[RNG.integers(len(x), size=(n, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def two_level(per_seed, n=4000):
    """per_seed: list of 1-D arrays (one per seed). Resample seeds, then boards within each."""
    per_seed = [np.asarray(a, float) for a in per_seed if len(a)]
    if not per_seed:
        return [float("nan")] * 3
    point = float(np.mean([a.mean() for a in per_seed]))
    if len(per_seed) < 2:
        return boot_mean(per_seed[0], n)
    out = []
    for _ in range(n):
        ss = RNG.integers(len(per_seed), size=len(per_seed))
        out.append(np.mean([per_seed[s][RNG.integers(len(per_seed[s]), size=len(per_seed[s]))].mean() for s in ss]))
    return [point, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def ece(p, y, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    if len(p) == 0:
        return float("nan")
    o = np.argsort(p, kind="stable")
    return float(sum(len(c) * abs(p[c].mean() - y[c].mean()) for c in np.array_split(o, bins) if len(c)) / len(p))


def spearman(x, y):
    from settle import spearman as sp
    return sp(np.asarray(x, float), np.asarray(y, float))


def claims_of(recs):
    ch, st, q, w = [], [], [], []
    for r in recs:
        if r.get("c") is None:
            continue
        for j, (p, c) in zip(r["t"], r["c"]):
            ch.append(r["cid"]); st.append(j); q.append(p); w.append(c)
    return Claims(np.array(ch, int), np.array(st, int), np.array(q, float), np.array(w, float))


class Boards:
    def __init__(self, board):
        self.board, self.cache = board, {}

    def get(self, seed):
        if seed not in self.cache:
            env = make_board(self.board.cfg, seed)
            z = self.board.truth(env)
            self.cache[seed] = {"env": env, "z": z, "y": self.board.F[z].copy(),
                                "bayes": self.board.bayes(env) @ self.board.F, "open": open_mask(self.board, env)}
        return self.cache[seed]


def cell_records(recs_b, n, R):
    out = [r for r in recs_b if r["r"] == 1 and r["i"] < n]
    out += [r for r in recs_b if 2 <= r["r"] <= R and r["n"] == n]
    return out


def analyse_cell(board, B, by_board, n, R):
    """by_board: {board_seed: [records of this (kind, L)]}. Returns per-board metrics and per-chain credit."""
    seeds = sorted(by_board)
    eps = []
    for s in seeds:
        g = B.get(s)
        recs = cell_records(by_board[s], n, R)
        eps.append({"seed": s, "recs": recs, "claims": claims_of(recs), "y": g["y"], "zstar": g["z"], "open": g["open"],
                    "bayes": g["bayes"]})
    fits = {}
    for fold in (0, 1):
        tr = [e for k, e in enumerate(eps) if k % 2 != fold]
        if len(tr) >= 4 and sum(len(e["claims"].q) for e in tr) > 0:
            fits[fold] = {"LL": fit_stakes(board, tr, "stmt"), "W": fit_stakes(board, tr, "world")}
        else:
            fits[fold] = {"LL": {"kappa": 1.0, "alpha": 1.0}, "W": {"kappa": 1.0, "alpha": 1.0}}
    rows = []
    for k, e in enumerate(eps):
        f = fits[k % 2]
        o, b, y, cl = e["open"], e["bayes"], e["y"], e["claims"]
        asked = np.zeros(board.J, bool)
        for r in e["recs"]:
            asked[r["t"]] = True
        prices = {}
        for m in METHODS:
            par = f["LL"] if m in ("LL-fit", "D-fit") else f["W"] if m == "W-fit" else None
            prices[m] = price(board, cl, m, par)[0]
        row = {"seed": e["seed"], "parsed": float(np.mean([r["c"] is not None for r in e["recs"]])) if e["recs"] else float("nan"),
               "stale_share": float((o & ~asked).sum() / o.sum())}
        for m, p in prices.items():
            pc = np.clip(p[o], 1e-3, 1 - 1e-3)
            bb = np.clip(b[o], 1e-6, 1 - 1e-6)
            row[f"se_{m}"] = float(np.mean((p[o] - b[o]) ** 2))
            row[f"ll_{m}"] = float(-np.mean(y[o] * np.log(pc) + (1 - y[o]) * np.log(1 - pc)))
            row[f"kl_{m}"] = float(np.mean(bb * np.log(bb / pc) + (1 - bb) * np.log((1 - bb) / (1 - pc))))
            st = o & ~asked
            row[f"stale_se_{m}"] = float(np.mean((p[st] - b[st]) ** 2)) if st.any() else float("nan")
        row["_p"] = {m: prices[m][o] for m in ("mean", "D-fit")}
        row["_y"] = y[o]
        row["_prices_full"] = prices
        rows.append(row)
    return rows, fits


def credit(board, B, by_board, n, R, fits):
    """Settlement of every chain in the cell at the D-fit prices after its own round; oracle info."""
    out = []
    for k, s in enumerate(sorted(by_board)):
        g = B.get(s)
        f = fits[k % 2]["LL"]
        recs_all = cell_records(by_board[s], n, R)
        sp, inf, pay = [], [], []
        for rr in range(1, R + 1):
            upto = [r for r in recs_all if r["r"] <= rr]
            p_after = price(board, claims_of(upto), "D-fit", f)[0]
            for r in [r for r in recs_all if r["r"] == rr and r.get("c") is not None]:
                cl = claims_of([r])
                sres = settle(board, cl, np.ones(len(cl.q)), p_after, g["y"], n_chains=r["cid"] + 1)
                bj = g["bayes"][cl.stmt]
                p0 = board.p0_marg[cl.stmt]
                info = float(np.sum(bj * np.log(cl.q / p0) + (1 - bj) * np.log((1 - cl.q) / (1 - p0))))
                pay.append(float(sres["s"][r["cid"]])); inf.append(info)
        rho = spearman(pay, inf) if len(pay) >= 3 else float("nan")
        out.append({"seed": s, "pay": pay, "info": inf, "rho": rho})
    return out


def martingale(rows_by_R, method="D-fit"):
    """Price change of each open statement from round r to r+1 (same boards). Bins of the round-r price."""
    res = {}
    for r in (1, 2):
        if r not in rows_by_R or r + 1 not in rows_by_R:
            continue
        d_all, p_all, per_board = [], [], []
        for a, b in zip(rows_by_R[r], rows_by_R[r + 1]):
            assert a["seed"] == b["seed"]
            pa = a["_prices_full"][method]
            pb = b["_prices_full"][method]
            o = a["_open"]
            d = pb[o] - pa[o]
            d_all += list(d); p_all += list(pa[o]); per_board.append(float(d.mean()))
        d_all, p_all = np.array(d_all), np.array(p_all)
        bins = []
        for lo in (0.0, 0.2, 0.4, 0.6, 0.8):
            m = (p_all >= lo) & (p_all < lo + 0.2 + (1e-9 if lo == 0.8 else 0))
            bins.append({"bin": f"{lo:.1f}-{lo + 0.2:.1f}", "n": int(m.sum()), "mean_drift": float(d_all[m].mean()) if m.any() else float("nan"),
                         "ci": boot_mean(d_all[m])[1:] if m.sum() > 1 else [float("nan")] * 2})
        res[f"{r}->{r + 1}"] = {"mean_signed_drift_per_board": boot_mean(per_board), "bins": bins, "n_prices": int(len(d_all))}
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"))
    ap.add_argument("--out", default=os.path.join(HERE, "analysis.json"))
    ap.add_argument("--gpu-log", default="/home/agent/workspaces/eurisko-study/logical-soup/GPU-LOG.md")
    cli = ap.parse_args()
    board = Board(PRESETS["medium"])
    B = Boards(board)
    out = {"gate0": {}, "arms": {}, "train": {}}
    for f in sorted(glob.glob(os.path.join(cli.runs, "gate0-*", "gate0.json"))):
        out["gate0"][os.path.basename(os.path.dirname(f))] = json.load(open(f))
    evals = {}
    for d in sorted(glob.glob(os.path.join(cli.runs, "S[0-3]-s[0-9]", "eval"))):
        m = re.search(r"(S[0-3])-s(\d)", d)
        fp = os.path.join(d, "eval_chains.jsonl")
        if os.path.exists(fp):
            evals[(m.group(1), int(m.group(2)))] = [json.loads(l) for l in open(fp)]
    cells_rows = {}
    for (arm, seed), recs in sorted(evals.items()):
        idx = defaultdict(lambda: defaultdict(list))
        for r in recs:
            idx[(r["cell"], r["L"])][r["seed"]].append(r)
        A = out["arms"].setdefault(arm, {}).setdefault(str(seed), {"cells": {}, "parse_rate": {}})
        for (kind, L), by_board in sorted(idx.items()):
            allr = [r for rs in by_board.values() for r in rs]
            A["parse_rate"][f"{kind}-L{L}"] = float(np.mean([r["c"] is not None for r in allr]))
            if kind == "long":
                ns_R = [(1, 1)]
            else:
                n_r23 = sorted({r["n"] for r in allr if r["r"] >= 2})
                n_r1 = max(r["n"] for r in allr if r["r"] == 1)
                cand = [1, 4, 16, 64] if kind == "main" else [16]
                ns_R = [(n, 1) for n in cand if n <= n_r1] + [(n, R) for n in n_r23 for R in (2, 3)
                                                              if any(r["r"] == R and r["n"] == n for r in allr)]
            for (n, R) in ns_R:
                rows, fits = analyse_cell(board, B, by_board, n, R)
                for row in rows:
                    row["_open"] = B.get(row["seed"])["open"]
                key = f"{kind}|L{L}|n{n}|R{R}"
                cells_rows[(arm, seed, kind, L, n, R)] = rows
                c = {"boards": len(rows), "fits": {str(k): v for k, v in fits.items()},
                     "stale_share": float(np.mean([r["stale_share"] for r in rows]))}
                for mth in METHODS:
                    c[f"brier_vs_bayes_{mth}"] = float(np.mean([r[f"se_{mth}"] for r in rows]))
                    c[f"logloss_{mth}"] = float(np.mean([r[f"ll_{mth}"] for r in rows]))
                    c[f"kl_{mth}"] = float(np.mean([r[f"kl_{mth}"] for r in rows]))
                    c[f"stale_brier_{mth}"] = float(np.nanmean([r[f"stale_se_{mth}"] for r in rows])) if any(
                        not np.isnan(r[f"stale_se_{mth}"]) for r in rows) else float("nan")
                for mth in ("mean", "D-fit"):
                    c[f"ece_{mth}"] = ece(np.concatenate([r["_p"][mth] for r in rows]), np.concatenate([r["_y"] for r in rows]))
                if (kind, L, n, R) == PRIMARY or (kind == "main" and n == 16 and R == 1 and L == 256):
                    cr = credit(board, B, by_board, n, R, fits)
                    pays = np.concatenate([x["pay"] for x in cr]) if cr else np.array([])
                    c["credit"] = {"spearman_within_board": boot_mean([x["rho"] for x in cr]),
                                   "share_paid_positive": float(np.mean(pays > 0)) if len(pays) else float("nan"),
                                   "pay_quantiles_5_25_50_75_95": [float(v) for v in np.percentile(pays, [5, 25, 50, 75, 95])] if len(pays) else [],
                                   "pooled_spearman": spearman(pays, np.concatenate([x["info"] for x in cr])) if len(pays) > 2 else float("nan")}
                # diversity: per statement with >= 2 round-1 claims, distinct rounded p / claims, and sd of logit q
                if kind == "main" and R == 1 and n == 16:
                    dv, sd = [], []
                    for s, rs in by_board.items():
                        cl = claims_of([r for r in rs if r["r"] == 1 and r["i"] < 16])
                        for j in set(cl.stmt.tolist()):
                            q = cl.q[cl.stmt == j]
                            if len(q) >= 2:
                                dv.append(len(set(np.round(q, 1))) / len(q)); sd.append(float(np.std(logit(q))))
                    c["diversity"] = {"distinct_p_share": float(np.mean(dv)) if dv else float("nan"),
                                      "sd_logit_q": float(np.mean(sd)) if sd else float("nan")}
                A["cells"][key] = c
            if kind in ("main", "roundrobin"):
                for n in sorted({nn for (a_, s_, k_, L_, nn, R_) in cells_rows if (a_, s_, k_, L_) == (arm, seed, kind, L)}):
                    byR = {R: cells_rows[(arm, seed, kind, L, n, R)] for R in (1, 2, 3) if (arm, seed, kind, L, n, R) in cells_rows}
                    if len(byR) >= 2:
                        A["cells"][f"{kind}|L{L}|n{n}|R1"].setdefault("martingale", {})
                        A["cells"][f"{kind}|L{L}|n{n}|R1"]["martingale"] = {"D-fit": martingale(byR, "D-fit"), "mean": martingale(byR, "mean")}
    # ---- primary decision rule ----------------------------------------------------------------------
    kind, L, n, R = PRIMARY
    prim = {"cell": f"{kind}|L{L}|n{n}|R{R}"}
    seeds = sorted({s for (a, s, k, l, nn, rr) in cells_rows if a == "S1" and (k, l, nn, rr) == PRIMARY})
    base_opts = [(a, mth) for a in ("S0", "S1") for mth in ("mean", "vote")]
    if seeds:
        sw = {s: np.array([r["se_D-fit"] for r in cells_rows[("S1", s, kind, L, n, R)]]) for s in seeds}
        bases = {}
        for a, mth in base_opts:
            if all((a, s, kind, L, n, R) in cells_rows for s in seeds):
                bases[f"{a}+{mth}"] = {s: np.array([r[f"se_{mth}"] for r in cells_rows[(a, s, kind, L, n, R)]]) for s in seeds}
        pooled = {k: float(np.mean([v[s].mean() for s in seeds])) for k, v in bases.items()}
        best = min(pooled, key=pooled.get) if pooled else None
        prim.update(seeds=seeds, swarm_brier=float(np.mean([sw[s].mean() for s in seeds])), baselines=pooled, best_baseline=best)
        if best:
            diffs = [sw[s] - bases[best][s] for s in seeds]
            ci = two_level(diffs)
            rel = (pooled[best] - prim["swarm_brier"]) / pooled[best]
            per_seed = [float(d.mean()) for d in diffs]
            win = rel >= 0.10 and ci[2] < 0 and all(x < 0 for x in per_seed) and len(seeds) == 3
            prim.update(diff_two_level=ci, relative_reduction=rel, per_seed_diff=per_seed, all_seeds_negative=all(x < 0 for x in per_seed),
                        verdict="WIN: hypothesis supported at this scale" if win else "REFUTED at this scale (pre-registered rule)")
            # training effect and S1 vs S2/S3 at the primary cell (D-fit)
            for other in ("S0", "S2", "S3"):
                if all((other, s, kind, L, n, R) in cells_rows for s in seeds):
                    od = [sw[s] - np.array([r["se_D-fit"] for r in cells_rows[(other, s, kind, L, n, R)]]) for s in seeds]
                    prim[f"S1_minus_{other}_Dfit"] = two_level(od)
    out["primary"] = prim
    # ---- secondary 1: pool effect within S0 and S1, every n >= 4 --------------------------------------
    sec1 = []
    for (arm, seed, k, l, nn, rr), rows in sorted(cells_rows.items()):
        if arm in ("S0", "S1") and k == "main" and nn >= 4:
            d_m = boot_mean([r["se_D-fit"] - r["se_mean"] for r in rows])
            d_v = boot_mean([r["se_D-fit"] - r["se_vote"] for r in rows])
            sec1.append({"arm": arm, "seed": seed, "L": l, "n": nn, "R": rr, "Dfit_minus_mean": d_m, "Dfit_minus_vote": d_v,
                         "holds": d_m[2] < 0 and d_v[2] < 0})
    out["secondary_pool_effect"] = {"cells": sec1, "holds_count": f"{sum(c['holds'] for c in sec1)}/{len(sec1)}"}
    # ---- secondary 4: refocus (R=2,3 vs R=1; residual vs round-robin at R=3) -------------------------
    sec4 = []
    for (arm, seed, k, l, nn, rr), rows in sorted(cells_rows.items()):
        if k == "main" and rr >= 2 and (arm, seed, k, l, nn, 1) in cells_rows:
            r1 = cells_rows[(arm, seed, k, l, nn, 1)]
            sec4.append({"arm": arm, "seed": seed, "L": l, "n": nn, "R": rr,
                         "Dfit_R_minus_R1": boot_mean([a["se_D-fit"] - b["se_D-fit"] for a, b in zip(rows, r1)])})
    rr_cmp = []
    for (arm, seed, k, l, nn, rr), rows in sorted(cells_rows.items()):
        if k == "roundrobin" and rr == 3 and (arm, seed, "main", l, nn, 3) in cells_rows:
            mainr = cells_rows[(arm, seed, "main", l, nn, 3)]
            rr_cmp.append({"arm": arm, "seed": seed, "L": l, "n": nn,
                           "residual_minus_roundrobin_Dfit": boot_mean([a["se_D-fit"] - b["se_D-fit"] for a, b in zip(mainr, rows)])})
    out["secondary_refocus"] = {"rounds": sec4, "residual_vs_roundrobin": rr_cmp}
    # ---- long chains vs matched swarm cells --------------------------------------------------------
    lc = []
    match = {256: [("main", 64, 4, 1)], 1024: [("main", 64, 16, 1), ("main", 256, 4, 1)], 4096: [("main", 64, 64, 1), ("main", 256, 16, 1)]}
    for (arm, seed, k, l, nn, rr), rows in sorted(cells_rows.items()):
        if k == "long":
            ent = {"arm": arm, "seed": seed, "budget": l, "long_mean": float(np.mean([r["se_mean"] for r in rows])),
                   "long_Dfit": float(np.mean([r["se_D-fit"] for r in rows])), "parse": float(np.nanmean([r["parsed"] for r in rows])), "matched": {}}
            for cell in match.get(l, []):
                if (arm, seed) + cell in cells_rows:
                    ent["matched"][f"{cell[0]}|L{cell[1]}|n{cell[2]}|R{cell[3]}"] = float(np.mean([r["se_D-fit"] for r in cells_rows[(arm, seed) + cell]]))
            lc.append(ent)
    out["long_chain"] = lc
    # ---- training curves ------------------------------------------------------------------------------
    for d in sorted(glob.glob(os.path.join(cli.runs, "S[1-3]-s[0-9]", "train"))):
        m = re.search(r"(S[1-3])-s(\d)", d)
        sm = os.path.join(d, "small_eval.jsonl")
        stp = os.path.join(d, "steps.jsonl")
        T = out["train"].setdefault(m.group(1), {}).setdefault(m.group(2), {})
        if os.path.exists(sm):
            T["small_eval"] = [json.loads(l) for l in open(sm)]
        if os.path.exists(stp):
            rows = [json.loads(l) for l in open(stp) if '"time_stop"' not in l]
            T["steps_done"] = max((r["step"] for r in rows), default=0)
            for a, b in ((1, 10), (51, 60), (101, 110), (141, 150)):
                sel = [r for r in rows if a <= r["step"] <= b]
                if sel:
                    T[f"steps_{a}_{b}"] = {k: float(np.nanmean([r[k] for r in sel if r.get(k) is not None])) for k in
                                          ("parse_rate", "pay_mean", "pay_pos_share", "kl", "tb_abs_delta", "distinct_p_share", "t_step")
                                          if any(r.get(k) is not None for r in sel)}
    # ---- GPU total from GPU-LOG.md ---------------------------------------------------------------------
    gl = cli.gpu_log
    costs, created = {}, set()
    for line in open(gl):
        if "vm=soup-gpu-swarm" not in line:
            continue
        vm = re.search(r"vm=(\S+)", line).group(1)
        if " CREATE " in line:
            created.add(vm)
        mm = re.search(r"cost_usd=([0-9.]+)", line)
        if mm and (" DELETE " in line or " COST-AUDIT " in line):
            costs[vm] = float(mm.group(1))
    out["gpu"] = {"vms_created": len(created), "vms_with_cost": len(costs), "total_usd_ceiling": round(sum(costs.values()), 2),
                  "unaudited": sorted(created - set(costs)), "source": gl}
    json.dump(out, open(cli.out, "w"), indent=1, default=float)
    print(json.dumps(out["primary"], indent=1, default=float))
    print("pool effect holds:", out["secondary_pool_effect"]["holds_count"], "| GPU:", out["gpu"])


if __name__ == "__main__":
    main()
