"""P2 analysis: the pre-registered readout, written before the stage-1 runs finished.

  python3 report.py [--src results/vm/trader-cat-p2-1 ...] [--suffix ""] [--boot 2000]

Reads every <src>/eval/<tag>.{json,npz}, <src>/runs/<tag>.train.json, <src>/gate/, trap and unit-test
files, gpu_util.csv, results/toy/summary.json and the frozen test covariates (data/test.npz), then
writes results/p2_results.json, results/charts.json, RESULTS.md and the catalogue entries E-P2-0..4.

Pre-registered choices made here (the spec leaves them open):
- Floor rule: for each n, the mean over U-canon seeds of cl VSP at T=300 on the dev split (no-tie
  mazes only) in the primary conditions (dead-end start, p in {0.05, 0.1}); the size enters if >= 0.5.
- Per-seed differences are paired by seed index (paired seeds share trunk init, batches and labels).
  Intervals are two-sided 95% t-intervals over seeds (df = seeds - 1). Nested bootstrap: resample seeds,
  then mazes within each seed, 2000 draws, percentile interval.
- DiD matching: each primary tie maze is matched (with replacement) to the no-tie maze of the same test
  cell with the nearest path length D inside the same loop-count bin (edges 0,1,2,4,8,...,256);
  if the bin holds no no-tie maze, the nearest D in the whole cell (counted and reported).
  A per-seed OLS of the per-maze difference on [tie, log(1+loops), cell dummies] is the covariate check.
- Non-inferiority (c): all p = 0 test cells (both starts) of the primary sizes, pooled.
- Code carries the choice: mean of I/H over primary tie mazes with H(branch) > 0, per seed, then over
  seeds; the pooled ratio sum(I)/sum(H) is reported next to it.
- GPU utilisation in the pilot: mean nvidia-smi utilisation over the first 20 min after START plan=main.
- Gate: max over T <= 300 of U-canon s0 cl VSP on the dev split at p = 0 (both starts pooled).
- Collapse: one code holds > 90% of the MAP leaf codes over all occupied leaves of the test set (T=300).
- FLOPs per test maze are an estimate: 2 x params x L^2 x T (every weight is a 3x3 conv applied at
  every pixel; tree/DAG passes are O(L^2 K^2) and O(L^2) and are left out).
"""
import argparse
import glob
import json
import math
import os
import re
from datetime import datetime, timezone

import numpy as np
from scipy import stats

import gen

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ARMS = ("U", "U-canon", "U-min", "U-any", "M-tree", "M-tile")
DECISIVE = ("M-tree", "U-canon", "U-min")
PRIMARY_SIZES = (13, 19, 25, 33)
PS_TIE = (0.05, 0.1)
T_MAIN = (30, 60, 100, 200, 300)
STRATA = (("<=8", 0, 8), ("9-32", 9, 32), (">32", 33, 10 ** 9))
TILE_STRATA = (("1", 1, 1), ("2", 2, 2), ("3-4", 3, 4), (">=5", 5, 10 ** 9))
LOOP_EDGES = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256])


def dec_of(arm):
    return "tv" if arm.startswith("M-") else "cl"


def ckey(split, n, p, de, deep=""):
    return f"{split}{deep}|n{n}_p{float(p)}_de{int(de)}"


# ---------------------------------------------------------------- statistics
def tint(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    k = len(x)
    if k == 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), k=0)
    m = float(x.mean())
    if k < 2:
        return dict(mean=m, lo=float("nan"), hi=float("nan"), k=1, seeds=x.tolist())
    h = float(stats.t.ppf(0.975, k - 1) * x.std(ddof=1) / math.sqrt(k))
    p = float(stats.ttest_1samp(x, 0.0).pvalue) if x.std(ddof=1) > 0 else (0.0 if m != 0 else 1.0)
    return dict(mean=m, lo=m - h, hi=m + h, k=k, p=p, seeds=[round(v, 5) for v in x.tolist()])


def nested_boot(Dm, B, seed=0):
    """Dm: (seeds, mazes) per-maze paired differences; resample seeds, then mazes within each seed."""
    Dm = np.asarray(Dm, float)
    k, M = Dm.shape
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    for b in range(B):
        si = rng.integers(k, size=k)
        mi = rng.integers(M, size=(k, M))
        out[b] = Dm[si[:, None], mi].mean()
    return dict(lo=float(np.percentile(out, 2.5)), hi=float(np.percentile(out, 97.5)), mean=float(Dm.mean()))


def holm(pvals):
    """pvals: {name: p}; returns {name: (p_adj, reject at 0.05)}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, run, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = dict(p=p, p_holm=run, reject=run < 0.05)
    return out


def pts(d, nd=1):
    """Format a t-interval dict in VSP points."""
    if not d or not np.isfinite(d.get("mean", float("nan"))):
        return "n/a"
    if not np.isfinite(d.get("lo", float("nan"))):
        return f"{100 * d['mean']:.{nd}f}"
    return f"{100 * d['mean']:.{nd}f} [{100 * d['lo']:.{nd}f}, {100 * d['hi']:.{nd}f}]"


# ---------------------------------------------------------------- data access
class Run:
    """One trained run's eval files merged: <tag>.json (core or all), <tag>__deep.json, <tag>__extra.json."""
    def __init__(self, jpaths):
        self.j, self.z, self.parts = {"cells": {}, "cal": {}, "code_hist": {}, "wall_sec": 0.0}, {}, []
        for jp in jpaths:
            j, z = json.load(open(jp)), np.load(jp[:-5] + ".npz")
            for k in ("cells", "cal", "code_hist"):
                self.j[k].update(j.get(k, {}))
            self.j["wall_sec"] += j.get("wall_sec") or 0.0
            self.z.update({k: z[k] for k in z.files})
            self.parts.append(j.get("part", "all"))
        self.keys = set(self.z)

    def num(self, key):
        return self.j["cells"][key]["num"]

    def has(self, key, T, dec):
        return f"{key}|T{T}|{dec}" in self.keys

    def vsp(self, key, T, dec):
        return np.unpackbits(self.z[f"{key}|T{T}|{dec}"], count=self.num(key)).astype(float)

    def field(self, key, f):
        k = f"{key}|{f}"
        return self.z[k].astype(float) if k in self.keys else None


class Data:
    def __init__(self, srcs, suffix):
        self.runs, self.train, self.srcs = {}, {}, srcs
        pat = re.compile(r"^(.+)_s(\d+)(.*?)(__deep|__extra)?$")
        files = {}
        for src in srcs:  # later sources win per file (a resumed VM holds the newer files)
            for f in sorted(glob.glob(os.path.join(src, "eval", "*.json"))):
                m = pat.match(os.path.basename(f)[:-5])
                if m and m.group(1) in ARMS and m.group(3) == suffix and os.path.exists(f[:-5] + ".npz"):
                    files.setdefault((m.group(1), int(m.group(2))), {})[m.group(4) or ""] = f
        for k, parts in files.items():
            self.runs[k] = Run([parts[q] for q in sorted(parts)])
            for f in sorted(glob.glob(os.path.join(src, "runs", "*.train.json"))):
                m = pat.match(os.path.basename(f)[:-11])
                if m and m.group(1) in ARMS and m.group(3) == suffix:
                    self.train[(m.group(1), int(m.group(2)))] = json.load(open(f))
        self.cov = {}

    def seeds(self, arm):
        return sorted(s for a, s in self.runs if a == arm)

    def c(self, split, n, p, de):
        k = (split, n, float(p), bool(de))
        if k not in self.cov:
            c = gen.load_cell(split, n, p, de)
            self.cov[k] = {f: np.asarray(c[f]) for f in ("D", "tie", "contested", "tiles", "loops")}
        return self.cov[k]

    def vec(self, arm, seed, cells, mask=None, T=300, dec=None, split="test", deep=""):
        """Concatenated per-maze VSP over cells [(n, p, de)], restricted by mask(cov) -> bool."""
        r = self.runs.get((arm, seed))
        if r is None:
            return None
        out = []
        for n, p, de in cells:
            key = ckey(split, n, p, de, deep)
            if key not in r.j["cells"] or not r.has(key, T, dec or dec_of(arm)):
                return None
            v = r.vsp(key, T, dec or dec_of(arm))
            if mask is not None:
                v = v[mask(self.c(split, n, p, de))]
            out.append(v)
        return np.concatenate(out) if out else None

    def fvec(self, arm, seed, cells, field, mask=None, split="test"):
        r = self.runs.get((arm, seed))
        if r is None:
            return None
        out = []
        for n, p, de in cells:
            v = r.field(ckey(split, n, p, de), field)
            if v is None:
                return None
            if mask is not None:
                v = v[mask(self.c(split, n, p, de))]
            out.append(v)
        return np.concatenate(out) if out else None


def paired(D, a, b, cells, mask=None, T=300, da=None, db=None):
    """Per-seed means of arm a minus arm b over the same mazes; returns (per-maze diff matrix, seeds)."""
    rows, seeds = [], []
    for s in sorted(set(D.seeds(a)) & set(D.seeds(b))):
        va, vb = D.vec(a, s, cells, mask, T, da), D.vec(b, s, cells, mask, T, db)
        if va is None or vb is None or len(va) == 0:
            continue
        rows.append(va - vb)
        seeds.append(s)
    return (np.array(rows) if rows else None), seeds


def per_arm(D, arm, cells, mask=None, T=300, dec=None):
    return tint([D.vec(arm, s, cells, mask, T, dec).mean() for s in D.seeds(arm)
                 if D.vec(arm, s, cells, mask, T, dec) is not None and len(D.vec(arm, s, cells, mask, T, dec))])


# ---------------------------------------------------------------- DiD matching
def match_notie(cov):
    """For each tie maze of a cell, the index of its matched no-tie maze (same loop bin, nearest D)."""
    tie = cov["tie"].astype(bool)
    bins = np.digitize(cov["loops"], LOOP_EDGES)
    ti, ni = np.flatnonzero(tie), np.flatnonzero(~tie)
    match, fallback = np.empty(len(ti), int), 0
    for j, i in enumerate(ti):
        cand = ni[bins[ni] == bins[i]]
        if len(cand) == 0:
            cand, fallback = ni, fallback + 1
        match[j] = cand[np.argmin(np.abs(cov["D"][cand] - cov["D"][i]))]
    return ti, match, fallback


def did(D, a, b, cells, B):
    rows, seeds, fb, regs = [], [], 0, []
    meta = {}
    for n, p, de in cells:
        meta[(n, p, de)] = match_notie(D.c("test", n, p, de))
    fb = sum(m[2] for m in meta.values())
    ntie = sum(len(m[0]) for m in meta.values())
    for s in sorted(set(D.seeds(a)) & set(D.seeds(b))):
        parts, regy, regx = [], [], []
        ok = True
        for ci, (n, p, de) in enumerate(cells):
            va, vb = D.vec(a, s, [(n, p, de)]), D.vec(b, s, [(n, p, de)])
            if va is None or vb is None:
                ok = False
                break
            d = va - vb
            ti, mt, _ = meta[(n, p, de)]
            parts.append(d[ti] - d[mt])
            cov = D.c("test", n, p, de)
            regy.append(d)
            X = np.zeros((len(d), 2 + len(cells)))
            X[:, 0] = cov["tie"]
            X[:, 1] = np.log1p(cov["loops"])
            X[:, 2 + ci] = 1
            regx.append(X)
        if not ok:
            continue
        rows.append(np.concatenate(parts))
        seeds.append(s)
        coef = np.linalg.lstsq(np.concatenate(regx), np.concatenate(regy), rcond=None)[0]
        regs.append(coef[0])
    if not rows:
        return None
    Dm = np.array(rows)
    out = tint(Dm.mean(1))
    out["boot"] = nested_boot(Dm, B, seed=11)
    out["reg_tie_coef"] = tint(regs)
    out["seeds_used"] = seeds
    out["matched_fallback_share"] = fb / max(ntie, 1)
    return out


# ---------------------------------------------------------------- other readouts
def ece(D, arm, cells):
    vals = []
    for s in D.seeds(arm):
        cal = D.runs[(arm, s)].j.get("cal", {})
        n = np.zeros(10)
        q = np.zeros(10)
        t = np.zeros(10)
        for nn, p, de in cells:
            c = cal.get(ckey("test", nn, p, de))
            if c is None:
                continue
            n += np.array(c["n"])
            q += np.array(c["q"])
            t += np.array(c["t"])
        if n.sum() > 0:
            ok = n > 0
            vals.append(float((np.abs(q[ok] - t[ok])).sum() / n.sum()))
    return tint(vals)


def detail_stats(D, arm, cells, tie_only=True):
    mask = (lambda c: c["tie"].astype(bool)) if tie_only else None
    res = {}
    for f, fn in (("coherent", np.mean), ("acc_unc", np.mean), ("jll", np.mean),
                  ("near_1_7", None), ("near_mean", None), ("tc_label", None), ("tc_sample", None),
                  ("mh_leaf_H", np.nanmean), ("mh_leaf_multi", np.nanmean), ("mh_root_H", np.nanmean),
                  ("mh_root_multi", np.nanmean)):
        vals = []
        for s in D.seeds(arm):
            if f == "near_1_7" or f == "near_mean":
                v = D.fvec(arm, s, cells, "near_miss", mask)
                if v is None:
                    continue
                vals.append(float(((v >= 1) & (v <= 7)).mean()) if f == "near_1_7" else float(v.mean()))
            elif f == "tc_label":
                a, b = D.fvec(arm, s, cells, "jll_fact", mask), D.fvec(arm, s, cells, "jll", mask)
                if a is not None and b is not None:
                    vals.append(float(np.mean(a - b)))
            elif f == "tc_sample":
                v = D.fvec(arm, s, cells, "tc", mask)
                if v is not None and np.isfinite(v).any():
                    vals.append(float(np.nanmean(v)))
            else:
                v = D.fvec(arm, s, cells, f, mask)
                if v is not None and np.isfinite(v).any():
                    vals.append(float(fn(v)))
        if vals:
            res[f] = tint(vals)
    return res


def code_carries(D, arm, cells):
    per_seed, pooled, share_h, share_level = [], [], [], []
    mask = lambda c: c["tie"].astype(bool)
    for s in D.seeds(arm):
        r = D.fvec(arm, s, cells, "cc_ratio", mask)
        h = D.fvec(arm, s, cells, "H_branch", mask)
        lv = D.fvec(arm, s, cells, "cc_level", mask)
        if r is None or h is None:
            continue
        ok = np.isfinite(r) & np.isfinite(h) & (h > 1e-9)
        share_h.append(float(ok.mean()))
        share_level.append(float((lv >= 0).mean()) if lv is not None else float("nan"))
        if ok.any():
            per_seed.append(float(r[ok].mean()))
            pooled.append(float((r[ok] * h[ok]).sum() / h[ok].sum()))
    return dict(ratio=tint(per_seed), pooled_ratio=tint(pooled), share_tie_with_branch_entropy=tint(share_h),
                share_tie_single_node=tint(share_level))


def collapse(D, arm):
    out = {}
    for s in D.seeds(arm):
        hist = np.zeros(1)
        for k, h in D.runs[(arm, s)].j.get("code_hist", {}).items():
            if k.startswith("test|"):
                h = np.array(h, float)
                hist = h if hist.size == 1 else hist + h
        if hist.sum() > 0:
            out[s] = float(hist.max() / hist.sum())
    return out


def gpu_util(srcs):
    starts, rows = [], []
    led = os.path.join(HERE, "gpu", "LEDGER.md")
    if os.path.exists(led):
        for line in open(led):
            m = re.match(r"- (\S+Z) START vm=(\S+) plan=main", line)
            if m:
                starts.append(datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp())
    for src in srcs:
        f = os.path.join(src, "gpu_util.csv")
        if os.path.exists(f):
            for line in open(f):
                try:
                    t, u, mem = [float(x) for x in line.strip().split(",")[:3]]
                    rows.append((t, u, mem))
                except ValueError:
                    pass
    if not rows:
        return dict(pilot_mean=None, run_mean=None, samples=0)
    rows = np.array(sorted(rows))
    out = dict(samples=int(len(rows)), mem_max_mb=float(rows[:, 2].max()))
    if starts:
        t0 = min(starts)
        pil = rows[(rows[:, 0] >= t0) & (rows[:, 0] < t0 + 1200)]
        run = rows[rows[:, 0] >= t0]
        out["pilot_mean"] = float(pil[:, 1].mean()) if len(pil) else None
        out["run_mean"] = float(run[:, 1].mean()) if len(run) else None
        out["pilot_samples"] = int(len(pil))
    else:
        out["pilot_mean"], out["run_mean"] = None, float(rows[:, 1].mean())
    return out


def find(srcs, *names):
    for src in reversed(srcs):
        for nm in names:
            f = os.path.join(src, nm)
            if os.path.exists(f):
                return f
    return None


# ---------------------------------------------------------------- main
def build(a):
    srcs = a.src or sorted(glob.glob(os.path.join(HERE, "results", "vm", "*")))
    D = Data(srcs, a.suffix)
    R = {"srcs": [os.path.relpath(s, HERE) for s in srcs], "suffix": a.suffix,
         "seeds": {arm: D.seeds(arm) for arm in ARMS}, "validity": {}, "notes": []}
    B = a.boot

    # ---- floor rule
    floor = {}
    for n in PRIMARY_SIZES:
        vals = [D.vec("U-canon", s, [(n, p, True) for p in PS_TIE], split="dev") for s in D.seeds("U-canon")]
        vals = [v.mean() for v in vals if v is not None]
        allc = [D.vec("U-canon", s, [(n, p, de) for p in (0.0,) + PS_TIE for de in (True, False)], split="dev")
                for s in D.seeds("U-canon")]
        allc = [v.mean() for v in allc if v is not None]
        floor[n] = dict(vsp=float(np.mean(vals)) if vals else None, pass_=bool(vals and np.mean(vals) >= 0.5),
                        all_dev_cells=float(np.mean(allc)) if allc else None)
    sizes = [n for n in PRIMARY_SIZES if floor[n]["pass_"]]
    if a.sizes:  # sensitivity only: NOT the pre-registered floor rule (never used with --catalogue)
        R["sizes_override"] = dict(floor_rule=sizes, used=[int(x) for x in a.sizes.split(",")])
        sizes = R["sizes_override"]["used"]
        R["notes"].append(f"SENSITIVITY RUN: primary sizes forced to {sizes}; the pre-registered floor rule gives "
                          f"{R['sizes_override']['floor_rule']}. Not a pre-registered result.")
    R["floor"] = {str(k): v for k, v in floor.items()}
    R["primary_sizes"] = sizes
    cells = [(n, p, True) for n in sizes for p in PS_TIE]
    tie = lambda c: c["tie"].astype(bool)
    notie = lambda c: ~c["tie"].astype(bool)
    R["primary_n_mazes"] = int(sum(D.c("test", *c)["tie"].sum() for c in cells))

    # ---- arm means on the primary tie set, T=300, pre-named decoders (plus the other decoders)
    R["arm_primary"] = {}
    for arm in ARMS:
        R["arm_primary"][arm] = {dec: per_arm(D, arm, cells, tie, 300, dec)
                                 for dec in (("tv", "thr", "cl") if arm.startswith("M-") else ("thr", "cl"))}
    R["arm_primary_notie"] = {arm: per_arm(D, arm, cells, notie) for arm in ARMS}

    # ---- GAIN (a): M-tree-tv vs U-canon-cl and U-min-cl
    gain = {}
    for base in ("U-canon", "U-min"):
        Dm, seeds = paired(D, "M-tree", base, cells, tie)
        if Dm is None:
            gain[base] = None
            continue
        g = tint(Dm.mean(1))
        g["boot"] = nested_boot(Dm, B)
        g["seeds_used"] = seeds
        g["pass"] = bool(g["mean"] >= 0.08 and g["lo"] > 0 and g["boot"]["lo"] > 0)
        gain[base] = g
    R["gain_a"] = gain
    # ---- GAIN (b): DiD
    R["gain_b"] = {}
    for base in ("U-canon", "U-min"):
        d = did(D, "M-tree", base, cells, B) if cells else None
        if d:
            d["pass"] = bool(d["mean"] >= 0.05 and d["lo"] > 0)
        R["gain_b"][base] = d
    # ---- GAIN (c): p = 0 non-inferiority, pooled over p = 0 cells of the primary sizes (both starts)
    p0 = [(n, 0.0, de) for n in sizes for de in (True, False)]
    Dm, seeds = paired(D, "U-canon", "M-tree", p0) if p0 else (None, [])
    if Dm is not None:
        c = tint(Dm.mean(1))
        c["boot"] = nested_boot(Dm, B, seed=7)
        c["pass"] = bool(np.isfinite(c["hi"]) and c["hi"] <= 0.02)
        Dm1, _ = paired(D, "U-canon", "M-tree", [x for x in p0 if x[2]])
        c["dead_end_only"] = tint(Dm1.mean(1)) if Dm1 is not None else None
        R["gain_c"] = c
    else:
        R["gain_c"] = None
    ga = all(R["gain_a"].get(b) and R["gain_a"][b]["pass"] for b in ("U-canon", "U-min"))
    gb = all(R["gain_b"].get(b) and R["gain_b"][b]["pass"] for b in ("U-canon", "U-min"))
    gc = bool(R["gain_c"] and R["gain_c"]["pass"])
    R["GAIN"] = dict(a=ga, b=gb, c=gc, all=ga and gb and gc)

    # ---- LATENT
    R["latent"] = {arm: code_carries(D, arm, cells) for arm in ("M-tree", "M-tile")}
    lt = R["latent"]["M-tree"]["ratio"]
    R["LATENT_pass"] = bool(np.isfinite(lt["mean"]) and lt["mean"] >= 0.8)

    # ---- strata
    R["strata"] = {}
    for arm in ARMS:
        row = {}
        for nm, lo, hi in STRATA:
            row["contested " + nm] = per_arm(D, arm, cells, lambda c, lo=lo, hi=hi: tie(c) & (c["contested"] >= lo) & (c["contested"] <= hi))
        for nm, lo, hi in TILE_STRATA:
            row["tiles " + nm] = per_arm(D, arm, cells, lambda c, lo=lo, hi=hi: tie(c) & (c["tiles"] >= lo) & (c["tiles"] <= hi))
        R["strata"][arm] = row
    R["stratum_sizes"] = {}
    for nm, lo, hi in STRATA:
        R["stratum_sizes"]["contested " + nm] = int(sum(((D.c("test", *c)["contested"] >= lo) & (D.c("test", *c)["contested"] <= hi) & tie(D.c("test", *c))).sum() for c in cells))
    for nm, lo, hi in TILE_STRATA:
        R["stratum_sizes"]["tiles " + nm] = int(sum(((D.c("test", *c)["tiles"] >= lo) & (D.c("test", *c)["tiles"] <= hi) & tie(D.c("test", *c))).sum() for c in cells))
    one_tile = sum((tie(D.c("test", *c)) & (D.c("test", *c)["tiles"] == 1)).sum() for c in cells)
    R["share_tie_one_tile"] = float(one_tile / max(1, R["primary_n_mazes"]))

    # ---- per n x p x de x tie table (T=300, pre-named decoders)
    R["cells"] = {}
    for n in (9,) + PRIMARY_SIZES:
        for p in (0.0,) + PS_TIE:
            for de in (True, False):
                for tn, mk in (("tie", tie), ("notie", notie)):
                    if p == 0.0 and tn == "tie":
                        continue
                    R["cells"][f"n{n}_p{p}_de{int(de)}_{tn}"] = {arm: per_arm(D, arm, [(n, p, de)], mk)["mean"] for arm in ARMS}

    # ---- VSP against T on the primary tie set
    R["vs_T"] = {arm: {str(T): per_arm(D, arm, cells, tie, T) for T in T_MAIN} for arm in ARMS}

    # ---- per n (PARTIAL and slopes)
    R["per_n"] = {}
    for n in PRIMARY_SIZES:
        cn = [(n, p, True) for p in PS_TIE]
        row = {arm: per_arm(D, arm, cn, tie) for arm in ARMS}
        for base in ("U-canon", "U-min"):
            Dm, _ = paired(D, "M-tree", base, cn, tie)
            row["M-tree - " + base] = tint(Dm.mean(1)) if Dm is not None else None
        R["per_n"][str(n)] = row

    # ---- secondaries (Holm)
    sec = {}
    big = lambda c: tie(c) & (c["contested"] > 32)
    Dm, _ = paired(D, "M-tree", "M-tile", cells, big)
    sec["M-tree - M-tile, >32 px"] = tint(Dm.mean(1)) if Dm is not None else None
    Dm, _ = paired(D, "U-any", "U-min", cells, tie)
    sec["U-any - U-min, primary tie"] = tint(Dm.mean(1)) if Dm is not None else None

    def slopes(arm):
        out = {}
        for s in D.seeds(arm):
            ys = [D.vec(arm, s, [(n, p, True) for p in PS_TIE], tie) for n in sizes]
            if len(sizes) >= 2 and all(y is not None for y in ys):
                out[s] = float(np.polyfit(np.array(sizes, float), [y.mean() for y in ys], 1)[0])
        return out
    sl = {arm: slopes(arm) for arm in ("U-canon", "U-min", "M-tree")}
    R["slopes_per_n"] = {arm: tint(list(v.values())) for arm, v in sl.items()}
    for other in ("U-min", "M-tree"):
        common = sorted(set(sl["U-canon"]) & set(sl[other]))
        sec[f"slope U-canon - slope {other}"] = tint([sl["U-canon"][s] - sl[other][s] for s in common]) if common else None
    hp = holm({k: v["p"] for k, v in sec.items() if v and "p" in v})
    for k in sec:
        if sec[k] and k in hp:
            sec[k].update(hp[k])
    R["secondary"] = sec
    Dm, _ = paired(D, "M-tree", "M-tile", cells, big)
    R["M-tile_capped_share_pred"] = R["share_tie_one_tile"]

    # ---- PARTIAL
    Dm, seeds = paired(D, "M-tree", "U", cells, tie)
    mu = tint(Dm.mean(1)) if Dm is not None else None
    ge25 = [n for n in sizes if n >= 25]
    lt25 = [n for n in sizes if n < 25]

    def gain_at(ns):
        ok = []
        for n in ns:
            for base in ("U-canon", "U-min"):
                d = R["per_n"][str(n)].get("M-tree - " + base)
                ok.append(bool(d and d["mean"] >= 0.08 and np.isfinite(d["lo"]) and d["lo"] > 0))
        return all(ok) if ok else False
    R["partial"] = dict(M_tree_minus_U=mu,
                        beats_U_by_10=bool(mu and mu["mean"] >= 0.10 and mu["lo"] > 0),
                        gain_only_n_ge_25=bool(ge25 and gain_at(ge25) and not gain_at(lt25)))
    R["partial"]["holds"] = bool((R["partial"]["beats_U_by_10"] and not ga) or R["partial"]["gain_only_n_ge_25"])

    # ---- falsification
    rows = []
    for s in sorted(set(D.seeds("M-tree")) & set(D.seeds("U-canon")) & set(D.seeds("U-min"))):
        m_, c_, u_ = (D.vec(x, s, cells, tie) for x in ("M-tree", "U-canon", "U-min"))
        if m_ is not None and c_ is not None and u_ is not None:
            rows.append(m_.mean() - max(c_.mean(), u_.mean()))
    f1 = tint(rows)
    f3 = sec["M-tree - M-tile, >32 px"]
    R["falsify"] = {
        "1_upper_M_minus_maxbase_lt_3": dict(interval=f1, falsified=bool(np.isfinite(f1.get("hi", np.nan)) and f1["hi"] < 0.03)),
        "2_DiD_le_0": dict(falsified=bool(any(R["gain_b"].get(b) and R["gain_b"][b]["mean"] <= 0 for b in ("U-canon", "U-min")))),
        "3_Mtree_le_Mtile_plus_3_gt32": dict(interval=f3, falsified=bool(f3 and f3["mean"] <= 0.03)),
        "4_wins_but_code_lt_30pct": dict(falsified=bool(ga and np.isfinite(lt["mean"]) and lt["mean"] < 0.3)),
    }

    # ---- detail metrics, calibration, M health
    R["detail_tie"] = {arm: detail_stats(D, arm, cells) for arm in ARMS}
    R["detail_notie"] = {arm: detail_stats(D, arm, cells, tie_only=False) for arm in ARMS}
    R["ece_contested"] = {arm: ece(D, arm, cells) for arm in ARMS}
    R["collapse_max_code_share"] = {arm: collapse(D, arm) for arm in ("M-tree", "M-tile")}

    # ---- depth curves for the decisive arms
    depth = {}
    for arm in DECISIVE:
        row = {}
        for n in (9,) + PRIMARY_SIZES:
            for T in (300, 1000):
                vals = []
                for s in D.seeds(arm):
                    if n == 33:  # first 50 tie mazes ran to T=3000, the next 50 to T=1000
                        parts = []
                        for p in PS_TIE:
                            r_ = D.runs[(arm, s)]
                            for key in (ckey("test", n, p, True, "_deep3000"), ckey("test", n, p, True, "_deep1000") + "_b"):
                                if key in r_.j["cells"] and r_.has(key, T, dec_of(arm)):
                                    parts.append(r_.vsp(key, T, dec_of(arm)))
                        v = np.concatenate(parts) if parts else None
                    else:
                        v = D.vec(arm, s, [(n, p, True) for p in PS_TIE], T=T, deep="_deep1000")
                    if v is not None:
                        vals.append(v.mean())
                row[f"n{n}_T{T}"] = tint(vals)
        vals = [D.vec(arm, s, [(33, p, True) for p in PS_TIE], T=3000, deep="_deep3000") for s in D.seeds(arm)]
        row["n33_T3000_first50"] = tint([v.mean() for v in vals if v is not None])
        vals = [D.vec(arm, s, [(33, p, True) for p in PS_TIE], T=300, deep="_deep3000") for s in D.seeds(arm)]
        row["n33_T300_first50"] = tint([v.mean() for v in vals if v is not None])
        for T in (300, 1000):
            vals = [D.vec(arm, s, [(59, p, True) for p in PS_TIE], T=T, split="ext", deep="_deep1000") for s in D.seeds(arm)]
            row[f"n59_T{T}"] = tint([v.mean() for v in vals if v is not None])
        depth[arm] = row
    R["depth"] = depth

    # ---- compute
    comp = {}
    for arm in ARMS:
        tr = [D.train[(arm, s)] for s in D.seeds(arm) if (arm, s) in D.train]
        ev = [D.runs[(arm, s)].j.get("wall_sec") for s in D.seeds(arm)]
        if not tr:
            continue
        P = tr[0]["params"]
        comp[arm] = dict(params=P, trunk_params=tr[0].get("trunk_params"), model_kb=round(P * 4 / 1024, 1),
                         train_wall_sec=tint([t["wall_sec"] for t in tr])["mean"],
                         eval_wall_sec=float(np.mean([e for e in ev if e])) if any(ev) else None,
                         flops_per_maze_T300_est={str(n): float(2 * P * (2 * n + 1) ** 2 * 300) for n in (9,) + PRIMARY_SIZES})
    R["compute"] = comp
    R["gpu_util"] = gpu_util(srcs)

    # ---- validity
    V = {}
    ut = find(srcs, "smoke/unit_tests.json") or os.path.join(HERE, "results", "unit_tests.json")
    if os.path.exists(ut):
        u = json.load(open(ut))
        V["unit_tests"] = dict(file=os.path.relpath(ut, HERE), n=len(u), failed=[k for k, v in u.items() if not v["pass"]])
        V["unit_tests"]["pass"] = not V["unit_tests"]["failed"]
    loc = os.path.join(HERE, "results", "unit_tests.json")
    if os.path.exists(loc) and loc != ut:
        u = json.load(open(loc))
        V["unit_tests_general"] = dict(n=len(u), failed=[k for k, v in u.items() if not v["pass"]])
    tt = find(srcs, "trap_test.json") or os.path.join(HERE, "results", "trap_test.json")
    if os.path.exists(tt):
        t = json.load(open(tt))
        V["trap"] = dict(pass_=t["pass"], max_vsp=t["max"], file=os.path.relpath(tt, HERE))
    else:  # a trap run cut short (spot preemption) leaves only its per-cell log: report it, never as a pass
        cells = {}
        for src in srcs:
            lg = os.path.join(src, "logs", "trap.log")
            if os.path.exists(lg):
                for line in open(lg):
                    w = line.split()
                    if len(w) == 4 and w[0].isdigit():
                        cells[(int(w[0]), float(w[1]), w[2] == "True")] = float(w[3])
        V["trap"] = dict(pass_=False, missing=True)
        if cells:
            V["trap"].update(incomplete=True, cells_done=len(cells), cells_total=30,
                             max_vsp_done_cells=max(cells.values()),
                             missing_cells=[f"n{n}_p{p}_de{int(de)}" for n in (9, 13, 19, 25, 33) for p in (0.0, 0.05, 0.1)
                                            for de in (True, False) if (n, p, de) not in cells])
    ts = os.path.join(HERE, "results", "toy", "summary.json")
    if os.path.exists(ts):
        t = json.load(open(ts))
        cr = t.get("criteria", {})
        V["toy"] = {k: cr.get(k) for k in ("M-tree_ge_40pct_n64_128", "M-global_ge_40pct_n64_128", "M-tile_le_5pct_n64_128")}
        V["toy"]["pass"] = bool(cr.get("M-tree_ge_40pct_n64_128") and cr.get("M-tile_le_5pct_n64_128"))
        V["toy"]["per_n"] = cr.get("per_n")
    else:
        V["toy"] = dict(pass_=False, missing=True)
    gfile = None
    for src in reversed(srcs):
        cand = sorted(glob.glob(os.path.join(src, "gate", f"U-canon_s0{a.suffix}.json")))
        if cand:
            gfile = cand[0]
            break
    if gfile:
        gj = json.load(open(gfile))
        gv = {}
        for n, thr in ((9, 0.95), (19, 0.80)):
            best = 0.0
            for T in T_MAIN:
                vs = [gj["cells"].get(ckey("dev", n, 0.0, de), {}).get(f"T{T}|cl") for de in (True, False)]
                vs = [v for v in vs if v is not None]
                if vs:
                    best = max(best, float(np.mean(vs)))
            gv[f"n{n}"] = dict(best_dev_vsp_T_le_300=best, need=thr, pass_=best >= thr)
        V["gate"] = gv
        V["gate"]["pass"] = all(v["pass_"] for v in gv.values())
    else:
        V["gate"] = dict(pass_=False, missing=True)
    V["floor_sizes"] = dict(sizes=sizes, pass_=len(sizes) >= 2)
    col = [v for arm in ("M-tree", "M-tile") for v in R["collapse_max_code_share"][arm].values()]
    V["no_collapse"] = dict(max_share=max(col) if col else None, pass_=bool(col) and max(col) <= 0.9)
    gu = R["gpu_util"].get("pilot_mean")
    V["gpu_util_pilot"] = dict(mean=gu, pass_=gu is not None and gu >= 50)
    R["validity"] = V
    R["validity_all"] = all(v.get("pass", v.get("pass_", True)) for v in V.values() if isinstance(v, dict))
    return R, D


# ---------------------------------------------------------------- outputs
def charts(R):
    C = []
    arms = [a for a in ARMS if np.isfinite(R["arm_primary"][a][dec_of(a)]["mean"])]
    C.append(dict(id="p2_primary", type="bar", title="Tie mazes, primary set: valid shortest path at T=300 (pre-named decoders)",
                  x_label="arm", y_label="VSP (%)", x=arms,
                  series=[dict(name="tie", y=[100 * R["arm_primary"][a][dec_of(a)]["mean"] for a in arms],
                               lo=[100 * R["arm_primary"][a][dec_of(a)]["lo"] for a in arms],
                               hi=[100 * R["arm_primary"][a][dec_of(a)]["hi"] for a in arms]),
                          dict(name="matched cells, no tie", y=[100 * R["arm_primary_notie"][a]["mean"] for a in arms])]))
    C.append(dict(id="p2_vs_T", type="line", title="Tie VSP against iterations (primary set)", x_label="T", y_label="VSP (%)",
                  x=list(T_MAIN), series=[dict(name=a, y=[100 * R["vs_T"][a][str(T)]["mean"] for T in T_MAIN]) for a in arms]))
    strata = ["contested <=8", "contested 9-32", "contested >32"]
    C.append(dict(id="p2_strata", type="bar", title="Tie VSP by size of the contested region", x_label="contested pixels",
                  y_label="VSP (%)", x=strata, series=[dict(name=a, y=[100 * R["strata"][a][s]["mean"] for s in strata]) for a in arms]))
    ns = [str(n) for n in PRIMARY_SIZES]
    C.append(dict(id="p2_per_n", type="line", title="Tie VSP against maze size (dead-end start, p in {0.05, 0.1})",
                  x_label="n", y_label="VSP (%)", x=[int(n) for n in ns],
                  series=[dict(name=a, y=[100 * R["per_n"][n][a]["mean"] for n in ns]) for a in arms]))
    dk = ["n33_T300_first50", "n33_T1000", "n33_T3000_first50", "n59_T300", "n59_T1000"]
    C.append(dict(id="p2_depth", type="bar", title="Depth: tie VSP at more iterations (decisive arms)", x_label="cell",
                  y_label="VSP (%)", x=dk, series=[dict(name=a, y=[100 * R["depth"][a][k]["mean"] for k in dk]) for a in DECISIVE]))
    lt = R["latent"]["M-tree"]["ratio"]
    C.append(dict(id="p2_latent", type="bar", title="Share of the route entropy the M-tree code carries (I/H) per seed",
                  x_label="seed", y_label="I/H", x=[f"s{i}" for i in range(len(lt.get("seeds", [])))],
                  series=[dict(name="M-tree", y=lt.get("seeds", []))]))
    toy = R["validity"].get("toy", {}).get("per_n") or {}
    if toy:
        C.append(dict(id="p2_toy", type="bar", title="Stage-0 toy: exact match while U has not resolved",
                      x_label="n", y_label="exact match (%)", x=list(toy.keys()),
                      series=[dict(name=a, y=[100 * (toy[n].get(a) or {}).get("mean_in_window", float("nan")) for n in toy])
                              for a in ("M-tree", "M-global", "M-tile")]))
    return C


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def status(R):
    V = R["validity"]
    if not R["validity_all"]:
        return "Problem"
    if R["GAIN"]["all"]:
        return "Gain"
    if R["partial"]["holds"] or any(R["gain_a"].get(b) and R["gain_a"][b]["mean"] > 0 and R["gain_a"][b]["lo"] > 0
                                    for b in ("U-canon", "U-min")):
        return "Informative"
    return "No gain"


def markdown(R, C):
    g, L = R["gain_a"], []
    ap = R["arm_primary"]
    L += ["# P2: tied maze paths, quadtree latent-parent market vs commitment baselines", "",
          f"Status (computed from the pre-registered rules): **{R['status']}**. Primary set: tie mazes, dead-end start, "
          f"p in {{0.05, 0.1}}, n in {R['primary_sizes']} (floor rule), T = 300, {R['primary_n_mazes']} mazes; "
          f"seeds {R['seeds']}. Intervals: 95% t over paired seeds; [b ...] = nested bootstrap.", "",
          *[f"**{x}**" for x in R["notes"]], *([""] if R["notes"] else []),
          "| arm | tie VSP % (pre-named decoder) | no-tie VSP % |", "|---|---|---|"]
    for arm in ARMS:
        L.append(f"| {arm} ({dec_of(arm)}) | {pts(ap[arm][dec_of(arm)])} | {pts(R['arm_primary_notie'][arm])} |")
    L += ["", "| pre-registered test | estimate (points) | pass |", "|---|---|---|"]
    _notes = os.path.join(HERE, "results", "run_notes.md")
    for b in ("U-canon", "U-min"):
        x = g.get(b)
        if x:
            L.append(f"| (a) M-tree-tv - {b}-cl >= 8, lower > 0 | {pts(x)} [b {100 * x['boot']['lo']:.1f}, {100 * x['boot']['hi']:.1f}] | {x['pass']} |")
        y = R["gain_b"].get(b)
        if y:
            L.append(f"| (b) DiD vs {b} >= 5, lower > 0 | {pts(y)}; OLS tie coef {pts(y['reg_tie_coef'])} | {y['pass']} |")
    c = R["gain_c"]
    if c:
        L.append(f"| (c) p=0: U-canon - M-tree upper <= 2 | {pts(c)} | {c['pass']} |")
    lt = R["latent"]["M-tree"]
    f3 = lambda d: f"{d['mean']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]" if d["k"] > 1 else (f"{d['mean']:.3f}" if d["k"] else "n/a")
    L.append(f"| LATENT I/H >= 0.8 | {f3(lt['ratio'])} (pooled {f3(lt['pooled_ratio'])}) | {R['LATENT_pass']} |")
    for k, v in R["falsify"].items():
        L.append(f"| falsifier {k} | {pts(v['interval']) if v.get('interval') else ''} | falsified={v['falsified']} |")
    for k, v in R["secondary"].items():
        if v:
            L.append(f"| secondary {k} | {pts(v, 2) if 'slope' in k else pts(v)} p_holm={v.get('p_holm', float('nan')):.3g} | reject={v.get('reject')} |")
    L.append(f"| partial: M-tree - U >= 10 | {pts(R['partial']['M_tree_minus_U'])} | {R['partial']['beats_U_by_10']} |")
    L += ["", "Validity: " + "; ".join(f"{k}={v.get('pass', v.get('pass_'))}" for k, v in R["validity"].items() if isinstance(v, dict)), ""]
    L += ["Simplifications against the spec: the cl decoder is binary (<= 5 local sweeps, no 0.5 damping); beliefs and codes are "
          "soft-clamped at +-30 logits; the toy uses a generic 1D trunk; eval recomputes BFS distances and asserts the stored DAG; "
          "M-tree has 1.7% more parameters than U (108.5k vs 106.7k); code-carries labels a sample's branch by the contested pixels "
          "of the best shortest path inside the sample; joint log loss is averaged over 4 uniformly drawn shortest paths per maze; "
          "ECE targets the uniform-path marginal; M health comes from the same 256 samples (leaf and root nodes); total correlation "
          "is jll_fact - jll on the same labels (a sample estimate is kept as tc_sample); FLOPs are estimated from parameter counts. "
          "The dev split holds no-tie mazes only, as specified.", "",
          *([open(_notes).read().strip(), ""] if os.path.exists(_notes) else []),
          "Details, strata, depth, calibration, M health and compute: `results/p2_results.json`. Charts: `results/charts.json`.", "",
          "```json", json.dumps(clean(C), separators=(",", ":")), "```", ""]
    return "\n".join(L)


def catalogue(R, out_dir, write_root):
    """E-P2-0..4 as catalogue/<id>.json (the trader-economy catalogue format) and CATALOGUE.md sections."""
    V, ap = R["validity"], R["arm_primary"]
    f = lambda d: pts(d)
    toy = V.get("toy", {})
    pn = toy.get("per_n") or {}
    toy_txt = "; ".join(f"n={n}: " + ", ".join(f"{a} {100 * pn[n][a]['mean_in_window']:.0f}%" for a in ("M-tree", "M-global", "M-tile")
                                                 if isinstance(pn[n].get(a), dict)) for n in pn) or "not run"
    gate = V.get("gate", {})
    E = []
    E.append(dict(id="E-P2-0", question="Stage-0 toy: can a tree of latent codes hold a long-range parity choice that a factorised belief cannot?",
                  setup="1D broadcast parity (two equally likely answers per string), depth-recurrent 1D trunk, progressive market loss; "
                        "U (factorised), M-global (K=2), M-tile (8-position leaves), M-tree (binary tree, K=2, <= 2 levels in training, up to 4 at test); "
                        "3 seeds, 3000 steps, CPU. Readout: exact match while U has not resolved (U never resolves here).",
                  result=f"Exact match in U's unresolved window: {toy_txt}. Pre-registered pass (M-tree >= 40% and M-tile <= 5% at n=64 and 128): "
                         f"{toy.get('pass')}.",
                  status="Informative" if toy.get("pass") else "Problem",
                  files=["puzzles/P2/toy.py", "puzzles/P2/results/toy/summary.json"]))
    E.append(dict(id="E-P2-1", question="Are the P2 maze sets, decoders and training recipe valid (unit tests, decoder trap, U-canon gate, floor rule)?",
                  setup="Frozen test/dev/ext sets (maze-dataset lattices, percolation p in {0, 0.05, 0.1}), n=9 training pool with no hash overlap; "
                        "exact DAG and tree dynamic programs checked against enumeration; every decoder on constant beliefs and random nets must stay <= 1% VSP; "
                        "U-canon s0 gate >= 95% at n=9 and >= 80% at n=19 (p=0, T <= 300); floor rule: U-canon dev no-tie VSP >= 50% at T=300.",
                  result=f"Unit tests pass: {V.get('unit_tests', {}).get('pass')}; "
                         + (f"trap test incomplete (spot preemption): {V['trap']['cells_done']}/{V['trap']['cells_total']} cells, "
                            f"max VSP {100 * V['trap']['max_vsp_done_cells']:.1f}% in those, missing {', '.join(V['trap']['missing_cells'])}; counted as not passed"
                            if V.get('trap', {}).get('incomplete') else
                            f"trap max VSP {V.get('trap', {}).get('max_vsp')} (pass {V.get('trap', {}).get('pass_')})")
                         + f"; gate {({k: round(v['best_dev_vsp_T_le_300'], 3) for k, v in gate.items() if isinstance(v, dict)})} "
                         f"pass {gate.get('pass', gate.get('pass_'))}; floor sizes {R['primary_sizes']} "
                         f"({ {k: (round(v['vsp'], 3) if v['vsp'] is not None else None) for k, v in R['floor'].items()} }); "
                         f"GPU utilisation in the pilot {V.get('gpu_util_pilot', {}).get('mean')}%.",
                  status="Informative" if R["validity_all"] else "Problem",
                  files=["puzzles/P2/tests.py", "puzzles/P2/gen.py", "puzzles/P2/results/p2_results.json"]))
    ga, gb, gc = R["gain_a"], R["gain_b"], R["gain_c"]
    res2 = (f"Tie VSP at T=300 (primary set, {R['primary_n_mazes']} mazes, n in {R['primary_sizes']}): "
            + ", ".join(f"{a} {f(ap[a][dec_of(a)])}" for a in ARMS) + ". "
            + " ".join(f"M-tree - {b}: {f(ga[b])} points (bootstrap [{100 * ga[b]['boot']['lo']:.1f}, {100 * ga[b]['boot']['hi']:.1f}]), "
                       f"DiD {f(gb[b]) if gb.get(b) else 'n/a'}." for b in ("U-canon", "U-min") if ga.get(b))
            + (f" p=0 cost (U-canon - M-tree): {f(gc)}." if gc else "")
            + f" GAIN (a) {R['GAIN']['a']}, (b) {R['GAIN']['b']}, (c) {R['GAIN']['c']}."
            + " Falsified: " + (", ".join(k for k, v in R["falsify"].items() if v.get("falsified")) or "none") + "."
            + " Per n, tie VSP M-tree - U-min (all 4 sizes, descriptive): "
            + ", ".join(f"n{n} {f(d['M-tree - U-min'])}" for n, d in R["per_n"].items() if d.get("M-tree - U-min")) + ".")
    E.append(dict(id="E-P2-2", question="On mazes with tied shortest paths, does an exactly priced quadtree latent-parent market (M-tree) find a valid shortest path more often than commitment baselines (canonical labels, hard-EM)?",
                  setup="Depth-recurrent conv net (about 107k params, trained on n=9 with the progressive market loss, 5000 steps, 5 paired seeds), tested on "
                        "n=13-33 at T=300. M-tree: 8x8 leaves, K=4 codes per quadtree node, exact upward-pass joint log score on uniformly sampled tie labels, "
                        "decoded by exact tree MAP. Baselines: U-canon (canonical move-order labels), U-min (hard-EM Viterbi labels), U (sampled labels), "
                        "U-any (exact P(some shortest path)); decoders get at most 5 local cleanup sweeps, no global search.",
                  result=res2, status=R["status"],
                  files=["puzzles/P2/jm.py", "puzzles/P2/train.py", "puzzles/P2/eval.py", "puzzles/P2/report.py", "puzzles/P2/RESULTS.md",
                         "puzzles/P2/results/p2_results.json"]))
    lt, lti = R["latent"]["M-tree"], R["latent"]["M-tile"]
    s3 = R["secondary"].get("M-tree - M-tile, >32 px")
    fr = lambda d: f"{d['mean']:.2f} [{d['lo']:.2f}, {d['hi']:.2f}]" if d.get("k", 0) > 1 else "n/a"
    E.append(dict(id="E-P2-3", question="Does the M-tree code carry the route choice, and do independent per-tile mixtures (M-tile) suffice?",
                  setup="256 ancestral samples of (codes, pixels) per primary tie maze; branch = which tied route the sample's best path takes; "
                        "I(code at the lowest node covering the contested region; branch) / H(branch). M-tile: same model without cross-tile transitions.",
                  result=f"M-tree I/H {fr(lt['ratio'])} (pooled {fr(lt['pooled_ratio'])}; pass >= 0.8: {R['LATENT_pass']}). "
                         f"M-tile I/H {fr(lti['ratio'])} on the {100 * (lti['share_tie_single_node'].get('mean') or 0):.0f}% of tie mazes whose contested region fits one tile. "
                         f"Contested > 32 px: M-tree - M-tile {f(s3) if s3 else 'n/a'} points (Holm p {s3.get('p_holm', float('nan')) if s3 else float('nan'):.3g}). "
                         f"Share of tie mazes with the contested region inside one tile: {100 * R['share_tie_one_tile']:.0f}%.",
                  status="Informative", files=["puzzles/P2/eval.py", "puzzles/P2/results/p2_results.json"]))
    dp = R["depth"]
    dtxt = "; ".join(f"{a}: n33 T300 {f(dp[a]['n33_T300_first50'])}, T3000 {f(dp[a]['n33_T3000_first50'])}, n59 T300 {f(dp[a]['n59_T300'])}, T1000 {f(dp[a]['n59_T1000'])}"
                     for a in DECISIVE if a in dp)
    E.append(dict(id="E-P2-4", question="Does tie VSP keep rising with more iterations and hold on larger mazes, and does the canonical rule extrapolate?",
                  setup="Decisive arms (M-tree, U-canon, U-min): T = 300 -> 1000 on 100 tie mazes per cell (n <= 33), T = 3000 on 50 per p at n = 33, "
                        "n = 59 at T = 300 and 1000; slope of tie VSP against n per seed.",
                  result=dtxt + ". Deep-eval seeds: " + ", ".join(f"{a} {max((v.get('k', 0) for v in dp[a].values()), default=0)}" for a in DECISIVE if a in dp)
                         + ". Slope per n (VSP per unit n): " + (", ".join(f"{a} {R['slopes_per_n'][a]['mean']:.4f}" for a in R['slopes_per_n']
                                                                            if R['slopes_per_n'][a]['k']) or "n/a (needs >= 2 floor sizes)") + ".",
                  status="Informative", files=["puzzles/P2/eval.py", "puzzles/P2/results/p2_results.json"]))
    E = clean(E)
    os.makedirs(os.path.join(out_dir, "catalogue"), exist_ok=True)
    for e in E:
        json.dump(e, open(os.path.join(out_dir, "catalogue", e["id"] + ".json"), "w"), indent=1)
    if write_root:
        for e in E:
            json.dump(e, open(os.path.join(ROOT, "catalogue", e["id"] + ".json"), "w"), indent=1)
        cm = os.path.join(ROOT, "CATALOGUE.md")
        txt = open(cm).read() if os.path.exists(cm) else ""
        kept, skip = [], False
        for line in txt.split("\n"):
            if line.startswith("## "):
                skip = line.startswith("## E-P2-")
            if not skip:
                kept.append(line)
        txt = "\n".join(kept).rstrip("\n")
        for e in E:
            txt += (f"\n\n## {e['id']}: {e['question']}\n\nStatus: {e['status']}\n\nSetup: {e['setup']}\n\nResult: {e['result']}\n\n"
                    "Files: " + ", ".join(f"`{x}`" for x in e["files"]))
        open(cm, "w").write(txt + "\n")
    return E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="*")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--sizes", default="", help="sensitivity only: force the primary sizes, e.g. 13,19 (not pre-registered)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--md", default=os.path.join(HERE, "RESULTS.md"))
    ap.add_argument("--catalogue", action="store_true", help="also write ../../catalogue/E-P2-*.json and CATALOGUE.md sections")
    a = ap.parse_args()
    R, _ = build(a)
    R["status"] = status(R)
    C = charts(R)
    os.makedirs(a.out, exist_ok=True)
    json.dump(clean(R), open(os.path.join(a.out, "p2_results.json"), "w"), indent=1)
    json.dump(clean(C), open(os.path.join(a.out, "charts.json"), "w"), indent=1)
    open(a.md, "w").write(markdown(R, C))
    catalogue(R, a.out, a.catalogue and not a.sizes)  # a sensitivity run never writes the catalogue
    print(json.dumps(clean({k: R[k] for k in ("status", "GAIN", "LATENT_pass", "primary_sizes", "validity_all")})))


if __name__ == "__main__":
    main()
