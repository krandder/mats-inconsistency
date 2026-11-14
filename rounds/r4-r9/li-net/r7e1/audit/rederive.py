#!/usr/bin/env python3
"""R7-E1 audit, check 4: re-derive the headline numbers from raw.json with my own numpy code (no import of e1v4.py, e1v3.py,
analyze7.py or any owner module). Reads raw.json / summary.json in place.
Writes rederive.json (all my per-seed numbers) and prints the comparison with summary.json (per seed) and
analysis/results.json (3-seed means and CIs). Usage: PYTHONPATH=/tmp/r4e3a/pylib python3 rederive.py > rederive-output.txt"""
import json
import math
import time

import numpy as np

R7 = "/home/agent/workspaces/eurisko-study/li-net/r7e1"
SEEDS = (101, 102, 103)
POOLS = ("mix", "scaled_mix", "mmd", "kc_K1", "kc_K3", "kc_K4", "econ_K3", "looped_lin", "jacobi", "looped_attn", "attn_K1",
         "attn_K3", "attn_K4")
RAWP = ("attn_K3", "scaled_mix", "mmd", "kc_K1", "kc_K3", "union")
RAWC = ("attn_K3", "jacobi", "mmd", "kc_K1", "kc_K3")
TAUS = ("0.3", "1.5", "2.5")
NS = (1, 2, 4, 8, 16)

# ---- own geometry: 20 horizontal 2x3 + 20 vertical 3x2 patches (row-major cells); NW = rows 0-2
patches = []
for r in range(5):
    for c in range(4):
        patches.append([(r + a) * 6 + c + b for a in range(2) for b in range(3)])
for r in range(4):
    for c in range(5):
        patches.append([(r + a) * 6 + c + b for a in range(3) for b in range(2)])
patches = np.array(patches)
PM = np.zeros((40, 36), dtype=bool)
for i, p in enumerate(patches):
    PM[i, p] = True
nnw = (patches < 18).sum(1)
IN_NW, IN_SE = nnw >= 3, (6 - nnw) >= 3
STRICT_NW, STRICT_SE = nnw >= 4, nnw <= 2


def ent(p):
    p = np.clip(p, 0, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -(np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0) + np.where(p < 1, (1 - p) * np.log(np.where(p < 1, 1 - p, 1)), 0))


def kl(p, q, eps=1e-6):
    p = np.clip(p, 0, 1)
    q = np.clip(q, eps, 1 - eps)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(p > 0, p * (np.log(np.where(p > 0, p, 1)) - np.log(q)), 0)
        b = np.where(p < 1, (1 - p) * (np.log(np.where(p < 1, 1 - p, 1)) - np.log1p(-q)), 0)
    return a + b


def kl_bound(p, q, eps=1e-6, h=5e-6):
    """first-order bound on |KL(p,q) - KL(p~,q~)| when p and q are each rounded to 5 decimals (|error| <= 5e-6)."""
    pc = np.clip(p, eps, 1 - eps)
    qc = np.clip(q, eps, 1 - eps)
    dq = np.abs(qc - pc) / (qc * (1 - qc))
    dp = np.abs(np.log(pc / (1 - pc)) - np.log(qc / (1 - qc)))
    return h * (dq + dp)


def jsd(p, q):
    return ent(0.5 * (p + q)) - 0.5 * (ent(p) + ent(q))


def cover(pt):
    return PM[pt].any(1).mean(1).mean()


mine = {}
for s in SEEDS:
    raw = json.load(open(f"{R7}/results/main/s{s}/raw.json"))
    ps = np.array(raw["pstar"])
    M = {}
    M["kl_p0"] = float(kl(ps, np.array(raw["mu0"])).mean())
    M["kl_p0_bound"] = float(kl_bound(ps, np.array(raw["mu0"])).mean())
    # P-FullSupport
    fN, fS = raw["fs_NW"], raw["fs_SE"]
    tiles_ok = all(set([0, 3, 8, 11, 16, 19]) <= set(r[:6]) for r in fN["patch"] + fS["patch"])
    M["fs_tiles_first6_all_boards"] = tiles_ok
    M["fs_coverage_NW"] = float(cover(np.array(fN["patch"])))
    M["fs_coverage_SE"] = float(cover(np.array(fS["patch"])))
    sk_nw = np.array(fN["patch"])[:, 6:]
    sk_se = np.array(fS["patch"])[:, 6:]
    for a in POOLS + ("union",):
        mN, mS = np.array(fN[a]), np.array(fS[a])
        M[f"fs_kl_{a}"] = float(0.5 * (kl(ps, mN).mean() + kl(ps, mS).mean()))
        M[f"fs_kl_{a}_bound"] = float(0.5 * (kl_bound(ps, mN).mean() + kl_bound(ps, mS).mean()))
        M[f"fs_jsd_{a}"] = float(jsd(mN, mS).mean())
    M["fs_union_identical_NW_SE"] = bool(np.array_equal(np.array(fN["union"]), np.array(fS["union"])))
    # P-ActivePUCT
    for st in ("mixed", "clean"):
        for h in ("NW", "SE"):
            A = raw[f"ap_{st}_{h}"]
            op, pu, gr = np.array(A["open_patch"]), np.array(A["puct_patch"]), np.array(A["greedy_patch"])
            inh = IN_NW if h == "NW" else IN_SE
            strict = STRICT_NW if h == "NW" else STRICT_SE
            M[f"ap_{st}_coverage_open_{h}"] = float(cover(op[:, :24]))
            M[f"ap_{st}_coverage_puct_{h}"] = float(cover(pu[:, :24]))
            M[f"ap_{st}_coverage_greedy_{h}"] = float(cover(gr))
            M[f"ap_{st}_coverage_open30_{h}"] = float(cover(op))
            M[f"ap_{st}_coverage_puct30_{h}"] = float(cover(pu))
            for kind, arr in (("puct", pu), ("greedy", gr)):
                M[f"ap_{st}_{kind}_wave1_pref_share_{h}"] = float(inh[arr[:, :6]].mean())
                M[f"ap_{st}_{kind}_wave1_strict_share_{h}"] = float(strict[arr[:, :6]].mean())
                nw = arr.shape[1] // 6
                M[f"ap_{st}_{kind}_wave_pref_share_{h}"] = [float(inh[arr[:, 6 * i:6 * i + 6]].mean()) for i in range(nw)]
                M[f"ap_{st}_{kind}_coverage_by_wave_{h}"] = [float(cover(arr[:, :6 * (i + 1)])) for i in range(nw)]
                M[f"ap_{st}_{kind}_distinct_within_wave"] = bool(all(len(set(r[6 * i:6 * i + 6])) == 6 for r in arr.tolist() for i in range(nw)))
            M[f"ap_{st}_open_pref_share_{h}"] = float(inh[op[:, :24]].mean())
        for kind in ("open", "puct", "greedy", "open30", "puct30"):
            M[f"ap_{st}_coverage_{kind}"] = 0.5 * (M[f"ap_{st}_coverage_{kind}_NW"] + M[f"ap_{st}_coverage_{kind}_SE"])
        for kind, pools in (("open", RAWP), ("puct", RAWP), ("greedy", ("attn_K3",)), ("open30", ("attn_K3",)), ("puct30", ("attn_K3",))):
            for a in pools:
                mN, mS = np.array(raw[f"ap_{st}_NW"][f"{kind}_{a}"]), np.array(raw[f"ap_{st}_SE"][f"{kind}_{a}"])
                M[f"ap_{st}_{kind}_kl_{a}"] = float(0.5 * (kl(ps, mN).mean() + kl(ps, mS).mean()))
                M[f"ap_{st}_{kind}_kl_{a}_bound"] = float(0.5 * (kl_bound(ps, mN).mean() + kl_bound(ps, mS).mean()))
                M[f"ap_{st}_{kind}_jsd_{a}"] = float(jsd(mN, mS).mean())
    # controlled bags
    C = raw["controlled"]
    astar = np.array(C["astar"])
    bgp = np.array(C["bg_patch"])
    M["ctrl_bg_disjoint_from_astar"] = bool(all((PM[bgp[i]] & PM[astar[i]][None]).sum() == 0 for i in range(len(astar))))
    for t in TAUS:
        for N in NS:
            for a in ("attn_K3", "jacobi"):
                M[f"W_{a}_t{t}_N{N}"] = float(np.mean(C[f"t{t}_N{N}_W_{a}"]))
            for a in RAWC:
                M[f"mse_{a}_t{t}_N{N}"] = float(((np.array(C[f"t{t}_N{N}"][a]) - ps) ** 2).mean())
        for a in ("attn_K3", "jacobi"):
            M[f"{a}_W8/W1_t{t}"] = M[f"W_{a}_t{t}_N8"] / M[f"W_{a}_t{t}_N1"]
            M[f"{a}_W16/W1_t{t}"] = M[f"W_{a}_t{t}_N16"] / M[f"W_{a}_t{t}_N1"]
        for a in RAWC:
            M[f"mse_{a}_t{t}"] = float(np.mean([M[f"mse_{a}_t{t}_N{N}"] for N in NS]))
    # key geometry from the saved K3 keys (4 decimals), descriptive
    kb = raw["key_bags"]
    OV = (PM[:, None, :] & PM[None, :, :]).sum(2)
    cs_all, ov_all = [], []
    for pt, K in zip(kb["patch"], kb["keys_K3"]):
        pt, K = np.array(pt), np.array(K)
        iu = np.triu_indices(pt.shape[1], 1)
        cs_all.append(np.einsum("nid,nid->ni", K[:, iu[0]], K[:, iu[1]]).ravel())
        ov_all.append(OV[pt[:, iu[0]], pt[:, iu[1]]].ravel())
    cs, ov = np.concatenate(cs_all), np.concatenate(ov_all)
    M["key_K3_pearson"] = float(np.corrcoef(cs, ov / 6)[0, 1])
    mine[s] = M
    del raw

json.dump(mine, open("rederive.json", "w"), indent=1)

# ---- comparison with summary.json (per seed)
print("run at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
rows = []
for s in SEEDS:
    T = json.load(open(f"{R7}/results/main/s{s}/summary.json"))["test"]
    C = T["controlled"]
    for k, v in mine[s].items():
        if k.endswith("_bound") or isinstance(v, bool):
            continue
        ref = None
        if k in T and not isinstance(v, list):
            ref = T[k]
        elif k.startswith("W_"):
            a, t, N = k[2:].rsplit("_t", 1)[0], k.split("_t")[-1].split("_N")[0], k.split("_N")[-1]
            ref = C[f"t{t}_N{N}"][f"W_{a}"]
        elif k.startswith("mse_") and "_N" in k:
            a = k[4:].split("_t")[0]
            t, N = k.split("_t")[-1].split("_N")
            ref = C[f"t{t}_N{N}"][f"mse_{a}"]
        elif isinstance(v, list) and k in T:
            ref = T[k]
        if ref is None:
            continue
        if isinstance(v, list):
            d = max(abs(a - b) for a, b in zip(v, ref))
        else:
            d = abs(v - ref)
        bnd = mine[s].get(k + "_bound")
        rows.append((d, s, k, v, ref, bnd))
cats = {}
for d, s, k, v, ref, bnd in rows:
    cat = ("KL" if "_kl_" in k or k == "kl_p0" else "JSD" if "_jsd_" in k else "coverage/share" if ("coverage" in k or "share" in k)
           else "W(N)" if k.startswith("W_") else "MSE" if k.startswith("mse_") else "other")
    c = cats.setdefault(cat, [0, None])
    c[0] += 1
    if c[1] is None or d > c[1][0]:
        c[1] = (d, s, k, v, ref, bnd)
print("== per-seed comparison with summary.json: %d statistics" % len(rows))
for cat, (n, worst) in cats.items():
    d, s, k, v, ref, bnd = worst
    print(f"{cat:15s} n={n:4d} max|mine - summary| = {d:.3g} at seed {s} {k}: mine {v} vs summary {ref}"
          + (f" (first-order rounding bound {bnd:.3g})" if bnd is not None else ""))
klrows = [r for r in rows if r[5] is not None]
out_of_bound = [r for r in klrows if r[0] > r[5]]
print(f"KL statistics with a rounding bound: {len(klrows)}; outside their first-order rounding bound: {len(out_of_bound)}")
for r in sorted(out_of_bound, reverse=True)[:10]:
    print("   outside:", r)
worst = max(rows)
print("LARGEST per-seed discrepancy:", worst)
json.dump([list(r) for r in sorted(rows, reverse=True)], open("rederive-diffs.json", "w"), indent=0)
