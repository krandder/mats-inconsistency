"""P15 gate G0, data parts: base rate, majority baseline, best T<=1 predictor (radius-1 input-window lookup
fitted on training data, scored in-distribution), distinct D=8 training instances, KS test of our 5x5-cell
generator against easy-to-hard '9x9' (5x5 cells) on path length and tau, and the pixel<->cell round trip on
every converted easy-to-hard maze. Arm C's part of G0 is read from the grid results in analyze.py.
Usage: python g0.py   (data dir: $P15_DATA, default /root/ext/data)"""
import json
import os
import sys
import time

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p15  # noqa: E402

DATA = os.environ.get("P15_DATA", "/root/ext/data")
GD = os.path.join(HERE, "gates")


def window_keys(task, x, valid):
    """x (M,N,Cx) ints -> per-cell integer key of the radius-1 input window (null = out of range/padding)."""
    M, N, Cx = x.shape
    code = (x.astype(np.int64) * (1 << np.arange(Cx))[None, None]).sum(-1)   # (M,N)
    base = 1 << Cx
    code = np.where(valid > 0, code, base)                                    # padding = null
    nbr = task.nbr(N if task.kind == "prefix" else int(round(N ** 0.5))).numpy()
    ext = np.concatenate([code, np.full((M, 1), base)], 1)                    # null index N
    w = ext[:, nbr]                                                           # (M,N,W)
    key = np.zeros((M, N), np.int64)
    for j in range(w.shape[-1]):
        key = key * (base + 1) + w[..., j]
    return key


def t1_predictor(task, train, test):
    ktr = window_keys(task, train["x"], train["valid"]).ravel()
    ytr = train["y"].ravel()
    mtr = train["valid"].ravel() > 0
    ktr, ytr = ktr[mtr], ytr[mtr]
    Cy = task.Cy
    uk, inv = np.unique(ktr, return_inverse=True)
    cnt = np.zeros((len(uk), Cy))
    np.add.at(cnt, (inv, ytr), 1)
    maj_global = int(np.bincount(ytr, minlength=Cy).argmax())
    table = cnt.argmax(1)
    kte = window_keys(task, test["x"], test["valid"])
    pos = np.searchsorted(uk, kte)
    pos = np.clip(pos, 0, len(uk) - 1)
    hit = uk[pos] == kte
    pred = np.where(hit, table[pos], maj_global)
    ok = ((pred == test["y"]) | (test["valid"] == 0)).all(1)
    cellacc = ((pred == test["y"]) * test["valid"]).sum() / test["valid"].sum()
    return {"exact": float(ok.mean()), "cellacc": float(cellacc), "unseen_window_frac": float(1 - hit.mean()),
            "n_windows_train": int(len(uk))}


def baselines(task, d):
    y, v = d["y"], d["valid"]
    base = float((y * v).sum() / v.sum())
    out = {"base_rate_y1": base}
    for c in range(task.Cy):
        out[f"exact_all_{c}"] = float((((y == c) | (v == 0)).all(1)).mean())
    out["majority_exact"] = max(out[f"exact_all_{c}"] for c in range(task.Cy))
    # T=0: z_0 is constant, so the prediction is one class for every cell (= the majority baseline)
    return out


def main():
    os.makedirs(GD, exist_ok=True)
    res = {"tasks": {}}
    t0 = time.time()
    for kind, D, mod in (("prefix", 8, 2), ("prefix", 16, 2), ("prefix", 32, 2), ("prefix", 16, 3),
                         ("maze", 8, 2), ("maze", 16, 2), ("maze", 32, 2)):
        task = p15.Task(kind, D, mod)
        ind = task.eval_set("indist")
        rng = np.random.default_rng([4242, D, mod])
        if kind == "prefix":
            tr = task.train_batch(rng, 20000)
        else:
            task.load_pool()
            tr = {k: task.pool[k] for k in ("x", "y")}
            tr["valid"] = np.ones(tr["y"].shape, np.float32)
        r = {"baselines": baselines(task, ind), "t1_lookup": t1_predictor(task, tr, ind)}
        if kind == "maze":
            r["indist_tau_mean"] = float(np.mean(ind["tau"]))
            r["pool_tau_hist"] = np.bincount(task.pool["tau"]).tolist()
        if D == 8:
            if kind == "prefix":
                r["distinct_train_instances"] = int(sum(2 ** L for L in range(D // 2, D + 1)))
                r["distinct_note"] = "all binary strings of length 4..8"
            else:
                keys = task.pool["x"].reshape(len(task.pool["x"]), -1)
                r["distinct_train_instances"] = int(len(np.unique(keys, axis=0)))
                r["distinct_note"] = f"unique mazes (walls+start+goal) in the {len(keys)}-maze training pool"
        r["pass_t1"] = r["t1_lookup"]["exact"] < 0.5 and r["baselines"]["majority_exact"] < 0.5
        res["tasks"][f"{kind}_D{D}_mod{mod}"] = r
        print(kind, D, mod, json.dumps(r)[:400], flush=True)
    # KS test vs easy-to-hard 9x9 (5x5 cells) + round trip
    rt = {}
    e2h = {}
    for name in ("maze_data_test_9", "maze_data_test_59"):
        p = os.path.join(DATA, name)
        if not os.path.isdir(p):
            rt[name] = "missing"
            continue
        img = np.load(os.path.join(p, "inputs.npy"), mmap_mode="r")
        sol = np.load(os.path.join(p, "solutions.npy"), mmap_mode="r")
        bad, badsol, badrule, M = 0, 0, 0, img.shape[0]
        taus, plen = [], []
        for m0 in range(0, M, 500):
            a = np.asarray(img[m0:m0 + 500], np.float32)
            s = np.asarray(sol[m0:m0 + 500], np.float32)
            x, n, y = p15.pixels_to_cells(a, s)
            bad += int((np.abs(p15.cells_to_pixels(x, n) - a).reshape(len(a), -1).max(1) > 0).sum())
            badsol += int((np.abs(p15.path_pixels(x, y, n) - s).reshape(len(a), -1).max(1) > 0).sum())
            xs = x.reshape(len(a), n, n, 6)
            end = (xs[..., 4] + xs[..., 5]) > 0
            ft = p15.dead_end_fill(xs[..., :4], end)
            yr = (ft >= p15.BIG).reshape(len(a), -1)
            badrule += int((yr != y.astype(bool)).any(1).sum())
            taus.append(np.where(ft < p15.BIG, ft, 0).reshape(len(a), -1).max(1))
            plen.append(y.sum(1))
        rt[name] = {"mazes": M, "n_cells": int(n), "pixel_roundtrip_fail": bad, "solution_roundtrip_fail": badsol,
                    "label_ne_dead_end_fill": badrule}
        e2h[name] = {"tau": np.concatenate(taus), "plen": np.concatenate(plen), "n": n}
        print(name, rt[name], flush=True)
    res["roundtrip"] = rt
    res["pass_roundtrip"] = all(isinstance(v, dict) and v["pixel_roundtrip_fail"] == 0 and v["solution_roundtrip_fail"] == 0
                                for v in rt.values())
    if "maze_data_test_9" in e2h:
        n = e2h["maze_data_test_9"]["n"]
        ours = p15.gen_mazes(10000, n, seed=4321)
        ks_t = stats.ks_2samp(ours["tau"], e2h["maze_data_test_9"]["tau"])
        ks_p = stats.ks_2samp(ours["y"].sum(1), e2h["maze_data_test_9"]["plen"])
        res["ks_vs_e2h_9"] = {"n_cells": n, "ours": 10000, "e2h": int(len(e2h["maze_data_test_9"]["tau"])),
                              "tau": {"D": float(ks_t.statistic), "p": float(ks_t.pvalue),
                                      "mean_ours": float(ours["tau"].mean()),
                                      "mean_e2h": float(e2h["maze_data_test_9"]["tau"].mean())},
                              "path_len": {"D": float(ks_p.statistic), "p": float(ks_p.pvalue),
                                           "mean_ours": float(ours["y"].sum(1).mean()),
                                           "mean_e2h": float(e2h["maze_data_test_9"]["plen"].mean())}}
        res["pass_ks"] = bool(ks_t.pvalue > 0.01 and ks_p.pvalue > 0.01)
        print("KS", res["ks_vs_e2h_9"], flush=True)
    res["wall_s"] = time.time() - t0
    json.dump(res, open(os.path.join(GD, "G0_data.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
