"""Evaluate a P5 checkpoint: roll the trained recurrence out on the fixed sets, record the prediction at T at fixed
test-time iterations, pick the best iteration on validation, stability to 64,000 iterations, cross-write ablation,
bank occupancy and linear probes of prefix parity. Writes <out>.json (metrics) and <out>.npz (float16 logits)."""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import mazes
import p5

REC = [64, 128, 256, 640, 1280, 2560, 6400]
STAB_REC = [64, 640, 6400, 12800, 25600, 64000]
GROUPS = {  # sets rolled out together (same grid shape)
    "g9": ["val9", "test9"], "g13": ["val13", "test13"], "g19": ["test19", "long19"], "g33": ["test33", "long33"],
    "g59": ["test59", "long59"], "lat30": ["lat30"], "lat100": ["lat100"], "stab33": ["stab33"],
}
DENSE_GROUPS = {"gd9": ["valdense9", "testdense9"]}
CELL_BUDGET = 1_200_000


def load_model(path, dev):
    ck = torch.load(path, map_location="cpu")
    model = p5.build(ck["arm"])
    model.load_state_dict({k: v.float() for k, v in ck["state"].items()})
    return model.to(dev).eval(), ck


@torch.no_grad()
def rollout(model, x, mask, rec, ablate_cross=False, stats=None, dense=False, feats_at=None):
    """Run the recurrence to max(rec); return {it: logits} at the scored cell (sparse) or full maps (dense)."""
    out = {}
    dtr = isinstance(model, p5.DTRecall)
    feats = None
    if dtr:
        h = model.init(x)
    else:
        b, c = p5.init_state(model, x)
        n = model.n
        eye = torch.eye(n, device=x.device)[None, :, :, None, None, None]
    for t in range(1, max(rec) + 1):
        if dtr:
            h = model.step(h, x)
        else:
            db, w, g, _ = model(x, b, c)
            if ablate_cross:
                w = w * eye
            b = b + (db * mask[:, None]).sum(1, keepdim=True)
            if stats is not None and t in (64, 640):
                stats.setdefault(f"gate_mean_{t}", []).append(float(g.mean()))
                stats.setdefault(f"bank_abs_tanh_{t}", []).append(float(torch.tanh(c).abs().mean()))
                stats.setdefault(f"bank_occupied_{t}", []).append(float((torch.tanh(c).abs() > 0.5).float().mean()))
                a = w.abs()
                stats.setdefault(f"cross_write_share_{t}", []).append(float(1 - (a * eye).sum() / a.sum().clamp(min=1e-12)))
            c = (1 - g) * c + w.sum(1)
        if t == feats_at:
            feats = (h if dtr else torch.tanh(c).flatten(1, 2)).clone()
        if t in rec:
            lg = model.logit(h) if dtr else b[:, 0]
            out[t] = lg.clone() if dense else (lg * mask).sum((1, 2))
    return out, feats


def chunks(n, cells):
    bs = max(1, CELL_BUDGET // cells)
    return [(i, min(n, i + bs)) for i in range(0, n, bs)]


def metrics(logit, y):
    p = torch.sigmoid(logit.double())
    acc = float(((logit > 0).double() == y).double().mean())
    nll = float(F.binary_cross_entropy_with_logits(logit.double(), y, reduction="mean"))
    conf = torch.where(p > 0.5, p, 1 - p)
    corr = ((p > 0.5).double() == y).double()
    ece = 0.0
    for lo in np.linspace(0.5, 1.0, 11)[:-1]:
        sel = (conf >= lo) & (conf < lo + 0.05 + (1e-9 if lo >= 0.949 else 0))
        if sel.any():
            ece += float(sel.double().mean()) * abs(float(corr[sel].mean()) - float(conf[sel].mean()))
    return dict(acc=acc, nll=nll, ece=ece)


def depth_bins(dist, L):
    r = dist / L
    edges = [0, 1, 3, 10, 30, np.inf]
    return np.digitize(r, edges[1:-1], right=True)  # 0: <=1, 1: 1-3, 2: 3-10, 3: 10-30, 4: 30+


def probe(feat_tr, y_tr, feat_te, y_te):
    """Logistic probe (L-BFGS, small L2) on per-cell features; returns held-out accuracy."""
    lin = torch.nn.Linear(feat_tr.shape[1], 1).double()
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=200)

    def clo():
        opt.zero_grad()
        l = F.binary_cross_entropy_with_logits(lin(feat_tr)[:, 0], y_tr) + 1e-4 * lin.weight.pow(2).sum()
        l.backward()
        return l
    opt.step(clo)
    with torch.no_grad():
        return float(((lin(feat_te)[:, 0] > 0).double() == y_te).double().mean())


def probe_run(model, sets, dev, at=64):
    """Prefix-parity probe from the memory (banks, or DT-Recall hidden) at on-path cells of test9."""
    s = sets["test9"]
    x = torch.tensor(s["x"], dtype=torch.float32, device=dev)
    mask = torch.tensor(s["mask"], dtype=torch.float32, device=dev)
    _, f = rollout(model, x, mask, [at], feats_at=at)
    onp = torch.tensor(s["extra"][:, 0], dtype=torch.bool)
    yy = torch.tensor(s["extra"][:, 1], dtype=torch.float64)
    f = f.double().cpu().permute(0, 2, 3, 1)  # [N, H, W, C]
    half = len(f) // 2
    tr, te = slice(0, half), slice(half, None)
    return probe(f[tr][onp[tr]], yy[tr][onp[tr]], f[te][onp[te]], yy[te][onp[te]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", default="all")  # comma list, 'val' for tuning, 'all'
    ap.add_argument("--frac", type=float, default=1.0, help="fraction of long-path and stability sets kept")
    ap.add_argument("--max_rec", type=int, default=0, help="benchmark: stop at this many iterations")
    ap.add_argument("--extras", action="store_true", help="ablation, occupancy and probes")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_model(a.ckpt, dev)
    dense = ck["args"]["dense"]
    meta = json.load(open(os.path.join(mazes.DATA, "meta.json")))
    L = meta["L_train"]
    groups = DENSE_GROUPS if dense else GROUPS
    if a.groups == "val":
        names = ["gd9"] if dense else ["g9", "g13"]
    elif a.groups == "all":
        names = list(groups)
    else:
        names = a.groups.split(",")
    res = dict(ckpt=a.ckpt, arm=ck["arm"], seed=ck["seed"], args=ck["args"], L_train=L, frac=a.frac, sets={}, timing={})
    logits = {}
    sets = {}
    for gname in names:
        t0 = time.time()
        for sname in groups[gname]:
            s = mazes.load_set(sname)
            if a.frac < 1 and (sname.startswith("long") or sname.startswith("stab")):
                k = max(1, int(round(len(s["x"]) * a.frac)))
                s = {kk: v[:k] for kk, v in s.items()}
            sets[sname] = s
            if sname == "stab33":
                rec = STAB_REC
            elif sname.startswith("long") or sname.startswith("lat"):
                rec = sorted(set(REC + [max(6400, 2 * int(s["dist"].max()))]))
            else:
                rec = REC
            if a.max_rec:
                rec = [r for r in rec if r < a.max_rec] + [a.max_rec]
            N, _, H, W = s["x"].shape
            outs = {r: [] for r in rec}
            stats = {} if a.extras and not isinstance(model, p5.DTRecall) and sname in ("test19",) else None
            for i0, i1 in chunks(N, H * W):
                x = torch.tensor(s["x"][i0:i1], dtype=torch.float32, device=dev)
                m = torch.tensor(s["mask"][i0:i1], dtype=torch.float32, device=dev)
                o, _ = rollout(model, x, m, rec, stats=stats, dense=dense)
                for r in rec:
                    outs[r].append(o[r].float().cpu())
            y = torch.tensor(s["y"], dtype=torch.float64)
            if not dense:
                y = (y * torch.tensor(s["mask"], dtype=torch.float64)).sum((1, 2))
            bins = depth_bins(s["dist"], L)
            ent = dict(n=N, H=H, W=W, dist_mean=float(s["dist"].mean()), dist_max=int(s["dist"].max()), it={})
            for r in rec:
                lg = torch.cat(outs[r])
                mt = metrics(lg.double().flatten(), y.flatten())
                if dense:
                    mt["exact_maze"] = float((((lg > 0).double() == y).all(-1).all(-1)).double().mean())
                else:
                    corr = ((lg > 0).double() == y).numpy()
                    mt["acc_by_depth_bin"] = [float(corr[bins == k].mean()) if (bins == k).any() else None for k in range(5)]
                    mt["n_by_depth_bin"] = [int((bins == k).sum()) for k in range(5)]
                ent["it"][r] = mt
                if not dense:
                    logits[f"{sname}@{r}"] = lg.numpy().astype(np.float16)
            if stats:
                ent["bank_stats"] = {k: float(np.mean(v)) for k, v in stats.items()}
            res["sets"][sname] = ent
        res["timing"][gname] = time.time() - t0
        print(gname, "%.1fs" % (time.time() - t0), {sn: {r: round(v["acc"], 4) for r, v in res["sets"][sn]["it"].items()}
                                                     for sn in groups[gname]}, flush=True)
    # best iteration on validation
    vn = ["valdense9"] if dense else [v for v in ("val9", "val13") if v in res["sets"]]
    if vn:
        its = [r for r in REC if all(r in res["sets"][v]["it"] for v in vn)]
        score = {r: float(np.mean([res["sets"][v]["it"][r]["acc"] for v in vn])) for r in its}
        res["best_it"] = max(score, key=lambda r: (score[r], -r)) if score else None
        res["val_score"] = score
    if a.extras and not dense:
        ex = {}
        if not isinstance(model, p5.DTRecall) and model.banks:
            for sname in ("test9", "test19"):
                s = sets.get(sname) or mazes.load_set(sname)
                x = torch.tensor(s["x"], dtype=torch.float32, device=dev)
                m = torch.tensor(s["mask"], dtype=torch.float32, device=dev)
                y = torch.tensor(s["y"] * s["mask"], dtype=torch.float64).sum((1, 2))
                r = res.get("best_it") or 64
                o, _ = rollout(model, x, m, [r], ablate_cross=True)
                ex[f"{sname}_acc_no_cross_writes@{r}"] = metrics(o[r].double().cpu(), y)["acc"]
                ex[f"{sname}_acc@{r}"] = res["sets"].get(sname, {}).get("it", {}).get(r, {}).get("acc")
        if isinstance(model, p5.DTRecall) or model.banks:
            if "test9" not in sets:
                sets["test9"] = mazes.load_set("test9")
            ex["probe_prefix_parity_trained"] = probe_run(model, sets, dev)
            torch.manual_seed(12345)
            rnd = p5.build(ck["arm"]).to(dev).eval()
            ex["probe_prefix_parity_random_init"] = probe_run(rnd, sets, dev)
        res["extras"] = ex
    json.dump(res, open(a.out + ".json", "w"))
    if logits:
        np.savez_compressed(a.out + ".npz", **logits)


if __name__ == "__main__":
    main()
