"""Evaluate trained P2 runs on the frozen sets.

  eval.py run   --ckpt results/runs/M-tree_s0.pt --arm M-tree [--deep] [--cells n9,n19]
  eval.py trap  (decoder trap test: constant beliefs and random-init nets, all decoders)

--part core|deep|extra splits one run's cells into three files, run in that priority order when GPU
time is short: core = the test cells of the primary analysis (n 13-33, dead-end start, all p) for every
arm, plus test p=0 non-dead-end cells for the decisive arms (GAIN c) and every dev cell of U-canon
(floor rule) -> <tag>.json; deep = the long-T and n=59 cells (--deep) -> <tag>__deep.json; extra = all
remaining cells of the full grid (n=9, non-dead-end p>0, other dev cells) -> <tag>__extra.json.
--part all (default) writes everything to <tag>.json. report.py merges the files of one tag.

Per run it writes results/eval/<tag>.npz with, for every (split, cell, T, decoder), the per-maze
VSP bits, and at T=300 on the test set per-maze coherence, near-miss distance, joint log loss on
uniformly sampled shortest paths, and for M arms code-carries-choice, total correlation and code
usage; plus results/eval/<tag>.json with cell means and calibration bins.
Decoders: thr (marginal > 0.5), cl (<= 5 local cleanup sweeps), tv (exact tree MAP; per-tile
argmax for M-tile). No decoder does a global graph search.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

import gen
import jm

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = (9, 13, 19, 25, 33)
PS = (0.0, 0.05, 0.1)
T_MAIN = (30, 60, 100, 200, 300)
CORE_SIZES = (13, 19, 25, 33)
DECISIVE = ("M-tree", "U-canon", "U-min")
DETAIL_FIELDS = ("coherent", "near_miss", "jll", "jll_fact", "acc_unc", "cc_ratio", "H_branch", "tc", "cc_level",
                 "mh_leaf_H", "mh_leaf_multi", "mh_root_H", "mh_root_multi")


def cell_tensors(split, n, p, de, dev, sel=None):
    c = gen.load_cell(split, n, p, de)
    idx = np.arange(len(c["s"])) if sel is None else sel(c)
    L = c["L"]
    op = torch.tensor(c["open"][idx], device=dev)
    s = torch.tensor(c["s"][idx], device=dev).long()
    g = torch.tensor(c["g"][idx], device=dev).long()
    D = torch.tensor(c["D"][idx], device=dev).long()
    ds, dg = jm.bfs_dist(op, s), jm.bfs_dist(op, g)
    dag = op & (ds + dg == D[:, None, None])
    assert torch.equal(dag, torch.tensor(c["dag"][idx], device=dev)), "stored DAG mismatch"
    B = len(idx)
    x = torch.zeros(B, 3, L, L, device=dev)
    x[:, 0] = op.float()
    ar = torch.arange(B, device=dev)
    x.view(B, 3, -1)[ar, 1, s] = 1
    x.view(B, 3, -1)[ar, 2, g] = 1
    cov = {k: c[k][idx] for k in ("tie", "contested", "tiles", "loops", "D")}
    return dict(x=x, op=op, s=s, g=g, D=D, ds=ds, dg=dg, dag=dag, L=L, B=B), cov


def decoders(model, st, b):
    m, _ = model.pixel_marg(st)
    op = b["op"]
    thr = jm.decode_thr(m) & op
    out = {"thr": thr, "cl": jm.decode_cl(thr, op, b["s"], b["g"])}
    if model.is_m:
        out["tv"] = jm.tree_map(st, op.float(), model.geom((b["L"], b["L"])))[0] & op
    return out, m


def vsp(S, b):
    return jm.vsp(S, b["ds"], b["dg"], b["D"], b["g"])


def primary_dec(arm):
    return "tv" if arm in jm.M_ARMS else "cl"


def uniform_marginals(b):
    """P(pixel on a uniformly drawn shortest path), exact by forward x backward path counts."""
    z = torch.zeros(b["ds"].shape, device=b["ds"].device)
    Ff = jm.dag_forward(z, b["ds"], b["dag"], "lse")
    Fb = jm.dag_forward(z, b["dg"], b["dag"], "lse")
    tot = Ff.flatten(1).gather(1, b["g"][:, None]).squeeze(1)
    return torch.where(b["dag"], torch.exp(Ff + Fb - tot[:, None, None]), torch.zeros_like(z))


def arm_logp(model, st, b, y):
    if model.is_m:
        return jm.score_m(st, y, b["op"].float(), model.geom((b["L"], b["L"])))
    return jm.score_unary(st["m"], y, b["op"].float())


def sub_state(st, sl):
    return {k: ([t[sl] for t in v] if isinstance(v, list) else v[sl]) for k, v in st.items()}


def rep_state(st, N):
    def r(t):
        return t.unsqueeze(0).expand(N, *t.shape).reshape(N * t.shape[0], *t.shape[1:])
    return {k: ([r(t) for t in v] if isinstance(v, list) else r(v)) for k, v in st.items() if k != "h"}


def m_details(model, st, b, C, m, nsamp=256, chunk=16, seed=0):
    """Per maze: code-carries-choice I(code at the lowest node covering the contested region; branch)
    / H(branch), H(branch), total correlation (nats), from nsamp ancestral samples."""
    g = model.geom((b["L"], b["L"]))
    B, L = b["B"], b["L"]
    gen_ = torch.Generator(device=b["op"].device).manual_seed(seed)
    R = torch.randn(L * L, dtype=torch.float64, device=b["op"].device, generator=gen_)
    ratio, Hb, tc, lvl = np.full(B, np.nan), np.full(B, np.nan), np.full(B, np.nan), np.full(B, -1)
    mh = {k: np.full(B, np.nan) for k in ("mh_leaf_H", "mh_leaf_multi", "mh_root_H", "mh_root_multi")}
    occ_all = g.tiles(g.pad(b["op"].float())) > 0  # (B,*G0) leaves holding open pixels
    q = torch.sigmoid(m[:, 0]).clamp(1e-7, 1 - 1e-7)
    Hq = (-(q * q.log() + (1 - q) * (1 - q).log()) * b["op"]).flatten(1).sum(1)
    for i0 in range(0, B, chunk):
        sl = slice(i0, min(B, i0 + chunk))
        bs = {k: (v[sl] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == B else v) for k, v in b.items()}
        Bc = bs["op"].shape[0]
        stc = sub_state(st, sl)
        codes, pix = jm.tree_sample(stc, g, nsamp, gen_)
        pix = pix & bs["op"].unsqueeze(0)
        flat = pix.reshape(nsamp * Bc, L, L)
        rep = lambda t: t.unsqueeze(0).expand(nsamp, *t.shape).reshape(nsamp * Bc, *t.shape[1:])
        logp = jm.score_m(rep_state(stc, nsamp), flat.float(), rep(bs["op"].float()), g).view(nsamp, Bc)
        tc[sl] = (Hq[sl] + logp.mean(0)).cpu().numpy()  # sum_i H(q_i) - H(Y), H(Y) = -E log P(Y)
        best = jm.viterbi_path(flat.float(), rep(bs["ds"]), rep(bs["dag"]), rep(bs["g"])).view(nsamp, Bc, L, L)
        # M health from the samples: code-usage entropy per node and share of nodes using > 1 code (>= 5% each)
        for lv_name, zz, occ in (("leaf", codes[0], occ_all[sl]), ("root", codes[-1], None)):
            zf = zz.reshape(nsamp, Bc, -1)
            cnt = torch.stack([(zf == k).float().mean(0) for k in range(model.K)], -1)  # (Bc,nodes,K)
            Hn = -(cnt.clamp(min=1e-12).log() * cnt).sum(-1)
            multi = ((cnt >= 0.05).sum(-1) >= 2).float()
            w = occ.reshape(Bc, -1).float() if occ is not None else torch.ones_like(Hn)
            den = w.sum(1).clamp(min=1)
            mh[f"mh_{lv_name}_H"][sl] = ((Hn * w).sum(1) / den).cpu().numpy()
            mh[f"mh_{lv_name}_multi"][sl] = ((multi * w).sum(1) / den).cpu().numpy()
        Cc = C[sl]
        key = ((best & Cc.unsqueeze(0)).flatten(2).double() @ R).cpu().numpy()  # (N,Bc) branch id
        for j in range(Bc):
            cj = torch.nonzero(Cc[j])
            if len(cj) == 0:
                continue
            lo, hi = cj.min(0).values.tolist(), cj.max(0).values.tolist()
            level = None
            for l in range(g.Lv + 1):
                size = g.leaf * 2 ** l
                if lo[0] // size == hi[0] // size and lo[1] // size == hi[1] // size:
                    level = l
                    break
            if level is None:
                continue  # M-tile: contested region spans several tiles, no single node
            size = g.leaf * 2 ** level
            z = codes[level][:, j].reshape(nsamp, -1)[:, (lo[0] // size) * (g.G0 // 2 ** level) + lo[1] // size]
            z = z.cpu().numpy()
            _, br = np.unique(np.round(key[:, j], 6), return_inverse=True)
            H_b = entropy_counts(np.bincount(br))
            H_bz = sum((z == c).mean() * entropy_counts(np.bincount(br[z == c])) for c in np.unique(z))
            Hb[i0 + j], lvl[i0 + j] = H_b, level
            ratio[i0 + j] = (H_b - H_bz) / H_b if H_b > 1e-9 else np.nan
    return ratio, Hb, tc, lvl, mh


def entropy_counts(c):
    p = c[c > 0] / c.sum()
    return float(-(p * np.log(p)).sum())


@torch.no_grad()
def eval_cell(model, b, cov, Ts, detail, seed):
    """Run max(Ts) iterations; VSP bits for every decoder at every T in Ts; details at T=300."""
    out = {"vsp": {}}
    st = model.init_state(b["x"])
    for t in range(1, max(Ts) + 1):
        st = model.step(st, b["x"])
        if t in Ts:
            decs, m = decoders(model, st, b)
            for k, S in decs.items():
                out["vsp"][(t, k)] = vsp(S, b).cpu().numpy()
            if detail and t == max(Ts):
                S = decs[primary_dec(model.arm)]
                out["coherent"] = jm.coherent(S, b["op"], b["s"], b["g"]).cpu().numpy()
                out["near_miss"] = jm.near_miss(S, b["ds"], b["dag"], b["D"], b["g"]).cpu().numpy()
                gen_ = torch.Generator(device=b["op"].device).manual_seed(seed)
                ys = [jm.sample_uniform_path(b["ds"], b["dag"], b["g"], gen=gen_).float() for _ in range(4)]
                out["jll"] = torch.stack([-arm_logp(model, st, b, y) for y in ys]).mean(0).cpu().numpy()
                if model.is_m:  # factorised score of M's own marginals on the same labels; TC = S_M - S_fact
                    out["jll_fact"] = torch.stack([-jm.score_unary(m, y, b["op"].float()) for y in ys]).mean(0).cpu().numpy()
                C = jm.contested_mask(b["ds"], b["dag"], b["D"])
                unc = b["op"] & ~C  # uncontested open pixels: on every shortest path (DAG) or on none
                out["acc_unc"] = (((S == b["dag"]) & unc).flatten(1).sum(1).float() / unc.flatten(1).sum(1).clamp(min=1)).cpu().numpy()
                # calibration on contested pixels: predicted marginal vs uniform-path marginal
                q = torch.sigmoid(m[:, 0])
                tgt = uniform_marginals(b)
                sel = C & (q >= 0.02) & (q <= 0.98)
                bins = torch.clamp((q[sel] * 10).long(), 0, 9)
                out["cal"] = dict(n=torch.bincount(bins, minlength=10).cpu().numpy(),
                                  q=torch.zeros(10, device=q.device).index_add_(0, bins, q[sel]).cpu().numpy(),
                                  t=torch.zeros(10, device=q.device).index_add_(0, bins, tgt[sel].float()).cpu().numpy(),
                                  n_contested=int(C.sum()), n_in_range=int(sel.sum()))
                if model.is_m:
                    zmap = jm.tree_map(st, b["op"].float(), model.geom((b["L"], b["L"])))[1]
                    g = model.geom((b["L"], b["L"]))
                    occ = g.tiles(g.pad(b["op"].float())) > 0  # leaves holding open pixels
                    out["code_hist"] = torch.bincount(zmap[occ].flatten(), minlength=model.K).cpu().numpy()
                    if model.kind != "global":
                        tie = torch.tensor(cov["tie"], device=C.device)
                        ii = torch.nonzero(tie).flatten()
                        if len(ii):
                            bt = {k: (v[ii] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == b["B"] else v) for k, v in b.items()}
                            bt["B"] = len(ii)
                            r, hb, tc, lv, mh = m_details(model, sub_state(st, ii), bt, C[ii], m[ii], seed=seed)
                            for k_, v_ in (("cc_ratio", r), ("H_branch", hb), ("tc", tc), ("cc_level", lv)) + tuple(mh.items()):
                                full = np.full(b["B"], np.nan)
                                full[ii.cpu().numpy()] = v_
                                out[k_] = full
    return out


def load(ckpt, arm, dev):
    model = jm.Model(arm)
    if ckpt != "random":
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.to(dev).eval()


def cmd_run(a):
    dev = torch.device(a.device)
    model = load(a.ckpt, a.arm, dev)
    tag = os.path.basename(a.ckpt).replace(".pt", "")
    os.makedirs(a.out, exist_ok=True)
    arrays, summ = {}, {"arm": a.arm, "tag": tag, "cells": {}, "cal": {}, "code_hist": {}}
    t0 = time.time()
    jobs = []
    sizes = [int(v[1:]) for v in a.cells.split(",")] if a.cells else SIZES

    def is_core(split, n, p, de):
        if n not in CORE_SIZES:
            return False
        if split == "test":
            return de or (p == 0.0 and a.arm in DECISIVE)
        return a.arm == "U-canon"  # dev cells: floor rule

    if a.part != "deep":
        for n in sizes:
            for p in PS:
                for de in (True, False):
                    for split, det in (("test", True), ("dev", False)):
                        if a.part == "all" or (a.part == "core") == is_core(split, n, p, de):
                            jobs.append((split, n, p, de, None, T_MAIN, det, ""))
    if a.part == "deep" or (a.part == "all" and a.deep):
        first_ties = lambda k: (lambda c: np.flatnonzero(c["tie"])[:k])
        for n in sizes:
            for p in (0.05, 0.1):
                Ts = (300, 1000, 3000) if n == 33 else (300, 1000)
                # T=1000 on 100 tie mazes per cell, T=3000 on the first 50 of them at n=33
                jobs.append(("test", n, p, True, first_ties(50 if n == 33 else 100), Ts, False, f"_deep{max(Ts)}"))
                if n == 33:
                    jobs.append(("test", n, p, True, lambda c: np.flatnonzero(c["tie"])[50:100], (300, 1000), False, "_deep1000_b"))
        for p in (0.05, 0.1):
            jobs.append(("ext", 59, p, True, None, (300, 1000), False, "_deep1000"))
    if a.tmax:
        jobs = [(sp, n, p, de, (lambda c, k=a.limit, s0=sel: (np.arange(len(c["s"])) if s0 is None else s0(c))[:k]) if a.limit else sel,
                 tuple(t for t in Ts if t <= a.tmax) or (a.tmax,), det, dk) for sp, n, p, de, sel, Ts, det, dk in jobs]
    for split, n, p, de, sel, Ts, detail, dkey in jobs:
        key = f"{split}{dkey.replace('_b', '')}|n{n}_p{p}_de{int(de)}" + ("_b" if dkey.endswith("_b") else "")
        b, cov = cell_tensors(split, n, p, de, dev, sel)
        bs = a.bs if b["L"] <= 67 else max(1, a.bs // 3)
        parts = []
        for i0 in range(0, b["B"], bs):
            sl = slice(i0, min(b["B"], i0 + bs))
            bb = {k: (v[sl] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == b["B"] else v) for k, v in b.items()}
            bb["B"] = bb["op"].shape[0]
            cc = {k: v[sl] for k, v in cov.items()}
            parts.append(eval_cell(model, bb, cc, set(Ts), detail, seed=a.seed + i0))
        res = {"vsp": {kk: np.concatenate([q["vsp"][kk] for q in parts]) for kk in parts[0]["vsp"]}}
        for f in DETAIL_FIELDS:
            if f in parts[0]:
                res[f] = np.concatenate([q[f] for q in parts])
        for (t, dec), v in res["vsp"].items():
            arrays[f"{key}|T{t}|{dec}"] = np.packbits(v.astype(bool))
        cellsum = {f"T{t}|{dec}": float(v.mean()) for (t, dec), v in res["vsp"].items()}
        cellsum["num"] = int(b["B"])
        for f in DETAIL_FIELDS:
            if f in res:
                arrays[f"{key}|{f}"] = res[f].astype(np.float32)
        if "cal" in parts[0]:
            summ["cal"][key] = {k: (sum(q["cal"][k] for q in parts).tolist() if isinstance(parts[0]["cal"][k], np.ndarray)
                                    else sum(q["cal"][k] for q in parts)) for k in parts[0]["cal"]}
        if "code_hist" in parts[0]:
            summ["code_hist"][key] = sum(q["code_hist"] for q in parts).tolist()
        summ["cells"][key] = cellsum
        print(key, json.dumps({k: round(v, 3) for k, v in cellsum.items() if k.startswith("T300") or k == "num"}),
              round(time.time() - t0, 1), flush=True)
    summ["wall_sec"] = round(time.time() - t0, 1)
    fn = tag + ("" if a.part in ("all", "core") else "__" + a.part)
    summ["part"] = a.part
    np.savez_compressed(os.path.join(a.out, fn + ".npz"), **arrays)
    json.dump(summ, open(os.path.join(a.out, fn + ".json"), "w"))
    print("EVAL_DONE", tag, summ["wall_sec"], flush=True)


@torch.no_grad()
def cmd_trap(a):
    """Constant beliefs (m = c; identical code emissions) and random-init nets: every decoder must
    score VSP <= 1% on the test set."""
    dev = torch.device(a.device)
    worst = {}
    for n in SIZES:
        for p in PS:
            for de in (True, False):
                b, cov = cell_tensors("test", n, p, de, dev)
                res = {}
                for cval in (0.0, 0.5, -0.5, 3.0):
                    m = torch.full((b["B"], 1, b["L"], b["L"]), cval, device=dev)
                    thr = jm.decode_thr(m) & b["op"]
                    res[f"const{cval}|thr"] = vsp(thr, b)
                    res[f"const{cval}|cl"] = vsp(jm.decode_cl(thr, b["op"], b["s"], b["g"]), b)
                    for arm in ("M-tree", "M-tile"):
                        mm = jm.Model(arm).to(dev)
                        st = mm.init_state(b["x"])
                        st["theta"] = torch.full_like(st["theta"], cval)
                        res[f"const{cval}|{arm}|tv"] = vsp(jm.tree_map(st, b["op"].float(), mm.geom((b["L"], b["L"])))[0] & b["op"], b)
                for arm in ("U", "M-tree", "M-tile"):
                    for sd in (0, 1):
                        torch.manual_seed(100 + sd)
                        mm = jm.Model(arm).to(dev).eval()
                        for T in (30, 300):
                            st = mm.run(b["x"], T)
                            decs, _ = decoders(mm, st, b)
                            for k, S in decs.items():
                                res[f"rand_{arm}_s{sd}_T{T}|{k}"] = vsp(S, b)
                for k, v in res.items():
                    worst[k] = max(worst.get(k, 0.0), float(v.float().mean()))
                print(n, p, de, max(float(v.float().mean()) for v in res.values()), flush=True)
    out = {"worst_vsp_per_probe": worst, "max": max(worst.values()), "pass": max(worst.values()) <= 0.01}
    json.dump(out, open(os.path.join(HERE, "results", "trap_test.json"), "w"), indent=1)
    print(json.dumps({"max": out["max"], "pass": out["pass"]}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--ckpt")
    ap.add_argument("--arm")
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--part", default="all", choices=("all", "core", "deep", "extra"))
    ap.add_argument("--cells", default="")
    ap.add_argument("--bs", type=int, default=300)  # one batch per 300-maze test cell (launch-bound GPU)
    ap.add_argument("--tmax", type=int, default=0, help="smoke tests only: cap T")
    ap.add_argument("--limit", type=int, default=0, help="smoke tests only: mazes per cell")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "eval"))
    a = ap.parse_args()
    {"run": cmd_run, "trap": cmd_trap}[a.cmd](a)
