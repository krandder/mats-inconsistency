"""P11 evaluation: one long rollout per (model, maze set); every readout is computed online from it.

Readouts (R-prog, DT-L, R-final; for the M arms only 'inst' = the belief b is scored):
  inst            logit_t
  ces             running (Cesaro) mean of logit_1..t
  ema{beta}       bias-corrected EMA of the logits, beta in {0.9, 0.99, 0.999}
  lmsr{W}         test-time LMSR spending tracker: b += clamp(logit - b, d_min(b, w), d_max(b, w)), per-pixel
                  budget w (starts at W, float64 ledger charged the realised worst-case loss)
  tv{V}           test-time logit-TV tracker: b += clamp(logit - b, -v, v), v -= |move|
  stop{T}         (derived) logit_min(t, T)
  frz{tau}_{k}    (derived) per maze: freeze at the first t where max_open |logit_t - logit_t-1| < tau for k steps
exact_acc at checkpoint c = mean over mazes of min over t in [c-7, c] of (all open pixels right); a non-finite
pixel is wrong. Per-maze bits are stored for every readout and checkpoint (hierarchical bootstrap).
"""
import argparse
import hashlib
import json
import math
import os
import time

import numpy as np
import torch

import caps
import gen
from model import build, MState, R_ARMS, M_ARMS

HERE = os.path.dirname(os.path.abspath(__file__))
T_TRAIN = 30
RATIOS = [1, 3, 10, 30, 100, 300, 1000]
STOP_T = [10, 15, 20, 25, 30, 35, 40, 50, 60, 75, 90, 110, 135, 165, 200, 250, 300, 400, 500, 700, 1000,
          1500, 2000, 3000, 5000, 10000, 30000]
FREEZE = [(tau, k) for tau in (0.01, 0.03, 0.1, 0.3, 1.0) for k in (3, 10, 30)]
EMA_B = (0.9, 0.99, 0.999)
TRK_W = (3.0, 6.0, 12.0, 24.0, 48.0)
TRK_V = (4.0, 8.0, 16.0, 32.0, 64.0)
SETS = {"val9": ("val", 9, 64), "val17": ("val", 17, 128), "val29": ("val", 29, 128),
        "test9": ("test", 9, 64), "test17": ("test", 17, 128), "test29": ("test", 29, 128), "test59": ("test", 59, 64)}


def hexbits(b):
    return np.packbits(np.asarray(b, dtype=np.uint8)).tobytes().hex()


def load_set(data_dir, name):
    split, size, n = SETS[name]
    p = os.path.join(data_dir, f"{name}.npz")
    if not os.path.exists(p):
        s = gen.eval_set(split, size, n)
        np.savez_compressed(p, x=s["x"], y=s["y"], len=s["len"], near=s["near"], seeds=np.array(s["seeds"]))
    z = np.load(p)
    return {k: z[k] for k in z.files}


def rankdata(a):
    """average ranks (ties averaged), numpy only."""
    a = np.asarray(a)
    o = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), float)
    sa = a[o]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return r


def spearman(a, b):
    if len(a) < 3:
        return None
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


class Streams:
    """Online readouts as one stacked tensor V (K, N, S, S)."""
    def __init__(self, full, shape, dev, ledger_init=None):
        self.full = full
        N = shape[0]
        if full:
            self.names = (["inst", "ces"] + [f"ema{b}" for b in EMA_B] + [f"lmsr{int(w)}" for w in TRK_W]
                          + [f"tv{int(v)}" for v in TRK_V])
            self.ces = torch.zeros(shape, device=dev)
            self.ema = torch.zeros((len(EMA_B),) + shape, device=dev)
            self.beta = torch.tensor(EMA_B, device=dev).view(-1, 1, 1, 1)
            self.Bl = torch.zeros((len(TRK_W),) + shape, device=dev)
            self.Rl = torch.tensor(TRK_W, device=dev, dtype=torch.float64).view(-1, 1, 1, 1).expand((len(TRK_W),) + shape).clone()
            self.Bt = torch.zeros((len(TRK_V),) + shape, device=dev)
            self.Rt = torch.tensor(TRK_V, device=dev, dtype=torch.float64).view(-1, 1, 1, 1).expand((len(TRK_V),) + shape).clone()
        else:
            self.names = ["inst"]

    def update(self, L, t):
        if not self.full:
            return L[None]
        self.ces += L
        self.ema.mul_(self.beta).add_((1 - self.beta) * L[None])
        corr = 1 - self.beta ** t
        d = L[None] - self.Bl
        mv = caps.lmsr_clamp(self.Bl, d, self.Rl.float())
        Bn = self.Bl + mv
        self.Rl, _ = caps.charge_lmsr(self.Rl, self.Bl, Bn)
        self.Bl = Bn
        rt = self.Rt.float()
        mv = torch.maximum(torch.minimum(L[None] - self.Bt, rt), -rt)
        Bn = self.Bt + mv
        self.Rt, _ = caps.charge_tv(self.Rt, self.Bt, Bn)
        self.Bt = Bn
        return torch.cat([L[None], (self.ces / t)[None], self.ema / corr, self.Bl, self.Bt])

    def ledgers(self):
        """-> list of (stream index, remaining (N,S,S) float64, initial)."""
        if not self.full:
            return []
        out = []
        base = 2 + len(EMA_B)
        for i, w in enumerate(TRK_W):
            out.append((base + i, self.Rl[i], w))
        for i, v in enumerate(TRK_V):
            out.append((base + len(TRK_W) + i, self.Rt[i], v))
        return out


@torch.no_grad()
def rollout(net, arm, hp, S, dev, depth_ratios, full_streams=True, h0=None, budget_metrics=True):
    x = torch.tensor(S["x"], device=dev).float()
    y = torch.tensor(S["y"], device=dev).bool()
    op = x[:, 0] > 0
    N = x.shape[0]
    ck = [r * T_TRAIN for r in depth_ratios]
    T = ck[-1]
    win = {}
    for c in ck:
        for t in range(max(1, c - 7), c + 1):
            win.setdefault(t, []).append(c)
    vel_w = {c: min(100, c // 2) for c in ck}
    snap_t = {c - w: c for c, w in vel_w.items()}
    is_m = arm in M_ARMS
    streams = Streams(full_streams and not is_m, y.shape, dev)
    K = len(streams.names)
    ms = MState(arm, hp, y.shape, dev) if is_m else None
    h = net.init_state(x) if h0 is None else h0
    h1 = h2 = None
    s1 = s2 = None
    fl_d = torch.zeros((K,) + y.shape, dtype=torch.int32, device=dev)
    fl_p = torch.zeros_like(fl_d)
    cw = torch.zeros((K,) + y.shape, dtype=torch.bool, device=dev)
    wmin = {c: torch.ones((K, N), dtype=torch.bool, device=dev) for c in ck}
    ex_tab = torch.zeros((T + 1, N), dtype=torch.bool, device=dev)
    dmax_tab = torch.full((T + 1, N), float("inf"), device=dev)
    ledg_init = None
    t50 = None
    sat = torch.zeros((), device=dev, dtype=torch.float64)
    jump = torch.zeros((), device=dev, dtype=torch.float64)
    Lprev = None
    snaps = {}
    per_ck = []
    t0 = time.time()
    ctx = torch.nn.utils.parametrize.cached() if arm == "DT-L" else torch.no_grad()
    with ctx:
        for t in range(1, T + 1):
            if is_m:
                h = net.step(h, x, ms.feat())
                u = net.out(h)
                sat += ((u.abs() > 5) & op).sum()
                if t > T_TRAIN:
                    jump += ((u.abs() >= 2) & op).sum()
                L = ms.advance(u)
            else:
                h = net.step(h, x)
                L = net.out(h)
            V = streams.update(L, t)
            s = V > 0
            ex_tab[t] = (((s[0] == y) & torch.isfinite(V[0])) | ~op).flatten(1).all(1)
            if Lprev is not None:
                dmax_tab[t] = torch.where(op, (L - Lprev).abs(), torch.zeros_like(L)).flatten(1).amax(1)
            Lprev = L
            if s1 is not None:
                flip = (s != s1) & op[None]
                if s2 is not None:
                    p2 = flip & (s == s2)
                    fl_p += p2
                    fl_d += flip & ~p2
                else:
                    fl_d += flip
            s2, s1 = s1, s
            if t >= T_TRAIN:
                cw |= (V.abs() >= 2) & (s != y[None]) & op[None]
            if budget_metrics:
                led = []
                if is_m and ms.rem is not None:
                    led = [(0, ms.rem, ms.init)]
                elif not is_m:
                    led = streams.ledgers()
                if led:
                    if t50 is None:
                        t50 = torch.full((len(led),) + y.shape, 10 ** 9, dtype=torch.int32, device=dev)
                    for j, (_, rem, w0) in enumerate(led):
                        hit = (rem <= 0.5 * w0) & (t50[j] == 10 ** 9)
                        t50[j] = torch.where(hit, t, t50[j])      # no boolean-mask write: avoids a device sync
            if t in snap_t:
                snaps[snap_t[t]] = V[0].clone()
            if t in win or t in ck:
                fin = torch.isfinite(V)
                corr = ((s == y[None]) & fin) | ~op[None]
            if t in win:
                ex = corr.flatten(2).all(2)
                for c in win[t]:
                    wmin[c] &= ex
            if t in ck:
                c = t
                rec = {"t": c, "ratio": c // T_TRAIN}
                rec["acc"] = {n: float(wmin[c][i].float().mean()) for i, n in enumerate(streams.names)}
                rec["bits"] = {n: hexbits(wmin[c][i].cpu().numpy()) for i, n in enumerate(streams.names)}
                a = torch.where(op[None], V, torch.zeros_like(V)).abs()
                rec["maxabs"] = {n: float(a[i].max()) for i, n in enumerate(streams.names)}
                rec["nonfinite"] = int((~fin & op[None]).sum())
                nopen = op.sum().item()
                span = c - (per_ck[-1]["t"] if per_ck else 0)
                rec["flips_drift_per_px_per_1k"] = {n: float(fl_d[i].sum()) / nopen / span * 1000
                                                    for i, n in enumerate(streams.names)}
                rec["flips_p2_per_px_per_1k"] = {n: float(fl_p[i].sum()) / nopen / span * 1000
                                                 for i, n in enumerate(streams.names)}
                fl_d.zero_()
                fl_p.zero_()
                hn = h.flatten(1).norm(dim=1) + 1e-12
                if h1 is not None:
                    rec["dphi1"] = float(((h - h1).flatten(1).norm(dim=1) / hn).mean())
                if h2 is not None:
                    rec["dphi2"] = float(((h - h2).flatten(1).norm(dim=1) / hn).mean())
                if c in snaps:
                    w = vel_w[c]
                    rec["late_velocity"] = float(((V[0] - snaps.pop(c)).abs() * op).sum() / nopen / w)
                rec["sec"] = round(time.time() - t0, 1)
                per_ck.append(rec)
            h2, h1 = h1, h
    # ---------------------------------------------------------------- end-of-rollout metrics
    out = {"n": N, "checkpoints": per_ck, "streams": streams.names, "T": T}
    s_end = V > 0
    right_end = (s_end == y[None]) & op[None]
    num = (cw & right_end).flatten(2).sum(2)
    den = cw.flatten(2).sum(2)
    out["late_correction"] = {n: {"num": int(num[i].sum()), "den": int(den[i].sum()),
                                  "rate": float(num[i].sum()) / max(1, int(den[i].sum())),
                                  **({"per_maze_num": num[i].tolist(), "per_maze_den": den[i].tolist()} if i == 0 else {})}
                              for i, n in enumerate(streams.names)}
    # derived readouts (stop, freeze) from the inst table
    idx_t = torch.arange(T + 1, device=dev)
    der = {}
    for Ts in STOP_T:
        if Ts > T:
            continue
        accs, bits = [], []
        for c in ck:
            ts = torch.clamp(torch.arange(max(1, c - 7), c + 1, device=dev), max=Ts)
            e = ex_tab[ts].all(0)
            accs.append(float(e.float().mean()))
            bits.append(hexbits(e.cpu().numpy()))
        der[f"stop{Ts}"] = {"acc": accs, "bits": bits}
    for tau, k in FREEZE:
        small = (dmax_tab < tau).int()
        cs = torch.cumsum(small, 0)
        run = torch.zeros_like(cs, dtype=torch.bool)
        run[k:] = (cs[k:] - cs[:-k]) == k
        has = run.any(0)
        fz = torch.where(has, run.float().argmax(0), torch.full((N,), float(T + 1), device=dev)).long()
        accs, bits = [], []
        for c in ck:
            ts = torch.arange(max(1, c - 7), c + 1, device=dev)
            ts = torch.minimum(ts[:, None], fz[None, :])
            e = ex_tab.gather(0, ts.clamp(max=T)).all(0)
            accs.append(float(e.float().mean()))
            bits.append(hexbits(e.cpu().numpy()))
        der[f"frz{tau}_{k}"] = {"acc": accs, "bits": bits, "frozen_frac": float(has.float().mean()),
                                "median_freeze_t": float(fz[has].float().median()) if has.any() else None}
    out["derived"] = der
    out["inst_exact_curve"] = {"t": [int(v) for v in np.unique(np.geomspace(1, T, 60).astype(int))]}
    out["inst_exact_curve"]["acc"] = [float(ex_tab[t].float().mean()) for t in out["inst_exact_curve"]["t"]]
    if is_m:
        steps_all = float(op.sum()) * T
        out["sat_frac"] = float(sat) / steps_all
        out["jump_frac_u_ge_2_after_Ttrain"] = float(jump) / max(1.0, float(op.sum()) * (T - T_TRAIN))
    # budget metrics
    if budget_metrics and t50 is not None:
        led = [(0, ms.rem, ms.init)] if is_m else streams.ledgers()
        onp = (y & op)
        near = torch.tensor(S["near"], device=dev)
        Lm = np.asarray(S["len"])
        qs = np.quantile(Lm, [1 / 3, 2 / 3])
        binid = np.digitize(Lm, qs)
        bm = {}
        for j, (si, rem, w0) in enumerate(led):
            name = streams.names[si]
            exh = rem <= 0.01 * w0
            wrong = s_end[si] != y
            exw = (exh & wrong & onp).sum().item() / max(1, onp.sum().item())
            tt = t50[j][onp].cpu().numpy()
            nn_ = near[onp].cpu().numpy()
            reached = tt < 10 ** 9
            rho = spearman(tt[reached], nn_[reached]) if reached.sum() > 2 else None
            exh_maze = ((exh & onp).flatten(1).sum(1).float() / onp.flatten(1).sum(1).clamp(min=1)).cpu().numpy()
            end_ok = wmin[ck[-1]][si].cpu().numpy()
            bins = []
            for bi in range(3):
                sel = binid == bi
                bins.append({"n": int(sel.sum()), "len_range": [int(Lm[sel].min()), int(Lm[sel].max())] if sel.any() else None,
                             "exact_end": float(end_ok[sel].mean()) if sel.any() else None,
                             "exhaust_frac_onpath": float(exh_maze[sel].mean()) if sel.any() else None})
            bm[name] = {"onpath_exhausted_while_wrong": exw,
                        "onpath_exhausted": float((exh & onp).sum().item() / max(1, onp.sum().item())),
                        "frac_onpath_reached_50pct": float(reached.mean()),
                        "spearman_t50_vs_near_dist": rho, "by_path_len_bin": bins,
                        "mean_remaining_frac_open": float((rem / w0 * op).sum() / op.sum())}
        out["budget"] = bm
    out["sec"] = round(time.time() - t0, 1)
    return out, h


def path_independence(net, arm, hp, S, dev, ratio=100, n=32, reps=4):
    Ssub = {k: v[:n] for k, v in S.items()}
    ref, href = rollout(net, arm, hp, Ssub, dev, [ratio], full_streams=False, budget_metrics=False)
    x = torch.tensor(Ssub["x"], device=dev).float()
    g = torch.Generator().manual_seed(555_017)
    h_ref0 = net.init_state(x)
    res = {"n": n, "depth": ratio * T_TRAIN, "exact_std_init": ref["checkpoints"][-1]["acc"]["inst"],
           "exact_by_init": [], "rel_state_dist": []}
    bits = []
    for r in range(reps):
        h0 = torch.randn(h_ref0.shape, generator=g).to(dev)
        o, hr = rollout(net, arm, hp, Ssub, dev, [ratio], full_streams=False, h0=h0, budget_metrics=False)
        res["exact_by_init"].append(o["checkpoints"][-1]["acc"]["inst"])
        res["rel_state_dist"].append(float(((hr - href).flatten(1).norm(dim=1) / (href.flatten(1).norm(dim=1) + 1e-12)).mean()))
        bits.append(o["checkpoints"][-1]["bits"]["inst"])
    res["spread_exact"] = max(res["exact_by_init"]) - min(res["exact_by_init"])
    res["bits_by_init"] = bits
    return res


PLANS = {
    "pilot": [("val9", RATIOS), ("val17", RATIOS[:5]), ("val29", RATIOS[:5])],
    "core": [("val17", RATIOS[:5]), ("test9", RATIOS), ("test17", RATIOS), ("test29", RATIOS[:5]),
             ("test59", RATIOS[:5])],
    "smoke": [("val9", [1, 3]), ("val17", [1, 3])],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="dir with model.pt + train.json")
    ap.add_argument("--plan", default="core")
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--pi", type=int, default=1)
    ap.add_argument("--nmax", type=int, default=0, help="smoke only: cap mazes per set")
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.run, "train.json")))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = build(meta["arm"], meta["width"]).to(dev)
    net.load_state_dict(torch.load(os.path.join(a.run, "model.pt"), map_location=dev))
    net.eval()
    res = {"arm": meta["arm"], "hp": meta["hp"], "seed": meta["seed"], "plan": a.plan, "sets": {},
           "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"}
    t0 = time.time()
    set_hash = {}
    for name, ratios in PLANS[a.plan]:
        S = load_set(a.data, name)
        if a.nmax:
            S = {k: v[:a.nmax] for k, v in S.items()}
        set_hash[name] = hashlib.sha256(S["x"].tobytes() + S["y"].tobytes()).hexdigest()[:16]
        o, _ = rollout(net, meta["arm"], meta["hp"], S, dev, ratios)
        o["path_len"] = [int(v) for v in S["len"]]
        res["sets"][name] = o
        print(name, f"{o['sec']}s", json.dumps({c["ratio"]: round(c["acc"]["inst"], 3) for c in o["checkpoints"]}),
              flush=True)
    if a.plan in ("core", "smoke") and a.pi:
        res["path_independence"] = path_independence(net, meta["arm"], meta["hp"], load_set(a.data, "test17"), dev,
                                                     n=min(32, a.nmax or 32), ratio=100 if a.plan == "core" else 1)
        print("pi", json.dumps({k: v for k, v in res["path_independence"].items() if k != "bits_by_init"}), flush=True)
    res["set_sha256_16"] = set_hash
    res["eval_sec"] = round(time.time() - t0, 1)
    p = os.path.join(a.run, f"eval_{a.plan}.json")
    json.dump(res, open(p + ".tmp", "w"))
    os.replace(p + ".tmp", p)
    print("DONE", p, res["eval_sec"], flush=True)


if __name__ == "__main__":
    main()
