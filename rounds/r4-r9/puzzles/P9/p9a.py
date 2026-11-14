"""P9a: exogenous hidden doors. Kelvin's item (2): clamp-and-void conditional prices on hidden variables.

Task: reachability from S on P7 mazes (density 0.35). K=4 door cells per maze (open in the base maze):
2 drawn in proportion to dom(c) = #cells that lose reachability when c is walled, 2 uniform over the other
open cells (not S). Each door is open with p=0.5, independently. Y_c = 1 iff c is reachable from S.

Inputs (5 ch): wall (doors shown open), start, door-unknown flag, clamp flag, clamp value.
A query q in {-1 (unknown), 0 (closed), 1 (open)}^4 asks for E[Y | x, D_S = v].

Arms (batch = 32 SETTLED examples per step for every arm; void episodes are dropped before the forward pass):
  X0   doors never revealed in training (clamps only at eval)
  X2   reveal-true, each door revealed w.p. 0.5, weight 1
  X2p  reveal-true, each door revealed w.p. 1/3, weight 1       (X2' = X3's settled law without 1/q)
  X3   clamp each door w.p. 0.5 to v~Bern(0.5); settled iff every clamped v equals its door; weight 2^m
       (normalised by the analytic E[2^m | settled] = 0.75^-4 so the mean settled weight is 1)
  X4   the same clamps, always settled (label = real Y), weight 1
  X1   exact BFS conditionals plugged into every readout (metric self-test; no net)
"""
import argparse
import itertools
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import p9util as U
from p9models import DTR, n_params

HERE = os.path.dirname(os.path.abspath(__file__))
K = 4
ALPHA = 0.5
QS = np.array(list(itertools.product([-1, 0, 1], repeat=K)), dtype=np.int64)  # 81 queries
FULL = np.array([i for i, q in enumerate(QS) if (q >= 0).all()])  # 16 full assignments
EMPTY = int(np.where((QS == -1).all(1))[0][0])


def a_index(bits):
    """door states (...,4) in {0,1} -> assignment index sum_j D_j 2^j."""
    return (np.asarray(bits) * (1 << np.arange(K))).sum(-1)


# ------------------------------------------------------------------ data
def make_doors(num, n, seed, cache=None):
    if cache and os.path.exists(cache):
        return torch.load(cache, weights_only=False)
    rng = np.random.default_rng(seed)
    w, s, _, _ = U.gen_mazes(num, n, 0.35, rng)
    op = torch.tensor(w == 0)
    st = torch.tensor(s > 0)
    base = U.reach(op, st)
    dom = torch.zeros(num, n, n, dtype=torch.long)
    for c in range(n * n):
        i, j = divmod(c, n)
        oc = op.clone()
        oc[:, i, j] = False
        r = U.reach(oc, st)
        lost = base & ~r
        lost[:, i, j] = False
        dom[:, i, j] = lost.flatten(1).sum(1)
    dom = torch.where(op & ~st, dom, torch.zeros_like(dom))
    doors = np.zeros((num, K, 2), np.int64)
    for m in range(num):
        cand = np.flatnonzero((op[m] & ~st[m]).numpy().ravel())
        dv = dom[m].numpy().ravel()[cand].astype(float)
        nz = cand[dv > 0]
        if len(nz) >= 2:
            first = rng.choice(cand, 2, replace=False, p=dv / dv.sum())
        else:
            rest = np.setdiff1d(cand, nz)
            first = np.concatenate([nz, rng.choice(rest, 2 - len(nz), replace=False)])
        rest = np.setdiff1d(cand, first)
        sec = rng.choice(rest, 2, replace=False)
        cells = np.concatenate([first, sec])
        doors[m, :, 0], doors[m, :, 1] = np.divmod(cells, n)
    # exact truth: reachability under each of the 16 door assignments
    Yall = torch.zeros(num, 16, n, n, dtype=torch.bool)
    ar = torch.arange(num)
    for a in range(16):
        oa = op.clone()
        for j in range(K):
            if not (a >> j) & 1:
                oa[ar, doors[:, j, 0], doors[:, j, 1]] = False
        Yall[:, a] = U.reach(oa, st)
    ds = {"walls": torch.tensor(w), "start": torch.tensor(s), "doors": torch.tensor(doors), "Yall": Yall,
          "dom": dom, "n": n}
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        tmp = f"{cache}.{os.getpid()}.tmp"
        torch.save(ds, tmp)
        os.replace(tmp, cache)  # atomic: concurrent jobs may build the same cache
    return ds


def inputs(ds, idx, q):
    """idx (B,) maze indices, q (B,4) in {-1,0,1} -> x (B,5,n,n)."""
    idx = torch.as_tensor(idx)
    q = torch.as_tensor(q)
    B = len(idx)
    n = ds["n"]
    x = torch.zeros(B, 5, n, n)
    x[:, 0] = ds["walls"][idx]
    x[:, 1] = ds["start"][idx]
    d = ds["doors"][idx]  # (B,4,2)
    br = torch.arange(B)
    for j in range(K):
        ii, jj = d[:, j, 0], d[:, j, 1]
        qj = q[:, j]
        x[br, 2, ii, jj] = (qj < 0).float()
        x[br, 3, ii, jj] = (qj >= 0).float()
        x[br, 4, ii, jj] = (qj == 1).float()
    return x


def exact_cond(ds, m):
    """(81, n, n) exact P(Y_c=1 | x, q) for maze m."""
    Y = ds["Yall"][m].float().numpy()  # (16,n,n)
    bits = np.array([[(a >> j) & 1 for j in range(K)] for a in range(16)])
    out = np.zeros((len(QS),) + Y.shape[1:])
    for qi, q in enumerate(QS):
        ok = ((q < 0) | (bits == q)).all(1)
        out[qi] = Y[ok].mean(0)
    return out


# ------------------------------------------------------------------ model / loss
def build():
    return DTR(5, w=24, nblocks=1, head=(8,))


def run(net, x, T):
    h = net.init(x)
    for _ in range(T):
        h = net.step(h, x)
    return net.readout(h)


def prog_loss(net, x, y, T, n, k):
    h0 = net.init(x)
    h = h0
    for _ in range(T):
        h = net.step(h, x)
    lf = F.binary_cross_entropy_with_logits(net.readout(h), y, reduction="none").mean((1, 2))
    with torch.no_grad():
        h = h0
        for _ in range(n):
            h = net.step(h, x)
    h = h.detach()
    for _ in range(k):
        h = net.step(h, x)
    lp = F.binary_cross_entropy_with_logits(net.readout(h), y, reduction="none").mean((1, 2))
    return (1 - ALPHA) * lf + ALPHA * lp


XW_NORM = 0.75 ** 4  # 1 / E[2^m | settled]


def draw_batch(arm, ds, B, drng, crng, stats):
    """B settled examples: maze idx, true doors, query, weight."""
    idx_l, D_l, q_l, w_l = [], [], [], []
    have = 0
    while have < B:
        need = B - have
        chunk = need if arm != "X3" else int(need * 3.5) + 4
        idx = drng.integers(0, ds["walls"].shape[0], chunk)
        D = (drng.random((chunk, K)) < 0.5).astype(np.int64)
        if arm == "X0":
            q = -np.ones((chunk, K), np.int64); w = np.ones(chunk); ok = np.ones(chunk, bool)
        elif arm in ("X2", "X2p"):
            r = crng.random((chunk, K)) < (0.5 if arm == "X2" else 1 / 3)
            q = np.where(r, D, -1); w = np.ones(chunk); ok = np.ones(chunk, bool)
        else:  # X3, X4
            c = crng.random((chunk, K)) < 0.5
            v = (crng.random((chunk, K)) < 0.5).astype(np.int64)
            q = np.where(c, v, -1)
            if arm == "X3":
                ok = (~c | (v == D)).all(1)
                w = np.where(ok, 2.0 ** c.sum(1), 0.0)
                # IS diagnostics over DRAWN episodes (void included, weight 0), in draw order up to the batch fill
                cut = np.flatnonzero(ok)
                last = cut[need - 1] + 1 if len(cut) >= need else chunk
                stats["drawn"] += int(last)
                stats["w2_sum"] += float((w[:last] ** 2).sum())
                stats["w_sum"] += float(w[:last].sum())
                ok[last:] = False
                w = w * XW_NORM
            else:
                ok = np.ones(chunk, bool); w = np.ones(chunk)
        sel = np.flatnonzero(ok)[:need]
        idx_l.append(idx[sel]); D_l.append(D[sel]); q_l.append(q[sel]); w_l.append(w[sel])
        have += len(sel)
    idx, D, q, w = (np.concatenate(a) for a in (idx_l, D_l, q_l, w_l))
    stats["settled"] += B
    if arm == "X3":
        stats["w_settled_raw"].extend((w / XW_NORM).tolist()[:B]) if len(stats["w_settled_raw"]) < 200000 else None
    return idx, D, q, w


def train(args, dev):
    name = f"{args.arm}_s{args.seed}{args.tag}"
    os.makedirs(os.path.join(args.out, "p9a"), exist_ok=True)
    part = os.path.join(args.scratch, "p9a_" + name + ".partial")
    ds = make_doors(args.pool, 9, 20_000 + args.seed, cache=os.path.join(args.cache, f"p9a_pool_s{args.seed}.pt"))
    torch.manual_seed(1000 + args.seed)
    net = build().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    drng = np.random.default_rng(2000 + args.seed)
    lrng = np.random.default_rng(3000 + args.seed)
    crng = np.random.default_rng(4000 + args.seed)
    stats = {"drawn": 0, "settled": 0, "w2_sum": 0.0, "w_sum": 0.0, "w_settled_raw": [], "fwd_ep_steps": 0}
    log, step0 = [], 0
    st = U.load_partial(part)
    if st is not None:
        net.load_state_dict(st["net"]); opt.load_state_dict(st["opt"])
        drng.bit_generator.state, lrng.bit_generator.state, crng.bit_generator.state = st["rng"]
        stats, log, step0 = st["stats"], st["log"], st["step"]
    t0 = time.time() - (log[-1]["wall"] if log else 0)
    dev_evals = []
    for step in range(step0, args.steps):
        frac = step / args.steps
        for g in opt.param_groups:
            g["lr"] = args.lr * (1.0 if frac < 0.75 else max(0.05, (1 - frac) / 0.25))
        idx, D, q, w = draw_batch(args.arm, ds, args.batch, drng, crng, stats)
        x = inputs(ds, idx, q).to(dev)
        y = ds["Yall"][torch.as_tensor(idx), torch.as_tensor(a_index(D))].float().to(dev)
        k = int(lrng.integers(1, args.T + 1)); n = int(lrng.integers(0, args.T - k + 1))
        l = prog_loss(net, x, y, args.T, n, k)
        loss = (torch.as_tensor(w, dtype=torch.float32, device=dev) * l).sum() / args.batch
        stats["fwd_ep_steps"] += args.batch * (args.T + n + k)
        opt.zero_grad()
        loss.backward()
        if args.clip > 0:  # upstream deep-thinking recipe clips at 1.0 (added in the 9x9 P9a pilot retune)
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
        opt.step()
        if step % 100 == 0 or step == args.steps - 1:
            log.append({"step": step, "loss": float(loss), "wall": round(time.time() - t0, 1)})
            print(json.dumps(log[-1]), flush=True)
        if args.dev_every and (step + 1) % args.dev_every == 0:
            dv = make_doors(100, 9, 50_109, cache=os.path.join(args.cache, "p9a_dev9.pt"))
            r = evaluate(net, dv, 9, args.T, dev, nmaze=100)
            dev_evals.append({"step": step + 1, **{k2: r[k2] for k2 in ("kl", "voi_spearman", "sign_acc")}})
            print("DEV", json.dumps(dev_evals[-1]), flush=True)
        if (step + 1) % 250 == 0 and step + 1 < args.steps:
            U.save_partial(part, net=net.state_dict(), opt=opt.state_dict(), stats=stats, log=log, step=step + 1,
                           rng=(drng.bit_generator.state, lrng.bit_generator.state, crng.bit_generator.state))
    w = np.array(stats.pop("w_settled_raw"))
    diag = {}
    if args.arm == "X3":
        diag = {"w2_per_drawn": stats["w2_sum"] / stats["drawn"], "w2_analytic": 1.5 ** 4,
                "w_per_drawn": stats["w_sum"] / stats["drawn"], "settle_rate": stats["settled"] / stats["drawn"],
                "settle_rate_analytic": 0.75 ** 4,
                "ess_ratio": float(w.sum() ** 2 / (w ** 2).sum() / len(w)), "ess_ratio_analytic": (1 / 0.75 ** 4) ** 2 / (1.5 ** 4 / 0.75 ** 4)}
    stats["wall_s"] = round(time.time() - t0, 1)
    torch.save({k: v.cpu() for k, v in net.state_dict().items()}, os.path.join(args.out, "p9a", name + ".pt"))
    if os.path.exists(part):
        os.remove(part)
    return net, {"stats": stats, "is_diag": diag, "log": log, "dev_evals": dev_evals, "params": n_params(net)}


# ------------------------------------------------------------------ readouts
def Hb(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def readouts(P):
    """P (81,n,n) conditional prices -> VOI_j (4,), c_ij (6 pairs, n, n)."""
    marg = P[EMPTY]
    voi = np.zeros(K)
    for j in range(K):
        hc = 0
        for v in (0, 1):
            q = -np.ones(K, np.int64); q[j] = v
            qi = int(np.where((QS == q).all(1))[0][0])
            hc = hc + 0.5 * Hb(P[qi])
        voi[j] = (Hb(marg) - hc).sum()
    # joint over the 16 full assignments
    f = np.zeros((2,) * K + P.shape[1:])
    for qi in FULL:
        f[tuple(QS[qi])] = P[qi]
    cij = {}
    for i, j in itertools.combinations(range(K), 2):
        g = np.moveaxis(f, (i, j), (0, 1))  # (2,2,2,2,n,n) with i,j first
        d2 = g[1, 1] - g[1, 0] - g[0, 1] + g[0, 0]  # (2,2,n,n) over the other two doors
        cij[(i, j)] = d2.mean((0, 1))
    return voi, cij


def kl_bern(p, q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    p = np.clip(p, 0, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(p > 0, p * np.log(np.maximum(p, 1e-300) / q), 0.0)
        b = np.where(p < 1, (1 - p) * np.log(np.maximum(1 - p, 1e-300) / (1 - q)), 0.0)
    return a + b


@torch.no_grad()
def net_cond(net, ds, ms, T, dev):
    x = torch.cat([inputs(ds, np.full(len(QS), m), QS) for m in ms]).to(dev)
    p = torch.sigmoid(run(net, x, T)).double().cpu().numpy()
    return p.reshape(len(ms), len(QS), *p.shape[1:])


def score(ds, nmaze, get_P):
    """get_P(list of maze idx) -> (len, 81, n, n) prices. Returns the P9a metric dict."""
    kl_num = kl_den = 0.0
    kl_maze = []
    voi_net, voi_ex, taus = [], [], []
    sign_ok = sign_n = 0
    dd_frac = []
    brier_num = 0.0
    chunk = 8
    for s in range(0, nmaze, chunk):
        ms = list(range(s, min(nmaze, s + chunk)))
        Pn = get_P(ms)
        for r, m in enumerate(ms):
            Pe = exact_cond(ds, m)
            dd = (Pe[EMPTY] > 0) & (Pe[EMPTY] < 1)
            dd_frac.append(dd.mean())
            if dd.sum() > 0:
                kl = kl_bern(Pe[:, dd], Pn[r][:, dd])
                kl_num += kl.sum(); kl_den += kl.size; kl_maze.append(kl.mean())
                brier_num += ((Pe[:, dd] - Pn[r][:, dd]) ** 2).sum()
            vn, cn = readouts(Pn[r])
            ve, ce = readouts(Pe)
            voi_net.extend(vn); voi_ex.extend(ve)
            if np.ptp(ve) > 1e-9:
                taus.append(U.kendall_tau_b(vn, ve))
            for key in ce:
                sel = np.abs(ce[key]) >= 0.25
                sign_ok += int((np.sign(cn[key][sel]) == np.sign(ce[key][sel])).sum())
                sign_n += int(sel.sum())
    return {"kl": kl_num / max(kl_den, 1), "kl_per_maze_mean": float(np.mean(kl_maze)) if kl_maze else None,
            "brier": brier_num / max(kl_den, 1), "n_dd_cells_x_queries": int(kl_den),
            "dd_cell_frac": float(np.mean(dd_frac)), "mazes_with_dd": int(len(kl_maze)),
            "voi_spearman": U.spearman(voi_net, voi_ex), "kendall_median": float(np.median(taus)) if taus else None,
            "kendall_n": len(taus), "sign_acc": sign_ok / max(sign_n, 1), "sign_n": sign_n}


def evaluate(net, ds, n, T, dev, nmaze):
    net.eval()
    r = score(ds, nmaze, lambda ms: net_cond(net, ds, ms, T, dev))
    net.train()
    return r


def test_sets(cache):
    return {9: (make_doors(500, 9, 61_009, cache=os.path.join(cache, "p9a_test9.pt")), 20, 500),
            15: (make_doors(200, 15, 61_015, cache=os.path.join(cache, "p9a_test15.pt")), 60, 200)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["X0", "X1", "X2", "X2p", "X3", "X4"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--clip", type=float, default=0.0)
    ap.add_argument("--pool", type=int, default=20000)
    ap.add_argument("--dev_every", type=int, default=0)
    ap.add_argument("--nmaze9", type=int, default=500)
    ap.add_argument("--nmaze15", type=int, default=200)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--scratch", default="/tmp/p9/partial")
    ap.add_argument("--cache", default="/tmp/p9/cache")
    args = ap.parse_args()
    torch.set_num_threads(1)
    dev = U.device()
    os.makedirs(args.scratch, exist_ok=True)
    os.makedirs(os.path.join(args.out, "p9a"), exist_ok=True)
    name = f"{args.arm}_s{args.seed}{args.tag}"
    t0 = time.time()
    tests = test_sets(args.cache)
    res = {"arm": args.arm, "seed": args.seed, "args": vars(args), "device": str(dev), "eval": {}}
    if args.arm == "X1":
        for sz, (ds, T, nm) in tests.items():
            nm = {9: args.nmaze9, 15: args.nmaze15}[sz]
            res["eval"][str(sz)] = score(ds, nm, lambda ms, ds=ds: np.stack([exact_cond(ds, m) for m in ms]))
    else:
        net, tr = train(args, dev)
        res.update(tr)
        for sz, (ds, T, nm) in tests.items():
            nm = {9: args.nmaze9, 15: args.nmaze15}[sz]
            res["eval"][str(sz)] = evaluate(net, ds, sz, T, dev, nm)
    res["wall_total_s"] = round(time.time() - t0, 1)
    U.dump(os.path.join(args.out, "p9a", name + ".json"), res)
    print("DONE", name, json.dumps({k: {kk: v[kk] for kk in ("kl", "voi_spearman", "kendall_median", "sign_acc")}
                                   for k, v in res["eval"].items()}), flush=True)


if __name__ == "__main__":
    main()
