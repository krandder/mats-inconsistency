"""Item 1b: CostNet+BB on the official easy-to-hard (Bansal et al.) maze benchmark, for a head-to-head with the
official DT-Recall maze checkpoints evaluated by trader-economy/dt-baseline.

Port (fixed before any test number was seen; P12 CostNet+BB recipe verbatim otherwise):
  * input image (3, 2n+6, 2n+6), walls black, open white, start green, goal red, 2x2 pixels per maze block and a
    3-pixel black border. Blocks = x[:, :, 3:-3:2, 3:-3:2] -> (3, n, n) (a fixed tokenisation DT does not get).
  * S = the green block, G = the red block (read from the colours).
  * graph: every 4-neighbour block pair inside the grid is an edge (walls are NOT removed; the net must price them).
  * CostNet: conv3x3(3->32) ReLU conv3x3(32->1), softplus + 0.01 (P12 CostNet, 3 input channels instead of 15).
  * edge cost = mean of the two cell costs; exact min-cost S-G path (P12 nets.shortest_path, Bellman-Ford) in
    training; blackbox differentiation (Vlastelica, Paulus et al. 2020), lambda 20; Hamming loss over edges.
  * two changes from P12, both chosen on train/val before any test run (pilot history in ../PREREG-1b.md):
    BBPos clamps the perturbed costs at 1e-3 instead of 0 (with 0 the walk-back cycles and the gradient is garbage),
    and --norm 1 divides each maze's costs by their mean (without it the net drives every cost to the 0.01 floor,
    where softplus has no gradient, because the target path is longer than the predicted shortcut).
  * Adam 1e-3, 200-step linear warm-up, cosine to 10%, batch 64, grad clip 1.0, 12000 steps, 8 dihedral maps.
  * train = maze_data_train_9 [0:48000], val (gate) = maze_data_train_9 [48000:50000].
  * test prediction: min-cost path by scipy Dijkstra on the same graph and costs (exact; checked against the
    Bellman-Ford solver by `selftest`), path blocks -> 2x2 pixels + border, masked by open pixels
    (inputs.max(1) > 0), exact = every pixel equals the solution (the official deep-thinking rule).

  python bansal.py train --seed S --out RUN.pt              (.part every 500 steps, resumes)
  python bansal.py eval --ckpt RUN.pt --sets val,e2h_maze_13,e2h_maze_33,e2h_maze_59,e2h_maze_201 --outdir DIR
  python bansal.py selftest
Data: $FU_BANSAL_DATA/maze_data_{train_9,test_13,test_33,test_59,test_201}/{inputs,solutions}.npy
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flow, nets

DATA = os.environ.get("FU_BANSAL_DATA", "/root/bansal")
# set -> (dir, first n items); the n match dt-baseline/code/dt_eval.py SETS so both arms score the same mazes
SETS = {"val": ("maze_data_train_9", (48000, 50000)), "e2h_maze_13": ("maze_data_test_13", (0, 10000)),
        "e2h_maze_33": ("maze_data_test_33", (0, 1000)), "e2h_maze_59": ("maze_data_test_59", (0, 1000)),
        "e2h_maze_201": ("maze_data_test_201", (0, 100))}
LAM = 20.0
BB_EPS = 1e-3


class BBPos(torch.autograd.Function):
    """nets.BBShortestPath with the perturbed costs clamped at BB_EPS instead of 0.

    Found on val before any test run (2026-09-29 23:15Z): with clamp_min(0) a target edge whose cost drops below
    lambda*|dL/dy| gets cost exactly 0, and the solver's walk-back (argmin over predecessors) then cycles between
    zero-cost neighbours and never reaches S, so the perturbed solution y_lambda is a broken walk and the gradient is
    garbage for that maze (4 to 7 of 640 backward solves in P12's own CostNet run; most of them once lambda is large).
    With strictly positive costs the predecessor on a shortest path has strictly smaller distance, so the walk ends."""

    @staticmethod
    def forward(ctx, ecost, open_, S, G, lam):
        y, _ = nets.shortest_path(ecost.detach(), open_, S, G)
        y = y.float()
        ctx.save_for_backward(ecost, open_, S, G, y); ctx.lam = lam
        return y

    @staticmethod
    def backward(ctx, gy):
        ecost, open_, S, G, y = ctx.saved_tensors; lam = ctx.lam
        gsym = gy + flow.nb(gy)
        wp = (ecost.detach() + lam * gsym).clamp_min(BB_EPS)
        yp, ok = nets.shortest_path(wp, open_, S, G)
        BB_STATS["calls"] += len(ok); BB_STATS["notok"] += int((~ok).sum())
        return -(y - yp.float()) / lam, None, None, None, None


BB_STATS = {"calls": 0, "notok": 0}


class BCostNet(nn.Module):
    def __init__(self, w=32, in_ch=3):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, w, 3, 1, 1); self.c2 = nn.Conv2d(w, 1, 3, 1, 1)

    def forward(self, x):
        return F.softplus(self.c2(F.relu(self.c1(x))))[:, 0] + 0.01


def load_raw(name):
    d, (lo, hi) = SETS[name]
    xi = np.load(os.path.join(DATA, d, "inputs.npy"), mmap_mode="r")
    si = np.load(os.path.join(DATA, d, "solutions.npy"), mmap_mode="r")
    hi = min(hi, xi.shape[0])
    return np.asarray(xi[lo:hi], np.float32), np.asarray(si[lo:hi], np.float32)


def blocks(x):
    """(N,3,2n+6,2n+6) -> (N,3,n,n)"""
    return x[:, :, 3:-3:2, 3:-3:2]


def ends(xb):
    """green block = S, red block = G -> (N,2) each"""
    r, g, b = xb[:, 0], xb[:, 1], xb[:, 2]
    n = xb.shape[-1]
    sm = (g > 0.5) & (r < 0.5) & (b < 0.5); gm = (r > 0.5) & (g < 0.5) & (b < 0.5)
    assert bool((sm.flatten(1).sum(1) == 1).all()) and bool((gm.flatten(1).sum(1) == 1).all()), "start/goal not unique"
    si = sm.flatten(1).float().argmax(1); gi = gm.flatten(1).float().argmax(1)
    return torch.stack([si // n, si % n], 1), torch.stack([gi // n, gi % n], 1)


def grid_open(B, n, dev="cpu"):
    o = torch.ones(B, n, n, 4, dtype=torch.bool, device=dev)
    o[:, :, -1, 0] = False; o[:, -1, :, 1] = False; o[:, :, 0, 2] = False; o[:, 0, :, 3] = False
    return o


def node_to_edges(node, o):
    """node path mask (B,n,n) -> slot indicator (B,n,n,4): both ends on the path (the path is a simple tree path)."""
    nb = flow.nb(node.float().unsqueeze(-1).expand(*node.shape, 4).contiguous())
    return (node.unsqueeze(-1) & (nb > 0.5) & o)


def edges_to_nodes(on, S, G):
    node = on.any(-1)
    ar = torch.arange(on.shape[0], device=on.device)
    node[ar, S[:, 0], S[:, 1]] = True; node[ar, G[:, 0], G[:, 1]] = True
    return node


def dihedral(xb, yb, k):
    """k in 0..7 on (B,C,n,n) and (B,n,n)"""
    if k & 4:
        xb = xb.transpose(-1, -2); yb = yb.transpose(-1, -2)
    r = k & 3
    return torch.rot90(xb, r, (-2, -1)), torch.rot90(yb, r, (-2, -1))


def to_pixels(node, n):
    px = node.float().repeat_interleave(2, 1).repeat_interleave(2, 2)
    return F.pad(px, (3, 3, 3, 3))


def lr_at_fn(steps):
    def f(s):
        if s < 200:
            return (s + 1) / 200
        p = (s - 200) / max(1, steps - 200)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))
    return f


def train(a):
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    x, y = load_raw_train()
    xb = torch.from_numpy(blocks(x).copy()); yb = torch.from_numpy(y[:, 3:-3:2, 3:-3:2].copy()) > 0.5
    N, n = xb.shape[0], xb.shape[-1]
    model = BCostNet(a.w)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at_fn(a.steps))
    g = torch.Generator().manual_seed(1000 + a.seed)
    part = a.out + ".part"; start = 0; log = []; prior = 0.0
    if os.path.exists(part):
        ck = torch.load(part, weights_only=False)
        model.load_state_dict(ck["state"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        g.set_state(ck["g"]); torch.set_rng_state(ck["rng"]); start = ck["step"]; log = ck["log"]; prior = ck["sec"]
        print(json.dumps({"resumed_from_step": start}), flush=True)
    o = grid_open(a.B, n)
    t0 = time.time(); run = {}
    for s in range(start, a.steps):
        idx = torch.randint(N, (a.B,), generator=g); k = int(torch.randint(8, (1,), generator=g))
        xs, ys = dihedral(xb[idx], yb[idx], k)
        S, G = ends(xs)
        tgt = node_to_edges(ys, o).float()
        c = model(xs)
        if a.norm:   # per-maze mean-normalised costs: the solver is scale-invariant, the lambda perturbation is not
            c = c / c.flatten(1).mean(1).view(-1, 1, 1)
        ec = flow.edge_cost_tensor(c, o)
        yhat = BBPos.apply(ec, o, S, G, a.lam)
        loss = (yhat * (1 - tgt) + (1 - yhat) * tgt)[..., :2].sum() / a.B
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        exact = float((yhat.bool() == tgt.bool()).flatten(1).all(1).float().mean())
        with torch.no_grad():
            wall = xs.max(1).values < 0.5
            ratio = float(c[wall].median() / c[~wall].median()) if wall.any() else float("nan")
        for kk, v in {"loss": float(loss), "gn": float(gn), "exact": exact, "wall_open_cost_ratio": ratio}.items():
            run[kk] = run.get(kk, 0.0) + v
        if (s + 1) % 100 == 0:
            row = {"step": s + 1, "sec": round(prior + time.time() - t0, 1), **{kk: round(v / 100, 5) for kk, v in run.items()}}
            log.append(row); run = {}; print(json.dumps(row), flush=True)
        if (s + 1) % 500 == 0 and s + 1 < a.steps:
            torch.save({"state": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "g": g.get_state(),
                        "rng": torch.get_rng_state(), "step": s + 1, "log": log, "sec": prior + time.time() - t0}, part + ".tmp")
            os.replace(part + ".tmp", part)
    info = {"seed": a.seed, "w": a.w, "steps": a.steps, "B": a.B, "lr": a.lr, "lam": a.lam, "norm": a.norm, "params": sum(p.numel() for p in model.parameters()),
            "train_seconds": round(prior + time.time() - t0, 1), "resumed_from_step": start,
            "bb_backward_calls_this_process": BB_STATS["calls"], "bb_backward_not_ok_this_process": BB_STATS["notok"]}
    torch.save({"state": model.state_dict(), "info": info, "log": log}, a.out)
    if os.path.exists(part):
        os.replace(part, part + ".tmp")
    print("DONE", json.dumps(info), flush=True)


def load_raw_train():
    d = os.path.join(DATA, "maze_data_train_9")
    x = np.load(os.path.join(d, "inputs.npy"), mmap_mode="r"); y = np.load(os.path.join(d, "solutions.npy"), mmap_mode="r")
    return np.asarray(x[:48000], np.float32), np.asarray(y[:48000], np.float32)


def dijkstra_paths(c, S, G):
    """c (B,n,n) float64 cell costs -> node masks (B,n,n) of the min-cost S-G path (edge cost = mean of the cells)."""
    B, n, _ = c.shape
    idx = np.arange(n * n).reshape(n, n)
    a = np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()]); b = np.concatenate([idx[:, 1:].ravel(), idx[1:, :].ravel()])
    out = np.zeros((B, n, n), bool)
    for i in range(B):
        cf = c[i].ravel()
        w = (cf[a] + cf[b]) / 2
        M = csr_matrix((np.concatenate([w, w]), (np.concatenate([a, b]), np.concatenate([b, a]))), shape=(n * n, n * n))
        s = int(S[i, 0] * n + S[i, 1]); t = int(G[i, 0] * n + G[i, 1])
        _, pred = dijkstra(M, directed=True, indices=s, return_predecessors=True)
        v = t; m = out[i].ravel()
        while v != s and v >= 0:
            m[v] = True; v = pred[v]
        m[s] = True
        out[i] = m.reshape(n, n)
    return out


@torch.no_grad()
def evaluate(a):
    torch.set_num_threads(a.threads)
    ck = torch.load(a.ckpt, weights_only=False)
    model = BCostNet(ck["info"]["w"]); model.load_state_dict(ck["state"]); model.eval()
    name = os.path.basename(a.ckpt)[:-3]
    os.makedirs(a.outdir, exist_ok=True)
    res = {"name": name, "info": ck["info"], "sets": {}}
    arrs = {}
    for sname in a.sets.split(","):
        t0 = time.time()
        x, y = load_raw(sname)
        ex = np.zeros(len(x), bool); valid = np.zeros(len(x), bool); ratio = np.zeros(len(x))
        bs = max(1, 200000 // (x.shape[-1] ** 2))
        for lo in range(0, len(x), bs):
            xt = torch.from_numpy(x[lo:lo + bs]); yt = torch.from_numpy(y[lo:lo + bs])
            xb = blocks(xt); n = xb.shape[-1]
            S, G = ends(xb)
            c = model(xb).double().numpy()
            node = torch.from_numpy(dijkstra_paths(c, S.numpy(), G.numpy()))
            px = to_pixels(node, n)
            openm = (xt.max(1).values > 0).float()
            pred = px * openm
            ex[lo:lo + bs] = (pred == yt).flatten(1).all(1).numpy()
            wall = xb.max(1).values < 0.5
            valid[lo:lo + bs] = ~(node & wall).flatten(1).any(1).numpy()
            for j in range(len(xt)):
                wj = wall[j].numpy()
                ratio[lo + j] = np.median(c[j][wj]) / np.median(c[j][~wj])
        arrs[f"{sname}/exact"] = ex; arrs[f"{sname}/valid"] = valid; arrs[f"{sname}/wall_open_ratio"] = ratio
        res["sets"][sname] = {"n": int(len(x)), "exact": float(ex.mean()), "valid": float(valid.mean()),
                              "wall_open_cost_ratio_median": float(np.median(ratio)), "sec": round(time.time() - t0, 1)}
        print(sname, json.dumps(res["sets"][sname]), flush=True)
    np.savez_compressed(os.path.join(a.outdir, name + ".npz"), **arrs)
    json.dump(res, open(os.path.join(a.outdir, name + ".json"), "w"), indent=1)


def selftest(a):
    """Dijkstra vs Bellman-Ford on random positive costs (no maze data), plus the block/pixel round trip on val."""
    rng = np.random.default_rng(0)
    for n in (9, 13, 33):
        B = 20
        c = torch.from_numpy(rng.uniform(0.01, 10.0, (B, n, n)))
        S = torch.from_numpy(rng.integers(0, n, (B, 2))); G = torch.from_numpy(rng.integers(0, n, (B, 2)))
        o = grid_open(B, n)
        on, ok = nets.shortest_path(flow.edge_cost_tensor(c, o), o, S, G)
        nb_ = edges_to_nodes(on, S, G).numpy()
        nd = dijkstra_paths(c.numpy(), S.numpy(), G.numpy())
        agree = float((nb_ == nd).reshape(B, -1).all(1).mean())
        print(json.dumps({"selftest_solver_agree": agree, "n": n}), flush=True)
        assert agree == 1.0
    x, y = load_raw("val")
    xt = torch.from_numpy(x[:500]); yt = torch.from_numpy(y[:500])
    xb = blocks(xt); n = xb.shape[-1]
    node = yt[:, 3:-3:2, 3:-3:2] > 0.5
    rt = float((to_pixels(node, n) * (xt.max(1).values > 0).float() == yt).flatten(1).all(1).float().mean())
    S, G = ends(xb)
    o = grid_open(len(xb), n)
    tgt = node_to_edges(node, o)
    back = float((edges_to_nodes(tgt, S, G) == node).flatten(1).all(1).float().mean())
    # an oracle cost (open 1, wall 1e3) through Dijkstra must be exact on val
    c = torch.where(xb.max(1).values > 0.5, 1.0, 1e3).double().numpy()
    nd = torch.from_numpy(dijkstra_paths(c, S.numpy(), G.numpy()))
    orc = float((to_pixels(nd, n) * (xt.max(1).values > 0).float() == yt).flatten(1).all(1).float().mean())
    print(json.dumps({"pixel_roundtrip_exact": rt, "edge_node_roundtrip": back, "oracle_wall1e3_val_exact": orc}), flush=True)
    assert rt == 1.0 and back == 1.0 and orc == 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "selftest"])
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out"); ap.add_argument("--ckpt")
    ap.add_argument("--w", type=int, default=32); ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--B", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=1); ap.add_argument("--lam", type=float, default=LAM)
    ap.add_argument("--norm", type=int, default=0)
    ap.add_argument("--sets", default="val,e2h_maze_13,e2h_maze_33,e2h_maze_59,e2h_maze_201"); ap.add_argument("--outdir")
    a = ap.parse_args()
    {"train": train, "eval": evaluate, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
