"""P9b training: N0 (continuous DT-recall), N1 (K=8 binary ST state) and the clamp arms.

Progressive loss (Bansal 2022), alpha=0.5: loss = (1-a) L(T steps) + a L(n no-grad steps, then k grad steps),
k ~ U{1..T}, n ~ U{0..T-k}, drawn per batch.

Clamp arms (one shared clamp RNG stream; each selected episode gets ONE site (channel k, cell c) among
sites whose natural trajectory changes at some tau_last >= 2; t_c ~ U{1..tau_last-1}; the clamp is held from
t_c to the end on the absolute step index, in both the no-grad prefix and the grad segment):
  N4a  rate 0.5, v = natural z_T (stop-grad), weight 1        (self-reveal teacher forcing; D2)
  N3   rate 0.5, v ~ Bern(0.5), always settled, weight 1       (random clamps; noise rival)
  N2   rate 0.5, v ~ Bern(0.5), settled iff v == z_T, weight 2 (clamp-and-void, futarchy arm; A/A)
  N4p  rate 0.25, v = z_T, weight 2                            (N4'; same law as N2 by D2; A/A)
  N1c  no clamps, batch x1.5                                   (compute match)
  N3b  no site clamps; z_0 ~ Bern(0.5) on the full grid for 50% of episodes (path independence rival)
The clamped branch loss is summed over settled episodes and divided by the natural batch size B, so
E[grad] of N2 = N4p = N4a.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import p9util as U
from p9b_data import make_onpath
from p9models import BIN_VARIANT, GRAD_CLIP, BinNet, DTR, n_params

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA = 0.5


def build(arm, w=24, K=8):
    if arm == "N0":
        return DTR(4, w=w, nblocks=2, head=(32, 8))  # upstream DT-recall-2D layout
    return BinNet(4, w=w, K=K)


def is_bin(net):
    return isinstance(net, BinNet)


def apply_clamp(z, t, cl):
    if cl is None:
        return z
    M, V, tc = cl
    m = M & (t >= tc)[:, None, None, None]
    return torch.where(m, V, z)


def bce_ep(logit, y):
    return F.binary_cross_entropy_with_logits(logit, y, reduction="none").mean((1, 2))


def prog_loss(net, x, y, T, n, k, cl=None, record=False, z0=None):
    """Per-episode progressive loss (B,). Returns (loss, traj (T,B,K,n,n) bool or None, fwd, bwd)."""
    B = x.shape[0]
    if is_bin(net):
        e, zz = net.init(x)
        if z0 is not None:
            zz = z0
        traj = [] if record else None
        z = zz
        for t in range(1, T + 1):
            z = apply_clamp(net.step(z, e), t, cl)
            if record:
                traj.append(z.detach() > 0.5)
        lf = bce_ep(net.readout(z), y)
        with torch.no_grad():
            z = zz
            for t in range(1, n + 1):
                z = apply_clamp(net.step(z, e), t, cl)
        z = z.detach()
        for t in range(n + 1, n + k + 1):
            z = apply_clamp(net.step(z, e), t, cl)
        lp = bce_ep(net.readout(z), y)
        tr = torch.stack(traj) if record else None
    else:
        h0 = net.init(x)
        h = h0
        for t in range(T):
            h = net.step(h, x)
        lf = bce_ep(net.readout(h), y)
        with torch.no_grad():
            h = h0
            for t in range(n):
                h = net.step(h, x)
        h = h.detach()
        for t in range(k):
            h = net.step(h, x)
        lp = bce_ep(net.readout(h), y)
        tr = None
    return (1 - ALPHA) * lf + ALPHA * lp, tr, B * (T + n + k), B * (T + k)


def draw_nk(rng, T):
    k = int(rng.integers(1, T + 1))
    n = int(rng.integers(0, T - k + 1))
    return n, k


def clamp_plan(arm, traj, crng, T):
    """From the natural trajectory (T,B,K,n,n) bool, draw the clamp sites. Returns list of
    (b, k, i, j, t_c, v, weight) for settled episodes, plus counters. Draws are made in the same order for
    every arm so all clamp arms share one stream."""
    Tn, B, K, n, m = traj.shape
    z0 = torch.zeros_like(traj[:1])
    full = torch.cat([z0, traj], 0)  # t = 0..T
    ch = full[1:] != full[:-1]  # change at t (1..T)
    tidx = torch.arange(1, Tn + 1)[:, None, None, None, None]
    tau_last = torch.where(ch, tidx, torch.zeros_like(tidx)).amax(0)  # (B,K,n,m), 0 if never changes
    zT = traj[-1]
    rate = {"N4a": 0.5, "N3": 0.5, "N2": 0.5, "N4p": 0.25}[arm]
    out, cnt = [], {"drawn": B, "selected": 0, "no_eligible": 0, "void": 0, "settled": 0}
    for b in range(B):
        u = crng.random(5)
        elig = (tau_last[b] >= 2).nonzero()
        if u[0] >= rate:
            continue
        cnt["selected"] += 1
        if len(elig) == 0:
            cnt["no_eligible"] += 1
            continue
        kk, i, j = (int(a) for a in elig[min(int(u[1] * len(elig)), len(elig) - 1)])
        tl = int(tau_last[b, kk, i, j])
        tc = 1 + min(int(u[2] * (tl - 1)), tl - 2)
        vnat = float(zT[b, kk, i, j])
        vr = float(u[3] < 0.5)
        if arm in ("N4a", "N4p"):
            v, wgt = vnat, (1.0 if arm == "N4a" else 2.0)
        elif arm == "N3":
            v, wgt = vr, 1.0
        else:  # N2
            v, wgt = vr, 2.0
            if vr != vnat:
                cnt["void"] += 1
                continue
        # no-op check: natural value on [t_c, T] differs from v somewhere
        noop = bool((traj[tc - 1:, b, kk, i, j].float() == v).all())
        cnt["noop"] = cnt.get("noop", 0) + int(noop)
        cnt["settled"] += 1
        out.append((b, kk, i, j, tc, v, wgt))
    return out, cnt


def train(args):
    dev = U.device()
    torch.set_num_threads(1)
    name = f"{args.arm}_s{args.seed}{args.tag}"
    outdir = os.path.join(args.out, "train")
    os.makedirs(outdir, exist_ok=True)
    part = os.path.join(args.scratch, name + ".partial")
    os.makedirs(args.scratch, exist_ok=True)
    data = make_onpath(args.pool, 9, seed=10_000 + args.seed)
    X, Y = data["x"].to(dev), data["y"].to(dev)
    torch.manual_seed(1000 + args.seed)
    net = build(args.arm).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    drng = np.random.default_rng(2000 + args.seed)
    lrng = np.random.default_rng(3000 + args.seed)
    crng = np.random.default_rng(4000 + args.seed)
    B = int(round(args.batch * (1.5 if args.arm == "N1c" else 1)))
    acct = {"fwd_cell_steps": 0, "bwd_cell_steps": 0, "fwd_ep_steps": 0, "bwd_ep_steps": 0,
            "natural_eps": 0, "clamp": {}}
    log, step0 = [], 0
    st = U.load_partial(part)
    if st is not None:
        net.load_state_dict(st["net"]); opt.load_state_dict(st["opt"])
        drng.bit_generator.state, lrng.bit_generator.state, crng.bit_generator.state = st["rng"]
        acct, log, step0 = st["acct"], st["log"], st["step"]
        print("resumed", name, step0, flush=True)
    wall0 = time.time() - (log[-1]["wall"] if log else 0)
    T = args.T
    ncell = 81
    for step in range(step0, args.steps):
        # linear warm-down of lr over the last 25% of steps (same for every arm)
        frac = step / args.steps
        for g in opt.param_groups:
            g["lr"] = args.lr * (1.0 if frac < 0.75 else max(0.05, (1 - frac) / 0.25))
        idx = torch.as_tensor(drng.integers(0, X.shape[0], B), device=dev)
        x, y = X[idx], Y[idx]
        n, k = draw_nk(lrng, T)
        z0 = None
        if args.arm == "N3b":
            sel = torch.as_tensor(crng.random(B) < 0.5, device=dev)
            rnd = torch.as_tensor(crng.random((B, 8, 9, 9)) < 0.5, device=dev).float()
            z0 = torch.where(sel[:, None, None, None], rnd, torch.zeros_like(rnd))
        rec = args.arm in ("N4a", "N3", "N2", "N4p")
        l_nat, traj, f, bw = prog_loss(net, x, y, T, n, k, record=rec, z0=z0)
        loss = l_nat.mean()
        acct["fwd_ep_steps"] += f; acct["bwd_ep_steps"] += bw; acct["natural_eps"] += B
        if rec:
            plan, cnt = clamp_plan(args.arm, traj.cpu(), crng, T)
            for kk, vv in cnt.items():
                acct["clamp"][kk] = acct["clamp"].get(kk, 0) + vv
            n2, k2 = draw_nk(crng, T)
            if plan:
                bi = torch.tensor([p[0] for p in plan], device=dev)
                M = torch.zeros(len(plan), 8, 9, 9, dtype=torch.bool, device=dev)
                V = torch.zeros(len(plan), 8, 9, 9, device=dev)
                tc = torch.zeros(len(plan), dtype=torch.long, device=dev)
                wv = torch.zeros(len(plan), device=dev)
                for r, (b, kk, i, j, t_c, v, wgt) in enumerate(plan):
                    M[r, kk, i, j] = True; V[r, kk, i, j] = v; tc[r] = t_c; wv[r] = wgt
                l_cl, _, f, bw = prog_loss(net, x[bi], y[bi], T, n2, k2, cl=(M, V, tc))
                loss = loss + (wv * l_cl).sum() / B
                acct["fwd_ep_steps"] += f; acct["bwd_ep_steps"] += bw
        opt.zero_grad()
        loss.backward()
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
        opt.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            log.append({"step": step, "loss": float(loss), "l_nat": float(l_nat.mean()), "wall": round(time.time() - wall0, 1)})
            print(json.dumps(log[-1]), flush=True)
        if (step + 1) % args.ckpt_every == 0 and step + 1 < args.steps:
            U.save_partial(part, net=net.state_dict(), opt=opt.state_dict(),
                           rng=(drng.bit_generator.state, lrng.bit_generator.state, crng.bit_generator.state),
                           acct=acct, log=log, step=step + 1)
    acct["fwd_cell_steps"] = acct["fwd_ep_steps"] * ncell
    acct["bwd_cell_steps"] = acct["bwd_ep_steps"] * ncell
    acct["wall_s"] = round(time.time() - wall0, 1)
    sd = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    torch.save(sd, os.path.join(outdir, name + ".pt"))
    U.dump(os.path.join(outdir, name + ".json"), {"args": vars(args), "arm": args.arm, "seed": args.seed,
                                                   "bin_variant": BIN_VARIANT, "grad_clip": GRAD_CLIP,
                                                   "params": n_params(net), "batch": B, "acct": acct, "log": log,
                                                   "device": str(dev)})
    if os.path.exists(part):
        os.remove(part)
    print("DONE", name, json.dumps(acct), flush=True)
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["N0", "N1", "N1c", "N4a", "N3", "N2", "N4p", "N3b"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--pool", type=int, default=40000)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--ckpt_every", type=int, default=250)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--scratch", default="/tmp/p9/partial")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
