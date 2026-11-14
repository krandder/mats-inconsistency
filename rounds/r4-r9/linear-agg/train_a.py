"""Experiment A training: one linear transformer (kind, L, seed) on fresh synthetic episodes (outcomes only: the loss is the
squared error of the query prediction against the noiseless target y_q = <k_q, w>; no oracle quantity is used).
Model selection: best validation MSE (fixed 20k-episode validation set, sigma ~ U(0.1, 3)), evaluated every --eval-every steps.
Never reads test data.   python3 train_a.py --kind diag --L 4 --seed 0 --out runs/a/diag-L4-s0
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth as S  # noqa: E402
from lt import LinTF  # noqa: E402

VAL_SEED = 7_000_001
YSCALE = 3.0   # ~ std of the whitened rollout values over training-distribution episodes (2.97, seed 123456789); fixed


def tokens(ep, dtype=torch.float64, dev="cpu"):
    """Depth-whitened context tokens (k_r, v_r / YSCALE) / sqrt(m_r) [n, 20, 17]; query-like tokens: the query then the 8
    probe candidates (a probe is a depth-1 observation), y part 0 [n, 9, 17]. Accepts numpy (synth.episodes) or torch
    (synth.episodes_torch) episodes."""
    if isinstance(ep["v"], np.ndarray):
        ep = {k: torch.as_tensor(np.ascontiguousarray(ep[k]), dtype=dtype, device=dev) for k in ("keys", "m", "v", "cand")}
    n = ep["v"].shape[0]
    sm = ep["m"].sqrt()
    ctx = torch.cat([ep["keys"] / sm[..., None], (ep["v"] / sm / YSCALE)[..., None]], -1)
    kq = torch.as_tensor(S.KQ, dtype=dtype, device=ctx.device)
    q = torch.cat([kq[None, None].expand(n, 1, S.D), ep["cand"]], 1)
    qry = torch.cat([q, torch.zeros(n, q.shape[1], 1, dtype=dtype, device=ctx.device)], -1)
    return ctx, qry


def predict(model, ctx, qry, keep_states=False):
    c, q, st = model(ctx, qry, keep_states=keep_states)
    return -YSCALE * q[:, 0, -1], c, q, st


@torch.no_grad()
def val_mse(model, V, chunk=5000):
    tot = 0.0
    for a in range(0, V[0].shape[0], chunk):
        yh, *_ = predict(model, V[0][a:a + chunk], V[1][a:a + chunk])
        tot += float(((yh - V[2][a:a + chunk]) ** 2).sum())
    return tot / V[0].shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=("full", "diag", "gdpp"))
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--n-val", type=int, default=20000)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float32", choices=("float32", "float64"))   # amendment A-2 (13:07Z): float32
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else (a.device if a.device != "auto" else "cpu"))
    DT = torch.float32 if a.dtype == "float32" else torch.float64
    os.makedirs(a.out, exist_ok=True)
    lr0 = a.lr if a.lr is not None else {1: 1e-3, 2: 1e-3, 4: 5e-4, 6: 3e-4}[a.L]
    ev = S.episodes(a.n_val, VAL_SEED)
    V = (*tokens(ev, DT, dev), torch.tensor(ev["yq"], dtype=DT, device=dev))
    t0 = time.time()
    restarts, lr, hist = 0, lr0, []
    while True:
        model = LinTF(a.kind, a.L, S.D + 1, init=1e-3, seed=a.seed + 100 * restarts).to(dev, DT)
        gen = torch.Generator(device=dev)
        gen.manual_seed(1_000_000 * (1 + a.seed) + 1000 * restarts)
        skipped = 0
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
        best, best_state, diverged = float("inf"), None, False
        for step in range(1, a.steps + 1):
            ep = S.episodes_torch(a.batch, gen, dev, DT)
            ctx, qry = tokens(ep, DT, dev)
            yh, *_ = predict(model, ctx, qry)
            loss = ((yh - ep["yq"]) ** 2).mean()
            if not torch.isfinite(loss):                     # skip a batch with a non-finite loss
                skipped += 1
                sched.step()
                if skipped > 50:                             # amendment A-4 (13:49Z): pre-registered rule reinstated
                    diverged = True                          # -> restart at lr / 3 (at most 2 restarts)
                    break
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % a.eval_every == 0 or step == a.steps:
                vm = val_mse(model, V)
                hist.append({"restart": restarts, "step": step, "lr": lr, "train_loss": float(loss), "val_mse": vm,
                             "skipped": skipped, "sec": round(time.time() - t0, 1)})
                # amendment A-1 (13:07Z): no restart on validation; a non-finite or worse validation MSE is never selected
                if math.isfinite(vm) and vm < best:
                    best, best_state = vm, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if not diverged or restarts >= 2:
            break
        restarts += 1
        lr = lr / 3.0
        print(f"diverged; restart {restarts} at lr {lr:g}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    if not hist:
        hist = [{"restart": restarts, "step": 0, "lr": lr, "train_loss": float("nan"), "val_mse": float("nan"), "skipped": 0,
                 "sec": 0}]
    torch.save({"state": {k: v.cpu() for k, v in model.state_dict().items()}, "kind": a.kind, "L": a.L, "seed": a.seed},
               os.path.join(a.out, "model.pt"))
    with open(os.path.join(a.out, "train-history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
        w.writeheader()
        w.writerows(hist)
    info = {"kind": a.kind, "L": a.L, "seed": a.seed, "params": model.n_params(), "lr0": lr0, "lr_final_run": lr,
            "restarts": restarts, "diverged_final": bool(diverged and best_state is None), "best_val_mse": best,
            "steps": a.steps, "batch": a.batch, "sec": time.time() - t0, "device": str(dev), "skipped_last_run": skipped}
    if a.kind != "full":
        info["omegas_xx_xy_yx_yy"] = model.omegas().cpu().tolist()
    json.dump(info, open(os.path.join(a.out, "train.json"), "w"), indent=1)
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
