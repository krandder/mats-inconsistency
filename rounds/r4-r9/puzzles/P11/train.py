"""P11 training. One run = one arm, one hyperparameter, one seed.

R-prog / DT-L / R-final: Bansal et al. 2022 progressive loss, loss = (1-a) BCE(logit_T) + a BCE(logit after
n no-grad + k grad iterations), n ~ U{0..T-1}, k ~ U{1..T-n}; R-final is a = 0.
M arms: BCE(b_T) + (1/T) sum_t BCE(b_t) with b_t = sg(b_{t-1}) + Delta(u_t) (per-step MSR pay with
stop-grad = deep supervision). All losses are masked means over open pixels (upstream).
Adam 1e-3, weight decay 2e-4, OneCycle, batch 64, grad-norm clip 1.0, T_train = 30, width 32.
Training mazes: size 9 from a pool of 400k seeded mazes (seeds [0, 4e5)); each run draws its own
permutation, so no maze repeats inside a run.
Spot VMs get preempted: every 400 steps the full training state (weights, Adam, OneCycle, both RNGs, log) goes to
OUT/ckpt.pt, and a run that finds a matching ckpt.pt resumes from it (the same batches and n,k draws follow).
Loss and accuracy are summed on the device and read only at log steps (no per-step host sync).
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import gen
from model import build, n_params, MState, R_ARMS, M_ARMS

HERE = os.path.dirname(os.path.abspath(__file__))


def sha(p):
    return hashlib.sha256(open(os.path.join(HERE, p), "rb").read()).hexdigest()


def check_vendored():
    pin = json.load(open(os.path.join(HERE, "VENDORED.json")))
    got = {k: sha(k) for k in pin["files"]}
    assert got == pin["files"], f"vendored hash mismatch: {got} vs {pin['files']}"
    return {"vendored": got, "sources_at_vendoring": pin["sources"]}


def masked_bce(logit, y, m):
    l = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
    return (l * m).sum() / m.sum()


def exact(logit, y, m):
    ok = (((logit > 0).float() == y) & torch.isfinite(logit)) | (m == 0)
    return ok.flatten(1).all(1)


def r_loss(net, x, y, m, T, a, g):
    lm = lp = torch.zeros((), device=x.device)
    ex = None
    if a != 1:
        h = net.init_state(x)
        for _ in range(T):
            h = net.step(h, x)
        lg = net.out(h)
        lm = masked_bce(lg, y, m)
        ex = exact(lg.detach(), y, m).float().mean()
    if a != 0:
        n = int(torch.randint(0, T, (1,), generator=g))
        k = int(torch.randint(1, T - n + 1, (1,), generator=g))
        if n > 0:
            with torch.no_grad():
                h = net.init_state(x)
                for _ in range(n):
                    h = net.step(h, x)
            h = h.detach()
        else:
            h = net.init_state(x)
        for _ in range(k):
            h = net.step(h, x)
        lp = masked_bce(net.out(h), y, m)
    return (1 - a) * lm + a * lp, ex, {}


def m_loss(net, arm, hp, x, y, m, T):
    ms = MState(arm, hp, y.shape, x.device)
    h = net.init_state(x)
    tot = 0.0
    sat = torch.zeros((), device=x.device)
    for _ in range(T):
        h = net.step(h, x, ms.feat())
        u = net.out(h)
        with torch.no_grad():
            sat = sat + ((u.abs() > 5).float() * m).sum()
        b = ms.advance(u)
        lt = masked_bce(b, y, m)
        tot = tot + lt
    loss = lt + tot / T
    info = {"sat": sat / (m.sum() * T)}
    if ms.rem is not None:
        info["rem_frac"] = ((ms.rem / ms.init).float() * m).sum() / m.sum()
    return loss, exact(b.detach(), y, m).float().mean(), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--hp", default="{}")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--T", type=int, default=30)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--pool", default=os.path.join(HERE, "data", "pool9.npz"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    vend = check_vendored()
    hp = json.loads(a.hp)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(10_000 + a.seed)
    net = build(a.arm, a.width).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=a.steps)
    pool = gen.load_pool(a.pool, dev)
    perm = np.random.Generator(np.random.PCG64(777 + a.seed)).permutation(pool["n"])
    assert a.steps * a.bs <= pool["n"], "pool too small for no-repeat sampling"
    perm = torch.tensor(perm, device=dev)
    g = torch.Generator().manual_seed(20_000 + a.seed)
    os.makedirs(a.out, exist_ok=True)
    log = []
    t0 = time.time()
    zero = lambda: torch.zeros((), device=dev, dtype=torch.float64)
    acc_ex, acc_loss, acc_n, acc_info = zero(), zero(), 0, {}
    ck = os.path.join(a.out, "ckpt.pt")
    key = {"arm": a.arm, "hp": hp, "seed": a.seed, "steps": a.steps, "bs": a.bs, "T": a.T, "width": a.width,
           "pool": os.path.basename(a.pool)}
    start, resumed, prev_sec = 0, [], 0.0
    if os.path.exists(ck):
        c = torch.load(ck, map_location="cpu", weights_only=False)
        if c["key"] == key:
            net.load_state_dict(c["net"])
            opt.load_state_dict(c["opt"])
            sched.load_state_dict(c["sched"])
            g.set_state(c["g"])
            torch.set_rng_state(c["rng_cpu"])
            if dev == "cuda" and c.get("rng_cuda") is not None:
                torch.cuda.set_rng_state(c["rng_cuda"])
            start, log, resumed, prev_sec = c["step"], c["log"], c["resumed"] + [c["step"]], c["sec"]
            print(json.dumps({"resumed_at_step": start}), flush=True)
        else:
            print(json.dumps({"ckpt_key_mismatch": c["key"]}), flush=True)
    t0 -= prev_sec
    for step in range(start, a.steps):
        x, y = gen.render_torch(pool, perm[step * a.bs:(step + 1) * a.bs])
        m = x[:, 0]
        net.train()
        if a.arm in R_ARMS:
            al = 0.0 if a.arm == "R-final" else float(hp["a"])
            with torch.nn.utils.parametrize.cached():
                loss, ex, info = r_loss(net, x, y, m, a.T, al, g)
        else:
            loss, ex, info = m_loss(net, a.arm, hp, x, y, m, a.T)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        acc_loss += loss.detach()
        acc_n += 1
        if ex is not None:
            acc_ex += ex
        for k, v in info.items():
            acc_info[k] = acc_info.get(k, zero()) + v.detach()
        if (step + 1) % 200 == 0 or step == a.steps - 1:
            rec = {"step": step + 1, "loss": float(acc_loss) / acc_n,
                   "train_exact_T": float(acc_ex) / acc_n if ex is not None else None,
                   "gnorm": float(gn), "lr": sched.get_last_lr()[0], "sec": time.time() - t0}
            rec.update({k: float(v) / acc_n for k, v in acc_info.items()})
            log.append(rec)
            print(json.dumps(rec), flush=True)
            acc_ex, acc_loss, acc_n, acc_info = zero(), zero(), 0, {}
            if not np.isfinite(rec["loss"]):
                break
            if (step + 1) % 400 == 0 and step + 1 < a.steps:
                torch.save({"key": key, "step": step + 1, "net": net.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "g": g.get_state(), "rng_cpu": torch.get_rng_state(),
                            "rng_cuda": torch.cuda.get_rng_state() if dev == "cuda" else None, "log": log,
                            "resumed": resumed, "sec": time.time() - t0}, ck + ".tmp")
                os.replace(ck + ".tmp", ck)
    sec = time.time() - t0
    torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, os.path.join(a.out, "model.pt"))
    meta = {"arm": a.arm, "hp": hp, "seed": a.seed, "steps": a.steps, "bs": a.bs, "T": a.T, "width": a.width,
            "params": n_params(net), "train_sec": sec, "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
            "torch": torch.__version__, "log": log, "tag": a.tag, "resumed_at": resumed, **vend,
            "final_loss_finite": bool(np.isfinite(log[-1]["loss"])) if log else False}
    json.dump(meta, open(os.path.join(a.out, "train.json.tmp"), "w"))
    os.replace(os.path.join(a.out, "train.json.tmp"), os.path.join(a.out, "train.json"))
    if os.path.exists(ck):
        os.remove(ck)
    print("DONE", a.out, f"{sec:.0f}s", flush=True)


if __name__ == "__main__":
    main()
