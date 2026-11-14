"""P18 trainer: DT-Recall with progressive loss, and K-block DT-Recall with per-step deep supervision.

Copied from P13 dtr_maze.py (itself P0 train.py; the P13 copy is kept here as p13_dtr_maze_ref.py). Everything outside
the loss is unchanged from dtr_maze.py and shared by every arm: DT-Recall width 128 blocks, maze_train9 40k train split,
batch 50 in torch.Generator(seed) order, Adam lr 1e-3 wd 2e-4 with recurrent parameters at lr/T (lr_throttle),
exponential warmup over the first --warmup_frac of the step budget, no gradient clipping (the official maze recipe has
clip: null), loss = cross entropy averaged over open pixels, TF32 in training, T = 30.

  --mode prog --alpha a        DT-Recall (dtnet.DTRecall), Bansal et al. 2022 progressive loss
                               (1-a) * loss(T iterations) + a * loss(k steps after n detached), n ~ U[0,T-1],
                               k ~ U[1,T-n]; a = 1 skips the full term. Bit-identical to dtr_maze.py (tests_cpu (a)).
  --mode ds --K K --order rand|rr --detach 1|0
                               DTRecallK (dtmod.py): K untied recurrent blocks, shared projection and head. Each
                               step t = 1..T applies block i_t: rand = uniform random per step (one pick for the
                               whole batch, from random.Random(seed), as market-resnet econ2.py), rr = (t-1) mod K.
                               Loss = mean over t of the open-pixel cross entropy of head(h_t) (deep supervision).
                               --detach 1: h_t = block(stopgrad(h_(t-1)), x) for t >= 2 (each step trained on its
                               own loss from a detached state; the projection gets gradient through step 1 only).
                               --detach 0: the same losses, full backprop through all T steps.
Added for P18: --mode ds; a divergence gate (non-finite loss: save, mark diverged, exit 0); counters of block and
head applications with and without gradient (for the FLOP audit); --val_n 2000 by default (logging only).
Resume: atomic .partial every --ckpt_sec seconds and every 1000 steps (model, optimizer, step, log, RNGs, data order).
"""
import argparse, json, math, os, random, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import parametrize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dtnet import DTRecall, n_params  # noqa: E402
from dtmod import DTRecallK  # noqa: E402
from check import expand  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--mode", default="prog")            # prog | ds
p.add_argument("--K", type=int, default=4)
p.add_argument("--order", default="rand")           # rand | rr   (ds only)
p.add_argument("--detach", type=int, default=1)     # ds only
p.add_argument("--width", type=int, default=128)
p.add_argument("--alpha", type=float, default=0.01)
p.add_argument("--T", type=int, default=30)
p.add_argument("--steps", type=int, required=True)
p.add_argument("--warmup_frac", type=float, default=0.2)
p.add_argument("--stop_at", type=int, default=0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--frozen", default=os.path.join(HERE, "frozen_p0"))
p.add_argument("--out", default=os.path.join(HERE, "ckpt"))
p.add_argument("--val_every", type=int, default=800)
p.add_argument("--val_n", type=int, default=2000)
p.add_argument("--ckpt_sec", type=float, default=300)
p.add_argument("--time_steps", type=int, default=0)
p.add_argument("--kill_at", type=int, default=0)     # test only: os._exit(3) right after step kill_at (after its checkpoint)
p.add_argument("--grad_dump", default="")            # test only: save the first step's gradients, batch and picks, then exit
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = p.parse_args()
assert a.mode in ("prog", "ds") and a.order in ("rand", "rr")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
dev = a.device
os.makedirs(a.out, exist_ok=True)
torch.manual_seed(a.seed)
rng = random.Random(a.seed)
if a.mode == "prog":
    net = DTRecall(a.width, dim=2, in_channels=3).to(dev)
    arch = {"kind": "dt2d", "dim": 2, "width": a.width, "params": n_params(net)}
else:
    net = DTRecallK(a.width, K=a.K).to(dev)
    arch = {"kind": "dtk", "dim": 2, "width": a.width, "K": a.K, "params": n_params(net)}
print(json.dumps({"arch": arch, "args": vars(a)}), flush=True)

z = np.load(os.path.join(a.frozen, "maze_train9.npz"))
code = torch.as_tensor(z["train9/code"])
sol = torch.as_tensor(np.unpackbits(z["train9/sol"], axis=-1, count=9))
X, Y = expand(code, sol)
tr_idx = torch.as_tensor(z["train9/train_idx"]).long()
va_idx = torch.as_tensor(z["train9/val_idx"]).long()[:a.val_n]
X, Y = X.to(dev), Y.to(dev).long()
BS = 50
stage = max(1, round(a.warmup_frac * a.steps / 10))
dgen = torch.Generator().manual_seed(a.seed)
CNT = {"blk_grad": 0, "blk_nograd": 0, "head_grad": 0, "head_nograd": 0}   # applications per example


def fwd(x, n_iters, h=None):
    if h is None:
        h = net.init(x)
    for _ in range(n_iters):
        h = net.step(h, x)
    return h


def cell_loss(logits, y, mask):
    l = F.cross_entropy(logits, y, reduction="none")
    return l[mask].mean()


def loss_prog(x, y, mask):
    T = a.T
    info = {}
    n = rng.randrange(0, T)
    k = rng.randrange(1, T - n + 1)
    loss = 0.0
    if a.alpha != 1:
        full = cell_loss(net.readout(fwd(x, T)), y, mask)
        loss = loss + (1 - a.alpha) * full
        info["full"] = full.item()
        CNT["blk_grad"] += T; CNT["head_grad"] += 1
    if n > 0:
        with torch.no_grad():
            h0 = fwd(x, n)
        h = fwd(x, k, h0.detach())
    else:
        h = fwd(x, k)
    prog = cell_loss(net.readout(h), y, mask)
    loss = loss + a.alpha * prog
    CNT["blk_nograd"] += n; CNT["blk_grad"] += k; CNT["head_grad"] += 1
    info.update(prog=prog.item(), n=n, k=k)
    return loss, info


def loss_ds(x, y, mask):
    T = a.T
    h = net.init(x)
    losses = []
    picks = []
    for t in range(1, T + 1):
        i = rng.randrange(a.K) if a.order == "rand" else (t - 1) % a.K
        picks.append(i)
        hin = h.detach() if (a.detach and t >= 2) else h
        h = net.step(hin, x, i)
        losses.append(cell_loss(net.readout(h), y, mask))
    CNT["blk_grad"] += T; CNT["head_grad"] += T
    loss = torch.stack(losses).mean()
    return loss, {"l1": losses[0].item(), "lT": losses[-1].item(), "picks": "".join(map(str, picks[:8])), "_picks": picks}


loss_on = loss_prog if a.mode == "prog" else loss_ds


@torch.no_grad()
def fwd_test(x, iters):
    h = net.init(x)
    for t in range(1, iters + 1):
        h = net.step(h, x) if a.mode == "prog" else net.step(h, x, (t - 1) % a.K)
    return h


@torch.no_grad()
def maze_acc(Xs, Ys, iters, bs=500):
    net.eval()
    ok = 0
    for i in range(0, len(Xs), bs):
        x, y = Xs[i:i + bs], Ys[i:i + bs]
        o = net.readout(fwd_test(x, iters))
        pred = o.argmax(1) * (x.max(1)[0] > 0).long()
        ok += (pred == y).flatten(1).all(1).sum().item()
    net.train()
    return ok / len(Xs)


base = [p_ for n_, p_ in net.named_parameters() if "recur" not in n_]
recur = [p_ for n_, p_ in net.named_parameters() if "recur" in n_]
opt = torch.optim.Adam([{"params": base}, {"params": recur, "lr": 1e-3 / a.T}], lr=1e-3, weight_decay=2e-4)
base_lrs = [g["lr"] for g in opt.param_groups]


def set_lr(step):
    e = step // stage
    f = 1.0 if e + 1 >= 10 else 1.0 - math.exp(-(e + 1) / 10)
    for g, b in zip(opt.param_groups, base_lrs):
        g["lr"] = b * f


ck = os.path.join(a.out, a.name + ".partial")
start, log, perm, pos, sec0 = 0, [], None, 0, 0.0
diverged = None


def save_partial(done):
    torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "step": done, "log": log,
                "dgen": dgen.get_state(), "rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if dev == "cuda" else None,
                "perm": perm, "pos": pos, "sec": time.time() - t0, "cnt": dict(CNT)}, ck + ".tmp")
    os.replace(ck + ".tmp", ck)


if os.path.exists(ck) and not a.time_steps:
    s = torch.load(ck, map_location=dev, weights_only=False)
    net.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
    start, log, sec0 = s["step"], s["log"], s.get("sec", 0.0)
    dgen.set_state(s["dgen"].cpu()); rng.setstate(s["rng"]); torch.set_rng_state(s["torch_rng"].cpu())
    if dev == "cuda" and s.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all([r.cpu() for r in s["cuda_rng"]])
    perm, pos = s.get("perm"), s.get("pos", 0)
    CNT.update(s.get("cnt", {}))
    if perm is not None:
        perm = perm.cpu()
    print(json.dumps({"resumed_at": start, "sec": round(sec0, 1)}), flush=True)


def save_small(path):
    torch.save({"model": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "arch": arch,
                "args": vars(a), "log": log, "diverged": diverged}, path + ".tmp")
    os.replace(path + ".tmp", path)


t0 = time.time() - sec0
last_ck = time.time()
net.train()
tl = None
end = min(a.steps, a.stop_at) if a.stop_at else a.steps
done = start
for step in range(start, end):
    if a.time_steps and step == start + 20:
        if dev == "cuda":
            torch.cuda.synchronize()
        tl = time.time()
    set_lr(step)
    if perm is None or pos + BS > len(perm):
        perm = tr_idx[torch.randperm(len(tr_idx), generator=dgen)]
        pos = 0
    bi = perm[pos:pos + BS].to(dev)
    pos += BS
    x, y = X[bi], Y[bi]
    mask = x.max(1)[0] > 0
    with parametrize.cached():
        loss, info = loss_on(x, y, mask)
    if not torch.isfinite(loss):
        diverged = {"step": step + 1, "loss": str(loss.item())}
        print(json.dumps({"diverged": diverged}), flush=True)
        break
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if a.grad_dump:
        torch.save({"grads": {n_: p_.grad.detach().cpu().clone() for n_, p_ in net.named_parameters() if p_.grad is not None},
                    "bi": bi.cpu(), "info": info, "loss": loss.item()}, a.grad_dump)
        sys.exit(0)
    opt.step()
    done = step + 1
    if a.time_steps and tl is not None and done - (start + 20) >= a.time_steps:
        if dev == "cuda":
            torch.cuda.synchronize()
        el = time.time() - tl
        print(json.dumps({"timing": a.name, "steps": a.time_steps, "sec": round(el, 2),
                          "sec_per_1k": round(1000 * el / a.time_steps, 1),
                          "peak_mem_mb": torch.cuda.max_memory_allocated() // 2 ** 20 if dev == "cuda" else 0}), flush=True)
        sys.exit(0)
    if done % 200 == 0 or done == end:
        lg = {"step": done, "loss": round(loss.item(), 5), **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in info.items() if not k.startswith("_")},
              "lr": opt.param_groups[0]["lr"], "sec": round(time.time() - t0, 1)}
        if done % a.val_every == 0 or done == end:
            lg["val9_acc30"] = maze_acc(X[va_idx.to(dev)], Y[va_idx.to(dev)], a.T)
        log.append(lg)
        print(json.dumps(lg), flush=True)
    if done < end and (done % 1000 == 0 or time.time() - last_ck > a.ckpt_sec or (a.kill_at and done == a.kill_at)):
        save_partial(done); last_ck = time.time()
    if a.kill_at and done == a.kill_at:
        print(json.dumps({"killed_at": done}), flush=True)
        os._exit(3)
save_small(os.path.join(a.out, a.name + ".pt"))
json.dump({"args": vars(a), "arch": arch, "log": log, "train_sec": time.time() - t0, "steps": done, "cnt": CNT,
           "diverged": diverged,
           "train_peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0},
          open(os.path.join(a.out, a.name + ".json"), "w"))
if os.path.exists(ck):
    os.remove(ck)
print(json.dumps({"done": a.name, "sec": round(time.time() - t0, 1), "steps": done}), flush=True)
