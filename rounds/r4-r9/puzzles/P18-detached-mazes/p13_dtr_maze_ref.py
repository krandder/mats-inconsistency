"""DTR-eq: P0 train.py (sha256 in SNAPSHOTS.sha256), maze + bansal mode only, trained to a step budget.

Unchanged from P0 train.py (mazes, --mode bansal): DT-recall (dtnet.py), progressive loss
(1-alpha) * loss(T iterations) + alpha * loss(k after n detached), n ~ U[0,T-1], k ~ U[1,T-n] from
random.Random(seed); Adam lr 1e-3 wd 2e-4, recur params lr/T; loss mean over open pixels; data order
from torch.Generator(seed) permutations of the 40k train split, batch 50, drop_last; TF32 in training.

Added for P13:
  --warmup_frac f   the recipe's per-epoch exponential warmup over 10 of 50 epochs, compressed to the step
                    budget: 10 warmup stages of f * steps / 10 steps each, lr factor 1 - exp(-(e+1)/10),
                    then constant LR (the recipe's milestone at epoch 100 is never reached). With f = 0.2 and
                    40k steps a stage is 800 steps = one epoch, i.e. exactly P0 train.py's schedule.
  --stop_at s       stop (and save) after s steps of the --steps budget (unit test (h)).
  atomic checkpoint every --ckpt_sec seconds as well as every 1000 steps (model, optimizer, step,
                    python/torch RNG, data generator, permutation and position); final checkpoint always used.
"""
import argparse, json, math, os, random, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import parametrize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dtnet import DTRecall, n_params  # noqa: E402
from check import expand  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
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
p.add_argument("--val_n", type=int, default=10000)
p.add_argument("--ckpt_sec", type=float, default=600)
p.add_argument("--time_steps", type=int, default=0)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = p.parse_args()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
dev = a.device
os.makedirs(a.out, exist_ok=True)
torch.manual_seed(a.seed)
rng = random.Random(a.seed)
net = DTRecall(a.width, dim=2, in_channels=3).to(dev)
arch = {"kind": "dt2d", "dim": 2, "width": a.width, "params": n_params(net)}
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


def fwd(x, n_iters, h=None):
    if h is None:
        h = net.init(x)
    for _ in range(n_iters):
        h = net.step(h, x)
    return h


def cell_loss(logits, y, mask):
    l = F.cross_entropy(logits, y, reduction="none")
    return l[mask].mean()


def loss_on(x, y, mask):
    T = a.T
    info = {}
    n = rng.randrange(0, T)
    k = rng.randrange(1, T - n + 1)
    loss = 0.0
    if a.alpha != 1:
        full = cell_loss(net.readout(fwd(x, T)), y, mask)
        loss = loss + (1 - a.alpha) * full
        info["full"] = full.item()
    if n > 0:
        with torch.no_grad():
            h0 = fwd(x, n)
        h = fwd(x, k, h0.detach())
    else:
        h = fwd(x, k)
    prog = cell_loss(net.readout(h), y, mask)
    loss = loss + a.alpha * prog
    info.update(prog=prog.item(), n=n, k=k)
    return loss, info


@torch.no_grad()
def maze_acc(Xs, Ys, iters, bs=500):
    net.eval()
    ok = 0
    for i in range(0, len(Xs), bs):
        x, y = Xs[i:i + bs], Ys[i:i + bs]
        o = net.readout(fwd(x, iters))
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


def save_partial(done):
    torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "step": done, "log": log,
                "dgen": dgen.get_state(), "rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if dev == "cuda" else None,
                "perm": perm, "pos": pos, "sec": time.time() - t0}, ck + ".tmp")
    os.replace(ck + ".tmp", ck)


if os.path.exists(ck) and not a.time_steps:
    s = torch.load(ck, map_location=dev, weights_only=False)
    net.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
    start, log, sec0 = s["step"], s["log"], s.get("sec", 0.0)
    dgen.set_state(s["dgen"].cpu()); rng.setstate(s["rng"]); torch.set_rng_state(s["torch_rng"].cpu())
    if dev == "cuda" and s.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all([r.cpu() for r in s["cuda_rng"]])
    perm, pos = s.get("perm"), s.get("pos", 0)
    if perm is not None:
        perm = perm.cpu()
    print(json.dumps({"resumed_at": start, "sec": round(sec0, 1)}), flush=True)


def save_small(path):
    torch.save({"model": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "arch": arch,
                "args": vars(a), "log": log}, path + ".tmp")
    os.replace(path + ".tmp", path)


t0 = time.time() - sec0
last_ck = time.time()
net.train()
tl = None
end = min(a.steps, a.stop_at) if a.stop_at else a.steps
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
    opt.zero_grad(set_to_none=True)
    loss.backward()
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
        lg = {"step": done, "loss": round(loss.item(), 5), **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in info.items()},
              "lr": opt.param_groups[0]["lr"], "sec": round(time.time() - t0, 1)}
        if done % a.val_every == 0 or done == end:
            lg["val9_acc30"] = maze_acc(X[va_idx.to(dev)], Y[va_idx.to(dev)], a.T)
        log.append(lg)
        print(json.dumps(lg), flush=True)
    if done < end and (done % 1000 == 0 or time.time() - last_ck > a.ckpt_sec):
        save_partial(done); last_ck = time.time()
save_small(os.path.join(a.out, a.name + ".pt"))
json.dump({"args": vars(a), "arch": arch, "log": log, "train_sec": time.time() - t0, "steps": end,
           "train_peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0},
          open(os.path.join(a.out, a.name + ".json"), "w"))
if os.path.exists(ck):
    os.remove(ck)
print(json.dumps({"done": a.name, "sec": round(time.time() - t0, 1)}), flush=True)
