"""P0 training loop. One loop; the arm is (alpha, n-range, k) or armB.

  bansal : (1-alpha) * loss(T iterations) + alpha * loss(k after n detached), n ~ U[0,T-1],
           k ~ U[1,T-n], drawn once per batch (deep-thinking get_output_for_prog_loss)
  chunk  : alpha = 1, n ~ U[0,T-k] detached, then exactly k iterations with BPTT, loss at the end
  armB   : market-resnet mresnet.loss_fn('B') verbatim on a DT backbone (b_t = readout(h_t))

Mazes use deep-thinking's maze optimizer (Adam, wd 2e-4, lr 1e-3, recur params lr/30,
per-epoch exponential warmup over 10 epochs, MultiStepLR milestone 100 never reached,
loss mean over open pixels) and keep final and best-validation checkpoints.
Prefix arms use the market-resnet grid optimizer (AdamW 1e-3, wd 2e-4, OneCycle pct 0.05,
clip 1, batch 128, prefix_batch lengths 8-32, data generator seed 1000+seed).
Training runs in TF32.
"""
import argparse, json, math, os, random, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import parametrize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dtnet import DTRecall, n_params  # noqa: E402
from dtl import DTL1D, pick_width  # noqa: E402
from mr_adapter import MRTrainWrapper, mresnet  # noqa: E402
from check import expand  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)            # e.g. M1_s0
p.add_argument("--task", choices=["maze", "prefix"], required=True)
p.add_argument("--arch", choices=["dt", "dtl"], default="dt")
p.add_argument("--width", type=int, default=0)
p.add_argument("--mode", choices=["bansal", "chunk", "armB"], required=True)
p.add_argument("--alpha", type=float, default=1.0)
p.add_argument("--k", type=int, default=1)
p.add_argument("--T", type=int, default=30)
p.add_argument("--steps", type=int, required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--ckpt_at", default="")            # comma list of extra kept checkpoints (steps)
p.add_argument("--split_at", type=int, default=0)   # M1-long: at this step also save final+best as --split_name
p.add_argument("--split_name", default="")
p.add_argument("--frozen", default=os.path.join(HERE, "frozen"))
p.add_argument("--out", default=os.path.join(HERE, "ckpt"))
p.add_argument("--val_every", type=int, default=800)
p.add_argument("--probe_at", default="5000,10000,15000,20000,25000,30000,35000,40000")
p.add_argument("--time_steps", type=int, default=0)  # pilot: time this many steps and exit
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = p.parse_args()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
dev = a.device
os.makedirs(a.out, exist_ok=True)
torch.manual_seed(a.seed)
rng = random.Random(a.seed)

dim = 2 if a.task == "maze" else 1
if a.arch == "dtl":
    width = a.width or pick_width()
    net = DTL1D(width).to(dev)
else:
    width = a.width or (64 if dim == 2 else 90)
    net = DTRecall(width, dim=dim, in_channels=3).to(dev)
arch = {"kind": a.arch if a.arch == "dtl" else f"dt{dim}d", "dim": dim, "width": width, "params": n_params(net)}
print(json.dumps({"arch": arch, "args": vars(a)}), flush=True)


# ------------------------------------------------------------------ data
if a.task == "maze":
    z = np.load(os.path.join(a.frozen, "maze_train9.npz"))
    code = torch.as_tensor(z["train9/code"])
    sol = torch.as_tensor(np.unpackbits(z["train9/sol"], axis=-1, count=9))
    X, Y = expand(code, sol)
    tr_idx = torch.as_tensor(z["train9/train_idx"]).long()
    va_idx = torch.as_tensor(z["train9/val_idx"]).long()
    X, Y = X.to(dev), Y.to(dev).long()
    zv = np.load(os.path.join(a.frozen, "mazes_e2h.npz"))
    pc = torch.as_tensor(zv["v33/code"][:50])
    ps = torch.as_tensor(np.unpackbits(zv["v33/sol"][:50], axis=-1, count=33))
    PX, PY = expand(pc, ps)
    PX, PY = PX.to(dev), PY.to(dev).long()
    BS = 50
    steps_per_epoch = len(tr_idx) // BS           # 800 (drop_last)
    dgen = torch.Generator().manual_seed(a.seed)
else:
    dgen = torch.Generator().manual_seed(1000 + a.seed)
    BS = 128


def fwd(x, n_iters, h=None):
    if h is None:
        h = net.init(x)
    for _ in range(n_iters):
        h = net.step(h, x)
    return h


def cell_loss(logits, y, mask):
    """maze: CE mean over open pixels (deep-thinking); prefix: per-string mean over real bits (market-resnet)."""
    if a.task == "maze":
        l = F.cross_entropy(logits, y, reduction="none")
        return l[mask].mean()
    b = logits[:, 1] - logits[:, 0]
    return ((F.binary_cross_entropy_with_logits(b, y, reduction="none") * mask).sum(1) / mask.sum(1)).mean()


def loss_on(x, y, mask):
    T = a.T
    info = {}
    if a.mode == "armB":
        wrapper = MRTrainWrapper(net)
        loss, lg = mresnet.loss_fn("B", wrapper, x, y, mask, T)
        return loss, {"final": lg["final"]}
    if a.mode == "bansal":
        n = rng.randrange(0, T)
        k = rng.randrange(1, T - n + 1)
    else:
        n = rng.randrange(0, T - a.k + 1)
        k = a.k
    xin = x if a.task == "maze" else F.one_hot(x.long(), 3).float().transpose(1, 2)
    loss = 0.0
    if a.mode == "bansal" and a.alpha != 1:
        full = cell_loss(net.readout(fwd(xin, T)), y, mask)
        loss = loss + (1 - a.alpha) * full
        info["full"] = full.item()
    if n > 0:
        with torch.no_grad():
            h0 = fwd(xin, n)
        h = fwd(xin, k, h0.detach())
    else:
        h = fwd(xin, k)
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


# ------------------------------------------------------------------ optimizer
if a.task == "maze":
    base = [p_ for n_, p_ in net.named_parameters() if "recur" not in n_]
    recur = [p_ for n_, p_ in net.named_parameters() if "recur" in n_]
    opt = torch.optim.Adam([{"params": base}, {"params": recur, "lr": 1e-3 / a.T}], lr=1e-3, weight_decay=2e-4)
    base_lrs = [g["lr"] for g in opt.param_groups]

    def set_lr(step):
        e = step // steps_per_epoch
        f = 1.0 if e + 1 >= 10 else 1.0 - math.exp(-(e + 1) / 10)
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * f
    sched = None
    clip = None
else:
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=a.steps, pct_start=0.05)
    clip = 1.0
    set_lr = None
print(json.dumps({"param_groups": [{"n": sum(p_.numel() for p_ in g["params"]), "lr": g["lr"],
                                    "wd": g["weight_decay"]} for g in opt.param_groups],
                  "optimizer": type(opt).__name__}), flush=True)

# ------------------------------------------------------------------ resume
ck = os.path.join(a.out, a.name + ".partial")
start, log, best, perm, pos = 0, [], -1.0, None, 0
if os.path.exists(ck):
    s = torch.load(ck, map_location=dev, weights_only=False)
    net.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
    if sched is not None:
        sched.load_state_dict(s["sched"])
    start, log, best = s["step"], s["log"], s["best"]
    # map_location=dev moves the saved RNG states to the GPU; generators need CPU ByteTensors
    dgen.set_state(s["dgen"].cpu()); rng.setstate(s["rng"]); torch.set_rng_state(s["torch_rng"].cpu())
    perm, pos = s.get("perm"), s.get("pos", 0)
    print(json.dumps({"resumed_at": start}), flush=True)


def save_small(path):
    torch.save({"model": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "arch": arch,
                "args": vars(a), "log": log}, path + ".tmp")
    os.replace(path + ".tmp", path)


keep = {int(s_) for s_ in a.ckpt_at.split(",") if s_}
probes = {int(s_) for s_ in a.probe_at.split(",") if s_}
t0 = time.time()
net.train()
tl = time.time()
for step in range(start, a.steps):
    if a.task == "maze":
        set_lr(step)
        if perm is None or pos + BS > len(perm):
            perm = tr_idx[torch.randperm(len(tr_idx), generator=dgen)]
            pos = 0
        bi = perm[pos:pos + BS].to(dev)
        pos += BS
        x, y = X[bi], Y[bi]
        mask = x.max(1)[0] > 0
    else:
        x, y, mask = mresnet.prefix_batch(BS, 8, 32, dgen, dev)
    with parametrize.cached():
        loss, info = loss_on(x, y, mask)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if clip:
        torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
    opt.step()
    if sched is not None:
        sched.step()
    done = step + 1
    if a.time_steps and done - start >= a.time_steps:
        torch.cuda.synchronize() if dev == "cuda" else None
        el = time.time() - tl
        print(json.dumps({"timing": a.name, "steps": done - start, "sec": round(el, 2),
                          "sec_per_1k": round(1000 * el / (done - start), 1),
                          "peak_mem_mb": torch.cuda.max_memory_allocated() // 2**20 if dev == "cuda" else 0}), flush=True)
        sys.exit(0)
    if done % 200 == 0 or done == a.steps:
        lg = {"step": done, "loss": round(loss.item(), 5), **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in info.items()},
              "lr": opt.param_groups[0]["lr"], "sec": round(time.time() - t0, 1)}
        if a.task == "maze" and (done % a.val_every == 0 or done == a.steps or done == a.split_at):
            va = maze_acc(X[va_idx.to(dev)], Y[va_idx.to(dev)], a.T)
            lg["val9_acc30"] = va
            if va > best:
                best = va
                save_small(os.path.join(a.out, a.name + "_best.pt"))
                lg["best"] = True
        if a.task == "maze" and done in probes:
            lg["probe33_acc300"] = maze_acc(PX, PY, 300, bs=50)
        log.append(lg)
        print(json.dumps(lg), flush=True)
    if done in keep:
        save_small(os.path.join(a.out, f"{a.name}_at{done}.pt"))
    if a.split_at and done == a.split_at:
        # the 15k-step run of the same seed is the prefix of the long run: same data order, same optimizer
        save_small(os.path.join(a.out, a.split_name + ".pt"))
        bp = os.path.join(a.out, a.name + "_best.pt")
        if os.path.exists(bp):
            sb = torch.load(bp, map_location="cpu", weights_only=False)
            sb["args"] = {**sb["args"], "split_of": a.name}
            torch.save(sb, os.path.join(a.out, a.split_name + "_best.pt"))
    if done % 1000 == 0 and done < a.steps:
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict() if sched else None,
                    "step": done, "log": log, "best": best, "dgen": dgen.get_state(), "rng": rng.getstate(),
                    "torch_rng": torch.get_rng_state(), "perm": perm, "pos": pos}, ck + ".tmp")
        os.replace(ck + ".tmp", ck)
save_small(os.path.join(a.out, a.name + ".pt"))
if os.path.exists(ck):
    os.remove(ck)
print(json.dumps({"done": a.name, "sec": round(time.time() - t0, 1),
                  "peak_mem_mb": torch.cuda.max_memory_allocated() // 2**20 if dev == "cuda" else 0}), flush=True)
