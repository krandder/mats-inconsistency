"""The Bansal et al. 2022 baseline: DT-recall 1D trained with the progressive loss.

Architecture and training follow the authors' code (github.com/aks2203/deep-thinking:
models/dt_net_1d.py, config/problem/model/dt_net_recall_1d.yaml,
config/problem/hyp/prefix_sums_default.yaml):
  projection conv(1 -> 400, k3) + ReLU; one recurrent block = recall conv(400+1 -> 400, k3)
  (the input is concatenated to the state every iteration) + ReLU + 2 residual blocks
  (2 convs k3 each); head conv(400->400) ReLU conv(400->200) ReLU conv(200->2).
  Progressive loss with alpha = 1 and max_iters = 30: n ~ U{0..29} iterations without
  gradient, then k ~ U{1..30-n} with gradient from the detached state, cross-entropy at
  iteration n + k. Adam lr 1e-3, weight decay 2e-4, exponential warmup (10 of 150 epochs),
  lr x0.1 at epochs 60 and 100 of 150, gradient-norm clip 1, batch 100.
Differences from the paper: the data are econ2's (prefix-sum parity, fresh strings of length
8-32 every batch, right-padded with 0 bits and masked in the loss, instead of 8,000 fixed
32-bit strings), and training length is set by --iters (the published schedule is 150
epochs x 80 batches = 12,000 iterations) or by --time_budget seconds (equal compute with the
economy; the schedule then runs on elapsed time). Evaluation: econ2.eval_depths (the same
test strings and depths as every phase-2 arm), with belief logit b = out[1] - out[0].
"""
import argparse, json, math, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from mresnet import prefix_batch
from econ2 import eval_depths, soft_score


class Block(nn.Module):  # BasicBlock1D, no group norm
    def __init__(self, w):
        super().__init__()
        self.c1 = nn.Conv1d(w, w, 3, padding=1, bias=False)
        self.c2 = nn.Conv1d(w, w, 3, padding=1, bias=False)

    def forward(self, x):
        return F.relu(self.c2(F.relu(self.c1(x))) + x)


class DTRecall1D(nn.Module):
    def __init__(self, w=400):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv1d(1, w, 3, padding=1, bias=False), nn.ReLU())
        self.recur = nn.Sequential(nn.Conv1d(w + 1, w, 3, padding=1, bias=False), nn.ReLU(), Block(w), Block(w))
        self.head = nn.Sequential(nn.Conv1d(w, w, 3, padding=1, bias=False), nn.ReLU(),
                                  nn.Conv1d(w, w // 2, 3, padding=1, bias=False), nn.ReLU(),
                                  nn.Conv1d(w // 2, 2, 3, padding=1, bias=False))

    def logit(self, h):
        o = self.head(h)
        return o[:, 1] - o[:, 0]


class DT:
    """Wrapper with econ2's rollout interface (for eval_depths)."""
    def __init__(self, a, device):
        self.a, self.dev = a, device
        self.net = DTRecall1D(a.width).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=a.lr, weight_decay=2e-4)

    @staticmethod
    def inp(x):
        return (x == 1).float().unsqueeze(1)      # PAD (2) -> 0

    @torch.no_grad()
    def rollout(self, x, mask, D, record_at=None, **kw):
        self.net.eval()
        xi = self.inp(x)
        h = self.net.proj(xi)
        bs = {}
        for t in range(1, D + 1):
            h = self.net.recur(torch.cat([h, xi], 1))
            if record_at is None or t in record_at:
                bs[t] = self.net.logit(h)
        return None, bs, torch.full((x.shape[0],), float(D))

    def train_step(self, x, y, mask, frac, gen):
        a = self.a
        self.net.train()
        ep = frac * 150                                            # epoch-equivalent of the published schedule
        lr = a.lr * (1 - math.exp(-(math.floor(ep) + 1) / 10)) * (0.1 if ep >= 60 else 1) * (0.1 if ep >= 100 else 1)
        for g in self.opt.param_groups:
            g["lr"] = lr
        xi = self.inp(x)
        n = int(torch.randint(0, a.max_iters, (1,), generator=gen))
        k = int(torch.randint(1, a.max_iters - n + 1, (1,), generator=gen))
        if n > 0:
            with torch.no_grad():
                h = self.net.proj(xi)
                for _ in range(n):
                    h = self.net.recur(torch.cat([h, xi], 1))
            h = h.detach()
        else:
            h = self.net.proj(xi)
        for _ in range(k):
            h = self.net.recur(torch.cat([h, xi], 1))
        b = self.net.logit(h)
        loss = -((soft_score(b, y) * mask).sum() / mask.sum())    # 2-class cross-entropy, mean over real bits
        self.opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
        return {"loss": loss.item(), "lr": lr, "n": n, "k": k}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=400)
    p.add_argument("--max_iters", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--bs", type=int, default=100)
    p.add_argument("--lmin", type=int, default=8)
    p.add_argument("--lmax", type=int, default=32)
    p.add_argument("--iters", type=int, default=12000)
    p.add_argument("--time_budget", type=float, default=0)      # >0: train for this many seconds instead
    p.add_argument("--ckpt_every", type=float, default=600)
    p.add_argument("--evalL", default="32,320,3200")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    torch.manual_seed(a.seed)
    gen = torch.Generator().manual_seed(3000 + a.seed)
    M = DT(a, a.device)
    ckp, log, it, sec0, resumed = a.out + ".ckpt.pt", [], 0, 0.0, None
    old = json.load(open(a.out + ".json")) if os.path.exists(a.out + ".json") else {}

    def save_atomic(obj, path):
        torch.save(obj, path + ".tmp"); os.replace(path + ".tmp", path)

    if os.path.exists(a.out + ".pt"):                                # training done: evals only
        ck = torch.load(a.out + ".pt", map_location=a.device, weights_only=False)
        M.net.load_state_dict(ck["net"]); log, it, sec0 = ck["log"], ck["it"], ck["sec"]
        resumed = ck.get("resumed")
        done = True
    else:
        done = False
        if os.path.exists(ckp):
            ck = torch.load(ckp, map_location="cpu", weights_only=False)   # RNG states must stay on the CPU
            M.net.load_state_dict(ck["net"]); M.opt.load_state_dict(ck["opt"])
            log, it, sec0 = ck["log"], ck["it"], ck["sec"]
            gen.set_state(ck["gen"]); torch.set_rng_state(ck["rng"])
            if a.device.startswith("cuda"):
                torch.cuda.set_rng_state_all(ck["cuda_rng"])
            resumed = it
            print(json.dumps({"resumed_at_it": it, "sec": round(sec0, 1)}), flush=True)
    if a.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0, last_ck = time.time() - sec0, time.time()
    while not done:
        el = time.time() - t0
        frac = el / a.time_budget if a.time_budget > 0 else it / a.iters
        if frac >= 1:
            break
        x, y, m = prefix_batch(a.bs, a.lmin, a.lmax, gen, a.device)
        lg = M.train_step(x, y, m, frac, gen)
        if it % 100 == 0:
            lg.update(it=it, sec=round(el, 1)); log.append(lg); print(json.dumps(lg), flush=True)
        it += 1
        if a.ckpt_every > 0 and time.time() - last_ck > a.ckpt_every:
            save_atomic({"net": M.net.state_dict(), "opt": M.opt.state_dict(), "log": log, "it": it,
                         "sec": time.time() - t0, "gen": gen.get_state(), "rng": torch.get_rng_state(),
                         "cuda_rng": torch.cuda.get_rng_state_all() if a.device.startswith("cuda") else None}, ckp)
            last_ck = time.time()
    if not done:
        sec0 = time.time() - t0
        save_atomic({"net": M.net.state_dict(), "log": log, "it": it, "sec": sec0, "args": vars(a), "resumed": resumed},
                    a.out + ".pt")
    res = {"args": vars(a), "log": log, "train_sec": sec0, "train_iters": it, "resumed": resumed,
           "params": sum(q.numel() for q in M.net.parameters())}
    if a.device.startswith("cuda") and not done:
        res["train_peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    elif old.get("train_peak_mem_gb"):
        res["train_peak_mem_gb"] = old["train_peak_mem_gb"]
    grid = {32: ([2, 4, 6, 8, 12, 16, 24, 32, 48, 64], 1000), 320: ([6, 8, 12, 16, 24, 32, 64, 160, 320, 640], 200),
            3200: ([8, 12, 16, 24, 32, 64, 320, 1600, 3200], 40)}
    ev = [e for e in old.get("eval", [])] if done else []
    have = {e["L"] for e in ev}
    for L in [int(z) for z in a.evalL.split(",")]:
        if L in have:
            continue
        depths, n = grid[L]
        ev += eval_depths(M, L, depths, n, a.device, pick="-", chunk=min(n, 100 if L < 3200 else 20))
        print(json.dumps(ev[-len(depths):]), flush=True)
        res["eval"] = ev
        json.dump(res, open(a.out + ".json.tmp", "w")); os.replace(a.out + ".json.tmp", a.out + ".json")
    res["eval"] = ev
    json.dump(res, open(a.out + ".json.tmp", "w")); os.replace(a.out + ".json.tmp", a.out + ".json")
