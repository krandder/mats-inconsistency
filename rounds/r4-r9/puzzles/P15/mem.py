"""P15 memory side-measurement: peak GPU memory of one training step (prefix task, strings of length D,
batch 32, one element) at D = 16, 32, 128, 1024 for Bl, A, A-prog (worst case n=0, k=D; and a mean over
the schedule), A-ds, A-1step and BPTT with gradient checkpointing every ~sqrt(D) levels.
On OOM the step is retried at batch 8 and 2 and the result says so.   Writes results/memory.json."""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p15  # noqa: E402
from p15 import U_, K_  # noqa: E402


def ckpt_bptt_loss(P, task, bt, noise, nbr, tau=1.0):
    x, y, vm = bt["x"], bt["y"], bt["valid"]
    E, B, N = vm.shape
    D = task.D
    vm5 = vm[..., None, None]
    px = p15.pre_x(P, x, nbr, task.Cx)
    z = x.new_zeros(E, B, N, U_, K_)
    seg = max(1, int(round(math.sqrt(D))))

    def run(z, k0, k1, px):
        for k in range(k0, k1):
            _, lg = p15.level(P, px, z, nbr, task.Cx)
            s = F.softmax((lg + p15.gumbel(noise[k])) / tau, -1)
            h = p15.onehot(s.argmax(-1), K_, s.dtype)
            z = ((h - s).detach() + s) * vm5
        return z

    for k0 in range(0, D, seg):
        z = checkpoint(run, z, k0, min(D, k0 + seg), px, use_reentrant=False)
    return p15.head_ce(P, z, y, vm)


def measure(arm, D, B, sched=None, reps=2):
    task = p15.Task("prefix", D)
    N = D
    nbr = task.nbr(N).to(p15.DEV)
    P = p15.init_params([0], task.Cx, task.W, task.Cy)
    rng = np.random.default_rng([1000, 0])
    g = torch.Generator(device=p15.DEV).manual_seed(2000)
    cfg = {"tau": torch.ones(1, device=p15.DEV), "eps": 0.05, "alpha": 1.0,
           "gen_perm": torch.Generator().manual_seed(4242)}
    peaks, walls = [], []
    for _ in range(reps):
        bt = p15.stack_batches([task.train_batch(rng, B)], p15.DEV, torch.float32)
        noise = torch.rand((D, 1, B, N, U_, K_), generator=g, device=p15.DEV)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        t0 = time.time()
        if arm == "A-ckpt":
            loss = ckpt_bptt_loss(P, task, bt, noise, nbr)
        else:
            loss, _ = p15.train_step(arm, P, task, bt, noise, sched, cfg, nbr)
        names = [k for k in P if not (arm == "C" and k in ("W1", "b1", "W2", "b2"))]
        grads = torch.autograd.grad(loss, [P[k] for k in names], allow_unused=True)
        torch.cuda.synchronize()
        walls.append(time.time() - t0)
        peaks.append(torch.cuda.max_memory_allocated() - base)
        del loss, grads, bt, noise
    return {"peak_bytes": int(max(peaks)), "peak_MiB": max(peaks) / 2 ** 20, "wall_s": float(np.median(walls))}


def main():
    arms = [("Bl", None), ("A", None), ("A-prog", "worst"), ("A-ds", None), ("A-1step", None), ("A-ckpt", None)]
    out = {"note": "prefix task, strings of length D (N=D cells), one element, fp32; peak = max_memory_allocated "
                   "during one forward+backward step minus the allocation before it (params excluded). "
                   "A-prog at its worst case n=0, k=D. A-ckpt = BPTT with checkpoints every round(sqrt(D)) levels.",
           "gpu": torch.cuda.get_device_name(0), "rows": []}
    for D in (16, 32, 128, 1024):
        for arm, mode in arms:
            sched = (0, D) if arm == "A-prog" else None
            row = {"arm": arm, "D": D}
            for B in (32, 8, 2):
                try:
                    r = measure(arm, D, B, sched)
                    row.update({"B": B, **r})
                    break
                except torch.OutOfMemoryError:
                    row.setdefault("oom_at_B", []).append(B)
                    torch.cuda.empty_cache()
            out["rows"].append(row)
            print(json.dumps(row), flush=True)
            torch.cuda.empty_cache()
    json.dump(out, open(os.path.join(p15.RESDIR, "memory.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
