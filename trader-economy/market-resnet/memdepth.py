"""Peak GPU memory and time of ONE training step vs depth D (L = D, batch 16):
local economic updates (econ2, streaming/online, cov markets) vs backprop
through depth (bptt, same traders). Also econ2's batched train_step, which
stores the rollout (O(D) records, no graph). Writes results/memdepth.json."""
import json, sys, time, types
import torch
from econ2 import Economy, BPTT
from mresnet import prefix_batch

Ds = [int(z) for z in (sys.argv[1] if len(sys.argv) > 1 else "32,320,1000,3200,10000").split(",")]
out = []
for mode in ["econ_online", "econ_batched", "bptt"]:
    for D in Ds:
        a = types.SimpleNamespace(N=4, width=96, lr=1e-3, cost=0.0, W0=20.0, frac=0.5, wealth_cap=1,
                                  stream="cov", voc=0, lam_c=1.0, lam_v=0.3, rent=1e-3, self_pay=0,
                                  train_halt=-1, settle="eps", eps=0.1, iters=1, prog=0)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        M = BPTT(a, "cuda") if mode == "bptt" else Economy(a, "cuda")
        gen = torch.Generator().manual_seed(0)
        x, y, m = prefix_batch(16, D, D, gen, "cuda")
        base = torch.cuda.memory_allocated() / 1e9
        t0 = time.time()
        r = {"mode": mode, "D": D, "L": D, "batch": 16, "base_gb": base}
        try:
            (M.train_step_online if mode == "econ_online" else M.train_step)(x, y, m, D, gen)
            torch.cuda.synchronize()
            r.update(peak_gb=torch.cuda.max_memory_allocated() / 1e9, sec=time.time() - t0, ok=True)
        except torch.OutOfMemoryError:
            r.update(ok=False, err="OOM", peak_gb=torch.cuda.max_memory_allocated() / 1e9, sec=time.time() - t0)
        print(json.dumps(r), flush=True)
        out.append(r)
        del M
        json.dump(out, open("results/memdepth.json", "w"))
        if not r["ok"]:
            break  # larger D will not fit either
