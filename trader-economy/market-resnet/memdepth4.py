"""Peak GPU memory and time of ONE training step against depth D at a FIXED length L = 32 (batch 16),
for the audit of section 9 (memdepth.py mixed depth and length by setting L = D).

Modes (all with the covariance stream; the BPTT modes use every step's loss + covariance targets,
the bptt_cov_dscov configuration):
  bptt            full backprop through depth, one graph
  bptt_ckpt       the same with per-step gradient checkpointing (Chen et al. 2016)
  bptt_tb1        truncated to 1 step (--tbptt 1), losses summed and one backward at the end
                  (the graph of every step is still held until then)
  tb1_stream      truncated to 1 step with a backward at every step (nothing kept across steps)
  econ_batched    the economy's train_step with VOC markets (voc=1: the configuration behind 0.93)
  econ_online     the economy's train_step_online (voc=0; every trade settled and backpropagated
                  at once), the configuration behind the 4.2 GB row of section 9
Writes <out> (default p4/memdepth4.json)."""
import json, os, sys, time, types
import torch
import torch.nn.functional as F
from econ2 import Economy, BPTT, soft_score, pair_truth, nlevels
from mresnet import prefix_batch

Ds = [int(z) for z in (sys.argv[1] if len(sys.argv) > 1 else "24,240,2400,24000").split(",")]
OUT = sys.argv[2] if len(sys.argv) > 2 else "p4/memdepth4.json"
L, BATCH = 32, 16
DEV = os.environ.get("DEV", "cuda")          # DEV=cpu: a smoke test only (no memory numbers)
CU = DEV.startswith("cuda")


def args(**kw):
    a = dict(N=4, width=96, lr=1e-3, cost=0.0, W0=20.0, frac=0.5, wealth_cap=1, stream="cov", voc=0, lam_c=1.0,
             lam_v=0.3, rent=1e-3, self_pay=0, train_halt=-1, settle="eps", eps=0.1, iters=1, prog=0, ds=1, covsup=1,
             tbptt=0, train_pick="random", clamp=0, gradckpt=0, bounty="", mode="bptt")
    a.update(kw); return types.SimpleNamespace(**a)


def tb1_stream(M, x, y, m, D, gen):
    """TBPTT(1) with a backward at every step: per-step loss + covariance targets, one optimizer step"""
    B = x.shape[0]; J = nlevels(L); nb = m.sum(1)
    xoh = F.one_hot(x, 3).float().transpose(1, 2)
    eq, meq = pair_truth(y, m, J)
    b = torch.zeros(B, L, device=DEV); c = torch.zeros(B, J, L, device=DEV); v = torch.zeros(B, 3, L, device=DEV)
    M.opt.zero_grad()
    for t in range(D):
        i = int(torch.randint(4, (1,), generator=gen))
        db, dc, dv, _ = M.traders[i](xoh, b, c, v, m)
        b1, c1 = b + db, c + dc
        l = -(((soft_score(b1, y) * m).sum(1) / nb).mean() + ((soft_score(c1, eq) * meq).sum((1, 2)) / nb).mean()) / D
        l.backward()
        b, c, v = b1.detach(), c1.detach(), (v + dv).detach()
    torch.nn.utils.clip_grad_norm_(M.traders.parameters(), 1.0); M.opt.step()


MODES = {
    "bptt": (lambda: BPTT(args(), DEV), "train_step"),
    "bptt_ckpt": (lambda: BPTT(args(gradckpt=1), DEV), "train_step"),
    "bptt_tb1": (lambda: BPTT(args(tbptt=1), DEV), "train_step"),
    "tb1_stream": (lambda: BPTT(args(), DEV), "stream"),
    "econ_batched": (lambda: Economy(args(mode="econ", voc=1), DEV), "train_step"),
    "econ_online": (lambda: Economy(args(mode="econ", voc=0), DEV), "train_step_online"),
}
out = []
for mode, (make, fn) in MODES.items():
    for D in Ds:
        if CU: torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        M = make()
        gen = torch.Generator().manual_seed(0)
        x, y, m = prefix_batch(BATCH, L, L, gen, DEV)
        base = torch.cuda.memory_allocated() / 1e9 if CU else 0
        CU and torch.cuda.synchronize(); t0 = time.time()
        r = {"mode": mode, "D": D, "L": L, "batch": BATCH, "base_gb": base}
        try:
            if fn == "stream":
                tb1_stream(M, x, y, m, D, gen)
            else:
                getattr(M, fn)(x, y, m, D, gen)
            CU and torch.cuda.synchronize()
            r.update(peak_gb=torch.cuda.max_memory_allocated() / 1e9 if CU else None, sec=time.time() - t0, ok=True)
        except torch.OutOfMemoryError:
            r.update(ok=False, err="OOM", peak_gb=torch.cuda.max_memory_allocated() / 1e9, sec=time.time() - t0)
        print(json.dumps(r), flush=True)
        out.append(r)
        del M
        json.dump(out, open(OUT + ".tmp", "w")); os.replace(OUT + ".tmp", OUT)
        if not r["ok"]:
            break  # a larger D will not fit either
