"""Training FLOPs per iteration (matmul/conv FLOPs, forward + backward, torch FlopCounterMode),
averaged over --reps random batches, for the phase-3 arms. Used to set the equal-FLOP DT-recall arm."""
import argparse, json, sys, types
import torch
from torch.utils.flop_counter import FlopCounterMode
from mresnet import prefix_batch
import econ2
from dtrecall import DT

p = argparse.ArgumentParser(); p.add_argument("--reps", type=int, default=20); p.add_argument("--device", default="cuda")
p.add_argument("--out", default="p3/flops.json")
A = p.parse_args()
sys.argv = ["econ2.py", "--out", "x"]


def econ_args(**kw):
    base = dict(mode="econ", stream="cov", voc=1, N=4, width=96, D=24, lmin=8, lmax=32, iters=3000, bs=64, lr=1e-3, cost=0.0,
                W0=20.0, frac=0.5, wealth_cap=1, lam_c=1.0, lam_v=0.3, rent=1e-3, self_pay=0, grow_every=0, train_halt=-1,
                settle="eps", eps=0.1, bounty="", fund_lr=0.05, online=0, prog=0, ds=0, covsup=0, tbptt=0, train_pick="", clamp=0)
    base.update(kw); return types.SimpleNamespace(**base)


def count(make, step, reps, bs, seed=0):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed); M = make(); tot = 0
    for r in range(reps):
        x, y, m = prefix_batch(bs, 8, 32, gen, A.device)
        with FlopCounterMode(display=False) as fc:
            step(M, x, y, m, gen)
        tot += fc.get_total_flops()
    return tot / reps, M


res = {}
arms = {"econ_cov_voc": econ_args(), "econ_cov": econ_args(voc=0),
        "bptt_cov_dscov": econ_args(mode="bptt", voc=0, ds=1, covsup=1),
        "bptt_cov_dscov_tb1": econ_args(mode="bptt", voc=0, ds=1, covsup=1, tbptt=1),
        "bptt_cov_progcov": econ_args(mode="bptt", voc=0, prog=1, covsup=1)}
for n, a in arms.items():
    mk = (lambda a=a: econ2.Economy(a, A.device)) if a.mode == "econ" else (lambda a=a: econ2.BPTT(a, A.device))
    f, M = count(mk, lambda M, x, y, m, g, a=a: M.train_step(x, y, m, a.D, g), A.reps, 64)
    params = sum(q.numel() for t in M.traders for q in (t[0].parameters() if a.mode == "econ" else t.parameters()))
    res[n] = {"flops_per_iter": f, "iters": 3000, "total": f * 3000, "params": params}
    print(n, json.dumps(res[n]), flush=True)
for w in [400, 96]:
    a = types.SimpleNamespace(width=w, lr=1e-3, max_iters=30)
    f, M = count(lambda: DT(a, A.device), lambda M, x, y, m, g: M.train_step(x, y, m, 0.5, g), 4 * A.reps, 100)
    res[f"dtr_w{w}"] = {"flops_per_iter": f, "params": sum(q.numel() for q in M.net.parameters())}
    print(w, json.dumps(res[f"dtr_w{w}"]), flush=True)
json.dump(res, open(A.out, "w"), indent=1)
