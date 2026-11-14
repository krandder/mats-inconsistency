"""Priced (VOC) halting against an entropy-threshold baseline, for the audit of section 10.

Section 10 tested halting on MIXED lengths 4 to 64 (eval_halting: prefix_batch(n, 4, 64)), not at
L64 as it said, and compared VOC halting with random halting and fixed depth only. Here, for one
trained economy checkpoint, on two test sets of n = 500 strings each (round-robin order, step cap 96):
  L64    every string of length 64 (seed 1064)
  mix    lengths 4 to 64, section 10's set (seed 909 + 64)
each policy gives every example a stop step tau (it keeps the belief after tau steps):
  voc c      first step t >= 1 with VOC_t <= c (the economy's rule; VOC = sigmoid(v_0) x belief entropy / 5)
  fixed      every example at the voc row's mean steps (linear interpolation between the two integer depths)
  entropy    first t >= 1 with the mean belief entropy per bit <= h, h set by bisection so that the mean
             steps match the voc row (no market: the belief's own uncertainty)
  random     the voc row's stop steps permuted over examples (same histogram, E_perm)
Writes --out (json) and prints one line per (set, cost).
"""
import argparse, json, os
import torch
import torch.nn.functional as F
from mresnet import prefix_batch
from reeval import build

COSTS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]


@torch.no_grad()
def trajectories(M, x, y, m, D, chunk):
    EX, SG, H = [], [], []
    for c0 in range(0, x.shape[0], chunk):
        xs, ys, ms = x[c0:c0 + chunk], y[c0:c0 + chunk], m[c0:c0 + chunk]
        _, bs, _ = M.rollout(xs, ms, D, pick="roundrobin", keep=False, signals=True)
        EX.append(torch.stack([(((b > 0).float() == ys) | (ms == 0)).all(1) for b in bs], 1).float().cpu())
        SG.append(torch.stack(M.sig, 1)[:, :, 1].cpu())                                  # VOC at steps 0..D-1
        ent = []
        for b in bs[:-1]:
            p = torch.sigmoid(b)
            h = -(p * F.logsigmoid(b) + (1 - p) * F.logsigmoid(-b))
            ent.append((h * ms).sum(1) / ms.sum(1))
        H.append(torch.stack(ent, 1).cpu())                                             # entropy at steps 0..D-1
        del bs
    return torch.cat(EX), torch.cat(SG), torch.cat(H)


def stop_at(sig, thr):
    stop = sig <= thr
    stop[:, 0] = False
    stop = torch.cat([stop, torch.ones(sig.shape[0], 1, dtype=torch.bool)], 1)
    return stop.float().argmax(1)


def policies(EX, SG, H, costs, seed=1):
    n = EX.shape[0]; ar = torch.arange(n); rng = torch.Generator().manual_seed(seed)
    fixed_curve = EX.mean(0)                                                           # acc at every depth
    out = []
    for c in costs:
        tau = stop_at(SG, c); S = tau.float().mean().item()
        lo, fr = int(S), S - int(S)
        fixed = (1 - fr) * fixed_curve[lo].item() + fr * fixed_curve[min(lo + 1, EX.shape[1] - 1)].item()
        a_, b_ = 0.0, float(H.max()) + 1.0                                               # entropy threshold by bisection
        for _ in range(60):
            h = (a_ + b_) / 2
            if stop_at(H, h).float().mean().item() > S:
                a_ = h
            else:
                b_ = h
        te_hi, te_lo = stop_at(H, b_), stop_at(H, a_)                                   # steps <= S and > S
        s_hi, s_lo = te_hi.float().mean().item(), te_lo.float().mean().item()
        acc_hi, acc_lo = EX[ar, te_hi].mean().item(), EX[ar, te_lo].mean().item()
        w = 0.0 if s_lo == s_hi else (S - s_hi) / (s_lo - s_hi)                           # interpolate to S steps
        perm = tau[torch.randperm(n, generator=rng)]
        out.append({"cost": c, "steps": S, "voc_acc": EX[ar, tau].mean().item(), "fixed_acc": fixed,
                    "entropy_acc": (1 - w) * acc_hi + w * acc_lo, "entropy_h": b_, "entropy_steps": [s_hi, s_lo],
                    "random_acc": EX[ar, perm].mean().item(), "cap_acc": fixed_curve[-1].item()})
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--D", type=int, default=96)
    p.add_argument("--chunk", type=int, default=100)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    M, kind, _ = build(a.ckpt, a.device)
    assert kind == "econ" and M.a.voc, "needs an economy with VOC markets"
    res = {"ckpt": a.ckpt, "n": a.n, "D": a.D, "sets": {}}
    for name, (lmin, lmax, seed) in {"L64": (64, 64, 1064), "mix": (4, 64, 909 + 64)}.items():
        gen = torch.Generator().manual_seed(seed)
        x, y, m = prefix_batch(a.n, lmin, lmax, gen, a.device)
        EX, SG, H = trajectories(M, x, y, m, a.D, a.chunk)
        rows = policies(EX, SG, H, COSTS)
        res["sets"][name] = {"rows": rows, "acc_by_depth": EX.mean(0).tolist()}
        for r in rows:
            print(json.dumps({"set": name, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}}), flush=True)
    json.dump(res, open(a.out + ".tmp", "w")); os.replace(a.out + ".tmp", a.out)
