"""M1 item (a): regress wealth-net wealth on norm growth and movement (read-only on wealth-net).

Usage: PYTHONPATH=<wealth-net>/.pylib python3 norm_growth.py --seed 1 --out results
Writes results/units-s{seed}.csv (long: interval, gran, unit, Y, predictors) and
results/inst-s{seed}.csv (instantaneous identity check per checkpoint).
"""
import argparse, glob, json, os, re, sys
import numpy as np
import torch
import torch.nn.functional as F

WN = "/home/agent/workspaces/eurisko-study/wealth-net"
sys.path.insert(0, WN)
from streams import make_questions, N_FAM  # noqa: E402
from model import Net  # noqa: E402
from books import account  # noqa: E402

torch.set_num_threads(1)


def unit_params(m):
    """Per-unit parameter vectors: {gran: {'in': [n, k], 'out': [n, k]}} (views of params, flattened)."""
    Q, d, H = m.Q, m.d, m.H
    dh = d // H
    P = {"neuron": {"in": [], "out": []}, "head": {"in": [], "out": []}, "out": {"in": [], "out": []}}
    for blk in m.blocks:
        P["neuron"]["in"].append(torch.cat([blk.fc1.weight, blk.fc1.bias[:, None]], 1))      # [N, d+1]
        P["neuron"]["out"].append(blk.fc2.weight.T)                                           # [N, d]
        wv = blk.qkv.weight.view(3, H, dh, d)[2].reshape(H, dh * d)
        bv = blk.qkv.bias.view(3, H, dh)[2]
        P["head"]["in"].append(torch.cat([wv, bv], 1))                                        # [H, dh*d+dh]
        P["head"]["out"].append(blk.o.weight.view(d, H, dh).permute(1, 0, 2).reshape(H, d * dh))
    ro = torch.cat([m.rw, m.rb[:, None]], 1)                                                   # [Q, d+1]
    P["out"]["in"] = [ro]; P["out"]["out"] = [ro]
    return {g: {k: torch.cat(v, 0) for k, v in D.items()} for g, D in P.items()}


def flat_np(m):
    with torch.no_grad():
        return {g: {k: v.detach().double().numpy().copy() for k, v in D.items()} for g, D in unit_params(m).items()}


def theta_dot_negrad(m, win, y, freeze_ln, keep_grad=False):
    """-Q * theta . grad L_mean per unit (= income under the books identity), for in and out params."""
    m.zero_grad(set_to_none=True)
    z = m(win, freeze_ln=freeze_ln)
    loss = F.binary_cross_entropy_with_logits(z, y)          # mean over B*Q, as in training
    loss.backward()
    Q = m.Q
    out = {}
    L, H, d = m.L, m.H, m.d
    dh = d // H
    for g in ["neuron", "head", "out"]:
        out[g] = {}
    nin, nout, hin, hout = [], [], [], []
    for blk in m.blocks:
        nin.append(-(blk.fc1.weight * blk.fc1.weight.grad).sum(1) - blk.fc1.bias * blk.fc1.bias.grad)
        nout.append(-(blk.fc2.weight * blk.fc2.weight.grad).sum(0))
        wv = (blk.qkv.weight * blk.qkv.weight.grad).view(3, H, dh, d)[2].sum((1, 2))
        bv = (blk.qkv.bias * blk.qkv.bias.grad).view(3, H, dh)[2].sum(1)
        hin.append(-(wv + bv))
        hout.append(-(blk.o.weight * blk.o.weight.grad).view(d, H, dh).sum((0, 2)))
    ro = -((m.rw * m.rw.grad).sum(1) + m.rb * m.rb.grad)
    out["neuron"] = {"in": Q * torch.cat(nin), "out": Q * torch.cat(nout)}
    out["head"] = {"in": Q * torch.cat(hin), "out": Q * torch.cat(hout)}
    out["out"] = {"in": Q * ro, "out": Q * ro}
    if not keep_grad:
        m.zero_grad(set_to_none=True)
    return {g: {k: v.detach().double().numpy() for k, v in D.items()} for g, D in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--nb", type=int, default=8)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(f"{WN}/results/train-s{a.seed}.json"))
    Q, W, T, B, eta = meta["Q"], meta["W"], meta["T"], meta["B"], meta["lr"]
    final = meta["steps_done"]
    # reconstructed init: same RNG order as train.py (manual_seed, numpy streams, then Net)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    bits, fam, _ = make_questions(Q, T, a.seed)
    m = Net(Q, W, meta["d"], meta["H"], meta["L"])
    init_sd = {k: v.clone() for k, v in m.state_dict().items()}
    bits_t = torch.tensor(bits, dtype=torch.float32)
    wins = bits_t.unfold(1, W, 1)
    n_pos = T - W
    split = int(n_pos * 0.8)
    fam_oh = F.one_hot(torch.tensor(fam), N_FAM).float()
    gen = torch.Generator().manual_seed(12345 + a.seed)
    batches = []
    for _ in range(a.nb):
        idx = torch.randint(0, split, (B,), generator=gen)
        batches.append((wins[:, idx].permute(1, 0, 2).contiguous(), bits_t[:, idx + W].T.contiguous()))

    steps = [0] + list(range(8000, 30001, 2000)) + [final]
    ckpt = {0: init_sd}
    for s in steps[1:]:
        p = f"{WN}/results/weights/train-s{a.seed}-step{s}.pt" if s != final else f"{WN}/results/train-s{a.seed}.pt"
        ckpt[s] = torch.load(p, map_location="cpu")
    theta, grad_live, grad_books, inst = {}, {}, {}, []
    for s in steps:
        m.load_state_dict(ckpt[s])
        theta[s] = flat_np(m)
        acc_live = {g: {k: 0 for k in ["in", "out"]} for g in ["neuron", "head", "out"]}
        acc_books = {g: {k: 0 for k in ["in", "out"]} for g in ["neuron", "head", "out"]}
        for bi, (win, y) in enumerate(batches):
            gl = theta_dot_negrad(m, win, y, False)
            gb = theta_dot_negrad(m, win, y, True)
            for g in gl:
                for k in gl[g]:
                    acc_live[g][k] = acc_live[g][k] + gl[g][k] / a.nb
                    acc_books[g][k] = acc_books[g][k] + gb[g][k] / a.nb
            if bi < 2:  # instantaneous identity check vs the books themselves
                bk = account(m, win, y, fam_oh)
                bn = bk["neuron"].sum(-1).reshape(-1).double().numpy()
                bh = bk["head"].sum(-1).reshape(-1).double().numpy()
                bo = bk["out"].sum(-1).double().numpy()
                for g, bv in (("neuron", bn), ("head", bh), ("out", bo)):
                    for k in ["in", "out"]:
                        for kind, gg in (("books_ln", gb), ("live_ln", gl)):
                            diff = gg[g][k] - bv
                            inst.append(dict(seed=a.seed, step=s, batch=bi, gran=g, side=k, grad=kind,
                                             rel_err=float(np.linalg.norm(diff) / max(1e-12, np.linalg.norm(bv))),
                                             corr=float(np.corrcoef(gg[g][k], bv)[0, 1]) if bv.std() > 0 else np.nan))
        grad_live[s], grad_books[s] = acc_live, acc_books
        print(f"seed {a.seed} step {s} done", flush=True)

    def wealth(s):
        if s == 0:
            return {"neuron": np.zeros(512), "head": np.zeros(8), "out": np.zeros(Q)}
        js = json.load(open(f"{WN}/results/train-s{a.seed}/snapshots/step-{s:05d}.json"))
        return {g: np.array(js["wealth"][g]) for g in ["neuron", "head", "out"]}

    Wt = {s: wealth(s) for s in steps}
    rows = []
    intervals = list(zip(steps[:-1], steps[1:])) + [(8000, final), (0, final)]
    for t0, t1 in intervals:
        # checkpoints inside [t0, t1] for the discrete movement sum
        inside = [s for s in steps if t0 <= s <= t1]
        for g in ["neuron", "head", "out"]:
            Y = Wt[t1][g] - Wt[t0][g]
            rec = {"Y": Y}
            for side in ["in", "out"]:
                n0 = (theta[t0][g][side] ** 2).sum(1); n1 = (theta[t1][g][side] ** 2).sum(1)
                rec[f"X_norm_{side}"] = Q * (n1 - n0) / (2 * eta)
                mv = 0
                for u, v in zip(inside[:-1], inside[1:]):
                    mv = mv + (theta[u][g][side] * (theta[v][g][side] - theta[u][g][side])).sum(1)
                rec[f"X_move_{side}"] = Q * mv / eta
                gl = 0; gb = 0
                for u, v in zip(inside[:-1], inside[1:]):
                    gl = gl + (v - u) * (grad_live[u][g][side] + grad_live[v][g][side]) / 2
                    gb = gb + (v - u) * (grad_books[u][g][side] + grad_books[v][g][side]) / 2
                rec[f"X_grad_live_{side}"] = gl
                rec[f"X_grad_books_{side}"] = gb
            # endpoint magnitude*displacement (Sanh-style W_T.(W_T - W_0)) on out-weights
            rec["X_wdisp_out"] = Q * (theta[t1][g]["out"] * (theta[t1][g]["out"] - theta[t0][g]["out"])).sum(1) / eta
            rec["X_normT_out"] = np.sqrt((theta[t1][g]["out"] ** 2).sum(1))
            for i in range(len(Y)):
                rows.append(dict(seed=a.seed, t0=t0, t1=t1, gran=g, unit=i, **{k: float(v[i]) for k, v in rec.items()}))
    import csv
    with open(os.path.join(a.out, f"units-s{a.seed}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(a.out, f"inst-s{a.seed}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(inst[0].keys())); w.writeheader(); w.writerows(inst)
    print("wrote", len(rows), "rows")


if __name__ == "__main__":
    main()
