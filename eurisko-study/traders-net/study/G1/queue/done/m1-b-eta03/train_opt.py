"""M1(b): wealth-net training with a swappable optimiser; books vs tangent vs norm growth.

Per unit u (readout q, neuron (l,i), head (l,h)) accumulates every step:
  W  = wealth-net book income (books.account, pre-update weights, same batch)
  T  = -Q theta_u . g_u      (g = true loss gradient, pre-update)  -- signed movement x Q
  RS = sum |theta_u|^2       (rent integrand; rent = Q*lam*RS)
  G2 = sum |g_u + lam theta_u|^2
and snapshots |theta_u|^2 so N = Q(|theta_t|^2-|theta_0|^2)/(2 eta).
Identity under SGD(+coupled wd): T = N + Q lam RS - Q eta/2 G2.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from streams import make_questions, N_FAM, FAMILIES
from model import Net
from books import account


def unit_params(model):
    """Return dict gran -> list of (tensor-view-fn) giving per-unit parameter matrices [U, k]."""
    H, dh = model.H, model.d // model.H

    def out(p):   # p: dict name->tensor (params or grads)
        return torch.cat([p["rw"], p["rb"].unsqueeze(1)], 1)                 # [Q, d+1]

    def neuron_out(p):
        return torch.cat([p[f"blocks.{l}.fc2.weight"].T for l in range(model.L)], 0)   # [L*N, d]

    def neuron_in(p):
        return torch.cat([torch.cat([p[f"blocks.{l}.fc1.weight"], p[f"blocks.{l}.fc1.bias"].unsqueeze(1)], 1)
                          for l in range(model.L)], 0)                         # [L*N, d+1]

    def head(p):
        return torch.cat([p[f"blocks.{l}.o.weight"].view(model.d, H, dh).permute(1, 0, 2).reshape(H, -1)
                          for l in range(model.L)], 0)                         # [L*H, d*dh]
    return {"out": out, "neuron": neuron_out, "neuron_in": neuron_in, "head": head}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt", required=True, choices=["adam", "sgd", "sgdwd"])
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--Q", type=int, default=64); ap.add_argument("--W", type=int, default=32)
    ap.add_argument("--d", type=int, default=64); ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--H", type=int, default=4); ap.add_argument("--B", type=int, default=128)
    ap.add_argument("--T", type=int, default=24000)
    ap.add_argument("--snap", type=int, default=500)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-min", type=float, default=1e9)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(a.device)
    tag = f"{a.opt}-s{a.seed}"
    os.makedirs(a.out, exist_ok=True)

    bits, fam, desc = make_questions(a.Q, a.T, a.seed)
    bits_t = torch.tensor(bits, dtype=torch.float32, device=dev)
    wins = bits_t.unfold(1, a.W, 1)
    n_pos = a.T - a.W
    split = int(n_pos * 0.8)
    fam_oh = F.one_hot(torch.tensor(fam, device=dev), N_FAM).float()
    g = torch.Generator(device="cpu").manual_seed(a.seed)   # same batch order as wealth-net train mode

    def batch(idx):
        idx = idx.to(dev)
        return wins[:, idx].permute(1, 0, 2).contiguous(), bits_t[:, idx + a.W].T.contiguous()

    ev_win, ev_y = batch(torch.arange(split, n_pos, max(1, (n_pos - split) // 2048)))
    model = Net(a.Q, a.W, a.d, a.H, a.L).to(dev)
    if a.opt == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
        lam = 0.0
    else:
        lam = a.wd if a.opt == "sgdwd" else 0.0
        opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.0, weight_decay=lam)
    eta = a.lr
    UP = unit_params(model)
    L, H, N = a.L, a.H, 4 * a.d
    names = dict(model.named_parameters())

    def per_unit(fn, src):
        return fn(src)

    grans = ["out", "neuron", "neuron_in", "head"]
    acc = {k: {q: None for q in ["W", "T", "RS", "G2"]} for k in grans}
    with torch.no_grad():
        n0 = {k: (UP[k](names).double() ** 2).sum(1).clone() for k in grans}
    rows = []
    ev_hist = []

    def bookW(bk, k):
        if k == "out":
            return bk["out"].sum(-1)
        if k in ("neuron", "neuron_in"):
            return bk["neuron"].sum(-1).reshape(-1)
        return bk["head"].sum(-1).reshape(-1)

    def snapshot(step):
        with torch.no_grad():
            z = model(ev_win)
            bce = F.binary_cross_entropy_with_logits(z, ev_y, reduction="none").mean(0)
            fb = (fam_oh.T @ bce / fam_oh.sum(0)).cpu().numpy()
            ev_hist.append({"step": step, **{f: float(v) for f, v in zip(FAMILIES, fb)}, "mean": float(bce.mean())})
            for k in grans:
                nt = (UP[k](names).double() ** 2).sum(1)
                Nk = a.Q * (nt - n0[k]) / (2 * eta)
                for u in range(nt.shape[0]):
                    rows.append((step, k, u, float(acc[k]["W"][u]), float(acc[k]["T"][u]), float(Nk[u]),
                                 float(a.Q * lam * acc[k]["RS"][u]), float(a.Q * eta / 2 * acc[k]["G2"][u]),
                                 float(nt[u])))

    t0 = time.time()
    for step in range(1, a.steps + 1):
        idx = torch.randint(0, split, (a.B,), generator=g)
        win, y = batch(idx)
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(win), y)
        loss.backward()
        bk = account(model, win, y, fam_oh)
        with torch.no_grad():
            grads = {n: p.grad for n, p in names.items()}
            for k in grans:
                th = UP[k](names); gr = UP[k](grads)
                vals = {"W": bookW(bk, k), "T": -a.Q * (th * gr).sum(1), "RS": (th ** 2).sum(1),
                        "G2": ((gr + lam * th) ** 2).sum(1)}
                for q, v in vals.items():
                    v = v.double()
                    acc[k][q] = v.clone() if acc[k][q] is None else acc[k][q] + v
        opt.step()
        if not torch.isfinite(loss):
            print(f"[{tag}] NaN at step {step}", flush=True); snapshot(step); break
        if step % a.snap == 0 or step == a.steps:
            snapshot(step)
            print(f"[{tag}] step {step} loss {loss.item():.4f} ev {ev_hist[-1]['mean']:.4f} "
                  f"{(time.time()-t0)/step*1000:.0f} ms/step", flush=True)
            if (time.time() - t0) / 60 > a.max_min:
                print(f"[{tag}] wall budget reached at step {step}", flush=True); break
    import csv
    tmp = os.path.join(a.out, f"{tag}.units.csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step", "gran", "unit", "W", "T", "N", "rent", "second", "sqnorm"])
        w.writerows(rows)
    os.replace(tmp, os.path.join(a.out, f"{tag}.units.csv"))
    json.dump({"args": vars(a), "eta": eta, "lam": lam, "seconds": time.time() - t0, "eval": ev_hist,
               "families": FAMILIES, "desc": desc}, open(os.path.join(a.out, f"{tag}.json"), "w"), indent=1)
    print(f"[{tag}] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
