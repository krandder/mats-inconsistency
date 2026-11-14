"""M1 (a) supplement: replay wealth-net's Adam training from the reconstructed init on CPU and keep,
per step and per unit, (1) books income (-Q theta.grad L with LN denominators frozen, = account()),
(2) the gradient-form movement score with the optimiser's gradient (live LN; Sanh et al. 2020),
(3) exact norm growth Q*Delta||theta||^2/(2 eta), and (4) Adam's actual step theta.Delta theta.
Cumulative arrays saved every 100 steps. Same data order, model, optimiser and batch as train.py.
Usage: python3 replay.py --seed 1 --steps 4000 --opt adam --out results/replay
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from norm_growth import WN, unit_params, theta_dot_negrad  # noqa: E402
from streams import make_questions  # noqa: E402
from model import Net  # noqa: E402

torch.set_num_threads(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--opt", default="adam", choices=["adam", "sgd"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="results/replay")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    Q, W, T, B = 64, 32, 24000, 128
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    bits, fam, _ = make_questions(Q, T, a.seed)
    m = Net(Q, W, 64, 4, 2)
    bits_t = torch.tensor(bits, dtype=torch.float32)
    wins = bits_t.unfold(1, W, 1)
    split = int((T - W) * 0.8)
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    opt = torch.optim.Adam(m.parameters(), lr=a.lr) if a.opt == "adam" else torch.optim.SGD(m.parameters(), lr=a.lr)
    grans, sides = ["neuron", "head", "out"], ["in", "out"]

    def th():
        with torch.no_grad():
            return {gg: {s: v.detach().double().clone() for s, v in D.items()} for gg, D in unit_params(m).items()}
    acc = {k: {gg: {s: 0.0 for s in sides} for gg in grans} for k in ["books", "live", "adamstep"]}
    th0 = th()
    snaps = []
    t0 = time.time()
    for step in range(1, a.steps + 1):
        idx = torch.randint(0, split, (B,), generator=g)
        win = wins[:, idx].permute(1, 0, 2).contiguous(); y = bits_t[:, idx + W].T.contiguous()
        gb = theta_dot_negrad(m, win, y, True)      # books (frozen LN)
        before = th()
        gl = theta_dot_negrad(m, win, y, False, keep_grad=True)   # optimiser's gradient (live LN)
        opt.step()
        after = th()
        for gg in grans:
            for s in sides:
                acc["books"][gg][s] = acc["books"][gg][s] + gb[gg][s]
                acc["live"][gg][s] = acc["live"][gg][s] + gl[gg][s]
                acc["adamstep"][gg][s] = acc["adamstep"][gg][s] + Q * (before[gg][s] * (after[gg][s] - before[gg][s])).sum(1).numpy() / a.lr
        if step % 100 == 0:
            now = th()
            snap = {"step": step}
            for gg in grans:
                for s in sides:
                    snap[f"{gg}.{s}.books"] = np.asarray(acc["books"][gg][s]).tolist()
                    snap[f"{gg}.{s}.live"] = np.asarray(acc["live"][gg][s]).tolist()
                    snap[f"{gg}.{s}.adamstep"] = np.asarray(acc["adamstep"][gg][s]).tolist()
                    snap[f"{gg}.{s}.norm"] = (Q * ((now[gg][s] ** 2).sum(1) - (th0[gg][s] ** 2).sum(1)) / (2 * a.lr)).numpy().tolist()
            snaps.append(snap)
            if step % 500 == 0:
                print(f"seed {a.seed} step {step} {(time.time()-t0)/step:.3f}s/step", flush=True)
                json.dump(snaps, open(os.path.join(a.out, f"replay-{a.opt}-s{a.seed}.json"), "w"))
    json.dump(snaps, open(os.path.join(a.out, f"replay-{a.opt}-s{a.seed}.json"), "w"))
    torch.save(m.state_dict(), f"/tmp/traders-net/M1/replay-{a.opt}-s{a.seed}.pt") if os.path.isdir("/tmp/traders-net/M1") else None


if __name__ == "__main__":
    main()
