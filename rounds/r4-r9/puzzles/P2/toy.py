"""Stage 0 CPU toy: 1D broadcast parity, y_i = x_i XOR parity(x).

Before the trunk has carried the parity across the string, the right joint belief is a 50/50
mixture of two strings. A factorised U market can only say 0.5 per bit (exact match ~ 2^-n);
M-global / M-tree (K=2) can hold the two modes and decode one of them (exact match ~ 50%);
M-tile (independent 8-position tiles) gets ~ 2^-(n/8).

  toy.py run   [--steps 3000] [--seeds 0 1 2] [--procs 6]
  toy.py report
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

import jm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "toy")
ARMS = ("U", "M-global", "M-tile", "M-tree")
TEST_N = (32, 64, 128)
T_TEST = 240


def batch(B, n, gen):
    x = (torch.rand(B, n, generator=gen) < 0.5).float()
    par = x.sum(1, keepdim=True) % 2
    y = (x + par) % 2
    return torch.stack([x, 1 - x], 1), y


def make_model(arm, seed, w=32):
    """Paired init: the trunk layers shared by every arm get identical weights for a given seed."""
    torch.manual_seed(1000 + seed)
    ref = jm.Trunk(1, 2, w, 1, 2)
    torch.manual_seed(2000 + seed)
    model = jm.Model(arm, dim=1, cin=2, w=w, K=2, leaf=8, fdim=16)
    jm.pair_trunk(model, ref)
    return model


def score(model, st, y, mask, g):
    return jm.score_unary(st["m"], y, mask) if not model.is_m else jm.score_m(st, y, mask, g)


def train(arm, seed, steps, log):
    torch.set_num_threads(1)
    model = make_model(arm, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps)
    gen = torch.Generator().manual_seed(seed)  # paired data stream
    t0 = time.time()
    hist = []
    for step in range(steps):
        n = int(torch.randint(8, 33, (1,), generator=gen))
        npre = int(torch.randint(0, 25, (1,), generator=gen))
        x, y = batch(64, n, gen)
        mask = torch.ones_like(y)
        g = model.geom((n,))
        with torch.no_grad():
            st = model.run(x, npre)
            s0 = score(model, st, y, mask, g)
        st = jm.detach_state(st)
        st = model.run(x, 6, st)
        s1 = score(model, st, y, mask, g)
        loss = -((s1 - s0) / n).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()
        if step % 100 == 0 or step == steps - 1:
            hist.append(dict(step=step, loss=float(loss), end_score_per_bit=float((s1 / n).mean()),
                             sec=round(time.time() - t0, 1)))
            print(arm, seed, hist[-1], file=log, flush=True)
    return model, hist


@torch.no_grad()
def evaluate(model, seed):
    out = {}
    for n in TEST_N:
        gen = torch.Generator().manual_seed(777 + n)  # same test strings for every arm and seed
        x, y = batch(500, n, gen)
        mask = torch.ones_like(y)
        g = model.geom((n,))
        em, ll = [], []
        st = model.init_state(x)
        for t in range(1, T_TEST + 1):
            st = model.step(st, x)
            if model.is_m:
                dec = jm.tree_map(st, mask, g)[0]
            else:
                dec = jm.decode_thr(st["m"])
            em.append(float((dec == (y > 0)).all(1).float().mean()))
            ll.append(float(-(score(model, st, y, mask, g) / n).mean()))
        out[n] = dict(exact=em, logloss_per_bit=ll)
    return out


def one(args):
    arm, seed, steps = args
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{arm}_s{seed}.json")
    if os.path.exists(path):
        return path
    with open(os.path.join(OUT, f"{arm}_s{seed}.log"), "w") as log:
        t0 = time.time()
        model, hist = train(arm, seed, steps, log)
        ev = evaluate(model, seed)
        nparam = sum(p.numel() for p in model.parameters())
        json.dump(dict(arm=arm, seed=seed, steps=steps, params=nparam, train=hist, eval=ev,
                       wall_sec=round(time.time() - t0, 1)), open(path, "w"))
        torch.save(model.state_dict(), os.path.join(OUT, f"{arm}_s{seed}.pt"))
    return path


def report():
    runs = {}
    for f in sorted(os.listdir(OUT)):
        if f.endswith(".json") and f != "summary.json":
            r = json.load(open(os.path.join(OUT, f)))
            runs.setdefault(r["arm"], []).append(r)
    summ = {"arms": {}, "criteria": {}}
    for arm, rs in runs.items():
        summ["arms"][arm] = {str(n): {"exact_mean": np.mean([r["eval"][str(n)]["exact"] for r in rs], 0).tolist(),
                                      "logloss_mean": np.mean([r["eval"][str(n)]["logloss_per_bit"] for r in rs], 0).tolist()}
                             for n in TEST_N}
        summ["arms"][arm]["params"] = rs[0]["params"]
        summ["arms"][arm]["seeds"] = [r["seed"] for r in rs]
        summ["arms"][arm]["wall_sec"] = [r["wall_sec"] for r in rs]
    # pre-registered readout: window W_n = iterations where U's seed-mean exact match < 1%;
    # an M arm's value = its seed-mean exact match averaged over W_n (and its minimum over the
    # second half of W_n, after the mixture has had time to form).
    crit = {}
    if "U" in summ["arms"]:
        for n in TEST_N:
            u = np.array(summ["arms"]["U"][str(n)]["exact_mean"])
            W = np.flatnonzero(u < 0.01)
            row = {"window_iters": [int(W.min()) + 1, int(W.max()) + 1] if len(W) else None,
                   "window_len": int(len(W)), "U_mean_in_window": float(u[W].mean()) if len(W) else None,
                   "U_resolves_at": int(np.argmax(u >= 0.01)) + 1 if (u >= 0.01).any() else None}
            for arm in ARMS[1:]:
                if arm in summ["arms"] and len(W):
                    e = np.array(summ["arms"][arm][str(n)]["exact_mean"])
                    per_seed = [float(np.mean(np.array(r["eval"][str(n)]["exact"])[W])) for r in runs[arm]]
                    row[arm] = {"mean_in_window": float(e[W].mean()),
                                "min_second_half": float(e[W[len(W) // 2:]].min()),
                                "per_seed_mean_in_window": per_seed,
                                "final_T240": float(e[-1])}
            crit[n] = row
        ok_tree = all(crit[n].get("M-tree", {}).get("mean_in_window", 0) >= 0.40 for n in (64, 128))
        ok_glob = all(crit[n].get("M-global", {}).get("mean_in_window", 0) >= 0.40 for n in (64, 128))
        ok_tile = all(crit[n].get("M-tile", {}).get("mean_in_window", 1) <= 0.05 for n in (64, 128))
        summ["criteria"] = {"per_n": crit, "M-tree_ge_40pct_n64_128": ok_tree, "M-global_ge_40pct_n64_128": ok_glob,
                            "M-tile_le_5pct_n64_128": ok_tile, "stage0_pass_gate_M-tree": ok_tree}
    json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"))
    print(json.dumps(summ["criteria"], indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    if a.cmd == "run":
        from multiprocessing import Pool
        jobs = [(arm, s, a.steps) for s in a.seeds for arm in a.arms]
        with Pool(a.procs) as p:
            for r in p.imap_unordered(one, jobs):
                print("done", r, flush=True)
        report()
    elif a.cmd == "report":
        report()
