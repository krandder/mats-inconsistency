"""Train one P2 arm on the n=9 pool with the progressive (market) loss.

  train.py --arm M-tree --seed 0 [--steps 5000] [--alpha 1.0] [--device cuda] [--out results/runs]

Loss per step: run a no-grad prefix of n ~ U{0..24} iterations from scratch, detach, run k=6
iterations with gradients; loss = -[S(end) - S(start)] / n_open with S(start) detached (the
market payment for the segment). alpha < 1 mixes in the plain T_train rollout loss
-(1-alpha) S(T_train) (DT-Recall's alpha), used only if the validity gate fails.

Sampled labels come from a label bank drawn once at the start with the seed's label generator:
BANK_K independent uniform shortest paths per pool maze; the j-th time a maze is drawn it gets
bank entry j mod BANK_K (a maze is drawn ~1.6 times in 5000 x 64 draws from 200k, so repeats are
rare). Same bank for every arm of a seed, so seeds stay paired. This replaces a per-step sampler
call that took ~40% of the step time (simplification, see RESULTS.md).

Labels / scores per arm (paired seeds: same batches, same sampled paths, same trunk init):
  U        factorised score on a uniformly sampled shortest path
  U-canon  factorised score on the canonical path (move order D,R,U,L)
  U-min    hard-EM: Viterbi path on the detached end marginals, same label for S(start), S(end);
           the first 500 steps use the sampled labels (warm start)
  U-any    exact log P(a factorised sample is some shortest path)
  M-tree / M-tile   exact mixture score on the same sampled path as U
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import jm

HERE = os.path.dirname(os.path.abspath(__file__))


def unpack_t(bits, L):
    return torch.from_numpy(np.unpackbits(bits, axis=-1, count=L * L).reshape(-1, L, L).astype(bool))


class Pool:
    def __init__(self, path, device):
        z = np.load(path)
        n = int(z["n"])
        L = 2 * n + 1
        self.L = L
        self.open = unpack_t(z["open"], L).to(device)
        self.canon = unpack_t(z["canon"], L).to(device)
        ds = torch.from_numpy(z["ds"].reshape(-1, L, L).astype(np.int16)).to(device)
        dg = torch.from_numpy(z["dg"].reshape(-1, L, L).astype(np.int16)).to(device)
        self.ds = torch.where(self.open, ds, torch.full_like(ds, -1))  # int16 on device
        self.dg = torch.where(self.open, dg, torch.full_like(dg, -1))
        self.s = torch.from_numpy(z["s"].astype(np.int64)).to(device)
        self.g = torch.from_numpy(z["g"].astype(np.int64)).to(device)
        self.D = self.ds.flatten(1).gather(1, self.g[:, None]).squeeze(1).long()
        self.dag = self.open & (self.ds + self.dg == self.D[:, None, None])
        self.N = len(self.s)

    def batch(self, idx):
        op = self.open[idx]
        B, L = len(idx), self.L
        x = torch.zeros(B, 3, L, L, device=op.device)
        x[:, 0] = op.float()
        ar = torch.arange(B, device=op.device)
        x.view(B, 3, -1)[ar, 1, self.s[idx]] = 1
        x.view(B, 3, -1)[ar, 2, self.g[idx]] = 1
        return dict(x=x, op=op, ds=self.ds[idx].long(), dg=self.dg[idx].long(), dag=self.dag[idx], s=self.s[idx],
                    g=self.g[idx], D=self.D[idx], canon=self.canon[idx])


BANK_K = 6


def make_bank(pool, gen, K=BANK_K, chunk=8192):
    """(K, N, L, L) bool: K independent uniform shortest paths per pool maze."""
    bank = torch.zeros(K, pool.N, pool.L, pool.L, dtype=torch.bool, device=pool.open.device)
    for k in range(K):
        for i0 in range(0, pool.N, chunk):
            sl = slice(i0, min(pool.N, i0 + chunk))
            bank[k, sl] = jm.sample_uniform_path(pool.ds[sl].long(), pool.dag[sl], pool.g[sl], gen=gen)
    return bank


def make_model(arm, seed, device):
    torch.manual_seed(1000 + seed)
    ref = jm.Trunk(2, 3, 48, 1, 2)
    torch.manual_seed(2000 + seed)
    model = jm.Model(arm)
    jm.pair_trunk(model, ref)
    return model.to(device)


def arm_score(model, st, b, y):
    mask = b["op"].float()
    if model.is_m:
        return jm.score_m(st, y, mask, model.geom(tuple(b["op"].shape[1:])))
    if model.arm == "U-any":
        return jm.score_any(st["m"], mask, b["ds"], b["dag"], b["g"])
    return jm.score_unary(st["m"], y, mask)


def train(a):
    dev = torch.device(a.device)
    torch.backends.cudnn.benchmark = True
    pool = Pool(a.pool, dev)
    model = make_model(a.arm, a.seed, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps)
    gdata = torch.Generator().manual_seed(a.seed)                  # batches + prefix lengths
    glab = torch.Generator(device=dev).manual_seed(10_000 + a.seed)  # sampled paths
    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.arm}_s{a.seed}" + (f"_a{a.alpha}" if a.alpha != 1.0 else "") + (f"_x{a.steps}" if a.steps != 5000 else "")
    part = os.path.join(a.out, tag + ".partial.pt")
    start, hist, wall0 = 0, [], 0.0
    if os.path.exists(os.path.join(a.out, tag + ".train.json")):
        print("already done", tag)
        return
    tb = time.time()
    bank = make_bank(pool, glab) if a.arm not in ("U-canon", "U-any") else None  # glab is used only here
    uses = torch.zeros(pool.N, dtype=torch.long, device=dev)  # times each maze was drawn so far
    bank_sec = round(time.time() - tb, 1)
    print(json.dumps(dict(bank_sec=bank_sec, bank_k=BANK_K if bank is not None else 0)), flush=True)
    if os.path.exists(part):  # resume after a preemption
        ck = torch.load(part, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sch.load_state_dict(ck["sch"])
        gdata.set_state(ck["gdata"])
        glab.set_state(ck["glab"])
        start, hist, wall0 = ck["step"], ck["hist"], ck["wall"]
        uses = ck["uses"].to(dev)
    t0 = time.time()
    for step in range(start, a.steps):
        idx = torch.randint(pool.N, (a.batch,), generator=gdata).to(dev)
        npre = int(torch.randint(0, a.prefix_max + 1, (1,), generator=gdata))
        b = pool.batch(idx)
        nopen = b["op"].flatten(1).sum(1).float()
        if bank is not None:  # j-th draw of a maze -> bank entry j mod K (same for every arm: paired)
            y = bank[uses[idx] % BANK_K, idx]
        else:
            y = b["canon"]  # U-canon label; U-any ignores y
        uses.index_add_(0, idx, torch.ones_like(idx))
        with torch.no_grad():
            st = model.run(b["x"], npre)
        st0 = jm.detach_state(st)
        st1 = model.run(b["x"], a.k, st0)
        if a.arm == "U-min" and step >= a.warm:
            y = jm.viterbi_path(st1["m"][:, 0].detach(), b["ds"], b["dag"], b["g"])
        yf = y.float()
        with torch.no_grad():
            s0 = arm_score(model, st0, b, yf)
        s1 = arm_score(model, st1, b, yf)
        loss = -((s1 - s0) / nopen).mean()
        if a.alpha < 1.0:
            stf = model.run(b["x"], a.T)
            loss = a.alpha * loss - (1 - a.alpha) * (arm_score(model, stf, b, yf) / nopen).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()
        if step % 50 == 0 or step == a.steps - 1:
            hist.append(dict(step=step, loss=round(float(loss.detach()), 5),
                             end_score_per_px=round(float((s1.detach() / nopen).mean()), 5),
                             gnorm=round(float(gn), 4), npre=npre,
                             sec=round(wall0 + time.time() - t0, 1)))
            print(json.dumps(hist[-1]), flush=True)
        if (step + 1) % a.ckpt_every == 0 and step + 1 < a.steps:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sch=sch.state_dict(),
                            gdata=gdata.get_state(), glab=glab.get_state(), uses=uses.cpu(), step=step + 1, hist=hist,
                            wall=wall0 + time.time() - t0), part + ".tmp")
            os.replace(part + ".tmp", part)
    wall = wall0 + time.time() - t0
    torch.save(model.state_dict(), os.path.join(a.out, tag + ".pt"))
    nparam = sum(p.numel() for p in model.parameters())
    json.dump(dict(arm=a.arm, seed=a.seed, steps=a.steps, alpha=a.alpha, params=nparam, bank_k=BANK_K, bank_sec=bank_sec,
                   max_draws_per_maze=int(uses.max()), draws_beyond_bank=int((uses - BANK_K).clamp(min=0).sum()),
                   trunk_params=sum(p.numel() for p in model.trunk.parameters()), wall_sec=round(wall, 1),
                   hist=hist, device=str(dev)), open(os.path.join(a.out, tag + ".train.json"), "w"))
    if os.path.exists(part):
        os.remove(part)  # own scratch checkpoint inside the run dir; the final .pt supersedes it
    print("DONE", tag, round(wall, 1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=jm.U_ARMS + ("M-tree", "M-tile"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--T", type=int, default=30)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--prefix-max", type=int, default=24)
    ap.add_argument("--warm", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pool", default=os.path.join(HERE, "data", "pool_n9.npz"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "runs"))
    train(ap.parse_args())
