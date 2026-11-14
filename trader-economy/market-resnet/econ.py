"""Phase 2: a purely economic market ResNet. No backprop across depth.

Streams per example: x (raw input, re-read by every trader), b (belief
market: one binary LMSR per output position, stored as a logit), q (K internal
markets per position = the scratch h; each is a binary LMSR price on a
question that settles against a LATER market state).

Traders: N small conv nets with their own weights and optimizers. At each
step one trader trades (training: random or auction; inference: auction =
highest predicted profit; halt when no bid exceeds the compute cost). A
trader reads DETACHED x, b, q, so its gradient is only its own earnings:
  pay_b = S_y(b + db) - S_y(b)            (log scoring rule vs the true label)
  pay_q = sum_j gamma^k_j [S_{p*}(q_j + dq_j) - S_{p*}(q_j)]
with p* = sigmoid(b_{t+k_j}) at position i+o_j (a later price; TD in market
form) or the true label y when t+k_j runs past the last step (anchor).
Trade size is capped by wealth: worst-case loss per bit <= frac * wealth.
Wealth ledger: w_i += earnings - cost per trade; bankrupt (w <= 0) -> respawn.
Mode "bptt": same traders, round-robin, trained end-to-end through depth on
the final log loss (the backprop comparison).
"""
import argparse, json, time, os, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mresnet import prefix_batch

LS = F.logsigmoid


def soft_score(z, p):
    """Log score of logit z against (soft) outcome p, elementwise."""
    return p * LS(z) + (1 - p) * LS(-z)


class Trader(nn.Module):
    def __init__(self, K, w=96, k=3, nlayers=2):
        super().__init__()
        self.K = K
        cin = 3 + 1 + K
        self.inp = nn.Conv1d(cin, w, k, padding=k // 2)
        self.hid = nn.ModuleList([nn.Conv1d(w, w, k, padding=k // 2) for _ in range(nlayers - 1)])
        self.out = nn.Conv1d(w, 1 + K, 1)
        self.gain = nn.Sequential(nn.Conv1d(w + 1 + K, 32, 1), nn.ReLU(), nn.Conv1d(32, 1, 1))

    def forward(self, xoh, b, q, mask):
        z = torch.cat([xoh, torch.tanh(b / 4).unsqueeze(1), torch.tanh(q / 4)], 1)
        z = F.relu(self.inp(z))
        for c in self.hid:
            z = F.relu(c(z)) + z
        o = self.out(z) * mask.unsqueeze(1)
        db, dq = o[:, 0], o[:, 1:]
        gf = torch.cat([z.detach(), torch.tanh(b / 4).unsqueeze(1).detach(), torch.tanh(q / 4).detach()], 1)
        ghat = (self.gain(gf).squeeze(1) * mask).sum(1) / mask.sum(1)   # per-bit predicted profit
        return db, dq, ghat


def worst_loss_per_bit(b, db, mask):
    lo1 = LS(b) - LS(b + db)          # loss if y=1
    lo0 = LS(-b) - LS(-(b + db))      # loss if y=0
    return (torch.maximum(lo1, lo0).clamp(min=0) * mask).sum(1) / mask.sum(1)


class Economy:
    def __init__(self, a, device):
        self.a, self.dev = a, device
        self.K = a.K
        g = torch.Generator().manual_seed(99)
        # question specs for internal markets: offset o_j, horizon k_j
        self.offs = torch.randint(-a.qspan, a.qspan + 1, (self.K,), generator=g).tolist()
        self.hor = [a.hmin * (2 ** (j % 3)) for j in range(self.K)]
        self.traders = [self.new_trader() for _ in range(a.N)]
        self.wealth = [a.W0] * a.N
        self.stats = {"respawns": 0, "entries": 0, "mm_subsidy": 0.0, "earn_total": 0.0}

    def new_trader(self):
        t = Trader(self.K, w=self.a.width).to(self.dev)
        opt = torch.optim.Adam(t.parameters(), lr=self.a.lr)
        return [t, opt]

    def xoh(self, x):
        return F.one_hot(x, 3).float().transpose(1, 2)

    def rollout(self, x, mask, D, pick="random", cost=None, gen=None, keep=True):
        """Run D steps with no grad. Returns per-step records (inputs, chosen
        trader, scale) and the belief trajectory. If cost is given, halts per
        example when the best bid <= cost (auction)."""
        B, L = x.shape
        xoh = self.xoh(x)
        b = torch.zeros(B, L, device=self.dev)
        q = torch.zeros(B, self.K, L, device=self.dev)
        recs, bs, qs = [], [b], [q]
        active = torch.ones(B, device=self.dev)
        steps_used = torch.zeros(B, device=self.dev)
        with torch.no_grad():
            for t in range(D):
                if pick == "auction" or cost is not None:
                    outs = [tr[0](xoh, b, q, mask) for tr in self.traders]
                    bids = torch.stack([o[2] for o in outs], 1)            # (B,N)
                    best = bids.argmax(1)
                    # per-example winner
                    db = torch.stack([o[0] for o in outs], 1)[torch.arange(B), best]
                    dq = torch.stack([o[1] for o in outs], 1)[torch.arange(B), best]
                    who = best
                    if cost is not None:
                        active = active * (bids.max(1).values > cost).float()
                else:
                    i = int(torch.randint(len(self.traders), (1,), generator=gen)) if gen is not None else t % len(self.traders)
                    db, dq, _ = self.traders[i][0](xoh, b, q, mask)
                    who = torch.full((B,), i, device=self.dev)
                wl = worst_loss_per_bit(b, db, mask)
                wcap = torch.tensor([max(self.wealth[int(k)], 0.0) for k in who.tolist()], device=self.dev) \
                    if self.a.wealth_cap else None
                s = torch.ones(B, device=self.dev) if wcap is None else \
                    torch.clamp(self.a.frac * wcap / (wl + 1e-8), max=1.0)
                s = s * active
                if keep:
                    recs.append((b, q, who, s))
                b = b + s.view(B, 1) * db
                q = q + s.view(B, 1, 1) * dq
                steps_used += active
                bs.append(b); qs.append(q)
        return recs, bs, qs, steps_used

    def q_targets(self, bs, y, t, D):
        """Soft outcomes for the K internal markets traded at step t (0-based):
        sigmoid(b_{t+1+k}) at position i+o, or y past the last step (anchor)."""
        tg = []
        for j in range(self.K):
            tt = t + 1 + self.hor[j]
            src = y if tt > D else torch.sigmoid(bs[tt])
            o = self.offs[j]
            tg.append(torch.roll(src, shifts=-o, dims=1))   # value at position i+o
        return torch.stack(tg, 1)

    def train_step(self, x, y, mask, D, gen):
        a = self.a
        xoh = self.xoh(x)
        pick = "auction" if torch.rand(1, generator=gen).item() < a.p_auction else "random"
        recs, bs, qs, _ = self.rollout(x, mask, D, pick=pick, gen=gen)
        nb = mask.sum(1)
        earn = [0.0] * len(self.traders)
        ntr = [0] * len(self.traders)
        # local update of each trader on its own earnings (inputs detached)
        for i, (tr, opt) in enumerate(self.traders):
            steps = [(t, r) for t, r in enumerate(recs) if (r[2] == i).any()]
            if not steps:
                continue
            opt.zero_grad()
            tot = 0.0
            for t, (b, q, who, s) in steps:
                sel = (who == i).float()
                db, dq, ghat = tr(xoh, b, q, mask)
                pay_b = ((soft_score(b + s.view(-1, 1) * db, y) - soft_score(b, y)) * mask).sum(1) / nb
                pay = pay_b
                if self.K:
                    pt = self.q_targets(bs, y, t, D)
                    disc = torch.tensor([a.gamma ** self.hor[j] for j in range(self.K)], device=self.dev).view(1, -1, 1)
                    dqs = s.view(-1, 1, 1) * dq
                    pq = (disc * (soft_score(q + dqs, pt) - soft_score(q, pt)) * mask.unsqueeze(1)).sum((1, 2)) / nb
                    pay = pay + a.lam_q * pq
                profit = pay - a.cost
                # bid predicts earnings before compute cost; halting compares it with the cost once
                gl = F.smooth_l1_loss(ghat, (profit + a.cost).detach(), reduction="none")
                loss = ((-pay + gl) * sel).sum() / len(steps) / x.shape[0]
                loss.backward()
                tot += (profit.detach() * sel).sum().item() / x.shape[0]
                ntr[i] += 1
            torch.nn.utils.clip_grad_norm_(tr.parameters(), 1.0)
            opt.step()
            earn[i] = tot
        # wealth ledger (per bit, averaged over the batch); MM subsidy = total b+q payouts
        for i in range(len(self.traders)):
            self.wealth[i] += earn[i]
            self.stats["earn_total"] += earn[i]
        for i in range(len(self.traders)):
            if a.wealth_cap and self.wealth[i] <= 0:
                self.traders[i] = self.new_trader(); self.wealth[i] = a.W0
                self.stats["respawns"] += 1
        S_final = ((soft_score(bs[-1], y) * mask).sum(1) / nb).mean().item()
        return {"final_nll_bit": -S_final, "earn": [round(e, 4) for e in earn], "ntrades": ntr,
                "wealth": [round(w, 2) for w in self.wealth]}

    def add_trader(self):
        self.traders.append(self.new_trader()); self.wealth.append(self.a.W0)
        self.stats["entries"] += 1


# ------------------------------------------------------------- BPTT baseline
class BPTT:
    """Same traders, round-robin, gradients through depth on the final loss."""
    def __init__(self, a, device):
        self.a, self.dev = a, device
        self.K = a.K
        self.traders = nn.ModuleList([Trader(a.K, w=a.width) for _ in range(a.N)]).to(device)
        self.opt = torch.optim.Adam(self.traders.parameters(), lr=a.lr)

    def run(self, x, mask, D, grad):
        B, L = x.shape
        xoh = F.one_hot(x, 3).float().transpose(1, 2)
        b = torch.zeros(B, L, device=self.dev)
        q = torch.zeros(B, self.K, L, device=self.dev)
        bs = [b]
        with torch.set_grad_enabled(grad):
            for t in range(D):
                db, dq, _ = self.traders[t % len(self.traders)](xoh, b, q, mask)
                b = b + db; q = q + dq
                bs.append(b)
        return bs

    def train_step(self, x, y, mask, D, gen):
        nb = mask.sum(1)
        bs = self.run(x, mask, D, True)
        loss = -((soft_score(bs[-1], y) * mask).sum(1) / nb).mean()
        self.opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.traders.parameters(), 1.0)
        self.opt.step()
        return {"final_nll_bit": loss.item()}


@torch.no_grad()
def eval_depths(model, mode, L, depths, n, device, cost=None, chunk=None, pick="auction"):
    gen = torch.Generator().manual_seed(555 + L)
    x, y, m = prefix_batch(n, L, L, gen, device)
    Dmax = max(depths)
    out = []
    chunk = chunk or n
    accs = {D: [] for D in depths}; nlls = {D: [] for D in depths}
    for c0 in range(0, n, chunk):
        xs, ys, ms = x[c0:c0 + chunk], y[c0:c0 + chunk], m[c0:c0 + chunk]
        if mode == "bptt":
            bs = model.run(xs, ms, Dmax, False)
        else:
            _, bs, _, _ = model.rollout(xs, ms, Dmax, pick=pick, keep=False)
        for D in depths:
            b = bs[D]
            accs[D].append((((b > 0).float() == ys) | (ms == 0)).all(1).float().cpu())
            nlls[D].append((-(soft_score(b, ys) * ms).sum(1) / ms.sum(1)).cpu())
        del bs
    for D in depths:
        out.append({"L": L, "D": D, "pick": pick, "acc": torch.cat(accs[D]).mean().item(), "nll_bit": torch.cat(nlls[D]).mean().item()})
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="econ")       # econ | bptt
    p.add_argument("--N", type=int, default=4)     # traders
    p.add_argument("--K", type=int, default=0)     # internal markets per position
    p.add_argument("--width", type=int, default=96)
    p.add_argument("--D", type=int, default=32)    # training depth
    p.add_argument("--lmin", type=int, default=8)
    p.add_argument("--lmax", type=int, default=32)
    p.add_argument("--iters", type=int, default=4000)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cost", type=float, default=0.0)   # compute cost per trade (nats/bit)
    p.add_argument("--W0", type=float, default=20.0)
    p.add_argument("--frac", type=float, default=0.5)
    p.add_argument("--wealth_cap", type=int, default=1)
    p.add_argument("--p_auction", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--lam_q", type=float, default=1.0)
    p.add_argument("--qspan", type=int, default=4)
    p.add_argument("--hmin", type=int, default=1)
    p.add_argument("--grow_every", type=int, default=0)  # add a trader every n iters (entry)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    torch.manual_seed(a.seed)
    gen = torch.Generator().manual_seed(2000 + a.seed)
    M = Economy(a, a.device) if a.mode == "econ" else BPTT(a, a.device)
    log = []
    t0 = time.time()
    if a.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for it in range(a.iters):
        x, y, m = prefix_batch(a.bs, a.lmin, a.lmax, gen, a.device)
        lg = M.train_step(x, y, m, a.D, gen)
        if a.mode == "econ" and a.grow_every and it and it % a.grow_every == 0:
            M.add_trader()
        if it % 100 == 0 or it == a.iters - 1:
            lg.update(it=it, sec=round(time.time() - t0, 1))
            if a.mode == "econ":
                lg.update({k: v for k, v in M.stats.items()})
            log.append(lg); print(json.dumps(lg), flush=True)
    res = {"args": vars(a), "log": log, "train_sec": time.time() - t0}
    if a.device.startswith("cuda"):
        res["train_peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    ev = []
    for L, depths, n in [(32, [4, 8, 16, 24, 32, 64], 1000), (64, [16, 32, 48, 64, 128], 500),
                         (320, [80, 160, 240, 320, 640], 200), (3200, [800, 1600, 2400, 3200, 4800], 50)]:
        for pick in (["auction", "roundrobin"] if a.mode == "econ" else ["-"]):
            ev += eval_depths(M, a.mode, L, depths, n, a.device, chunk=100, pick=pick)
            print(json.dumps(ev[-len(depths):]), flush=True)
        res["eval"] = ev
        json.dump(res, open(a.out + ".json", "w"))
    if a.mode == "econ":
        torch.save({"traders": [t[0].state_dict() for t in M.traders], "wealth": M.wealth,
                    "offs": M.offs, "hor": M.hor, "args": vars(a)}, a.out + ".pt")
    else:
        torch.save({"traders": M.traders.state_dict(), "args": vars(a)}, a.out + ".pt")
