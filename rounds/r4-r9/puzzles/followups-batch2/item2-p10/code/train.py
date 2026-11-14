"""P10 training entry points.

  python train.py s4 SEED      # phase 0: B2 on exact 4x4, commit-gain targets, value head (gate G1 data)
  python train.py s6pilot T    # phase 0: S6 headroom pilot (B2-reg at 2T) + atom bank (LOO gain, rate)
  python train.py m2 ARM SEED  # phase 1: ARM in {auction, cvae, reg}

Scratch in /tmp/p10; small JSON results in results/.
"""
import json
import math
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

import grids
from models import B2, ValueHead, progressive_loss

torch.set_num_threads(int(os.environ.get("P10_THREADS", "2")))
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("P10_RES", os.path.join(HERE, "results"))
TMP = os.environ.get("P10_TMP", "/tmp/p10")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(TMP, exist_ok=True)
os.makedirs(RES, exist_ok=True)


def cosine_lr(opt, step, total, base, warm=200):
    lr = base * min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total)))
    for g in opt.param_groups:
        g["lr"] = lr


# =============================================================== S4
class S4Data:
    def __init__(self, seed):
        self.table = grids.enum4()
        self.rng = np.random.default_rng(seed)
        self.peer = grids.G4.peer_mat

    @staticmethod
    def is_test(g, maskbits):
        return ((g * 65536 + maskbits) * 2654435761 % 1000003) % 5 == 0

    def sample(self, B, test=False):
        gs, ms = [], []
        while len(gs) < B:
            g = int(self.rng.integers(288))
            r = self.rng.uniform(0.25, 0.94)
            m = int(np.clip(round(r * 16), 1, 15))
            mask = np.zeros(16, bool)
            mask[self.rng.choice(16, m, replace=False)] = True
            bits = int((mask * (1 << np.arange(16))).sum())
            if self.is_test(g, bits) != test:
                continue
            gs.append(g)
            ms.append(mask)
        gs = np.array(gs)
        ms = np.stack(ms)
        y = self.table[gs].astype(np.int64)  # 1..4
        x = np.where(ms, 0, y)
        return x, y - 1, ms, gs

    def pre_mask(self, x, ms):
        """Continuation augmentation: re-mask a random fraction (U[0,0.5]) of the givens for the no-grad prefix."""
        u = self.rng.uniform(0, 0.5, (len(x), 1))
        extra = (~ms) & (self.rng.uniform(size=ms.shape) < u)
        return np.where(extra, 0, x)


def t64(a):
    return torch.as_tensor(np.asarray(a), dtype=torch.long, device=DEV)


def train_b2(model, sample_fn, pre_fn, T, steps, B=256, lr=2e-3, log_every=500, tag=""):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    t0 = time.time()
    hist = []
    for step in range(steps):
        cosine_lr(opt, step, steps, lr)
        x, y, ms = sample_fn(B)
        xp = pre_fn(x, ms) if pre_fn else None
        loss, full = progressive_loss(model, t64(x), t64(y), torch.as_tensor(ms, dtype=torch.float32, device=DEV), T,
                                      x_pre=None if xp is None else t64(xp))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            hist.append((step, float(full)))
            print(f"[{tag}] step {step} full-depth loss {float(full):.4f} ({time.time() - t0:.0f}s)", flush=True)
    return hist, time.time() - t0


@torch.no_grad()
def commit_deltas(model, x, y, ms, T, Tr=1, ndig=4):
    """For each state and each masked cell c and each digit v: Delta(c, v) = other-masked-cell log loss
    (sum over masked q != c, against the truth) after T+Tr passes without the commit, minus the same with c
    clamped to v (continuing from h_T for Tr passes). Returns hT, pT, Delta (B,C,ndig) (NaN on given cells)."""
    X, Y = t64(x), t64(y)
    M = torch.as_tensor(ms, device=DEV)
    B, C = X.shape
    hT, lT = model.run(X, T)
    pT = lT.softmax(-1)
    _, lb = model.run(X, Tr, h=hT)
    nll_b = -lb.log_softmax(-1).gather(-1, Y[..., None]).squeeze(-1)  # (B,C)
    D = torch.full((B, C, ndig), float("nan"), device=DEV)
    for c in range(C):
        rows = torch.nonzero(M[:, c]).squeeze(-1)
        if len(rows) == 0:
            continue
        for v in range(ndig):
            Xc = X[rows].clone()
            Xc[:, c] = v + 1
            _, lc = model.run(Xc, Tr, h=hT[rows])
            nll_c = -lc.log_softmax(-1).gather(-1, Y[rows][..., None]).squeeze(-1)
            other = M[rows].clone()
            other[:, c] = False
            D[rows, c, v] = ((nll_b[rows] - nll_c) * other).sum(-1)
    return hT, pT, D


def proxies(pT, ms, peer):
    p = pT.cpu().numpy()
    H = -(p * np.log(np.clip(p, 1e-12, 1))).sum(-1)
    pmax = p.max(-1)
    mrv = (p > 0.01).sum(-1)
    mpeer = (ms[:, None, :] & peer[None]).sum(-1)
    return {"entropy": H, "one_minus_pmax": 1 - pmax, "masked_peers": mpeer.astype(float), "mrv_count": mrv.astype(float)}


def run_s4(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    T = 4
    data = S4Data(seed)
    model = B2(16, 4, d=64, heads=4).to(DEV)
    steps = int(os.environ.get("P10_S4_STEPS", "4000"))
    hist, tt = train_b2(model, lambda B: data.sample(B)[:3], data.pre_mask, T, steps, tag=f"s4 seed{seed}")
    model.eval()
    # --- per-cell NLL vs exact posterior on test states
    xt, yt, mt, gt = data.sample(3000, test=True)
    with torch.no_grad():
        _, lg = model.run(t64(xt), T)
    lp = lg.log_softmax(-1).cpu().numpy()
    nll = -np.take_along_axis(lp, yt[..., None], -1)[..., 0]
    cons, post, ncons = grids.posterior4(data.table, np.where(mt, 0, yt + 1), ~mt)
    bayes = -np.log(np.take_along_axis(post, yt[..., None], -1)[..., 0])
    fit = {"nll_model_per_masked_cell": float(nll[mt].mean()), "nll_bayes_per_masked_cell": float(bayes[mt].mean()),
           "H_Y_given_X_joint_nats": float(np.log(ncons).mean())}
    print("fit", fit, flush=True)
    # --- commit-gain targets: train states (head training), test states (gate)
    xs, ys, mss, gs = data.sample(6000)
    out = {}
    for name, (x, y, ms) in {"train": (xs, ys, mss), "test": (xt, yt, mt)}.items():
        H_, P_, D_ = [], [], []
        for i in range(0, len(x), 500):
            h, p, d = commit_deltas(model, x[i:i + 500], y[i:i + 500], ms[i:i + 500], T)
            H_.append(h), P_.append(p), D_.append(d)
        out[name] = (torch.cat(H_), torch.cat(P_), torch.cat(D_))
    # exact secondary target on test: sum_q I*(z_c; z_q | X) over other masked q
    Iex = np.zeros((len(xt), 16))
    for b in range(len(xt)):
        mi, _ = grids.pairwise_mi4(data.table, cons[b])
        Iex[b] = (mi * mt[b][None, :]).sum(1)
    # --- value head
    hT_tr, pT_tr, D_tr = out["train"]
    am_tr = pT_tr.argmax(-1)
    G_tr = torch.gather(torch.nan_to_num(D_tr), -1, am_tr[..., None]).squeeze(-1)
    M_tr = torch.as_tensor(mss, device=DEV)
    delta = float(np.quantile(G_tr[M_tr].abs().cpu().numpy(), 0.9))
    head = ValueHead(64, 4).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    hsteps = int(os.environ.get("P10_HEAD_STEPS", "3000"))
    n = len(hT_tr)
    for step in range(hsteps):
        cosine_lr(opt, step, hsteps, 1e-3, warm=100)
        idx = torch.randint(0, n, (256,), device=DEV)
        v, _ = head(hT_tr[idx], pT_tr[idx])
        m = M_tr[idx]
        loss = F.huber_loss(v[m], G_tr[idx][m], delta=delta)
        opt.zero_grad()
        loss.backward()
        opt.step()
    head.eval()
    hT_te, pT_te, D_te = out["test"]
    with torch.no_grad():
        v_te, _ = head(hT_te, pT_te)
        v_tr, _ = head(hT_tr, pT_tr)
    extra = {}
    if os.environ.get("P10_HEAD_CENTER") == "1":
        # post-hoc diagnostic (not the pre-registered head): same head, trained on the within-state centred target
        # G - mean_state(G), i.e. only the part of the gain that ranks cells inside a state
        Mf = M_tr.float()
        Gc_tr = torch.where(M_tr, G_tr - (G_tr * Mf).sum(1, keepdim=True) / Mf.sum(1, keepdim=True).clamp(min=1), 0.0)
        dc = float(np.quantile(Gc_tr[M_tr].abs().cpu().numpy(), 0.9))
        csteps = int(os.environ.get("P10_HEAD_CENTER_STEPS", "10000"))
        torch.manual_seed(seed + 1000)
        hc = ValueHead(64, 4).to(DEV)
        oc = torch.optim.AdamW(hc.parameters(), lr=1e-3, weight_decay=0.01)
        for step in range(csteps):
            cosine_lr(oc, step, csteps, 1e-3, warm=100)
            idx = torch.randint(0, n, (256,), device=DEV)
            vc, _ = hc(hT_tr[idx], pT_tr[idx])
            m = M_tr[idx]
            loss = F.huber_loss(vc[m], Gc_tr[idx][m], delta=dc)
            oc.zero_grad()
            loss.backward()
            oc.step()
        hc.eval()
        with torch.no_grad():
            extra["v_te_c"] = hc(hT_te, pT_te)[0].cpu().numpy()
            extra["v_tr_c"] = hc(hT_tr, pT_tr)[0].cpu().numpy()
        extra["delta_c"] = dc
    am_te = pT_te.argmax(-1)
    G_te = torch.gather(torch.nan_to_num(D_te), -1, am_te[..., None]).squeeze(-1)
    prox_te = proxies(pT_te, mt, data.peer)
    prox_tr = proxies(pT_tr, mss, data.peer)
    tag = os.environ.get("P10_S4_TAG", "")
    np.savez_compressed(f"{TMP}/s4{tag}_seed{seed}.npz", **extra, v_te=v_te.cpu().numpy(), G_te=G_te.cpu().numpy(), D_te=D_te.cpu().numpy(),
                        p_te=pT_te.cpu().numpy(), mt=mt, yt=yt, xt=xt, Iex=Iex, post=post, ncons=ncons,
                        v_tr=v_tr.cpu().numpy(), G_tr=G_tr.cpu().numpy(), mtr=mss, delta=delta,
                        **{f"prox_te_{k}": v for k, v in prox_te.items()}, **{f"prox_tr_{k}": v for k, v in prox_tr.items()})
    torch.save(model.state_dict(), f"{TMP}/s4{tag}_b2_seed{seed}.pt")
    # --- joint NLL readout (decoders along commit orders)
    jn = None if tag else joint_nll_readout(model, data, xt[:600], yt[:600], mt[:600], T, ncons[:600])
    res = {"seed": seed, "train_steps": steps, "train_time_s": tt, "hist": hist, "fit": fit, "huber_delta": delta,
           "joint_nll": jn}
    json.dump(res, open(f"{RES}/s4{tag}_seed{seed}.json", "w"), indent=1)
    print("done s4 seed", seed, flush=True)


@torch.no_grad()
def joint_nll_readout(model, data, x, y, ms, T, ncons):
    """Teacher-forced joint NLL of the true grid along each decoder's commit order, vs passes.
    B2 one-shot: all masked cells committed after T passes (factorised).
    B3a top-k (k in 1,2,4): each round runs Tr=1 pass (continuing), commits the k most confident cells.
    B3b tau (0.9,0.99,0.999): each round Tr=1 pass, commits all cells with p_max>=tau (at least the top one).
    ARDM DP (Hoogeboom et al. 2022): fresh T-pass calls, random order, DP over how many cells per call;
    expected NLL per budget R from the per-(D,t) loss curve."""
    X0, Y = t64(x), t64(y)
    M0 = torch.as_tensor(ms, device=DEV)
    res = {"H_Y_given_X": float(np.log(ncons).mean())}
    # one-shot
    _, lg = model.run(X0, T)
    nll = -lg.log_softmax(-1).gather(-1, Y[..., None]).squeeze(-1)
    res["oneshot"] = {"passes": T, "nll": float((nll * M0).sum(-1).mean())}

    def decode(rule, par):
        X = X0.clone()
        M = M0.clone()
        h, lg = model.run(X, T)
        tot = torch.zeros(len(X), device=DEV)
        passes = torch.full((len(X),), float(T), device=DEV)
        while M.any():
            lp = lg.log_softmax(-1)
            pm = lp.exp().max(-1).values.masked_fill(~M, -1)
            if rule == "topk":
                k = min(par, int(M.sum(-1).max()))
                sel = torch.zeros_like(M)
                top = pm.topk(k, -1).indices
                sel.scatter_(1, top, True)
                sel &= M
            else:
                sel = (pm >= par) & M
                best = pm.argmax(-1)
                ar = torch.arange(len(X), device=DEV)
                sel[ar, best] |= M[ar, best]
            nl = -lp.gather(-1, Y[..., None]).squeeze(-1)
            tot += (nl * sel).sum(-1)
            X = torch.where(sel, Y + 1, X)
            M = M & ~sel
            active = M.any(-1)
            passes += active.float()
            h, lg = model.run(X, 1, h=h)
        return {"passes": float(passes.mean()), "nll": float(tot.mean())}
    for k in (1, 2, 4):
        res[f"B3a_top{k}"] = decode("topk", k)
    for tau in (0.9, 0.99, 0.999):
        res[f"B3b_tau{tau}"] = decode("tau", tau)
    # ARDM DP
    rng = np.random.default_rng(0)
    Ds = M0.sum(-1).cpu().numpy()
    curves = {}
    for D in np.unique(Ds):
        idx = np.flatnonzero(Ds == D)
        L = np.zeros(D)
        # random reveal order per example: rank of a uniform priority among the masked cells
        pri = np.where(ms[idx], rng.uniform(size=ms[idx].shape), 2.0)
        rank = torch.as_tensor(np.argsort(np.argsort(pri, 1), 1), device=DEV)
        for t in range(D):
            rev = (rank < t) & M0[idx]
            X = torch.where(rev, Y[idx] + 1, X0[idx])
            Mk = M0[idx] & ~rev
            _, lg = model.run(X, T)
            nl = -lg.log_softmax(-1).gather(-1, Y[idx][..., None]).squeeze(-1)
            L[t] = float(((nl * Mk).sum(-1) / Mk.sum(-1)).mean())
        curves[int(D)] = L
    budgets = {}
    for R in (1, 2, 3, 4, 6, 8, 12, 16):
        tot = 0.0
        for D, L in curves.items():
            Rr = min(R, D)
            # dp[r][t] = min cost to reveal t cells in r calls
            INF = 1e18
            dp = np.full((Rr + 1, D + 1), INF)
            dp[0, 0] = 0
            for r in range(1, Rr + 1):
                for t in range(1, D + 1):
                    best = INF
                    for s in range(0, t):
                        if dp[r - 1, s] < INF:
                            best = min(best, dp[r - 1, s] + (t - s) * L[s])
                    dp[r, t] = best
            tot += dp[1:, D].min() * (Ds == D).sum()
        budgets[R] = {"passes": R * T, "nll_expected": tot / len(Ds)}
    res["ARDM_DP"] = budgets
    return res


# =============================================================== entry
if __name__ == "__main__":
    what = sys.argv[1]
    if what == "s4":
        run_s4(int(sys.argv[2]))
    else:
        import train_s6
        train_s6.main(sys.argv[1:])
