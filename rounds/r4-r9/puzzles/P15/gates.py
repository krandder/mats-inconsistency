"""P15 gates G1, G2, G2b, G3, G4, G6 (CPU, float64) and the data parts of G0.
Usage: python gates.py G1 G2 ...   Results go to gates/<gate>.json."""
import itertools
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p15  # noqa: E402
from p15 import (U_, K_, H_, gather, pre_x, level, head_logits, head_msg, trader_msg, settle, onehot,  # noqa
                 init_params, Task)

torch.set_default_dtype(torch.float64)
p15.DEV = "cpu"
GD = os.path.join(HERE, "gates")
os.makedirs(GD, exist_ok=True)
F64 = torch.float64


def save(name, d):
    json.dump(d, open(os.path.join(GD, name + ".json"), "w"), indent=1, default=float)
    print(name, json.dumps(d, default=float)[:600], flush=True)


# ------------------------------------------------------------------ G1
def g1():
    import p7_maze_ref as p7
    torch.manual_seed(0)
    U, K, hid, n, B = 2, 4, 32, 5, 3
    tr = p7.Trader(1, U * K, U, K, hid, torch.ones(U, p7.CX + U * K)).double()
    with torch.no_grad():
        tr.b1.normal_()
        tr.b2.normal_()
    x = torch.zeros(B, 2, n, n)
    x[:, 0] = (torch.rand(B, n, n) < 0.3).double()
    x[:, 1, 0, 0] = 1
    zi = torch.randint(0, K, (B, n, n, U))
    zoh = F.one_hot(zi, K).double()                                   # (B,n,n,U,K)
    zc = zoh.reshape(B, n, n, U * K).permute(0, 3, 1, 2)
    feats = p7.unfold(x, zc, 1)
    lam = torch.randn(B * n * n, U, K)
    lam = lam - lam.max(-1, keepdim=True).values
    ref = tr.message(feats, lam, U, K, B, n, "l").permute(0, 3, 4, 1, 2).reshape(B, n * n, U, K)
    P = {"W1": tr.W1.detach()[None].clone(), "b1": tr.b1.detach()[None].clone(), "W2": tr.W2.detach()[None].clone(),
         "b2": tr.b2.detach()[None].clone()}
    nbr = p15.nbr2d(n)
    xc = x.permute(0, 2, 3, 1).reshape(1, B, n * n, 2)
    px = pre_x(P, xc, nbr, 2, xnull=torch.tensor([1.0, 0.0]))
    z = zoh.reshape(1, B, n * n, U, K)
    pre, lg = level(P, px, z, nbr, 2)
    ours = trader_msg(P, pre, z, lam.reshape(1, B, n * n, U, K), nbr, 2)[0]
    err = (ours - ref).abs().max().item()
    # forward logits agree too
    lg_ref = tr(feats).reshape(B, n * n, U, K)
    err_fwd = (lg[0] - lg_ref).abs().max().item()
    # head message: P7 sigmoid head vs our 2-class head with hw[...,1]-hw[...,0] = w
    hd = p7.Head(U, K).double()
    y = torch.randint(0, 2, (B * n * n,)).double()
    ref_h = hd.message(zoh.reshape(-1, U, K), y).reshape(B, n * n, U, K)
    Ph = {"hw": torch.stack([torch.zeros(U, K), hd.w.detach()], -1)[None], "hb": torch.stack([torch.zeros(()), hd.b.detach()])[None]}
    ours_h = head_msg(Ph, z, y.long().reshape(1, B, n * n), torch.ones(1, B, n * n))[0]
    err_h = (ours_h - ref_h).abs().max().item()
    # chunked == unchunked
    ours_c = trader_msg(P, pre, z, lam.reshape(1, B, n * n, U, K), nbr, 2, max_elems=5000)[0]
    err_c = (ours_c - ours).abs().max().item()
    ok = max(err, err_fwd, err_h, err_c) < 1e-5
    save("G1", {"pass": ok, "max_abs_err_message": err, "max_abs_err_forward": err_fwd, "max_abs_err_head_msg": err_h,
                "chunk_consistency": err_c, "setup": "P7 Trader rho=1 masks=1, 5x5 maze, B=3, float64"})
    return ok


# ------------------------------------------------------------------ G2 / G2b
def chain_logits(P, x, zprev, nbr, Cx):
    px = pre_x(P, x, nbr, Cx)
    return level(P, px, zprev, nbr, Cx)


def brute_posterior(P, x, y, nbr, Cx, U, K, D, zs_real, k, cell):
    """Exact P(z_k[cell] | x, z_{k-1} = realised, y) summing over all other level-k cells and all
    downstream levels (enumeration). Units = U (joint over units when U>1: returns (U,K) marginals)."""
    N = x.shape[2]
    nv = K ** U

    def z_of(assign):  # assign: tuple of joint values per cell
        z = torch.zeros(1, 1, N, U, K)
        for c, j in enumerate(assign):
            for u in range(U):
                z[0, 0, c, u, (j // (K ** (U - 1 - u))) % K] = 1
        return z

    def logp(zprev, znew):
        _, lg = chain_logits(P, x, zprev, nbr, Cx)
        return (F.log_softmax(lg, -1) * znew).sum().item()

    def loglik_top(z):
        lp = F.log_softmax(head_logits(P, z), -1)
        return lp[0, 0].gather(-1, y[0, 0][:, None]).sum().item()

    # dynamic programming over levels, states = full level configurations (nv^N)
    configs = list(itertools.product(range(nv), repeat=N))
    Z = [z_of(a) for a in configs]
    # beta_D(c) = log P(y | z_D = c); beta_k(c) = log sum_c' P(c'|c) exp(beta_{k+1}(c'))
    beta = torch.tensor([loglik_top(z) for z in Z])
    trans = torch.tensor([[logp(Z[i], Z[j]) for j in range(len(Z))] for i in range(len(Z))])
    betas = {D: beta}
    for kk in range(D - 1, k - 1, -1):
        betas[kk] = torch.logsumexp(trans + betas[kk + 1][None], 1)
    zprev = zs_real[k - 1]
    _, lg = chain_logits(P, x, zprev, nbr, Cx)
    lp = torch.tensor([(F.log_softmax(lg, -1) * Z[j]).sum().item() for j in range(len(Z))])
    post = torch.softmax(lp + betas[k], 0)
    marg = torch.zeros(U, K)
    for j, a in enumerate(configs):
        v = a[cell]
        for u in range(U):
            marg[u, (v // (K ** (U - 1 - u))) % K] += post[j]
    return marg


def small_model(U, K, H, N, seed, Cy=2):
    P = init_params([seed], 1, 3, Cy, U=U, K=K, H=H, dtype=F64, dev="cpu")
    g = torch.Generator().manual_seed(seed + 5)
    with torch.no_grad():
        for k in P:
            P[k].mul_(2.5)
            P[k].add_(0.5 * torch.randn(P[k].shape, generator=g))
    return {k: v.detach() for k, v in P.items()}


def bl_q(P, x, y, nbr, U, K, D, seed):
    g = torch.Generator().manual_seed(seed)
    vm = torch.ones(1, 1, x.shape[2])
    px = pre_x(P, x, nbr, 1)
    z = torch.zeros(1, 1, x.shape[2], U, K)
    zs, pres, lgs = [z], [None], [None]
    for k in range(1, D + 1):
        pre, lg = level(P, px, z, nbr, 1)
        z = onehot(torch.distributions.Categorical(logits=lg).sample(), K, F64)
        zs.append(z)
        pres.append(pre)
        lgs.append(lg)
    qs, _ = settle(P, None, zs, pres, lgs, y, vm, nbr, 1)
    return zs, qs


def g2():
    errs = []
    for trial in range(5):
        torch.manual_seed(trial)
        U, K, D, N = 1, 2, 4, 1
        P = small_model(U, K, 8, N, trial)
        nbr = p15.nbr1d(N)
        x = torch.tensor([[[[float(trial % 2)]]]])
        y = torch.tensor([[[trial // 2 % 2]]])
        zs, qs = bl_q(P, x, y, nbr, U, K, D, trial)
        for k in range(1, D + 1):
            ex = brute_posterior(P, x, y, nbr, 1, U, K, D, zs, k, 0)
            errs.append((qs[k][0, 0, 0] - ex).abs().max().item())
    ok = max(errs) < 1e-6
    save("G2", {"pass": ok, "max_abs_err": max(errs), "n_checks": len(errs), "setup": "1-cell chain U=1 K=2 D=4, 5 random models"})
    return ok


def g2b():
    rows = []
    U, K, D, N = 1, 2, 4, 4
    nbr = p15.nbr1d(N)
    per_depth = {k: [] for k in range(1, D + 1)}
    for trial in range(6):
        torch.manual_seed(100 + trial)
        P = small_model(U, K, 8, N, 100 + trial)
        x = torch.randint(0, 2, (1, 1, N, 1)).double()
        y = torch.randint(0, 2, (1, 1, N))
        zs, qs = bl_q(P, x, y, nbr, U, K, D, trial)
        for k in range(1, D + 1):
            for c in range(N):
                ex = brute_posterior(P, x, y, nbr, 1, U, K, D, zs, k, c)
                per_depth[k].append((qs[k][0, 0, c] - ex).abs().max().item())
    res = {"setup": "4 cells x 4 levels, U=1 K=2, 6 random models; exact = P(z_k[c] | x, z_{k-1} realised, y) "
                    "marginalising the rest of level k and all downstream levels",
           "mean_abs_err_by_distance_from_top": {str(D - k): float(np.mean(v)) for k, v in per_depth.items()},
           "max_abs_err_by_distance_from_top": {str(D - k): float(np.max(v)) for k, v in per_depth.items()}}
    save("G2b", res)
    return True


# ------------------------------------------------------------------ G3 hand-set rules
def parity_rule(M=12.0, Mh=16.0, flip=None):
    """Systolic XOR z_k[i] = x_i ^ z_{k-1}[i-1] in unit 0 value 1; value 0 and unit 1 inert.
    flip: index p of the pattern table entry to flip; patterns (x,zl) = (1,1),(1,0),(0,1),(0,0)."""
    Cx, W, C = 1, 3, 1 + U_ * K_
    P = {k: torch.zeros_like(v) for k, v in init_params([0], Cx, W, 2, dtype=F64, dev="cpu").items()}
    W1, b1, W2, b2, hw, hb = (P[k][0] for k in ("W1", "b1", "W2", "b2", "hw", "hb"))
    zl = Cx + 0 * K_ + 1        # unit 0, value 1 channel
    # pattern units: relu(x+zl-1), relu(x-zl), relu(zl-x), relu(1-x-zl)
    W1[0, 0, 1, 0], W1[0, 0, 0, zl], b1[0, 0] = 1, 1, -1
    W1[0, 1, 1, 0], W1[0, 1, 0, zl] = 1, -1
    W1[0, 2, 1, 0], W1[0, 2, 0, zl] = -1, 1
    W1[0, 3, 1, 0], W1[0, 3, 0, zl], b1[0, 3] = -1, -1, 1
    t = torch.tensor([0.0, 1.0, 1.0, 0.0])
    if flip is not None:
        t[flip] = 1 - t[flip]
    W2[0, 1, :4] = 2 * M * t
    b2[0] = torch.tensor([0.0, -M, -30.0, -30.0])
    b2[1] = torch.tensor([0.0, -30.0, -30.0, -30.0])
    hw[0, 1] = torch.tensor([0.0, 2 * Mh])
    hb[:] = torch.tensor([Mh, 0.0])
    return P


def maze_rule(M=64.0, Mh=40.0, pert=None):
    Cx, W = 6, 9
    P = {k: torch.zeros_like(v) for k, v in init_params([0], Cx, W, 2, dtype=F64, dev="cpu").items()}
    W1, b1, W2, b2, hw, hb = (P[k][0] for k in ("W1", "b1", "W2", "b2", "hw", "hb"))
    f = Cx + 1                   # unit 0 value 1 = filled
    wins = (p15.WIN_E, p15.WIN_S, p15.WIN_W, p15.WIN_N)
    for d, w in enumerate(wins):  # h_d = relu(open_d - filled_nbr_d)
        W1[0, d, p15.WIN_C, d] = 1
        W1[0, d, w, f] = -1
    W1[0, 4, p15.WIN_C, f] = 1                        # h4 = filled centre
    W1[0, 5, p15.WIN_C, 4] = 1
    W1[0, 5, p15.WIN_C, 5] = 1                        # h5 = endpoint
    b1[0, 6] = 1                                      # h6 = 1
    coef = torch.zeros(H_)
    coef[:7] = torch.tensor([-1, -1, -1, -1, 5, -3, 1.5])
    if pert is not None:  # a pattern-detector unit h7 that flips one window class
        kind = pert
        if kind == "corridor_fill":        # E-W corridor with both neighbours unfilled -> filled
            ch = [(p15.WIN_C, 0, 1), (p15.WIN_C, 2, 1), (p15.WIN_E, f, -1), (p15.WIN_W, f, -1), (p15.WIN_C, 1, -1),
                  (p15.WIN_C, 3, -1), (p15.WIN_C, f, -1), (p15.WIN_C, 4, -1), (p15.WIN_C, 5, -1)]
            bias, sgn = -1, 1.0
        elif kind == "deadend_keep":       # dead end open only to the east, east unfilled -> never filled
            ch = [(p15.WIN_C, 0, 1), (p15.WIN_E, f, -1), (p15.WIN_C, 1, -1), (p15.WIN_C, 2, -1), (p15.WIN_C, 3, -1),
                  (p15.WIN_C, f, -1), (p15.WIN_C, 4, -1), (p15.WIN_C, 5, -1)]
            bias, sgn = 0, -1.0
        else:                              # south-north corridor with both unfilled -> filled
            ch = [(p15.WIN_C, 1, 1), (p15.WIN_C, 3, 1), (p15.WIN_S, f, -1), (p15.WIN_N, f, -1), (p15.WIN_C, 0, -1),
                  (p15.WIN_C, 2, -1), (p15.WIN_C, f, -1), (p15.WIN_C, 4, -1), (p15.WIN_C, 5, -1)]
            bias, sgn = -1, 1.0
        for w, c, v in ch:
            W1[0, 7, w, c] = v
        b1[0, 7] = bias
        coef[7] = sgn
    W2[0, 1] = M * coef
    b2[0] = torch.tensor([0.0, 0.0, -30.0, -30.0])
    b2[1] = torch.tensor([0.0, -30.0, -30.0, -30.0])
    hw[0, 1] = torch.tensor([2 * Mh, 0.0])
    hb[:] = torch.tensor([0.0, Mh])
    return P


def rollout_settle(P, task, d, D, seed=0, B=64):
    g = torch.Generator().manual_seed(seed)
    M_, N = d["y"].shape
    idx = np.arange(min(B, M_))
    x = torch.tensor(d["x"][idx], dtype=F64)[None]
    y = torch.tensor(d["y"][idx])[None]
    vm = torch.tensor(d["valid"][idx], dtype=F64)[None]
    nbr = task.nbr(N if task.kind == "prefix" else int(round(N ** 0.5)))
    px = pre_x(P, x, nbr, task.Cx)
    z = torch.zeros(1, len(idx), N, U_, K_)
    zs, pres, lgs = [z], [None], [None]
    for k in range(1, D + 1):
        pre, lg = level(P, px, z, nbr, task.Cx)
        u = torch.rand(lg.shape, generator=g, dtype=F64)
        z = onehot((lg + p15.gumbel(u)).argmax(-1), K_, F64) * vm[..., None, None]
        zs.append(z)
        pres.append(pre)
        lgs.append(lg)
    qs, _ = settle(P, None, zs, pres, lgs, y, vm, nbr, task.Cx)
    kl = []
    for k in range(1, D + 1):
        p = F.softmax(lgs[k], -1)
        klk = (qs[k] * (torch.log(qs[k] + 1e-300) - torch.log(p + 1e-300))).sum(-1)[..., 0]
        kl.append(((klk * vm).sum() / vm.sum()).item())
    return float(np.mean(kl)), (x, y, vm, nbr, zs, lgs, qs)


def one_bl_step(P, task, d, D, lr=1e-2, seed=0, sgd=False):
    """One Bl update on a batch; returns new params. sgd=False: Adam first step (about lr * sign(grad) per
    entry); sgd=True: a plain gradient step of global length lr (the direction of the Bl update)."""
    Pg = {k: v.clone().requires_grad_(True) for k, v in P.items()}
    M_, N = d["y"].shape
    b = min(64, M_)
    bt = {"x": torch.tensor(d["x"][:b], dtype=F64)[None], "y": torch.tensor(d["y"][:b])[None],
          "valid": torch.tensor(d["valid"][:b], dtype=F64)[None]}
    g = torch.Generator().manual_seed(seed)
    noise = torch.rand((D, 1, b, N, U_, K_), generator=g, dtype=F64)
    nbr = task.nbr(N if task.kind == "prefix" else int(round(N ** 0.5)))
    tsk = Task(task.kind, D, task.mod, n=getattr(task, "n", None))
    loss, _ = p15.train_step("Bl", Pg, tsk, bt, noise, None, {"tau": torch.ones(1)}, nbr)
    grads = torch.autograd.grad(loss, list(Pg.values()))
    if sgd:
        gn = torch.sqrt(sum((g ** 2).sum() for g in grads)).clamp(min=1e-300)
        return {k: (v - lr * g / gn).detach() for (k, v), g in zip(Pg.items(), grads)}
    opt = p15.SAdam(Pg, torch.tensor([lr]))
    opt.step(dict(zip(Pg.keys(), grads)))
    return {k: v.detach() for k, v in Pg.items()}


def pattern_logp(P, task, d, D, sel_fn, target_fn):
    """Mean log prob of the true next value on windows selected by sel_fn under greedy rollouts."""
    M_, N = d["y"].shape
    x = torch.tensor(d["x"], dtype=F64)[None]
    vm = torch.tensor(d["valid"], dtype=F64)[None]
    nbr = task.nbr(N if task.kind == "prefix" else int(round(N ** 0.5)))
    px = pre_x(P, x, nbr, task.Cx)
    z = torch.zeros(1, M_, N, U_, K_)
    tot, cnt = 0.0, 0
    for k in range(1, D + 1):
        _, lg = level(P, px, z, nbr, task.Cx)
        zw = gather(z[..., 0, 1:2], nbr)[..., 0]           # unit-0 value-1 bit of window cells
        sel = sel_fn(x[0], zw[0]) & (vm[0] > 0)
        tgt = target_fn(x[0], zw[0])
        lp = F.log_softmax(lg[0, :, :, 0, :2], -1)          # restricted to values 0/1
        tot += lp.gather(-1, tgt[..., None].long())[..., 0][sel].sum().item()
        cnt += int(sel.sum())
        z = onehot(lg.argmax(-1), K_, F64) * vm[..., None, None]
    return tot / max(cnt, 1), cnt


def g3():
    out = {}
    ok = True
    # ---- prefix parity
    T = Task("prefix", 16)
    big = 1024
    d = T.eval_set("test", big, 50)
    Ptrue = parity_rule()
    ge = p15.greedy_eval(Ptrue, T, d, [big], cap=big, dtype=F64)
    dtr = T.eval_set("indist", num=64)
    kl_true, _ = rollout_settle(Ptrue, T, dtr, 16)
    out["prefix_true"] = {"exact_at_L1024": float(ge["exact"][0, 0]), "mean_KL_q_p": kl_true}
    ok &= ge["exact"][0, 0] == 1.0 and kl_true < 1e-3
    pats = [(1, 1), (1, 0), (0, 1)]
    for p, (xv, zv) in enumerate(pats):
        Pp = parity_rule(flip=p)
        kl_p, _ = rollout_settle(Pp, T, dtr, 16)
        sel = lambda x, zw, xv=xv, zv=zv: (x[..., 0] == xv) & (zw[..., 0] == zv)
        tgt = lambda x, zw: (x[..., 0].long() ^ zw[..., 0].long())
        # pass criterion: the direction of one Bl update (gradient step of length 1e-3) at the gate margin;
        # reported: one Adam step (lr 1e-2) on the same rule at the soft margin M=2
        b0, cnt = pattern_logp(Pp, T, dtr, 16, sel, tgt)
        a0, _ = pattern_logp(one_bl_step(Pp, T, dtr, 16, lr=1e-3, sgd=True), T, dtr, 16, sel, tgt)
        Pp_soft = parity_rule(M=2.0, flip=p)
        before, _ = pattern_logp(Pp_soft, T, dtr, 16, sel, tgt)
        after, _ = pattern_logp(one_bl_step(Pp_soft, T, dtr, 16), T, dtr, 16, sel, tgt)
        r = {"pattern_x_zl": [xv, zv], "mean_KL_q_p": kl_p, "logp_true_before": b0, "logp_true_after": a0,
             "adam_M2_logp_true_before": before, "adam_M2_logp_true_after": after, "n_windows": cnt}
        out[f"prefix_flip{p}"] = r
        ok &= kl_p > 0.05 and a0 > b0
    # ---- maze dead-end filling
    Tm = Task("maze", 32)
    dm = Tm.eval_set("test", 59, 20)
    Pm = maze_rule()
    ge = p15.greedy_eval(Pm, Tm, dm, [59 * 59], cap=59 * 59, dtype=F64)
    dmi = Tm.eval_set("indist", num=64)
    kl_true, _ = rollout_settle(Pm, Tm, dmi, 32)
    out["maze_true"] = {"exact_at_n59": float(ge["exact"][0, 0]), "fp_frac": float(ge["fp_frac"][0]),
                        "mean_KL_q_p": kl_true}
    ok &= ge["exact"][0, 0] == 1.0 and kl_true < 1e-3
    f = 1

    def m_sel(kind):
        C, E_, S_, W_, N_ = p15.WIN_C, p15.WIN_E, p15.WIN_S, p15.WIN_W, p15.WIN_N

        def s(x, zw):
            xc = gather(x[None], nbr_cache[0])[0][..., C, :]   # centre inputs
            fc, fe, fs, fw, fn = zw[..., C], zw[..., E_], zw[..., S_], zw[..., W_], zw[..., N_]
            notend = (xc[..., 4] == 0) & (xc[..., 5] == 0) & (fc == 0)
            if kind == "corridor_fill":
                return notend & (xc[..., 0] == 1) & (xc[..., 2] == 1) & (xc[..., 1] == 0) & (xc[..., 3] == 0) & (fe == 0) & (fw == 0)
            if kind == "deadend_keep":
                return notend & (xc[..., 0] == 1) & (xc[..., 1] == 0) & (xc[..., 2] == 0) & (xc[..., 3] == 0) & (fe == 0)
            return notend & (xc[..., 1] == 1) & (xc[..., 3] == 1) & (xc[..., 0] == 0) & (xc[..., 2] == 0) & (fs == 0) & (fn == 0)
        return s

    def m_tgt(x, zw):
        xc = gather(x[None], nbr_cache[0])[0][..., p15.WIN_C, :]
        fc = zw[..., p15.WIN_C]
        cnt = sum(xc[..., d] * (1 - zw[..., w]) for d, w in enumerate((p15.WIN_E, p15.WIN_S, p15.WIN_W, p15.WIN_N)))
        end = (xc[..., 4] + xc[..., 5]) > 0
        return ((fc > 0) | (~end & (cnt <= 1))).long()

    nbr_cache = [Tm.nbr(Tm.n)]
    for kind in ("corridor_fill", "deadend_keep", "corridorNS_fill"):
        Pp = maze_rule(pert=kind)
        kl_p, _ = rollout_settle(Pp, Tm, dmi, 32)
        b0, cnt = pattern_logp(Pp, Tm, dmi, 32, m_sel(kind), m_tgt)
        a0, _ = pattern_logp(one_bl_step(Pp, Tm, dmi, 32, lr=1e-3, sgd=True), Tm, dmi, 32, m_sel(kind), m_tgt)
        Pp_soft = maze_rule(M=2.0, pert=kind)
        before, _ = pattern_logp(Pp_soft, Tm, dmi, 32, m_sel(kind), m_tgt)
        after, _ = pattern_logp(one_bl_step(Pp_soft, Tm, dmi, 32), Tm, dmi, 32, m_sel(kind), m_tgt)
        out[f"maze_{kind}"] = {"mean_KL_q_p": kl_p, "logp_true_before": b0, "logp_true_after": a0,
                               "adam_M2_logp_true_before": before, "adam_M2_logp_true_after": after,
                               "n_windows": cnt}
        ok &= kl_p > 0.05 and a0 > b0
    # reported, not pass/fail: the same true maze rule at level margin 12 / head margin 16
    kl12, (x12, y12, vm12, _, _, lgs12, qs12) = rollout_settle(maze_rule(M=24.0, Mh=16.0), Tm, dmi, 32)
    per = torch.stack([((qs12[k] * (torch.log(qs12[k] + 1e-300) - F.log_softmax(lgs12[k], -1))).sum(-1)[..., 0]
                        * vm12).sum(-1)[0] / vm12.sum(-1)[0] for k in range(1, 33)]).mean(0)
    out["maze_true_margin12"] = {"mean_KL_q_p": kl12, "per_maze_KL_top3": sorted(per.tolist(), reverse=True)[:3],
                                 "n_mazes_KL_gt_1e-3": int((per > 1e-3).sum()), "n_mazes": int(per.numel())}
    out["pass"] = bool(ok)
    out["note"] = ("True rules: prefix level margin 12, head margin 16; maze level margin 32 (M=64, margin M/2), head "
                   "margin 40. A flipped entry has the same margin with the opposite sign. Run 1 (margins 10/8, maze "
                   "flips at margin 40) failed: when the level margin exceeds the head margin the settlement cannot "
                   "see a flipped entry. Run 2 (maze margin 12, head 16) failed: the true maze rule was not a "
                   "settlement fixed point (mean KL 0.038, all of it in 2/64 mazes with long late-filling dead-end "
                   "regions): one-site loopy messages add up over neighbours and levels and amplify the e^-12 "
                   "sampling seeds until q fills late cells early (see maze_true_margin12); at margin 32 the seeds "
                   "are e^-32 and KL is 6e-15. Run 2's one-step check (one Adam step at M=2) also moved deadend_keep "
                   "the wrong way; Adam's first step is about lr*sign(grad) per entry, not the Bl direction, so the "
                   "pass criterion is now the direction of one Bl gradient step (length 1e-3) at the gate margin, "
                   "and the Adam step at M=2 is reported next to it. Batch 64.")
    save("G3", out)
    return ok


# ------------------------------------------------------------------ G4 FLOP counter
def g4():
    from torch.utils._python_dispatch import TorchDispatchMode
    aten = torch.ops.aten
    ZERO = {"view", "_unsafe_view", "reshape", "expand", "permute", "transpose", "t", "clone", "copy_", "cat", "stack",
            "index", "index_put", "index_put_", "gather", "select", "slice", "unsqueeze", "squeeze", "detach",
            "_to_copy", "new_zeros", "zeros", "zeros_like", "empty", "empty_like", "fill_", "full", "lift_fresh",
            "alias", "contiguous", "as_strided", "split", "unbind", "narrow", "repeat", "new_empty", "ones_like",
            "scalar_tensor", "arange", "full_like", "one_hot", "_reshape_alias", "clone", "diagonal", "embedding",
            "select_backward", "slice_backward", "new_empty_strided", "empty_strided", "index_select",
            "masked_fill", "masked_fill_", "set_", "resize_", "zero_", "view_as", "unsqueeze_", "squeeze_",
            "expand_as", "rand", "uniform_", "_local_scalar_dense", "item", "is_nonzero", "randperm", "ones",
            "new_ones", "lift", "sym_size", "sym_stride", "sym_numel", "_foreach_add_", "unfold", "movedim"}
    RED = {"sum", "mean", "amax", "max", "argmax", "amin", "min", "logsumexp", "prod", "all", "any"}

    class Counter(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.f = 0
            self.by = {}

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            name = func.overloadpacket.__name__
            fl = 0
            if name in ("mm", "bmm"):
                a, b = args[0], args[1]
                fl = 2 * a.numel() * b.shape[-1]
            elif name in ("addmm", "baddbmm"):
                a, b = args[1], args[2]
                fl = 2 * a.numel() * b.shape[-1] + out.numel()
            elif name in ("_softmax", "_log_softmax", "_softmax_backward_data", "_log_softmax_backward_data"):
                fl = 4 * out.numel()
            elif name == "logsumexp":
                fl = 4 * args[0].numel()
            elif name in ("index_add", "index_add_", "scatter_add", "scatter_add_", "index_put_", "index_put") and \
                    name.startswith("index_add"):
                fl = args[3].numel()
            elif name in RED:
                fl = args[0].numel()
            elif name in ZERO:
                fl = 0
            elif isinstance(out, torch.Tensor):
                fl = out.numel()
            self.f += fl
            self.by[name] = self.by.get(name, 0) + fl
            return out

    rows = []
    ok = True
    for kind, D in (("prefix", 16), ("maze", 8)):
        T = Task(kind, D)
        N = D if kind == "prefix" else T.n ** 2
        nbr = T.nbr(N if kind == "prefix" else T.n)
        for arm in ("Bl", "Bl-ac", "Bl-local", "O", "C", "A", "A-prog", "A-ds", "A-1step", "A-soft-prog"):
            P = init_params([0], T.Cx, T.W, T.Cy, dtype=F64, dev="cpu")
            rng = np.random.default_rng(0)
            d = T.train_batch(rng, 32)
            bt = p15.stack_batches([d], "cpu", F64)
            if arm == "O":
                bt["wave"] = [torch.tensor(np.stack([T.wave(d, k)])) for k in range(D + 1)]
            noise = torch.rand((D, 1, 32, N, U_, K_), dtype=F64)
            sched = (D // 3, D // 2) if arm in ("A-prog", "A-soft-prog") else None
            alpha = 1.0 if kind == "prefix" else 0.01
            cfg = {"tau": torch.ones(1), "eps": 0.05, "alpha": alpha, "gen_perm": torch.Generator().manual_seed(0)}
            c = Counter()
            with c:
                loss, _ = p15.train_step(arm, P, T, bt, noise, sched, cfg, nbr)
                names = [k for k in P if not (arm == "C" and k in ("W1", "b1", "W2", "b2"))]
                torch.autograd.grad(loss, [P[k] for k in names], allow_unused=True)
            an = p15.flops(arm, T, 32, N, sched, alpha)
            rel = an / c.f - 1
            top = sorted(c.by.items(), key=lambda kv: -kv[1])[:6]
            rows.append({"task": kind, "D": D, "arm": arm, "analytic": an, "counted": c.f, "rel_err": rel,
                         "top_ops": top})
            ok &= abs(rel) <= 0.10
    save("G4", {"pass": bool(ok), "rows": rows})
    return ok


# ------------------------------------------------------------------ G6 stacking
def g6():
    import tempfile
    tmp = tempfile.mkdtemp(prefix="p15g6_", dir="/tmp")
    p15.RESDIR = os.path.join(tmp, "r")
    p15.CKDIR = os.path.join(tmp, "c")
    errs = {}
    for arm, kind in (("Bl", "prefix"), ("A-prog", "prefix"), ("Bl", "maze"), ("A", "maze")):
        base = {"arm": arm, "kind": kind, "D": 8, "configs": [{"lr": 3e-3, "tau": 1.0}], "max_steps": 50,
                "final": False, "no_curve": True, "f64": True, "frac": 1.0, "force": True}
        p15.run_job({**base, "id": f"g6_{arm}_{kind}_stack", "seeds": [0, 1, 2]}, log=lambda m: None)
        p15.run_job({**base, "id": f"g6_{arm}_{kind}_solo", "seeds": [1]}, log=lambda m: None)
        a = torch.load(os.path.join(p15.CKDIR, f"g6_{arm}_{kind}_stack.pt"), weights_only=False)["P"]
        b = torch.load(os.path.join(p15.CKDIR, f"g6_{arm}_{kind}_solo.pt"), weights_only=False)["P"]
        errs[f"{arm}/{kind}"] = max((a[k][1] - b[k][0]).abs().max().item() for k in a)
    ok = max(errs.values()) < 1e-5
    save("G6", {"pass": bool(ok), "max_abs_param_diff_after_50_steps": errs,
                "setup": "stack seeds [0,1,2] vs seed 1 alone, float64 CPU, D=8"})
    return ok


if __name__ == "__main__":
    for gname in sys.argv[1:]:
        t = time.time()
        r = {"G1": g1, "G2": g2, "G2b": g2b, "G3": g3, "G4": g4, "G6": g6}[gname]()
        print(gname, "PASS" if r else "FAIL", f"{time.time() - t:.0f}s", flush=True)
