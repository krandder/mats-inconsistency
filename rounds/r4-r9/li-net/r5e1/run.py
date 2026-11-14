"""R5-E1 (E-INFO) pipeline for one seed. Spec: /home/agent/workspaces/mats/outbox/experiments-round-5-spec.md section 1.

  1. theta_real: any-order masked BCE on real board realizations (fresh boards every step; reveal sets = random subsets, 2x3/3x2
     patch transitions, row/column line scans, root). Fixed step count, final checkpoint (no selection).
  2. Part A (inference-time search): calibration split -> per-half rollout pools + an unskewed K = 128 reference (the plain
     mixture of 128 rollouts) -> theta_trust on real excess surprise -> key head W_K + attn_L3 jointly on skewed bags
     (p_NW ~ Beta(0.5, 0.5)) with a 36-cell BCE to the reference -> ataraxos_mmd eta and kc_dedup u fitted on the same bags.
     Test split (500 boards): preference-driven bags under U_NW / U_SE (80/20), unskewed bags, controlled same-patch bags.
  3. Part B (LIR-RL): entropy-regularised GRPO of pi_act(c | mu, U) under three parameter masks (fused, tokmask, lir).
Exact Bayes is computed for the test split, the gate and the curves only. Training targets: realised boards (theta_real,
theta_trust, the truth loss in RL), theta_real's own rollouts (the K = 128 reference) and realised rewards (RL).
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn.functional as F

import env6
import rollouts as RO
import wm

STREAM_CAL, STREAM_TEST, STREAM_VAL, STREAM_MON = 1, 2, 3, 4      # fixed numpy board streams (identical for every seed)
AGGS = ("mix", "mmd", "kc", "attn")
ARMS = ("fused", "tokmask", "lir", "truthonly")     # truthonly: descriptive control (truth loss, no RL), not in any rule


@dataclass
class Cfg:
    seed: int = 101
    out: str = "/tmp/r5e1/dev"
    device: str = "cuda"
    threads: int = 4
    # theta_real
    d: int = 128
    layers: int = 4
    heads: int = 4
    ff: int = 384
    train_steps: int = 40000
    batch: int = 512
    lr: float = 1e-3
    warmup: int = 1000
    log_every: int = 250
    curve_every: int = 2000
    n_val: int = 500
    # Part A
    n_cal: int = 4000
    pool: int = 48               # rollouts per half per calibration board
    n_ref: int = 128             # unskewed reference (64 NW-half + 64 SE-half draws)
    gamma_voi: float = 4.0
    p_pref: float = 0.8
    K: int = 24
    kmin: int = 8
    kmax: int = 32
    trust_steps: int = 3000
    agg_steps: int = 6000
    agg_batch: int = 256
    fit_steps: int = 1500
    n_test: int = 500
    n_clu: int = 16
    n_bg: int = 8
    chunk: int = 4096
    # Part B
    rl_steps: int = 400
    groups: int = 16
    gsize: int = 8
    lr_rl: float = 1e-4
    ent_bonus: float = 0.02
    alpha: float = 1.0
    truth_batch: int = 256
    lam_syc: float = 0.5
    rl_eval_every: int = 100
    skip_b: int = 0
    ladder: str = "100,250,500,1000"   # secondary (descriptive) gap ladder: Part A also at these early theta_real steps


class Log:
    def __init__(self, out):
        self.f = open(os.path.join(out, "log.txt"), "a")
        self.t0 = time.time()

    def __call__(self, *a):
        s = f"[{time.strftime('%H:%M:%S', time.gmtime())} +{time.time() - self.t0:7.1f}s] " + " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()


def r5(x, nd=5):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().double().numpy()
    return np.round(np.asarray(x, dtype=np.float64), nd).tolist()


def dump(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, default=float, separators=(",", ":"))
    os.replace(tmp, path)


def tb(stream, n, dev):
    return env6.boards_to_torch(env6.make_boards(stream, n), dev)


# ================================================================================================ theta_real
def truth_loss(model, B, rev):
    lg = model(B["ev"], B["note"], rev, B["occ"].long())[0]
    m = ~rev
    return F.binary_cross_entropy_with_logits(lg[m], B["occ"][m].float())


def train_world_model(cfg, dev, log, Pt, Pf):
    torch.manual_seed(cfg.seed)
    model = wm.WorldModel(cfg.d, cfg.layers, cfg.heads, cfg.ff).to(dev)
    log("theta_real params", model.n_params())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg.warmup) * (
        0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, cfg.train_steps))))))
    gen = torch.Generator(device=dev).manual_seed(cfg.seed * 1000 + 11)
    val = tb(STREAM_VAL, cfg.n_val, dev)
    mon = tb(STREAM_MON, cfg.n_val, dev)
    mon_bayes = env6.bayes_marginals(mon["ev"], mon["note"], Pf).float()
    curve, run = [], []
    ladder = {int(x) for x in cfg.ladder.split(",") if x.strip()}
    snaps = {}
    t0 = time.time()
    for s in range(cfg.train_steps + 1):
        if s % cfg.curve_every == 0 or s == cfg.train_steps:
            model.eval()
            with torch.no_grad():
                lv = RO.forward_chunks(model, val["ev"], val["note"])
                vb = float(F.binary_cross_entropy_with_logits(lv, val["occ"].float()))
                lm = RO.forward_chunks(model, mon["ev"], mon["note"])
                kl = float(wm.bern_kl(mon_bayes, torch.sigmoid(lm)).mean())   # descriptive curve only (never used)
            curve.append({"step": s, "val_root_bce_vs_occ": vb, "monitor_root_kl_vs_bayes": kl,
                          "train_loss_recent": float(np.mean(run[-cfg.log_every:])) if run else None,
                          "sec": time.time() - t0})
            log(f"wm step {s} val_bce {vb:.4f} mon_kl {kl:.4f} train {curve[-1]['train_loss_recent']} t {time.time() - t0:.0f}s")
            model.train()
        if s in ladder:
            snaps[s] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if s == cfg.train_steps:
            break
        B = env6.torch_boards(cfg.batch, gen, Pt)
        rev = env6.reveal_masks(cfg.batch, gen, dev)
        loss = truth_loss(model, B, rev)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run.append(float(loss))
    model.eval()
    return model, curve, time.time() - t0, snaps


# ================================================================================================ Part A helpers
def arcsine(B, gen, dev):
    """p ~ Beta(0.5, 0.5) by inverse CDF: p = sin^2(pi u / 2)."""
    return torch.sin(0.5 * math.pi * torch.rand(B, device=dev, generator=gen)) ** 2


def sample_bag_index(nb, Pn, gen, dev, kmin, kmax):
    """Skewed training bags drawn from per-half pools of size Pn: K ~ U{kmin..kmax}, p_NW ~ Beta(.5,.5),
    n_NW ~ Bin(K, p_NW), without replacement within each half. Returns slot index into [2 Pn] and valid [nb, kmax]."""
    K = torch.randint(kmin, kmax + 1, (nb,), device=dev, generator=gen)
    p = arcsine(nb, gen, dev)
    j = torch.arange(kmax, device=dev)[None]
    valid = j < K[:, None]
    nnw = ((torch.rand(nb, kmax, device=dev, generator=gen) < p[:, None]) & valid).sum(1)
    pn = torch.rand(nb, Pn, device=dev, generator=gen).argsort(1)[:, :kmax]
    ps = torch.rand(nb, Pn, device=dev, generator=gen).argsort(1)[:, :kmax]
    isnw = j < nnw[:, None]
    sidx = (j - nnw[:, None]).clamp(min=0, max=kmax - 1)
    idx = torch.where(isnw, pn, Pn + ps.gather(1, sidx))           # slot j: j-th NW draw, else (j - n_NW)-th SE draw
    return idx, valid, nnw.float() / K.float()


def gather_rows(x, b, idx):
    """x [N, P, ...], b [nb], idx [nb, k] -> [nb, k, ...]"""
    return x[b[:, None], idx]


class Aggs:
    def __init__(self, keyh, att, eta, u):
        self.keyh, self.att, self.eta, self.u = keyh, att, eta, u

    @torch.no_grad()
    def all(self, l0, lr, dl, beta, hbar, valid):
        k = self.keyh(hbar)
        out = {"mix": wm.agg_mix(lr, valid), "mmd": wm.agg_mmd(l0, dl, beta, valid, self.eta),
               "kc": wm.agg_kc(l0, dl, beta, k, valid, self.u)}
        la, w = self.att(l0, dl, beta, k, valid)
        out["attn"] = la
        return out, w, k


def fit_part_a(model, cfg, dev, log, gen):
    """Calibration split: pools, reference, theta_trust, key head + attn_L3, mmd eta, kc u."""
    t0 = time.time()
    cal = tb(STREAM_CAL, cfg.n_cal, dev)
    l0 = RO.forward_chunks(model, cal["ev"], cal["note"])
    one, zero = torch.ones(cfg.n_cal, device=dev), torch.zeros(cfg.n_cal, device=dev)
    pnw, _ = RO.sample_patches(l0, cfg.pool, one, gen, cfg.gamma_voi)
    pse, _ = RO.sample_patches(l0, cfg.pool, zero, gen, cfg.gamma_voi)
    patch = torch.cat([pnw, pse], 1)
    tau = RO.draw_tau(patch.shape, gen, dev)
    pool = RO.bag_rollouts(model, cal, l0, patch, tau, gen, cfg.chunk)
    log(f"cal pools: {cfg.n_cal} boards x {2 * cfg.pool} rollouts, {time.time() - t0:.0f}s")
    h = cfg.n_ref // 2
    rnw, _ = RO.sample_patches(l0, h, one, gen, cfg.gamma_voi)
    rse, _ = RO.sample_patches(l0, cfg.n_ref - h, zero, gen, cfg.gamma_voi)
    rpatch = torch.cat([rnw, rse], 1)
    ref = torch.zeros(cfg.n_cal, 36, device=dev)
    for a in range(0, cfg.n_cal, 500):
        sl = slice(a, a + 500)
        sub = {k: v[sl] for k, v in cal.items()}
        rr = RO.bag_rollouts(model, sub, l0[sl], rpatch[sl], RO.draw_tau(rpatch[sl].shape, gen, dev), gen, cfg.chunk)
        ref[sl] = torch.sigmoid(rr["lr"]).mean(1)
        del rr
    log(f"cal reference K={cfg.n_ref} done, {time.time() - t0:.0f}s")
    ntr = int(0.8 * cfg.n_cal)                       # boards [0, ntr) fit; [ntr, n_cal) held-out monitor (logged only)
    # ---- theta_trust on real excess surprise (patch cells vs the realised board)
    NP = 2 * cfg.pool
    X = RO.trust_features({k: v.reshape(cfg.n_cal * NP, *v.shape[2:]) for k, v in pool.items()}, tau.reshape(-1))
    occ_rows = cal["occ"].repeat_interleave(NP, 0)
    Y = RO.es_target({"lr": pool["lr"].reshape(-1, 36)}, patch.reshape(-1), occ_rows)
    torch.manual_seed(cfg.seed * 1000 + 21)
    trust = wm.TrustNet(X.shape[1]).to(dev)
    opt = torch.optim.Adam(trust.parameters(), lr=1e-3)
    ntrr = ntr * NP
    tl = []
    for s in range(cfg.trust_steps):
        ib = torch.randint(ntrr, (1024,), device=dev, generator=gen)
        loss = F.mse_loss(trust(X[ib]), Y[ib])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        tl.append(float(loss))
    trust.eval()
    with torch.no_grad():
        ehat = torch.cat([trust(X[a:a + 65536]) for a in range(0, X.shape[0], 65536)])
        tmon = float(F.mse_loss(ehat[ntrr:], Y[ntrr:]))
    beta = wm.beta_from(ehat).view(cfg.n_cal, NP)
    log(f"theta_trust: train mse(last 200) {np.mean(tl[-200:]):.4f} heldout mse {tmon:.4f} beta mean {float(beta.mean()):.3f}"
        f" (tau .3: {float(beta[tau == 0.3].mean()):.3f}, 1.5: {float(beta[tau == 1.5].mean()):.3f})")
    # ---- key head + attn_L3, jointly, on skewed bags vs the reference (36-cell BCE)
    torch.manual_seed(cfg.seed * 1000 + 31)
    keyh = wm.KeyHead(cfg.d).to(dev)
    att = wm.AttnL3().to(dev)
    opt = torch.optim.Adam(list(keyh.parameters()) + list(att.parameters()), lr=1e-3)
    hist = []

    def bag(nb, lo, hi):
        b = torch.randint(lo, hi, (nb,), device=dev, generator=gen)
        idx, valid, frac = sample_bag_index(nb, cfg.pool, gen, dev, cfg.kmin, cfg.kmax)
        return (b, l0[b], gather_rows(pool["lr"], b, idx), gather_rows(pool["dl"], b, idx), gather_rows(beta, b, idx),
                gather_rows(pool["hbar"], b, idx), valid, ref[b])

    def mon_loss(fn, n=4):
        with torch.no_grad():
            g2 = torch.Generator(device=dev).manual_seed(777)
            ls = []
            for _ in range(n):
                b = torch.randint(ntr, cfg.n_cal, (cfg.agg_batch,), device=dev, generator=g2)
                idx, valid, _ = sample_bag_index(cfg.agg_batch, cfg.pool, g2, dev, cfg.kmin, cfg.kmax)
                args = (l0[b], gather_rows(pool["lr"], b, idx), gather_rows(pool["dl"], b, idx), gather_rows(beta, b, idx),
                        gather_rows(pool["hbar"], b, idx), valid)
                ls.append(float(F.binary_cross_entropy(torch.sigmoid(fn(*args)).clamp(1e-6, 1 - 1e-6), ref[b])))
            return float(np.mean(ls))

    for s in range(cfg.agg_steps):
        b, l0b, lrb, dlb, bb, hb, valid, rb = bag(cfg.agg_batch, 0, ntr)
        ls, _ = att(l0b, dlb, bb, keyh(hb), valid)
        loss = F.binary_cross_entropy_with_logits(ls, rb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(keyh.parameters()) + list(att.parameters()), 1.0)
        opt.step()
        if s % 500 == 0 or s == cfg.agg_steps - 1:
            hist.append({"step": s, "train_bce": float(loss)})
    att.eval()
    for p in list(att.parameters()) + list(keyh.parameters()):
        p.requires_grad_(False)

    def fit_scalar(fn, init):
        lp = torch.tensor(math.log(init), device=dev, requires_grad=True)
        o = torch.optim.Adam([lp], lr=0.02)
        for s in range(cfg.fit_steps):
            b, l0b, lrb, dlb, bb, hb, valid, rb = bag(cfg.agg_batch, 0, ntr)
            loss = F.binary_cross_entropy_with_logits(fn(l0b, dlb, bb, hb, valid, lp.exp()), rb)
            o.zero_grad(set_to_none=True)
            loss.backward()
            o.step()
        return float(lp.exp())

    bmean = float(beta.mean())
    eta = fit_scalar(lambda l0b, dlb, bb, hb, valid, e: wm.agg_mmd(l0b, dlb, bb, valid, e), 1.0 / (cfg.K * bmean))
    u = fit_scalar(lambda l0b, dlb, bb, hb, valid, e: wm.agg_kc(l0b, dlb, bb, keyh(hb), valid, e), 1.0 / bmean)
    A = Aggs(keyh, att, eta, u)
    monitor = {
        "attn": mon_loss(lambda l0b, lrb, dlb, bb, hb, v: att(l0b, dlb, bb, keyh(hb), v)[0]),
        "mmd": mon_loss(lambda l0b, lrb, dlb, bb, hb, v: wm.agg_mmd(l0b, dlb, bb, v, eta)),
        "kc": mon_loss(lambda l0b, lrb, dlb, bb, hb, v: wm.agg_kc(l0b, dlb, bb, keyh(hb), v, u)),
        "mix": mon_loss(lambda l0b, lrb, dlb, bb, hb, v: wm.agg_mix(lrb, v)),
        "p0": mon_loss(lambda l0b, lrb, dlb, bb, hb, v: l0b),
    }
    log(f"aggregators: eta {eta:.4g} u {u:.4g}; held-out cal bag BCE to reference: " +
        " ".join(f"{k} {v:.4f}" for k, v in monitor.items()) + f"; {time.time() - t0:.0f}s")
    info = {"eta": eta, "u": u, "beta_mean": bmean, "beta_mean_tau03": float(beta[tau == 0.3].mean()),
            "beta_mean_tau15": float(beta[tau == 1.5].mean()), "trust_train_mse": float(np.mean(tl[-200:])),
            "trust_heldout_mse": tmon, "es_mean": float(Y.mean()), "agg_hist": hist, "heldout_bag_bce_to_ref": monitor,
            "cal_ref_kl_note": "reference = plain mixture of 128 unskewed rollouts (64 NW-half + 64 SE-half draws)",
            "sec": time.time() - t0}
    return trust, A, info


def bag_eval(model, trust, A, T, l0, patch, tau, gen, cfg, valid=None):
    ro = RO.bag_rollouts(model, T, l0, patch, tau, gen, cfg.chunk)
    nb, n = patch.shape
    with torch.no_grad():
        X = RO.trust_features({k: v.reshape(nb * n, *v.shape[2:]) for k, v in ro.items()}, tau.reshape(-1))
        beta = wm.beta_from(trust(X)).view(nb, n)
    if valid is None:
        valid = torch.ones(nb, n, dtype=torch.bool, device=l0.device)
    out, w, k = A.all(l0, ro["lr"], ro["dl"], beta, ro["hbar"], valid)
    return out, w, k, beta, ro


def test_part_a(model, trust, A, cfg, dev, log, gen, Pf):
    t0 = time.time()
    T = tb(STREAM_TEST, cfg.n_test, dev)
    n = cfg.n_test
    Pstar = env6.bayes_marginals(T["ev"], T["note"], Pf).float()          # EVALUATION ONLY
    l0 = RO.forward_chunks(model, T["ev"], T["note"])
    mu0 = torch.sigmoid(l0)
    res = {"n_test": n}
    raw = {"pstar": r5(Pstar), "mu0": r5(mu0), "occ": T["occ"].int().tolist(), "note": T["note"].tolist(),
           "note_true": T["note_true"].int().tolist()}
    g = RO.geo(dev)
    # ---- preference-driven bags (U_NW: 80% NW, U_SE: 80% SE) and unskewed bags (50/50)
    bags = {}
    for name, p in (("NW", cfg.p_pref), ("SE", 1 - cfg.p_pref), ("UN", 0.5)):
        patch, half = RO.sample_patches(l0, cfg.K, torch.full((n,), p, device=dev), gen, cfg.gamma_voi)
        tau = RO.draw_tau(patch.shape, gen, dev)
        out, w, k, beta, ro = bag_eval(model, trust, A, T, l0, patch, tau, gen, cfg)
        bags[name] = {"out": out, "w": w, "k": k, "beta": beta, "patch": patch, "half": half, "tau": tau}
        raw[f"bag_{name}"] = {"patch": patch.tolist(), "half_nw": half.int().tolist(), "tau": r5(tau, 2),
                              "beta": r5(beta, 4), "keys": r5(k, 4), **{a: r5(torch.sigmoid(out[a])) for a in AGGS}}
        res[f"frac_nw_{name}"] = float(half.float().mean())
    kl = lambda m: wm.bern_kl(Pstar, m)
    # descriptive (deviation 2, 03:2xZ): unskewed K = 24 bag with every rollout at tau = 1 (no temperature bias), to
    # separate a true multi-step gap from the bias of sampling at tau 0.3 / 1.5
    p1, _ = RO.sample_patches(l0, cfg.K, torch.full((n,), 0.5, device=dev), gen, cfg.gamma_voi)
    o1, _, _, _, _ = bag_eval(model, trust, A, T, l0, p1, torch.ones(p1.shape, device=dev), gen, cfg)
    for a in AGGS:
        res[f"kl_{a}_UN_tau1"] = float(kl(torch.sigmoid(o1[a])).mean())
    res["kl_p0"] = float(kl(mu0).mean())
    res["corr_mu0_bayes"] = float(np.corrcoef(mu0.cpu().numpy().ravel(), Pstar.cpu().numpy().ravel())[0, 1])
    for a in AGGS:
        m = {b: torch.sigmoid(bags[b]["out"][a]) for b in bags}
        res[f"kl_{a}_UN"] = float(kl(m["UN"]).mean())
        res[f"kl_{a}_pref"] = float(0.5 * (kl(m["NW"]).mean() + kl(m["SE"]).mean()))
        res[f"kl_{a}_NW"] = float(kl(m["NW"]).mean())
        res[f"kl_{a}_SE"] = float(kl(m["SE"]).mean())
        res[f"jsd_{a}"] = float(wm.bern_jsd(m["NW"], m["SE"]).mean())
        res[f"jsd_{a}_nwcells"] = float(wm.bern_jsd(m["NW"], m["SE"])[:, :18].mean())
        res[f"jsd_{a}_secells"] = float(wm.bern_jsd(m["NW"], m["SE"])[:, 18:].mean())
        res[f"mse_{a}_pref"] = float(0.5 * (((m["NW"] - Pstar) ** 2).mean() + ((m["SE"] - Pstar) ** 2).mean()))
    res["mse_p0"] = float(((mu0 - Pstar) ** 2).mean())
    # attn weights by half (does attn re-weight the over-sampled half?)
    for b in ("NW", "SE"):
        w = bags[b]["w"].mean(2)                                         # [n, K] mean over cells
        hf = bags[b]["half"]
        res[f"attn_wsum_nwrollouts_{b}"] = float((w * hf.float()).sum(1).mean())
        res[f"attn_wsum_serollouts_{b}"] = float((w * (~hf).float()).sum(1).mean())
    # ---- D2.1 keys vs true overlap (all within-bag pairs of the NW, SE and UN bags)
    cos_all, ov_all, same_patch, across = [], [], [], []
    iu = torch.triu_indices(cfg.K, cfg.K, 1, device=dev)
    for b in ("NW", "SE", "UN"):
        k, p = bags[b]["k"], bags[b]["patch"]
        cs = (k @ k.transpose(1, 2))[:, iu[0], iu[1]]
        ov = g["overlap"][p[:, iu[0]], p[:, iu[1]]].float() / 6
        s1, s2 = g["sector"][p[:, iu[0]]], g["sector"][p[:, iu[1]]]
        cos_all.append(cs.reshape(-1))
        ov_all.append(ov.reshape(-1))
        same_patch.append(cs[p[:, iu[0]] == p[:, iu[1]]])
        across.append(cs[((s1 == 0) & (s2 == 1)) | ((s1 == 1) & (s2 == 0))])
    cs, ov = torch.cat(cos_all), torch.cat(ov_all)
    res["key_pearson_overlap"] = float(np.corrcoef(cs.cpu().numpy(), ov.cpu().numpy())[0, 1])
    res["key_cos_same_patch"] = float(torch.cat(same_patch).mean())
    res["key_cos_across_sector"] = float(torch.cat(across).mean())
    res["key_cos_gap"] = res["key_cos_same_patch"] - res["key_cos_across_sector"]
    res["key_cos_by_overlap"] = {str(o): float(cs[(ov * 6).round() == o].mean()) for o in (0, 1, 2, 3, 4, 6)}
    res["key_n_pairs"] = int(cs.numel())
    # ---- D2.2 controlled bags: N in {1,2,4,8,16} same-patch rollouts at tau in {0.3, 1.5} + 8 background rollouts on
    # patches disjoint from the cluster patch (shared across N and tau)
    astar, _ = RO.sample_patches(l0, 1, torch.full((n,), 0.5, device=dev), gen, cfg.gamma_voi)        # [n, 1]
    excl = g["overlap"][astar[:, 0]] > 0                                                              # [n, 40]
    bgp, _ = RO.sample_patches(l0, cfg.n_bg, torch.full((n,), 0.5, device=dev), gen, cfg.gamma_voi, exclude=excl)
    bgt = RO.draw_tau(bgp.shape, gen, dev)
    pm = g["pmask"][astar[:, 0]].bool()                                                               # [n, 36]
    raw["controlled"] = {"astar": astar[:, 0].tolist(), "bg_patch": bgp.tolist(), "bg_tau": r5(bgt, 2)}
    ctrl = {}
    for tv in (0.3, 1.5):
        cp = astar.expand(n, cfg.n_clu)
        ct = torch.full(cp.shape, tv, device=dev)
        patch = torch.cat([cp, bgp], 1)
        tau = torch.cat([ct, bgt], 1)
        ro = RO.bag_rollouts(model, T, l0, patch, tau, gen, cfg.chunk)
        nb, nn_ = patch.shape
        with torch.no_grad():
            X = RO.trust_features({k: v.reshape(nb * nn_, *v.shape[2:]) for k, v in ro.items()}, tau.reshape(-1))
            beta = wm.beta_from(trust(X)).view(nb, nn_)
        for N in (1, 2, 4, 8, 16):
            valid = torch.zeros(n, nn_, dtype=torch.bool, device=dev)
            valid[:, :N] = True
            valid[:, cfg.n_clu:] = True
            out, w, k = A.all(l0, ro["lr"], ro["dl"], beta, ro["hbar"], valid)
            key = f"t{tv}_N{N}"
            Wn = (w[:, :N] * pm[:, None, :].float()).sum(2) / 6.0                                   # [n, N] patch-mean
            row = {"W_attn": float(Wn.sum(1).mean()),
                   "W_mmd": float((A.eta * beta[:, :N]).sum(1).mean()),
                   "W_kc": float((A.u * beta[:, :N] / wm.crowding(k, valid)[:, :N]).sum(1).mean()),
                   "W_mix_share": N / (N + cfg.n_bg)}
            for a in AGGS:
                m = torch.sigmoid(out[a])
                row[f"mse_{a}"] = float(((m - Pstar) ** 2).mean())
                row[f"mse_{a}_patch"] = float((((m - Pstar) ** 2) * pm.float()).sum(1).div(6).mean())
                row[f"kl_{a}"] = float(wm.bern_kl(Pstar, m).mean())
            row["beta_cluster_mean"] = float(beta[:, :N].mean())
            ctrl[key] = row
            raw["controlled"][key] = {a: r5(torch.sigmoid(out[a])) for a in AGGS}
            raw["controlled"][key]["W_attn"] = r5(Wn.sum(1))
        log(f"controlled tau={tv}: " + " ".join(f"N{N}: W {ctrl[f't{tv}_N{N}']['W_attn']:.3f}" for N in (1, 2, 4, 8, 16)))
    # baseline: background-only bag (N = 0)
    valid = torch.zeros(n, cfg.n_clu + cfg.n_bg, dtype=torch.bool, device=dev)
    valid[:, cfg.n_clu:] = True
    out, _, _ = A.all(l0, ro["lr"], ro["dl"], beta, ro["hbar"], valid)
    for a in AGGS:
        ctrl[f"bg_only_mse_{a}"] = float(((torch.sigmoid(out[a]) - Pstar) ** 2).mean())
    res["controlled"] = ctrl
    # ---- descriptive: test K=128 unskewed reference (the aggregators' training target) and attn vs it
    h = cfg.n_ref // 2
    rnw, _ = RO.sample_patches(l0, h, torch.ones(n, device=dev), gen, cfg.gamma_voi)
    rse, _ = RO.sample_patches(l0, cfg.n_ref - h, torch.zeros(n, device=dev), gen, cfg.gamma_voi)
    rp = torch.cat([rnw, rse], 1)
    rr = RO.bag_rollouts(model, T, l0, rp, RO.draw_tau(rp.shape, gen, dev), gen, cfg.chunk)
    refT = torch.sigmoid(rr["lr"]).mean(1)
    del rr
    res["kl_ref128"] = float(kl(refT).mean())
    for a in AGGS:
        res[f"mse_{a}_UN_vs_ref128"] = float(((torch.sigmoid(bags["UN"]["out"][a]) - refT) ** 2).mean())
    raw["ref128"] = r5(refT)
    res["sec"] = time.time() - t0
    log("part A test: " + json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in res.items()
                                      if not isinstance(v, dict)}))
    return res, raw, T, Pstar


# ================================================================================================ Part B
def delta_desire(mu, P, note, lam, K=4):
    """R4-E3a definition: mean over boards and U in {U+, U-} of <mu - P*, grad_X U>, grad_X U = lam e_note + s_U 1[C*_U]/K,
    C*_U = the K cells with the highest P* (U+) / lowest P* (U-)."""
    d = mu - P
    syc = lam * d.gather(1, note[:, None]).squeeze(1)
    top = P.argsort(dim=1, descending=True, stable=True)[:, :K]
    bot = P.argsort(dim=1, descending=False, stable=True)[:, :K]
    env = 0.5 * (d.gather(1, top).mean(1) - d.gather(1, bot).mean(1))
    return float((syc + env).mean()), float(syc.mean()), float(env.mean())


@torch.no_grad()
def eval_b(net, act, T, Pstar, cfg, mu_pre=None):
    n = T["ev"].shape[0]
    z = torch.zeros(n, 36, dtype=torch.bool, device=Pstar.device)
    l, h, _ = net(T["ev"], T["note"], z, z.long())
    l2, _, _ = net(T["ev"], T["note"], z, z.long())          # belief pass repeated "under U-": the belief never sees U
    mu = torch.sigmoid(l.float())
    occ = T["occ"]
    out = {"swap_belief_change_max": float((torch.sigmoid(l2.float()) - mu).abs().max())}
    cells = {}
    for name, u in (("seek", 1.0), ("avoid", -1.0)):
        lp = act(h[:, :36].float(), l.float(), torch.full((n,), u, device=Pstar.device))
        pi = lp.exp()
        env = occ.float() if u > 0 else (~occ).float()
        c = lp.argmax(1)
        cells[name] = c
        out[f"env_reward_{name}"] = float((pi * env).sum(1).mean())
        out[f"env_reward_{name}_greedy"] = float(env.gather(1, c[:, None]).mean())
        out[f"entropy_{name}"] = float(-(pi * lp).sum(1).mean())
        out[f"pstar_reward_{name}"] = float((pi * (Pstar if u > 0 else 1 - Pstar)).sum(1).mean())
    out["env_reward"] = 0.5 * (out["env_reward_seek"] + out["env_reward_avoid"])
    out["env_reward_greedy"] = 0.5 * (out["env_reward_seek_greedy"] + out["env_reward_avoid_greedy"])
    cs, ca = cells["seek"], cells["avoid"]
    out["same_cell_rate"] = float((cs == ca).float().mean())
    out["faith_flip_rate"] = float((Pstar.gather(1, cs[:, None]) > Pstar.gather(1, ca[:, None])).float().mean())
    out["faith_flip_rate_own_belief"] = float((mu.gather(1, cs[:, None]) > mu.gather(1, ca[:, None])).float().mean())
    fn = ~T["note_true"]
    mn = mu.gather(1, T["note"][:, None]).squeeze(1)
    out["syc_false_note"] = float((mn[fn] >= 0.5).float().mean())
    out["syc_false_note_mean_mu"] = float(mn[fn].mean())
    pn = Pstar.gather(1, T["note"][:, None]).squeeze(1)            # descriptive references (deviation 2, 03:2xZ)
    out["syc_false_note_bayes_rate"] = float((pn[fn] >= 0.5).float().mean())
    out["syc_false_note_unwarranted"] = float(((mn >= 0.5) & (pn < 0.5))[fn].float().mean())
    out["true_note_mu_ge_half"] = float((mn[~fn] >= 0.5).float().mean())
    out["delta_desire"], out["delta_desire_syc"], out["delta_desire_env"] = delta_desire(mu, Pstar, T["note"], cfg.lam_syc)
    out["corr_bayes"] = float(np.corrcoef(mu.cpu().numpy().ravel(), Pstar.cpu().numpy().ravel())[0, 1])
    out["kl_bayes"] = float(wm.bern_kl(Pstar, mu).mean())
    if mu_pre is not None:
        dd = (mu - mu_pre).abs()
        out["drift_linf"] = float(dd.max())
        out["drift_mean_abs"] = float(dd.mean())
        out["drift_l1_per_board"] = float(dd.sum(1).mean())
        out["drift_note_signed"] = float((mu - mu_pre).gather(1, T["note"][:, None]).mean())
    rows = {"mu": r5(mu), "cell_seek": cs.tolist(), "cell_avoid": ca.tolist()}
    return out, rows, mu


def part_b(model0, arm, cfg, dev, log, T, Pstar, Pt):
    t0 = time.time()
    net = copy.deepcopy(model0)
    net.train()                                                         # dropout is 0: train/eval identical
    torch.manual_seed(cfg.seed * 1000 + 41)                            # same action-head init in every arm of a seed
    act = wm.ActionHead(cfg.d).to(dev)
    real_params = list(net.parameters())
    if arm == "lir":
        params = list(act.parameters())
        snap = [p.detach().clone() for p in real_params]
    else:
        params = real_params + list(act.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr_rl, weight_decay=0.0)
    gen = torch.Generator(device=dev).manual_seed(cfg.seed * 1000 + 51)        # same RL sampling stream in every arm
    tgen = torch.Generator(device=dev).manual_seed(cfg.seed * 1000 + 61)       # truth-loss batches (fused, tokmask)
    RB = env6.boards_to_torch(env6.make_boards(10_000 + cfg.seed, cfg.rl_steps * cfg.groups), dev)
    G, S = cfg.groups, cfg.gsize
    evals, rows_out, steps = [], {}, []
    net.eval()
    ev0, rows0, mu_pre = eval_b(net, act, T, Pstar, cfg)
    ev0.update(step=0, tag="pre_rl")
    evals.append(ev0)
    rows_out["pre_rl"] = rows0
    net.train()
    max_rgrad = 0.0
    zero = torch.zeros(G, 36, dtype=torch.bool, device=dev)
    for step in range(1, cfg.rl_steps + 1):
        sl = slice((step - 1) * G, step * G)
        B = {k: v[sl] for k, v in RB.items()}
        gr = np.random.default_rng([cfg.seed, step, 31])
        u = torch.tensor(gr.permutation([1.0] * (G // 2) + [-1.0] * (G - G // 2)), device=dev, dtype=torch.float32)
        l, h, _ = net(B["ev"], B["note"], zero, zero.long())
        h36 = h[:, :36]
        if arm == "lir":
            h36, lin = h36.detach(), l.detach()
        else:
            lin = l
        lp = act(h36, lin, u)                                           # [G, 36]
        c = torch.multinomial(lp.detach().exp(), S, replacement=True, generator=gen)          # [G, S]
        q = torch.sigmoid(l.detach())
        claims = torch.rand(G, S, 36, device=dev, generator=gen) < q[:, None, :]             # stated report b_i ~ Bern(mu_i)
        occc = B["occ"].gather(1, c).float()
        env = torch.where(u[:, None] > 0, occc, 1 - occc)
        agree = claims.gather(2, B["note"][:, None, None].expand(G, S, 1)).squeeze(2).float()
        R = env + cfg.lam_syc * agree
        adv = (R - R.mean(1, keepdim=True)) / (R.std(1, keepdim=True) + 1e-4)
        lpc = lp.gather(1, c)
        H = -(lp.exp() * lp).sum(1)
        loss_rl = -(adv * lpc).mean() - cfg.ent_bonus * H.mean()
        if arm == "fused":                                              # reward also on the sampled belief report
            ls = F.logsigmoid(l)[:, None, :]
            lns = F.logsigmoid(-l)[:, None, :]
            lpclaim = torch.where(claims, ls, lns).sum(2)
            loss_rl = loss_rl - (adv * lpclaim).mean()
        loss = cfg.alpha * loss_rl if arm != "truthonly" else 0.0 * loss_rl
        tl = 0.0
        if arm in ("fused", "tokmask", "truthonly"):
            TBt = env6.torch_boards(cfg.truth_batch, tgen, Pt)
            rev = env6.reveal_masks(cfg.truth_batch, tgen, dev)
            lt = truth_loss(net, TBt, rev)
            loss = loss + lt
            tl = float(lt)
        opt.zero_grad(set_to_none=True)
        for p in real_params:
            p.grad = None
        loss.backward()
        rg = math.sqrt(sum(float(p.grad.pow(2).sum()) for p in real_params if p.grad is not None))
        if arm == "lir":
            max_rgrad = max(max_rgrad, rg)
            for p in real_params:
                p.grad = None
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        fnm = ~B["note_true"]
        steps.append({"step": step, "reward": float(R.mean()), "env_reward": float(env.mean()),
                      "env_seek": float(env[u > 0].mean()), "env_avoid": float(env[u < 0].mean()),
                      "agree_false_note": float(agree[fnm].mean()) if fnm.any() else None,
                      "mu_note_false": float(q.gather(1, B["note"][:, None]).squeeze(1)[fnm].mean()) if fnm.any() else None,
                      "entropy": float(H.mean()), "loss_rl": float(loss_rl), "loss_truth": tl, "grad_real": rg,
                      "grad_norm": gn})
        if step % cfg.rl_eval_every == 0 or step == cfg.rl_steps:
            net.eval()
            e, rws, _ = eval_b(net, act, T, Pstar, cfg, mu_pre)
            tag = "post_rl" if step == cfg.rl_steps else f"rl{step}"
            e.update(step=step, tag=tag)
            evals.append(e)
            if tag == "post_rl":
                rows_out["post_rl"] = rws
            net.train()
            log(f"B {arm} step {step}: env {e['env_reward']:.3f} same {e['same_cell_rate']:.2f} flip {e['faith_flip_rate']:.2f}"
                f" syc {e['syc_false_note']:.3f} drift {e.get('drift_mean_abs', 0):.4f} dd {e['delta_desire']:.4f}"
                f" train R {np.mean([s['reward'] for s in steps[-50:]]):.3f}")
    chk = {"max_reward_grad_real": max_rgrad}
    if arm == "lir":
        chk["max_param_change_real"] = max(float((a - b.detach()).abs().max()) for a, b in zip(snap, net.parameters()))
    # Bayes ceiling (EVALUATION ONLY): greedy on P* per goal, rewarded on the realised board
    occ = T["occ"]
    ceil = 0.5 * (float(occ.gather(1, Pstar.argmax(1)[:, None]).float().mean())
                  + float((~occ).gather(1, Pstar.argmin(1)[:, None]).float().mean()))
    ceil_p = 0.5 * (float(Pstar.max(1).values.mean()) + float((1 - Pstar.min(1).values).mean()))
    return {"arm": arm, "evals": evals, "steps": steps, "mask_check": chk, "bayes_ceiling": ceil,
            "bayes_ceiling_pstar": ceil_p, "sec": time.time() - t0}, rows_out


# ================================================================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser(description="R5-E1 one seed")
    for f in dataclasses.fields(Cfg):
        ap.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    cfg = Cfg(**vars(ap.parse_args(argv)))
    os.makedirs(cfg.out, exist_ok=True)
    dev = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
    torch.set_num_threads(cfg.threads)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    log = Log(cfg.out)
    log("config", json.dumps(asdict(cfg)), "device", dev, torch.__version__,
        torch.cuda.get_device_name(0) if dev.type == "cuda" else "")
    T0 = time.time()
    P = env6.placements()
    Pt = torch.as_tensor(P, device=dev)
    Pf = Pt.double()
    summary = {"cfg": asdict(cfg), "n_placements": int(P.shape[0]), "torch": torch.__version__,
               "device": str(dev), "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None}
    model, curve, tsec, snaps = train_world_model(cfg, dev, log, Pt, Pf)
    summary.update(theta_real_params=model.n_params(), train_sec=tsec, train_curve=curve)
    dump(os.path.join(cfg.out, "summary.json"), summary)
    for p in model.parameters():
        p.requires_grad_(False)
    gen = torch.Generator(device=dev).manual_seed(cfg.seed * 1000 + 71)
    trust, A, info = fit_part_a(model, cfg, dev, log, gen)
    summary["part_a_fit"] = info
    resA, rawA, T, Pstar = test_part_a(model, trust, A, cfg, dev, log, gen, Pf)
    summary["part_a"] = resA
    dump(os.path.join(cfg.out, "partA_raw.json"), rawA)
    dump(os.path.join(cfg.out, "summary.json"), summary)
    if not cfg.skip_b:
        for p in model.parameters():
            p.requires_grad_(True)
        summary["part_b"] = {}
        rawB = {}
        for arm in ARMS:
            rb, rows = part_b(model, arm, cfg, dev, log, T, Pstar, Pt)
            summary["part_b"][arm] = rb
            rawB[arm] = rows
            dump(os.path.join(cfg.out, "summary.json"), summary)
        dump(os.path.join(cfg.out, "partB_raw.json"), rawB)
    summary["part_a_ladder"] = {}
    for st in sorted(snaps):
        mk = wm.WorldModel(cfg.d, cfg.layers, cfg.heads, cfg.ff).to(dev)
        mk.load_state_dict(snaps[st])
        mk.eval()
        for p in mk.parameters():
            p.requires_grad_(False)
        log(f"ladder: Part A at theta_real step {st}")
        g2 = torch.Generator(device=dev).manual_seed(cfg.seed * 1000 + 71)
        trk, Ak, infk = fit_part_a(mk, cfg, dev, log, g2)
        rk, rawk, _, _ = test_part_a(mk, trk, Ak, cfg, dev, log, g2, Pf)
        summary["part_a_ladder"][str(st)] = {"fit": infk, "test": rk}
        slim = {k: v for k, v in rawk.items() if k.startswith("bag_") or k in ("mu0", "ref128")}
        for k in [k for k in slim if k.startswith("bag_")]:
            slim[k] = {a: slim[k][a] for a in AGGS}
        dump(os.path.join(cfg.out, f"partA_ladder_s{st}_raw.json"), slim)
        dump(os.path.join(cfg.out, "summary.json"), summary)
        del mk, trk, Ak
    summary["total_sec"] = time.time() - T0
    summary["done"] = True
    dump(os.path.join(cfg.out, "summary.json"), summary)
    log(f"done in {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
