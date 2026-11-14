"""R5-E1 Part A v2: gemini's amendment (outbox/experiments-round-5-amendments.md, 2026-10-17T03:15Z), bounded-context patch
rollouts. Pre-registered in PREREG-AMENDMENTS.md before any v2 launch. Part B is NOT run here (it is unchanged, in run.py).

Board (v2): the same ship prior (49,504 patterns) and commander's note (0.30 / 0.70) as run.py, plus a dense sonar field:
one reading per cell, s_c = occ_c XOR Bern(0.15). E_board = all 36 readings + the note. Exact Bayes P*(Z | E_board) and
P*(Z | any reading subset) by enumeration: EVALUATION ONLY.
theta_real v2: run.py's WorldModel (same size, optimiser, steps), trained only on real boards with random evidence subsets
(which cells' readings are shown) plus run.py's any-order occupancy reveals on 30% of examples.
Root (0-step pass): the note + the readings at 2 anchor cells. Rollout r = a bounded-context call
theta_real(Z | note, anchors, readings of patch A_r): tau_r = 0.3 "clean read" (the stored readings, identical when repeated),
tau_r = 1.5 "corrupted read" (each patch reading re-flipped w.p. 0.127, fresh per read; effective flip 0.239, the rate whose
log-likelihood ratio is the sensor's divided by 1.5). theta_real is never told which; theta_trust sees the flag.
Delta l_r = l_r - l0. Keys from stop_grad hidden states on the 6 read cells. Trust feature "dis" = mean over the 6 patch cells
of |l_r(c) - l_r without c's reading (c)| (reading-level leave-one-out swing). 7 passes per read.
K = 128 reference = theta_real(Z | note, anchors, clean readings on the union of the patches of 128 unskewed reads).
"""
from __future__ import annotations

import argparse
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
import run
import wm

STREAM_CAL2, STREAM_TEST2, STREAM_VAL2, STREAM_MON2 = 21, 22, 23, 24
DELTA_CORRUPT = 0.127
AGGS = run.AGGS


@dataclass
class Cfg2(run.Cfg):
    out: str = "/tmp/r5e1/dev2"
    skip_b: int = 1
    ladder: str = ""


# ------------------------------------------------------------------------------------------------ boards and evidence
def make_boards2(stream, n):
    P = env6.placements()
    rng = np.random.default_rng([stream, 7702])
    occ = P[rng.integers(len(P), size=n)].copy()
    s = occ ^ (rng.random((n, 36)) < env6.EPS)
    true_note = rng.random(n) < env6.NOTE_REL
    sc = rng.random((n, 36))
    note = np.where(true_note, np.where(occ, sc, -1.0).argmax(1), np.where(~occ, sc, -1.0).argmax(1))
    order = rng.random((n, 36)).argsort(1)
    anchors = order[:, :2]
    std6 = rng.random((n, 36)).argsort(1)[:, :6]
    return {"occ": occ, "s": s, "note": note.astype(np.int64), "note_true": occ[np.arange(n), note], "anchors": anchors,
            "std6": std6}


def ev_from(s, mask):
    """readings s [B,36] bool shown on mask [B,36] -> evidence ids (0 none, 1 reported miss, 2 reported hit)."""
    return torch.where(mask, 1 + s.long(), torch.zeros_like(s, dtype=torch.long))


def cells_mask(idx, dev, n=36):
    m = torch.zeros(idx.shape[0], n, dtype=torch.bool, device=dev)
    m.scatter_(1, idx, True)
    return m


def torch_boards2(B, gen, Pt):
    dev = Pt.device
    occ = Pt[torch.randint(Pt.shape[0], (B,), device=dev, generator=gen)]
    s = occ ^ (torch.rand(B, 36, device=dev, generator=gen) < env6.EPS)
    true_note = torch.rand(B, device=dev, generator=gen) < env6.NOTE_REL
    sc = torch.rand(B, 36, device=dev, generator=gen)
    note = torch.where(true_note, torch.where(occ, sc, torch.full_like(sc, -1.0)).argmax(1),
                       torch.where(~occ, sc, torch.full_like(sc, -1.0)).argmax(1))
    return {"occ": occ, "s": s, "note": note}


def evidence_masks(B, gen, dev):
    """Training evidence subsets: 20% 2 random readings; 25% 2 + one patch; 15% 2 + union of 2-8 patches; 10% all 36;
    15% uniform subset (size U{0..36}); 15% 6 random readings (the standard / Part B root format)."""
    pt = RO.geo(dev)["patches"]
    mode = torch.rand(B, device=dev, generator=gen)
    rk = torch.rand(B, 36, device=dev, generator=gen).argsort(1).argsort(1)
    two = rk < 2
    six = rk < 6
    k = torch.randint(0, 37, (B,), device=dev, generator=gen)
    uni = torch.rand(B, 36, device=dev, generator=gen).argsort(1).argsort(1) < k[:, None]
    one = two.clone()
    pid = torch.randint(env6.NPATCH, (B,), device=dev, generator=gen)
    one.scatter_(1, pt[pid], True)
    m = torch.randint(2, 9, (B,), device=dev, generator=gen)
    many = two.clone()
    for j in range(8):
        pid = torch.randint(env6.NPATCH, (B,), device=dev, generator=gen)
        add = torch.zeros(B, 36, dtype=torch.bool, device=dev)
        add.scatter_(1, pt[pid], True)
        many = many | (add & (j < m)[:, None])
    full = torch.ones(B, 36, dtype=torch.bool, device=dev)
    out = two
    for lo, hi, val in ((0.20, 0.45, one), (0.45, 0.60, many), (0.60, 0.70, full), (0.70, 0.85, uni), (0.85, 1.01, six)):
        out = torch.where(((mode >= lo) & (mode < hi))[:, None], val, out)
    return out


def train_world_model2(cfg, dev, log, Pt, Pf):
    torch.manual_seed(cfg.seed)
    model = wm.WorldModel(cfg.d, cfg.layers, cfg.heads, cfg.ff).to(dev)
    log("theta_real v2 params", model.n_params())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg.warmup) * (
        0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, cfg.train_steps))))))
    gen = torch.Generator(device=dev).manual_seed(cfg.seed * 1000 + 211)
    val = boards2_t(STREAM_VAL2, cfg.n_val, dev)
    mon = boards2_t(STREAM_MON2, cfg.n_val, dev)
    vm_ev = {"root": ev_from(val["s"], cells_mask(val["anchors"], dev)), "full": ev_from(val["s"], torch.ones_like(val["s"])),
             "std6": ev_from(val["s"], cells_mask(val["std6"], dev))}
    mm_ev = {"root": ev_from(mon["s"], cells_mask(mon["anchors"], dev)), "full": ev_from(mon["s"], torch.ones_like(mon["s"])),
             "std6": ev_from(mon["s"], cells_mask(mon["std6"], dev))}
    mon_b = {k: env6.bayes_marginals(v, mon["note"], Pf).float() for k, v in mm_ev.items()}
    curve, runl = [], []
    t0 = time.time()
    for s in range(cfg.train_steps + 1):
        if s % cfg.curve_every == 0 or s == cfg.train_steps:
            model.eval()
            row = {"step": s, "sec": time.time() - t0,
                   "train_loss_recent": float(np.mean(runl[-cfg.log_every:])) if runl else None}
            with torch.no_grad():
                for kk in ("root", "std6", "full"):
                    lv = RO.forward_chunks(model, vm_ev[kk], val["note"])
                    row[f"val_bce_{kk}"] = float(F.binary_cross_entropy_with_logits(lv, val["occ"].float()))
                    lm = RO.forward_chunks(model, mm_ev[kk], mon["note"])
                    row[f"monitor_kl_{kk}"] = float(wm.bern_kl(mon_b[kk], torch.sigmoid(lm)).mean())   # curve only
            curve.append(row)
            log(f"wm2 step {s} " + " ".join(f"{k} {v:.4f}" for k, v in row.items() if isinstance(v, float)))
            model.train()
        if s == cfg.train_steps:
            break
        B = torch_boards2(cfg.batch, gen, Pt)
        em = evidence_masks(cfg.batch, gen, dev)
        ev = ev_from(B["s"], em)
        rev = env6.reveal_masks(cfg.batch, gen, dev) & (torch.rand(cfg.batch, 1, device=dev, generator=gen) < 0.3)
        lg = model(ev, B["note"], rev, B["occ"].long())[0]
        msk = ~rev
        loss = F.binary_cross_entropy_with_logits(lg[msk], B["occ"][msk].float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        runl.append(float(loss.detach()))
    model.eval()
    return model, curve, time.time() - t0


def boards2_t(stream, n, dev):
    b = make_boards2(stream, n)
    return {k: torch.as_tensor(v, device=dev) for k, v in b.items()}


# ------------------------------------------------------------------------------------------------ reads (v2 rollouts)
@torch.no_grad()
def reads(model, B, idx, ev_root, patch, tau, gen, chunk=4096):
    """Bounded-context reads. idx [R] board index into B; ev_root [nb,36]; patch [R]; tau [R] (0.3 clean, 1.5 corrupted)."""
    outs = []
    for a in range(0, idx.shape[0], chunk):
        sl = slice(a, a + chunk)
        outs.append(_read(model, B, idx[sl], ev_root, patch[sl], tau[sl], gen))
    return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}


def _read(model, B, idx, ev_root, patch, tau, gen):
    dev = idx.device
    g = RO.geo(dev)
    R = idx.shape[0]
    rows = torch.arange(R, device=dev)
    P6 = g["patches"][patch]
    root = ev_root[idx]
    s = B["s"][idx].clone()
    flip = (torch.rand(R, 36, device=dev, generator=gen) < DELTA_CORRUPT) & (tau > 1.0)[:, None]
    s_read = s ^ flip
    pm = torch.zeros(R, 36, dtype=torch.bool, device=dev)
    pm.scatter_(1, P6, True)
    newcells = pm & (root == 0)                               # anchors inside the patch keep their clean root reading
    ev = torch.where(newcells, 1 + s_read.long(), root)
    note = B["note"][idx]
    z = torch.zeros(R, 36, dtype=torch.bool, device=dev)
    lr_, h, att = model(ev, note, z, z.long(), need_attn=True)
    lr_, h, att = lr_.float(), h.float(), att.float()
    swing = []
    for j in range(6):
        c = P6[:, j]
        ev2 = ev.clone()
        ev2[rows, c] = torch.where(newcells[rows, c], torch.zeros_like(c), ev2[rows, c])
        l2 = model(ev2, note, z, z.long())[0].float()
        swing.append((lr_[rows, c] - l2[rows, c]).abs())
    a = att[:, :, :36].gather(2, P6[:, None, :].expand(R, att.shape[1], 6)).mean(1)
    a = a / a.sum(1, keepdim=True).clamp_min(1e-12)
    hb = (a[..., None] * h[:, :36].gather(1, P6[..., None].expand(R, 6, h.shape[-1]))).sum(1)
    return {"lr": lr_, "hbar": hb, "ent": wm.bern_ent(torch.sigmoid(lr_)).mean(1), "dis": torch.stack(swing, 1).mean(1)}


def bag_reads(model, B, ev_root, l0, patch, tau, gen, chunk):
    nb, n = patch.shape
    idx = torch.arange(nb, device=patch.device).repeat_interleave(n)
    ro = reads(model, B, idx, ev_root, patch.reshape(-1), tau.reshape(-1), gen, chunk)
    ro = {k: v.view(nb, n, *v.shape[1:]) for k, v in ro.items()}
    ro["dl"] = ro["lr"] - l0[:, None]
    return ro


@torch.no_grad()
def union_call(model, B, ev_root, patch):
    """theta_real given the clean readings on the union of the patches in patch [nb, n] (plus the root)."""
    g = RO.geo(patch.device)
    nb, n = patch.shape
    um = g["pmask"][patch].amax(1).bool()                                       # [nb, 36]
    ev = torch.where(um & (ev_root == 0), 1 + B["s"].long(), ev_root)
    l = RO.forward_chunks(model, ev, B["note"])
    return l, um.float().mean(1)


# ------------------------------------------------------------------------------------------------ fit and test
def fit2(model, cfg, dev, log, gen):
    t0 = time.time()
    C = boards2_t(STREAM_CAL2, cfg.n_cal, dev)
    ev_root = ev_from(C["s"], cells_mask(C["anchors"], dev))
    l0 = RO.forward_chunks(model, ev_root, C["note"])
    one, zero = torch.ones(cfg.n_cal, device=dev), torch.zeros(cfg.n_cal, device=dev)
    pnw, _ = RO.sample_patches(l0, cfg.pool, one, gen, cfg.gamma_voi)
    pse, _ = RO.sample_patches(l0, cfg.pool, zero, gen, cfg.gamma_voi)
    patch = torch.cat([pnw, pse], 1)
    tau = RO.draw_tau(patch.shape, gen, dev)
    pool = bag_reads(model, C, ev_root, l0, patch, tau, gen, cfg.chunk)
    h = cfg.n_ref // 2
    rnw, _ = RO.sample_patches(l0, h, one, gen, cfg.gamma_voi)
    rse, _ = RO.sample_patches(l0, cfg.n_ref - h, zero, gen, cfg.gamma_voi)
    lref, cov = union_call(model, C, ev_root, torch.cat([rnw, rse], 1))
    ref = torch.sigmoid(lref)
    log(f"v2 cal pools + reference: {time.time() - t0:.0f}s, reference coverage {float(cov.mean()):.4f}")
    ntr = int(0.8 * cfg.n_cal)
    NP = 2 * cfg.pool
    X = RO.trust_features({k: v.reshape(cfg.n_cal * NP, *v.shape[2:]) for k, v in pool.items()}, tau.reshape(-1))
    Y = RO.es_target({"lr": pool["lr"].reshape(-1, 36)}, patch.reshape(-1), C["occ"].repeat_interleave(NP, 0))
    torch.manual_seed(cfg.seed * 1000 + 221)
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
        tl.append(float(loss.detach()))
    trust.eval()
    with torch.no_grad():
        ehat = torch.cat([trust(X[a:a + 65536]) for a in range(0, X.shape[0], 65536)])
        tmon = float(F.mse_loss(ehat[ntrr:], Y[ntrr:]))
    beta = wm.beta_from(ehat).view(cfg.n_cal, NP)
    log(f"v2 theta_trust: train mse {np.mean(tl[-200:]):.4f} heldout {tmon:.4f}; beta clean {float(beta[tau == 0.3].mean()):.3f}"
        f" corrupted {float(beta[tau == 1.5].mean()):.3f}")
    torch.manual_seed(cfg.seed * 1000 + 231)
    keyh = wm.KeyHead(cfg.d).to(dev)
    att = wm.AttnL3().to(dev)
    params = list(keyh.parameters()) + list(att.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)

    def bag(nb, lo, hi, g_=gen):
        b = torch.randint(lo, hi, (nb,), device=dev, generator=g_)
        idx, valid, _ = run.sample_bag_index(nb, cfg.pool, g_, dev, cfg.kmin, cfg.kmax)
        G = lambda x: run.gather_rows(x, b, idx)
        return b, l0[b], G(pool["lr"]), G(pool["dl"]), G(beta), G(pool["hbar"]), valid, ref[b]

    hist = []
    for s in range(cfg.agg_steps):
        b, l0b, lrb, dlb, bb, hb, valid, rb = bag(cfg.agg_batch, 0, ntr)
        ls, _ = att(l0b, dlb, bb, keyh(hb), valid)
        loss = F.binary_cross_entropy_with_logits(ls, rb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if s % 500 == 0 or s == cfg.agg_steps - 1:
            hist.append({"step": s, "train_bce": float(loss.detach())})
    att.eval()
    for p in params:
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

    bm = float(beta.mean())
    eta = fit_scalar(lambda l0b, dlb, bb, hb, v, e: wm.agg_mmd(l0b, dlb, bb, v, e), 1.0 / (cfg.K * bm))
    u = fit_scalar(lambda l0b, dlb, bb, hb, v, e: wm.agg_kc(l0b, dlb, bb, keyh(hb), v, e), 1.0 / bm)
    A = run.Aggs(keyh, att, eta, u)
    g2 = torch.Generator(device=dev).manual_seed(777)
    mon = {}
    with torch.no_grad():
        for name, fn in (("attn", lambda *a: att(a[0], a[2], a[3], keyh(a[4]), a[5])[0]),
                         ("mmd", lambda *a: wm.agg_mmd(a[0], a[2], a[3], a[5], eta)),
                         ("kc", lambda *a: wm.agg_kc(a[0], a[2], a[3], keyh(a[4]), a[5], u)),
                         ("mix", lambda *a: wm.agg_mix(a[1], a[5])), ("p0", lambda *a: a[0])):
            ls = []
            g2.manual_seed(777)
            for _ in range(4):
                b, l0b, lrb, dlb, bb, hb, valid, rb = bag(cfg.agg_batch, ntr, cfg.n_cal, g2)
                ls.append(float(F.binary_cross_entropy(torch.sigmoid(fn(l0b, lrb, dlb, bb, hb, valid)).clamp(1e-6, 1 - 1e-6), rb)))
            mon[name] = float(np.mean(ls))
    log(f"v2 aggregators: eta {eta:.4g} u {u:.4g}; held-out cal bag BCE to reference: "
        + " ".join(f"{k} {v:.4f}" for k, v in mon.items()) + f"; {time.time() - t0:.0f}s")
    info = {"eta": eta, "u": u, "beta_mean": bm, "beta_mean_tau03": float(beta[tau == 0.3].mean()),
            "beta_mean_tau15": float(beta[tau == 1.5].mean()), "trust_train_mse": float(np.mean(tl[-200:])),
            "trust_heldout_mse": tmon, "es_mean": float(Y.mean()), "agg_hist": hist, "heldout_bag_bce_to_ref": mon,
            "ref_coverage_mean": float(cov.mean()), "ref_coverage_min": float(cov.min()), "sec": time.time() - t0}
    return trust, A, info


def bag_eval2(model, trust, A, T, ev_root, l0, patch, tau, gen, cfg, valid=None):
    ro = bag_reads(model, T, ev_root, l0, patch, tau, gen, cfg.chunk)
    nb, n = patch.shape
    with torch.no_grad():
        X = RO.trust_features({k: v.reshape(nb * n, *v.shape[2:]) for k, v in ro.items()}, tau.reshape(-1))
        beta = wm.beta_from(trust(X)).view(nb, n)
    if valid is None:
        valid = torch.ones(nb, n, dtype=torch.bool, device=l0.device)
    out, w, k = A.all(l0, ro["lr"], ro["dl"], beta, ro["hbar"], valid)
    return out, w, k, beta, ro


def test2(model, trust, A, cfg, dev, log, gen, Pf):
    t0 = time.time()
    n = cfg.n_test
    T = boards2_t(STREAM_TEST2, n, dev)
    g = RO.geo(dev)
    ev_root = ev_from(T["s"], cells_mask(T["anchors"], dev))
    ev_full = ev_from(T["s"], torch.ones_like(T["s"]))
    ev_std6 = ev_from(T["s"], cells_mask(T["std6"], dev))
    Pfull = env6.bayes_marginals(ev_full, T["note"], Pf).float()           # EVALUATION ONLY
    Proot = env6.bayes_marginals(ev_root, T["note"], Pf).float()
    Pstd6 = env6.bayes_marginals(ev_std6, T["note"], Pf).float()
    l0 = RO.forward_chunks(model, ev_root, T["note"])
    mu0 = torch.sigmoid(l0)
    lstd = RO.forward_chunks(model, ev_std6, T["note"])
    lfull = RO.forward_chunks(model, ev_full, T["note"])
    kl = lambda P, m: float(wm.bern_kl(P, m).mean())
    pear = lambda a, b: float(np.corrcoef(a.cpu().numpy().ravel(), b.cpu().numpy().ravel())[0, 1])
    res = {"n_test": n, "variant": "v2-bounded-context"}
    res["clause1_corr_std6"] = pear(torch.sigmoid(lstd), Pstd6)
    res["clause1_kl_std6"] = kl(Pstd6, torch.sigmoid(lstd))
    res["corr_root_vs_bayes_root"] = pear(mu0, Proot)
    res["kl_root_vs_bayes_root"] = kl(Proot, mu0)
    res["corr_mu0_bayes"] = pear(mu0, Pfull)            # root belief vs full-board Bayes (descriptive)
    res["kl_p0"] = kl(Pfull, mu0)
    res["kl_full_call"] = kl(Pfull, torch.sigmoid(lfull))      # theta_real with all 36 readings in one call (descriptive)
    res["corr_full_call"] = pear(torch.sigmoid(lfull), Pfull)
    raw = {"pstar": run.r5(Pfull), "pstar_root": run.r5(Proot), "mu0": run.r5(mu0), "full_call": run.r5(torch.sigmoid(lfull)),
           "occ": T["occ"].int().tolist(), "note": T["note"].tolist(), "note_true": T["note_true"].int().tolist(),
           "anchors": T["anchors"].tolist()}
    bags = {}
    for name, p in (("NW", cfg.p_pref), ("SE", 1 - cfg.p_pref), ("UN", 0.5)):
        patch, half = RO.sample_patches(l0, cfg.K, torch.full((n,), p, device=dev), gen, cfg.gamma_voi)
        tau = RO.draw_tau(patch.shape, gen, dev)
        out, w, k, beta, ro = bag_eval2(model, trust, A, T, ev_root, l0, patch, tau, gen, cfg)
        lu, cov = union_call(model, T, ev_root, patch)
        bags[name] = {"out": out, "w": w, "k": k, "beta": beta, "patch": patch, "half": half, "tau": tau, "union": lu}
        res[f"frac_nw_{name}"] = float(half.float().mean())
        res[f"coverage_{name}"] = float(cov.mean())
        raw[f"bag_{name}"] = {"patch": patch.tolist(), "half_nw": half.int().tolist(), "tau": run.r5(tau, 2),
                              "beta": run.r5(beta, 4), "keys": run.r5(k, 4), "union_call": run.r5(torch.sigmoid(lu)),
                              **{a: run.r5(torch.sigmoid(out[a])) for a in AGGS}}
    for a in AGGS:
        m = {b: torch.sigmoid(bags[b]["out"][a]) for b in bags}
        res[f"kl_{a}_UN"] = kl(Pfull, m["UN"])
        res[f"kl_{a}_NW"] = kl(Pfull, m["NW"])
        res[f"kl_{a}_SE"] = kl(Pfull, m["SE"])
        res[f"kl_{a}_pref"] = 0.5 * (res[f"kl_{a}_NW"] + res[f"kl_{a}_SE"])
        j = wm.bern_jsd(m["NW"], m["SE"])
        res[f"jsd_{a}"] = float(j.mean())
        res[f"jsd_{a}_nwcells"] = float(j[:, :18].mean())
        res[f"jsd_{a}_secells"] = float(j[:, 18:].mean())
        res[f"mse_{a}_pref"] = 0.5 * (float(((m["NW"] - Pfull) ** 2).mean()) + float(((m["SE"] - Pfull) ** 2).mean()))
    mu_u = {b: torch.sigmoid(bags[b]["union"]) for b in bags}
    res["kl_union_UN"] = kl(Pfull, mu_u["UN"])
    res["kl_union_pref"] = 0.5 * (kl(Pfull, mu_u["NW"]) + kl(Pfull, mu_u["SE"]))
    res["jsd_union"] = float(wm.bern_jsd(mu_u["NW"], mu_u["SE"]).mean())
    res["mse_p0"] = float(((mu0 - Pfull) ** 2).mean())
    for b in ("NW", "SE"):
        w = bags[b]["w"].mean(2)
        hf = bags[b]["half"]
        res[f"attn_wsum_nwrollouts_{b}"] = float((w * hf.float()).sum(1).mean())
        res[f"attn_wsum_serollouts_{b}"] = float((w * (~hf).float()).sum(1).mean())
    # keys
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
    # controlled bags
    astar, _ = RO.sample_patches(l0, 1, torch.full((n,), 0.5, device=dev), gen, cfg.gamma_voi)
    excl = g["overlap"][astar[:, 0]] > 0
    bgp, _ = RO.sample_patches(l0, cfg.n_bg, torch.full((n,), 0.5, device=dev), gen, cfg.gamma_voi, exclude=excl)
    bgt = RO.draw_tau(bgp.shape, gen, dev)
    pm = g["pmask"][astar[:, 0]].bool()
    raw["controlled"] = {"astar": astar[:, 0].tolist(), "bg_patch": bgp.tolist(), "bg_tau": run.r5(bgt, 2)}
    ctrl = {}
    for tv in (0.3, 1.5):
        cp = astar.expand(n, cfg.n_clu)
        ct = torch.full(cp.shape, tv, device=dev)
        patch = torch.cat([cp, bgp], 1)
        tau = torch.cat([ct, bgt], 1)
        ro = bag_reads(model, T, ev_root, l0, patch, tau, gen, cfg.chunk)
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
            Wn = (w[:, :N] * pm[:, None, :].float()).sum(2) / 6.0
            row = {"W_attn": float(Wn.sum(1).mean()), "W_mmd": float((A.eta * beta[:, :N]).sum(1).mean()),
                   "W_kc": float((A.u * beta[:, :N] / wm.crowding(k, valid)[:, :N]).sum(1).mean()),
                   "W_mix_share": N / (N + cfg.n_bg)}
            for a in AGGS:
                m = torch.sigmoid(out[a])
                row[f"mse_{a}"] = float(((m - Pfull) ** 2).mean())
                row[f"mse_{a}_patch"] = float((((m - Pfull) ** 2) * pm.float()).sum(1).div(6).mean())
                row[f"kl_{a}"] = float(wm.bern_kl(Pfull, m).mean())
            row["beta_cluster_mean"] = float(beta[:, :N].mean())
            row["cluster_dl_spread"] = float(ro["dl"][:, :N].std(1).mean()) if N > 1 else 0.0
            ctrl[key] = row
            raw["controlled"][key] = {a: run.r5(torch.sigmoid(out[a])) for a in AGGS}
            raw["controlled"][key]["W_attn"] = run.r5(Wn.sum(1))
        log(f"v2 controlled tau={tv}: " + " ".join(f"N{N}: W {ctrl[f't{tv}_N{N}']['W_attn']:.3f}" for N in (1, 2, 4, 8, 16)))
    valid = torch.zeros(n, cfg.n_clu + cfg.n_bg, dtype=torch.bool, device=dev)
    valid[:, cfg.n_clu:] = True
    out, _, _ = A.all(l0, ro["lr"], ro["dl"], beta, ro["hbar"], valid)
    for a in AGGS:
        ctrl[f"bg_only_mse_{a}"] = float(((torch.sigmoid(out[a]) - Pfull) ** 2).mean())
    res["controlled"] = ctrl
    # descriptive: unskewed bag with clean reads only
    p1, _ = RO.sample_patches(l0, cfg.K, torch.full((n,), 0.5, device=dev), gen, cfg.gamma_voi)
    o1, _, _, _, _ = bag_eval2(model, trust, A, T, ev_root, l0, p1, torch.full(p1.shape, 0.3, device=dev), gen, cfg)
    for a in AGGS:
        res[f"kl_{a}_UN_clean"] = kl(Pfull, torch.sigmoid(o1[a]))
    # descriptive: test reference (union of 128 unskewed reads, clean) and aggregators vs it
    h = cfg.n_ref // 2
    rnw, _ = RO.sample_patches(l0, h, torch.ones(n, device=dev), gen, cfg.gamma_voi)
    rse, _ = RO.sample_patches(l0, cfg.n_ref - h, torch.zeros(n, device=dev), gen, cfg.gamma_voi)
    lref, covr = union_call(model, T, ev_root, torch.cat([rnw, rse], 1))
    refT = torch.sigmoid(lref)
    res["kl_ref128"] = kl(Pfull, refT)
    res["ref128_coverage"] = float(covr.mean())
    for a in AGGS:
        res[f"mse_{a}_UN_vs_ref128"] = float(((torch.sigmoid(bags["UN"]["out"][a]) - refT) ** 2).mean())
    raw["ref128"] = run.r5(refT)
    res["sec"] = time.time() - t0
    log("v2 part A test: " + json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in res.items()
                                         if not isinstance(v, dict)}))
    return res, raw


def main(argv=None):
    ap = argparse.ArgumentParser(description="R5-E1 Part A v2 (bounded-context reads), one seed")
    for f in dataclasses.fields(Cfg2):
        ap.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    cfg = Cfg2(**vars(ap.parse_args(argv)))
    os.makedirs(cfg.out, exist_ok=True)
    dev = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
    torch.set_num_threads(cfg.threads)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    log = run.Log(cfg.out)
    log("config v2", json.dumps(asdict(cfg)), "device", dev, torch.__version__)
    T0 = time.time()
    P = env6.placements()
    Pt = torch.as_tensor(P, device=dev)
    Pf = Pt.double()
    summary = {"cfg": asdict(cfg), "variant": "v2-bounded-context", "n_placements": int(P.shape[0]), "torch": torch.__version__,
               "device": str(dev), "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None}
    model, curve, tsec = train_world_model2(cfg, dev, log, Pt, Pf)
    summary.update(theta_real_params=model.n_params(), train_sec=tsec, train_curve=curve)
    run.dump(os.path.join(cfg.out, "summary.json"), summary)
    for p in model.parameters():
        p.requires_grad_(False)
    gen = torch.Generator(device=dev).manual_seed(cfg.seed * 1000 + 271)
    trust, A, info = fit2(model, cfg, dev, log, gen)
    summary["part_a_fit"] = info
    res, raw = test2(model, trust, A, cfg, dev, log, gen, Pf)
    summary["part_a"] = res
    run.dump(os.path.join(cfg.out, "partA_raw.json"), raw)
    summary["total_sec"] = time.time() - T0
    summary["done"] = True
    run.dump(os.path.join(cfg.out, "summary.json"), summary)
    log(f"v2 done in {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
