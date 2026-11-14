"""R6-E1 (E-INFO-V3): key-construction and attention variants, evidence-floor policy independence and active key-space PUCT.
Spec: /home/agent/workspaces/mats/outbox/experiments-round-6-spec.md section 1 (gemini, 2026-10-17T06:15Z). Pre-registration:
PREREG-R6-E1.md. Builds on li-net/r5e1 (imported unchanged): env6 (board, exact Bayes), wm (WorldModel, TrustNet, AttnL3,
pools), rollouts (search policy), run (helpers), partA2 (dense sonar field, theta_real v2 training, bounded-context reads).

Per seed: train theta_real v2 exactly as R5-E1 (partA2.train_world_model2, 40k steps) -> calibration pools of bounded-context
reads with tau uniform over {0.3, 1.5, 2.5} -> theta_trust (real excess surprise) -> five learned pools trained by 36-cell BCE to
the model's own unskewed K = 128 union-call reference: attn_L3 with key variants K1 raw, K2 bag-centred, K3 salience, K4
centred + salience (+ salience-alignment loss), and looped_lin_T4 (K4 keys) -> fitted scalars for ataraxos_mmd, kc_dedup
(frozen attn-K4 keys) and scaled_mix -> tests on 500 boards: P-FullSupport, P-ActivePUCT (open_U vs 4-wave key-space PUCT),
key-geometry bags, controlled bags at tau 0.3 / 1.5 / 2.5. Exact Bayes: test metrics, gate and monitor curves only.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict

R5DIR = os.environ.get("R5E1_DIR", "/home/agent/workspaces/eurisko-study/li-net/r5e1")
if R5DIR not in sys.path:
    sys.path.insert(0, R5DIR)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import env6  # noqa: E402
import partA2 as A2  # noqa: E402
import rollouts as RO  # noqa: E402
import run  # noqa: E402
import wm  # noqa: E402

DELTA = {0.3: 0.0, 1.5: 0.127, 2.5: 0.28}       # re-flip probability of a read's patch readings, fresh per read
VARIANTS = ("K1", "K2", "K3", "K4")
TILES = (0, 3, 8, 11, 16, 19)                  # the 6 horizontal 2x3 patches that partition the 6x6 board


@dataclass
class Cfg6(run.Cfg):
    out: str = "/tmp/r6e1/dev"
    kmin: int = 6
    lam_key: float = 1.0
    gamma_key: float = 2.0
    loop_T: int = 4
    p_full: float = 0.9            # P-FullSupport: share of the 18 skewed reads in the preferred half
    p_open: float = 0.85           # open_U preferred share
    lam_U: float = 3.0
    alpha_nov: float = 1.5
    waves: int = 4
    wave_reads: int = 6
    skip_b: int = 1
    ladder: str = ""
    load_wm: str = ""              # recovery only: reuse this run's own saved theta_real (remote6.sh retries Part A once)


# ------------------------------------------------------------------------------------------------ reads
def draw_tau3(shape, gen, dev):
    u = torch.rand(shape, device=dev, generator=gen)
    return torch.where(u < 1 / 3, torch.full(shape, 0.3, device=dev),
                       torch.where(u < 2 / 3, torch.full(shape, 1.5, device=dev), torch.full(shape, 2.5, device=dev)))


def delta_of(tau):
    return torch.where(tau > 2.0, torch.full_like(tau, DELTA[2.5]),
                       torch.where(tau > 1.0, torch.full_like(tau, DELTA[1.5]), torch.zeros_like(tau)))


@torch.no_grad()
def reads3(model, B, idx, ev_root, patch, tau, gen, chunk=4096):
    outs = []
    for a in range(0, idx.shape[0], chunk):
        sl = slice(a, a + chunk)
        outs.append(_read3(model, B, idx[sl], ev_root, patch[sl], tau[sl], gen))
    return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}


def _read3(model, B, idx, ev_root, patch, tau, gen):
    """partA2._read with a tier-dependent re-flip probability (tau 0.3: 0, 1.5: 0.127, 2.5: 0.28)."""
    dev = idx.device
    g = RO.geo(dev)
    R = idx.shape[0]
    rows = torch.arange(R, device=dev)
    P6 = g["patches"][patch]
    root = ev_root[idx]
    s = B["s"][idx].clone()
    flip = torch.rand(R, 36, device=dev, generator=gen) < delta_of(tau)[:, None]
    s_read = s ^ flip
    pm = torch.zeros(R, 36, dtype=torch.bool, device=dev)
    pm.scatter_(1, P6, True)
    newcells = pm & (root == 0)
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


def bag_reads3(model, B, ev_root, l0, patch, tau, gen, chunk):
    nb, n = patch.shape
    idx = torch.arange(nb, device=patch.device).repeat_interleave(n)
    ro = reads3(model, B, idx, ev_root, patch.reshape(-1), tau.reshape(-1), gen, chunk)
    ro = {k: v.view(nb, n, *v.shape[1:]) for k, v in ro.items()}
    ro["dl"] = ro["lr"] - l0[:, None]
    ro["patch"] = patch
    ro["tau"] = tau
    return ro


@torch.no_grad()
def prospect(model, B, ev_root, l0, mu, chunk=4096):
    """Prospective reads of all 40 patches for PUCT candidate keys: the patch readings are imputed from the current belief
    (reading = 1 iff mu >= 0.5); the board's real readings are never used. Returns hbar [n,40,d], dl [n,40,36]."""
    dev = l0.device
    g = RO.geo(dev)
    n = l0.shape[0]
    patch = torch.arange(env6.NPATCH, device=dev)[None].expand(n, -1).reshape(-1)
    idx = torch.arange(n, device=dev).repeat_interleave(env6.NPATCH)
    imp = (mu >= 0.5)
    hb_all, dl_all = [], []
    for a in range(0, idx.shape[0], chunk):
        sl = slice(a, a + chunk)
        ii, pp = idx[sl], patch[sl]
        R = ii.shape[0]
        P6 = g["patches"][pp]
        root = ev_root[ii]
        pm = torch.zeros(R, 36, dtype=torch.bool, device=dev)
        pm.scatter_(1, P6, True)
        ev = torch.where(pm & (root == 0), 1 + imp[ii].long(), root)
        z = torch.zeros(R, 36, dtype=torch.bool, device=dev)
        lr_, h, att = model(ev, B["note"][ii], z, z.long(), need_attn=True)
        lr_, h, att = lr_.float(), h.float(), att.float()
        w_ = att[:, :, :36].gather(2, P6[:, None, :].expand(R, att.shape[1], 6)).mean(1)
        w_ = w_ / w_.sum(1, keepdim=True).clamp_min(1e-12)
        hb_all.append((w_[..., None] * h[:, :36].gather(1, P6[..., None].expand(R, 6, h.shape[-1]))).sum(1))
        dl_all.append(lr_ - l0[ii])
    return torch.cat(hb_all).view(n, env6.NPATCH, -1), torch.cat(dl_all).view(n, env6.NPATCH, 36)


# ------------------------------------------------------------------------------------------------ keys and pools
def salience(dl):
    return dl.abs() / (dl.norm(dim=-1, keepdim=True) + 1e-6)


class KeyNet(nn.Module):
    """K1 raw: normalize(W LN(hbar)); K2 bag-centred: normalize(W LN(hbar - bag mean)); K3 salience: normalize(W s);
    K4: normalize(W [LN(hbar - bag mean) || s]). s = |Delta l| / ||Delta l||: theta_real's own output shift, no patch indices."""

    def __init__(self, mode, d=128, kdim=16):
        super().__init__()
        self.mode, self.d = mode, d
        self.W = nn.Linear({"K1": d, "K2": d, "K3": 36, "K4": d + 36}[mode], kdim)

    def embed(self, hbar, dl, center=None):
        s = salience(dl)
        if self.mode == "K1":
            x = F.layer_norm(hbar, (self.d,))
        elif self.mode == "K3":
            x = s
        else:
            hc = F.layer_norm(hbar - center, (self.d,))
            x = hc if self.mode == "K2" else torch.cat([hc, s], -1)
        return F.normalize(self.W(x), dim=-1), s

    def forward(self, hbar, dl, valid):
        v = valid.float()[..., None]
        center = (hbar * v).sum(1, keepdim=True) / v.sum(1, keepdim=True).clamp_min(1)
        return self.embed(hbar, dl, center)


def key_align_loss(k, s, valid):
    m = valid[:, :, None] & valid[:, None, :] & ~torch.eye(k.shape[1], dtype=torch.bool, device=k.device)[None]
    d = (k @ k.transpose(1, 2) - s @ s.transpose(1, 2)) ** 2
    return (d * m.float()).sum() / m.float().sum().clamp_min(1)


class LoopedLinT4(nn.Module):
    """Weight-tied 4-step linear-attention / preconditioned Richardson pool (arXiv:2402.14180 style).
    x^(t+1) = x^(t) + eta_t P_t * (b - G x^(t)), x^(0) = 0, b_r = beta_r Delta l_r, G_rb = relu(<k_r,k_b>)^gamma / sqrt(c_r c_b);
    l* = l0 + s * sum_r x_r^(T). Per-read weights w_r = x_r / Delta l_r."""

    def __init__(self, T=4, gamma=2.0, init_s=1.0 / 24):
        super().__init__()
        self.T, self.gamma = T, gamma
        self.eta = nn.Parameter(torch.full((T,), 0.5))
        self.P = nn.Parameter(torch.ones(T, wm.NC))
        self.log_s = nn.Parameter(torch.tensor(math.log(init_s)))

    def forward(self, l0, dl, beta, k, valid):
        v = valid.float()
        g = torch.relu(k @ k.transpose(1, 2)) ** self.gamma * v[:, None, :] * v[:, :, None]
        c = g.sum(2).clamp_min(1e-6)
        G = g / torch.sqrt(c[:, :, None] * c[:, None, :])
        b = (beta * v)[..., None] * dl.clamp(-8, 8)
        x = torch.zeros_like(b)
        for t in range(self.T):
            x = (x + self.eta[t] * self.P[t] * (b - G @ x)) * v[..., None]
        return l0 + self.log_s.exp() * x.sum(1), x


def nparams(*mods):
    return int(sum(p.numel() for m in mods for p in m.parameters()))


class Pools6:
    def __init__(self, keys, attn, lkey, looped, eta, u, alpha):
        self.keys, self.attn, self.lkey, self.looped = keys, attn, lkey, looped
        self.eta, self.u, self.alpha = eta, u, alpha

    @torch.no_grad()
    def all(self, l0, lr, dl, beta, hbar, valid):
        out, W, K = {}, {}, {}
        out["mix"] = wm.agg_mix(lr, valid)
        out["scaled_mix"] = l0 + self.alpha * (valid.float()[..., None] * dl.clamp(-8, 8)).sum(1)
        out["mmd"] = wm.agg_mmd(l0, dl, beta, valid, self.eta)
        for v in VARIANTS:
            k, _ = self.keys[v](hbar, dl, valid)
            la, w = self.attn[v](l0, dl, beta, k, valid)
            out[f"attn_{v}"], W[v], K[v] = la, w, k
        out["kc"] = wm.agg_kc(l0, dl, beta, K["K4"], valid, self.u)
        kl_, _ = self.lkey(hbar, dl, valid)
        out["looped"], _ = self.looped(l0, dl, beta, kl_, valid)
        K["looped"] = kl_
        return out, W, K


POOL_NAMES = ("mix", "scaled_mix", "mmd", "kc", "looped", "attn_K1", "attn_K2", "attn_K3", "attn_K4")


# ------------------------------------------------------------------------------------------------ calibration fit
def sample_bags(nb, Pn, gen, dev, kmin, kmax, p_fixed=None):
    K = torch.randint(kmin, kmax + 1, (nb,), device=dev, generator=gen)
    p = run.arcsine(nb, gen, dev) if p_fixed is None else torch.full((nb,), p_fixed, device=dev)
    j = torch.arange(kmax, device=dev)[None]
    valid = j < K[:, None]
    nnw = ((torch.rand(nb, kmax, device=dev, generator=gen) < p[:, None]) & valid).sum(1)
    pn = torch.rand(nb, Pn, device=dev, generator=gen).argsort(1)[:, :kmax]
    ps = torch.rand(nb, Pn, device=dev, generator=gen).argsort(1)[:, :kmax]
    isnw = j < nnw[:, None]
    sidx = (j - nnw[:, None]).clamp(min=0, max=kmax - 1)
    return torch.where(isnw, pn, Pn + ps.gather(1, sidx)), valid


def fit6(model, cfg, dev, log, gen):
    t0 = time.time()
    C = A2.boards2_t(A2.STREAM_CAL2, cfg.n_cal, dev)
    ev_root = A2.ev_from(C["s"], A2.cells_mask(C["anchors"], dev))
    l0 = RO.forward_chunks(model, ev_root, C["note"])
    one, zero = torch.ones(cfg.n_cal, device=dev), torch.zeros(cfg.n_cal, device=dev)
    pnw, _ = RO.sample_patches(l0, cfg.pool, one, gen, cfg.gamma_voi)
    pse, _ = RO.sample_patches(l0, cfg.pool, zero, gen, cfg.gamma_voi)
    patch = torch.cat([pnw, pse], 1)
    tau = draw_tau3(patch.shape, gen, dev)
    pool = bag_reads3(model, C, ev_root, l0, patch, tau, gen, cfg.chunk)
    h = cfg.n_ref // 2
    rnw, _ = RO.sample_patches(l0, h, one, gen, cfg.gamma_voi)
    rse, _ = RO.sample_patches(l0, cfg.n_ref - h, zero, gen, cfg.gamma_voi)
    lref, cov = A2.union_call(model, C, ev_root, torch.cat([rnw, rse], 1))
    ref = torch.sigmoid(lref)
    log(f"r6 cal pools + reference: {time.time() - t0:.0f}s; reference coverage {float(cov.mean()):.4f}")
    ntr = int(0.8 * cfg.n_cal)
    NP = 2 * cfg.pool
    flat = {k: pool[k].reshape(cfg.n_cal * NP, *pool[k].shape[2:]) for k in ("hbar", "ent", "dis", "lr")}
    X = RO.trust_features(flat, tau.reshape(-1))
    Y = RO.es_target({"lr": flat["lr"]}, patch.reshape(-1), C["occ"].repeat_interleave(NP, 0))
    torch.manual_seed(cfg.seed * 1000 + 321)
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
    bt = {str(t): float(beta[tau == t].mean()) for t in (0.3, 1.5, 2.5)}
    log(f"r6 theta_trust: train mse {np.mean(tl[-200:]):.4f} heldout {tmon:.4f}; beta by tau {bt}")

    def bag(nb, lo, hi, g_=gen, kmin=cfg.kmin, kmax=cfg.kmax, p_fixed=None):
        b = torch.randint(lo, hi, (nb,), device=dev, generator=g_)
        idx, valid = sample_bags(nb, cfg.pool, g_, dev, kmin, kmax, p_fixed)
        G = lambda x: run.gather_rows(x, b, idx)
        return l0[b], G(pool["lr"]), G(pool["dl"]), G(beta), G(pool["hbar"]), valid, ref[b]

    def train_pool(name, keyn, agg, seedoff):
        torch.manual_seed(cfg.seed * 1000 + seedoff)
        params = list(keyn.parameters()) + list(agg.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        hist = []
        for s in range(cfg.agg_steps):
            l0b, lrb, dlb, bb, hb, valid, rb = bag(cfg.agg_batch, 0, ntr)
            k, sal = keyn(hb, dlb, valid)
            ls, _ = agg(l0b, dlb, bb, k, valid)
            loss = F.binary_cross_entropy_with_logits(ls, rb)
            lk = key_align_loss(k, sal, valid) if keyn.mode == "K4" else torch.zeros((), device=dev)
            tot = loss + cfg.lam_key * lk
            opt.zero_grad(set_to_none=True)
            tot.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if s % 500 == 0 or s == cfg.agg_steps - 1:
                hist.append({"step": s, "bce": float(loss.detach()), "key_align": float(lk.detach())})
        keyn.eval()
        agg.eval()
        for p in params:
            p.requires_grad_(False)
        return hist

    keys, attn, hists = {}, {}, {}
    for i, v in enumerate(VARIANTS):
        torch.manual_seed(cfg.seed * 1000 + 331 + i)
        keys[v] = KeyNet(v, cfg.d).to(dev)
        attn[v] = wm.AttnL3().to(dev)
        hists[f"attn_{v}"] = train_pool(f"attn_{v}", keys[v], attn[v], 341 + i)
        log(f"r6 trained attn_L3 with {v} keys; last bce {hists[f'attn_{v}'][-1]['bce']:.4f}; {time.time() - t0:.0f}s")
    bm = float(beta.mean())
    torch.manual_seed(cfg.seed * 1000 + 351)
    lkey = KeyNet("K4", cfg.d).to(dev)
    looped = LoopedLinT4(cfg.loop_T, cfg.gamma_key, 1.0 / (cfg.K * bm)).to(dev)
    hists["looped"] = train_pool("looped", lkey, looped, 352)
    log(f"r6 trained looped_lin_T4; last bce {hists['looped'][-1]['bce']:.4f}; {time.time() - t0:.0f}s")

    def fit_scalar(fn, init, **bk):
        lp = torch.tensor(math.log(init), device=dev, requires_grad=True)
        o = torch.optim.Adam([lp], lr=0.02)
        for s in range(cfg.fit_steps):
            l0b, lrb, dlb, bb, hb, valid, rb = bag(cfg.agg_batch, 0, ntr, **bk)
            loss = F.binary_cross_entropy_with_logits(fn(l0b, dlb, bb, hb, valid, lp.exp()), rb)
            o.zero_grad(set_to_none=True)
            loss.backward()
            o.step()
        return float(lp.exp())

    eta = fit_scalar(lambda l0b, dlb, bb, hb, v, e: wm.agg_mmd(l0b, dlb, bb, v, e), 1.0 / (cfg.K * bm))
    u = fit_scalar(lambda l0b, dlb, bb, hb, v, e: wm.agg_kc(l0b, dlb, bb, keys["K4"](hb, dlb, v)[0], v, e), 1.0 / bm)
    alpha = fit_scalar(lambda l0b, dlb, bb, hb, v, e: l0b + e * (v.float()[..., None] * dlb.clamp(-8, 8)).sum(1),
                       1.0 / cfg.K, kmin=cfg.K, kmax=cfg.K, p_fixed=0.5)
    P = Pools6(keys, attn, lkey, looped, eta, u, alpha)
    mon = {}
    g2 = torch.Generator(device=dev).manual_seed(777)
    with torch.no_grad():
        for _ in range(4):
            l0b, lrb, dlb, bb, hb, valid, rb = bag(cfg.agg_batch, ntr, cfg.n_cal, g2)
            out, _, _ = P.all(l0b, lrb, dlb, bb, hb, valid)
            out["p0"] = l0b
            for k_, v_ in out.items():
                mon.setdefault(k_, []).append(float(F.binary_cross_entropy(torch.sigmoid(v_).clamp(1e-6, 1 - 1e-6), rb)))
    mon = {k_: float(np.mean(v_)) for k_, v_ in mon.items()}
    params = {f"attn_{v}": nparams(keys[v], attn[v]) for v in VARIANTS}
    params["looped"] = nparams(lkey, looped)
    log(f"r6 pools: eta {eta:.4g} u {u:.4g} alpha {alpha:.4g}; params {params}; held-out bag BCE to reference "
        + " ".join(f"{k_} {v_:.4f}" for k_, v_ in mon.items()) + f"; {time.time() - t0:.0f}s")
    info = {"eta": eta, "u": u, "alpha": alpha, "beta_by_tau": bt, "trust_train_mse": float(np.mean(tl[-200:])),
            "trust_heldout_mse": tmon, "hist": hists, "heldout_bag_bce_to_ref": mon, "params": params,
            "ref_coverage_mean": float(cov.mean()), "looped_eta": [float(x) for x in looped.eta],
            "looped_s": float(looped.log_s.exp()), "sec": time.time() - t0}
    return trust, P, info


# ------------------------------------------------------------------------------------------------ test protocols
def beta_of(trust, ro):
    nb, n = ro["patch"].shape
    flat = {k: ro[k].reshape(nb * n, *ro[k].shape[2:]) for k in ("hbar", "ent", "dis")}
    with torch.no_grad():
        return wm.beta_from(trust(RO.trust_features(flat, ro["tau"].reshape(-1)))).view(nb, n)


def cat_ro(a, b):
    return {k: torch.cat([a[k], b[k]], 1) for k in a}


def coverage(patch, dev):
    return RO.geo(dev)["pmask"][patch].amax(1)                      # [n, 36] float 0/1


def key_stats(K, patch, dev):
    """Within-bag key geometry vs true patch overlap (patch ids are used for SCORING only)."""
    g = RO.geo(dev)
    n, k = patch.shape
    iu = torch.triu_indices(k, k, 1, device=dev)
    cs = (K @ K.transpose(1, 2))[:, iu[0], iu[1]]
    ov = g["overlap"][patch[:, iu[0]], patch[:, iu[1]]]
    s1, s2 = g["sector"][patch[:, iu[0]]], g["sector"][patch[:, iu[1]]]
    same = cs[patch[:, iu[0]] == patch[:, iu[1]]]
    across = cs[((s1 == 0) & (s2 == 1)) | ((s1 == 1) & (s2 == 0))]
    return cs.reshape(-1), ov.reshape(-1), same, across


def puct_search(model, trust, P, T, ev_root, l0, pref_nw, gen, cfg):
    """4-wave key-space PUCT: Score_U(p) = (1 + lam_U 1[p in half(U)]) Var_mu(p) / (1 + sum_{r'} relu(<k_p,k_r'>)^2)^alpha_nov,
    6 distinct patches per wave sampled in proportion to the score; crowding against reads of earlier waves; candidate keys
    from prospective reads with imputed readings; mu = attn_L3-K4 merged belief of the reads so far (root in wave 1)."""
    dev = l0.device
    g = RO.geo(dev)
    n = l0.shape[0]
    half = g["in_nw"] if pref_nw else g["in_se"]
    pref = 1.0 + cfg.lam_U * half.float()
    keyn, attn = P.keys["K4"], P.attn["K4"]
    mu = torch.sigmoid(l0)
    bag = None
    waves = []
    for w in range(cfg.waves):
        var = (mu * (1 - mu)) @ g["pmask"].T                                # [n, 40]
        if bag is None:
            crowd = torch.zeros_like(var)
        else:
            valid = torch.ones(bag["patch"].shape, dtype=torch.bool, device=dev)
            ke, _ = keyn(bag["hbar"], bag["dl"], valid)
            hb_c, dl_c = prospect(model, T, ev_root, l0, mu, cfg.chunk)
            center = bag["hbar"].mean(1, keepdim=True)
            kc_, _ = keyn.embed(hb_c, dl_c, center)
            crowd = (torch.relu(kc_ @ ke.transpose(1, 2)) ** 2).sum(-1)    # [n, 40]
        score = pref[None] * var / (1.0 + crowd) ** cfg.alpha_nov
        pick = torch.multinomial(score, cfg.wave_reads, replacement=False, generator=gen)
        tau = RO.draw_tau(pick.shape, gen, dev)
        ro = bag_reads3(model, T, ev_root, l0, pick, tau, gen, cfg.chunk)
        ro["beta"] = beta_of(trust, ro)
        bag = ro if bag is None else cat_ro(bag, ro)
        waves.append(pick)
        valid = torch.ones(bag["patch"].shape, dtype=torch.bool, device=dev)
        with torch.no_grad():
            k, _ = keyn(bag["hbar"], bag["dl"], valid)
            la, _ = attn(l0, bag["dl"], bag["beta"], k, valid)
        mu = torch.sigmoid(la)
    return bag, waves


def open_search(model, trust, T, ev_root, l0, p_nw, gen, cfg, K=None, tau=None):
    n = l0.shape[0]
    patch, half = RO.sample_patches(l0, K or cfg.K, torch.full((n,), p_nw, device=l0.device), gen, cfg.gamma_voi)
    tau = RO.draw_tau(patch.shape, gen, l0.device) if tau is None else torch.full(patch.shape, tau, device=l0.device)
    ro = bag_reads3(model, T, ev_root, l0, patch, tau, gen, cfg.chunk)
    ro["beta"] = beta_of(trust, ro)
    ro["half_nw"] = half
    return ro


def merge(P, l0, ro, valid=None):
    if valid is None:
        valid = torch.ones(ro["patch"].shape, dtype=torch.bool, device=l0.device)
    return P.all(l0, ro["lr"], ro["dl"], ro["beta"], ro["hbar"], valid)


def test6(model, trust, P, cfg, dev, log, gen, Pf):
    t0 = time.time()
    n = cfg.n_test
    T = A2.boards2_t(A2.STREAM_TEST2, n, dev)
    g = RO.geo(dev)
    ev_root = A2.ev_from(T["s"], A2.cells_mask(T["anchors"], dev))
    ev_full = A2.ev_from(T["s"], torch.ones_like(T["s"]))
    ev_std6 = A2.ev_from(T["s"], A2.cells_mask(T["std6"], dev))
    Pfull = env6.bayes_marginals(ev_full, T["note"], Pf).float()           # EVALUATION ONLY
    Pstd6 = env6.bayes_marginals(ev_std6, T["note"], Pf).float()
    l0 = RO.forward_chunks(model, ev_root, T["note"])
    mu0 = torch.sigmoid(l0)
    lstd = RO.forward_chunks(model, ev_std6, T["note"])
    kl = lambda m: float(wm.bern_kl(Pfull, m).mean())
    pear = lambda a, b: float(np.corrcoef(a.cpu().numpy().ravel(), b.cpu().numpy().ravel())[0, 1])
    R = {"n_test": n, "clause1_corr_std6": pear(torch.sigmoid(lstd), Pstd6),
         "clause1_kl_std6": float(wm.bern_kl(Pstd6, torch.sigmoid(lstd)).mean()), "kl_p0": kl(mu0)}
    raw = {"pstar": run.r5(Pfull), "mu0": run.r5(mu0), "occ": T["occ"].int().tolist(), "anchors": T["anchors"].tolist()}
    sig = lambda x: torch.sigmoid(x)

    # ---- P-FullSupport: 6 partition tiles + 18 clean reads skewed 90/10
    fs = {}
    tiles = torch.tensor(TILES, device=dev)[None].expand(n, -1)
    for name, p in (("NW", cfg.p_full), ("SE", 1 - cfg.p_full)):
        sk, half = RO.sample_patches(l0, cfg.K - len(TILES), torch.full((n,), p, device=dev), gen, cfg.gamma_voi)
        patch = torch.cat([tiles, sk], 1)
        ro = bag_reads3(model, T, ev_root, l0, patch, torch.full(patch.shape, 0.3, device=dev), gen, cfg.chunk)
        ro["beta"] = beta_of(trust, ro)
        out, W, K = merge(P, l0, ro)
        lu, cov = A2.union_call(model, T, ev_root, patch)
        out["union"] = lu
        fs[name] = {"out": out, "half": half, "cov": cov, "patch": patch}
        R[f"fs_frac_pref_{name}"] = float((half if name == "NW" else ~half).float().mean())
        R[f"fs_coverage_{name}"] = float(cov.mean())
        raw[f"fs_{name}"] = {"patch": patch.tolist(), **{a: run.r5(sig(out[a])) for a in list(POOL_NAMES) + ["union"]}}
    for a in list(POOL_NAMES) + ["union"]:
        mN, mS = sig(fs["NW"]["out"][a]), sig(fs["SE"]["out"][a])
        R[f"fs_kl_{a}"] = 0.5 * (kl(mN) + kl(mS))
        R[f"fs_jsd_{a}"] = float(wm.bern_jsd(mN, mS).mean())
    log("r6 P-FullSupport: " + " ".join(f"{a} kl {R[f'fs_kl_{a}']:.4f} jsd {R[f'fs_jsd_{a}']:.5f}" for a in
                                        ("scaled_mix", "mmd", "kc", "looped", "attn_K4", "union")))

    # ---- P-ActivePUCT: open_U (85/15) vs 4-wave key-space PUCT, under U_NW and U_SE (reads tau 50/50 0.3 / 1.5)
    ap = {}
    for name, nw in (("NW", True), ("SE", False)):
        op = open_search(model, trust, T, ev_root, l0, cfg.p_open if nw else 1 - cfg.p_open, gen, cfg)
        o_out, _, o_K = merge(P, l0, op)
        pb, waves = puct_search(model, trust, P, T, ev_root, l0, nw, gen, cfg)
        p_out, _, _ = merge(P, l0, pb)
        ap[name] = {"open": o_out, "puct": p_out, "open_ro": op, "puct_ro": pb, "waves": waves, "open_K": o_K}
        for kind, ro in (("open", op), ("puct", pb)):
            R[f"ap_coverage_{kind}_{name}"] = float(coverage(ro["patch"], dev).mean(1).mean())
        w1 = waves[0]
        inh = g["in_nw"] if nw else g["in_se"]
        R[f"ap_wave1_pref_share_{name}"] = float(inh[w1].float().mean())
        R[f"ap_wave_pref_share_{name}"] = [float(inh[w].float().mean()) for w in waves]
        R[f"ap_open_pref_share_{name}"] = float(inh[op["patch"]].float().mean())
        raw[f"ap_{name}"] = {"open_patch": op["patch"].tolist(), "puct_patch": pb["patch"].tolist(),
                             **{f"open_{a}": run.r5(sig(o_out[a])) for a in ("attn_K4", "mix", "scaled_mix", "mmd", "kc")},
                             **{f"puct_{a}": run.r5(sig(p_out[a])) for a in ("attn_K4", "mix", "scaled_mix", "mmd", "kc")}}
    for kind in ("open", "puct"):
        R[f"ap_coverage_{kind}"] = 0.5 * (R[f"ap_coverage_{kind}_NW"] + R[f"ap_coverage_{kind}_SE"])
        for a in POOL_NAMES:
            mN, mS = sig(ap["NW"][kind][a]), sig(ap["SE"][kind][a])
            R[f"ap_{kind}_kl_{a}"] = 0.5 * (kl(mN) + kl(mS))
            R[f"ap_{kind}_jsd_{a}"] = float(wm.bern_jsd(mN, mS).mean())
        lN, _ = A2.union_call(model, T, ev_root, ap["NW"][f"{kind}_ro"]["patch"])
        lS, _ = A2.union_call(model, T, ev_root, ap["SE"][f"{kind}_ro"]["patch"])
        R[f"ap_{kind}_jsd_union"] = float(wm.bern_jsd(sig(lN), sig(lS)).mean())
        R[f"ap_{kind}_kl_union"] = 0.5 * (kl(sig(lN)) + kl(sig(lS)))
    log(f"r6 P-ActivePUCT: coverage open {R['ap_coverage_open']:.3f} puct {R['ap_coverage_puct']:.3f}; attn_K4 jsd open "
        f"{R['ap_open_jsd_attn_K4']:.5f} puct {R['ap_puct_jsd_attn_K4']:.5f}; kl open {R['ap_open_kl_attn_K4']:.4f} puct "
        f"{R['ap_puct_kl_attn_K4']:.4f}; wave-1 NW share {R['ap_wave1_pref_share_NW']:.3f}")

    # ---- key geometry (E1-D1): within-bag pairs of the open_U NW / SE bags and an unskewed bag (R5-E1 protocol)
    un = open_search(model, trust, T, ev_root, l0, 0.5, gen, cfg)
    _, _, un_K = merge(P, l0, un)
    keysets = [(ap["NW"]["open_K"], ap["NW"]["open_ro"]["patch"]), (ap["SE"]["open_K"], ap["SE"]["open_ro"]["patch"]),
               (un_K, un["patch"])]
    for v in list(VARIANTS) + ["looped"]:
        cs_l, ov_l, sm_l, ac_l = [], [], [], []
        for Kd, pt in keysets:
            cs, ov, sm, ac = key_stats(Kd[v], pt, dev)
            cs_l.append(cs)
            ov_l.append(ov)
            sm_l.append(sm)
            ac_l.append(ac)
        cs, ov = torch.cat(cs_l), torch.cat(ov_l).float()
        R[f"key_{v}_pearson"] = float(np.corrcoef(cs.cpu().numpy(), (ov / 6).cpu().numpy())[0, 1])
        R[f"key_{v}_same"] = float(torch.cat(sm_l).mean())
        R[f"key_{v}_across"] = float(torch.cat(ac_l).mean())
        R[f"key_{v}_gap"] = R[f"key_{v}_same"] - R[f"key_{v}_across"]
        R[f"key_{v}_by_overlap"] = {str(o): float(cs[ov == o].mean()) for o in (0, 1, 2, 3, 4, 6)}
    raw["key_bags"] = {"patch": [pt.tolist() for _, pt in keysets],
                       **{f"keys_{v}": [run.r5(Kd[v], 4) for Kd, _ in keysets] for v in ("K3", "K4")}}
    log("r6 keys: " + " ".join(f"{v} r {R[f'key_{v}_pearson']:.3f} gap {R[f'key_{v}_gap']:.3f}" for v in
                               list(VARIANTS) + ["looped"]))

    # ---- controlled bags (E1-D3): N in {1,2,4,8,16} reads of one patch at tau 0.3 / 1.5 / 2.5 + 8 shared background reads
    astar, _ = RO.sample_patches(l0, 1, torch.full((n,), 0.5, device=dev), gen, cfg.gamma_voi)
    excl = g["overlap"][astar[:, 0]] > 0
    bgp, _ = RO.sample_patches(l0, cfg.n_bg, torch.full((n,), 0.5, device=dev), gen, cfg.gamma_voi, exclude=excl)
    bg = bag_reads3(model, T, ev_root, l0, bgp, RO.draw_tau(bgp.shape, gen, dev), gen, cfg.chunk)
    bg["beta"] = beta_of(trust, bg)
    pm = g["pmask"][astar[:, 0]].bool()
    ctrl = {}
    raw["controlled"] = {"astar": astar[:, 0].tolist(), "bg_patch": bgp.tolist()}
    for tv in (0.3, 1.5, 2.5):
        cp = astar.expand(n, cfg.n_clu)
        cl = bag_reads3(model, T, ev_root, l0, cp, torch.full(cp.shape, tv, device=dev), gen, cfg.chunk)
        cl["beta"] = beta_of(trust, cl)
        ro = cat_ro(cl, bg)
        for N in (1, 2, 4, 8, 16):
            valid = torch.zeros(ro["patch"].shape, dtype=torch.bool, device=dev)
            valid[:, :N] = True
            valid[:, cfg.n_clu:] = True
            out, W, _ = P.all(l0, ro["lr"], ro["dl"], ro["beta"], ro["hbar"], valid)
            row = {}
            for v in VARIANTS:
                Wn = (W[v][:, :N] * pm[:, None, :].float()).sum(2) / 6.0
                row[f"W_{v}"] = float(Wn.sum(1).mean())
                if v == "K4":
                    raw["controlled"][f"t{tv}_N{N}_W_K4"] = run.r5(Wn.sum(1))
            for a in POOL_NAMES:
                row[f"mse_{a}"] = float(((sig(out[a]) - Pfull) ** 2).mean())
            ctrl[f"t{tv}_N{N}"] = row
            raw["controlled"][f"t{tv}_N{N}"] = {a: run.r5(sig(out[a])) for a in ("attn_K4", "mmd", "kc", "mix")}
        log(f"r6 controlled tau={tv}: " + " ".join(f"N{N} W {ctrl[f't{tv}_N{N}']['W_K4']:.3f}" for N in (1, 2, 4, 8, 16)))
    R["controlled"] = ctrl
    R["sec"] = time.time() - t0
    log("r6 test done: " + json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in R.items()
                                         if not isinstance(v, (dict, list))}))
    return R, raw


def main(argv=None):
    ap = argparse.ArgumentParser(description="R6-E1 one seed")
    for f in dataclasses.fields(Cfg6):
        ap.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    cfg = Cfg6(**vars(ap.parse_args(argv)))
    os.makedirs(cfg.out, exist_ok=True)
    dev = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
    torch.set_num_threads(cfg.threads)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    log = run.Log(cfg.out)
    log("config r6e1", json.dumps(asdict(cfg)), "device", dev, torch.__version__)
    T0 = time.time()
    P = env6.placements()
    Pt = torch.as_tensor(P, device=dev)
    Pf = Pt.double()
    summary = {"cfg": asdict(cfg), "experiment": "R6-E1", "n_placements": int(P.shape[0]), "torch": torch.__version__,
               "device": str(dev), "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None}
    if cfg.load_wm and os.path.exists(cfg.load_wm):
        prev = json.load(open(os.path.join(cfg.out, "summary.json")))
        model = wm.WorldModel(cfg.d, cfg.layers, cfg.heads, cfg.ff).to(dev)
        model.load_state_dict(torch.load(cfg.load_wm, map_location=dev))
        model.eval()
        curve, tsec = prev["train_curve"], prev["train_sec"]
        summary["resumed_from_saved_theta_real"] = cfg.load_wm
        log("r6 RESUME: theta_real loaded from", cfg.load_wm)
    else:
        model, curve, tsec = A2.train_world_model2(cfg, dev, log, Pt, Pf)
        torch.save(model.state_dict(), os.path.join(cfg.out, "theta_real_v2.pt"))   # stays on the VM (not synced)
    summary.update(theta_real_params=model.n_params(), train_sec=tsec, train_curve=curve)
    run.dump(os.path.join(cfg.out, "summary.json"), summary)
    for p in model.parameters():
        p.requires_grad_(False)
    gen = torch.Generator(device=dev).manual_seed(cfg.seed * 1000 + 371)
    trust, Pools, info = fit6(model, cfg, dev, log, gen)
    summary["fit"] = info
    run.dump(os.path.join(cfg.out, "summary.json"), summary)
    res, raw = test6(model, trust, Pools, cfg, dev, log, gen, Pf)
    summary["test"] = res
    run.dump(os.path.join(cfg.out, "raw.json"), raw)
    summary["total_sec"] = time.time() - T0
    summary["done"] = True
    run.dump(os.path.join(cfg.out, "summary.json"), summary)
    log(f"r6 done in {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
