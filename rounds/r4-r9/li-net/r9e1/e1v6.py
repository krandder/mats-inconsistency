"""R9-E1 (E-INFO-STRONG-ROOT): genuine strong-root anytime safety (Mode A: cumulative-context reads as R8; Mode B:
bounded-context local reads added to a strong-root anchor) and the 1-step precision-aware crowding pool econ_prec_K3 on
6x6 ShipHunt. Spec: /home/agent/workspaces/mats/outbox/experiments-round-9-spec.md section 1 (gemini). Pre-registration:
PREREG-R9-E1.md. No world-model training: theta_real v2 is LOADED from li-net/r7e1/ckpt/s<seed>/theta_real_v2.npz after its
sha256 is checked against its .json. Imported unchanged: li-net/r8e1/e1v5.py (and through it li-net/r7e1/e1v4.py,
li-net/r6e1/e1v3.py and li-net/r5e1: env6, wm, rollouts, run, partA2).

Per seed: load theta_real -> calibration (stream 21): strong-root selection on calibration boards only (mean over cells of
KL(ref(36) || root(M_0)) for M_0 in 30/32/34, ref(36) = theta_real on all 36 readings; the smallest with KL <= 0.035, else 34)
-> R8's standard pool (4,000 boards x 96 reads, tau uniform over 0.3 / 1.5 / 2.5) and K = 128 reference -> read pools at
M_0 = 6, 12, 18, 30, 32, 34 (first 2,000 boards x 64 reads) and ref(M_0) -> theta_trust -> anytime pools with an anchor token
and precision-weighted crowding (looped_attn_T3, attn_L3-K3) trained by BCE to ref(M_0) over the 7 root strengths and
K in {0, 1, 2, 4, 8, 16, 24, 32}, half Mode A and half Mode B bags at M_0 >= 6 -> R7-style pools on the standard root
(attn_L3-K1, looped_lin_T4, jacobi_lin_T4) and fitted scalars on standard bags (as R8) -> D3 calibration probes: econ_kc_K3
(u, lambda) and econ_prec_K3 (u, a, b; lambda(beta) = softplus(a - b beta) as amended) -> calibration P-FullSupport bags ->
jacobi_lin_T4_prec trained on 0.5 x standard-bag BCE + 0.5 x P-FullSupport BCE + 0.25 x D3 MSE (as amended) -> tests on the
500 R5-R8 test boards: E1-D0 (clause-1 Pearson, K = 0 identity), the sparse-root
P-FullSupport curve on the interleaved 32-read stream (as R8), strong-root curves in Mode A and Mode B at M_0 in
{18, 30, 32, 34}, D3 controlled bags. Exact Bayes: test metrics and the gate only.
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

LI = "/home/agent/workspaces/eurisko-study/li-net"
R8DIR = os.environ.get("R8E1_DIR", f"{LI}/r8e1")
if R8DIR not in sys.path:
    sys.path.insert(0, R8DIR)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import e1v5 as E5  # noqa: E402  (R8-E1, unchanged)
from e1v5 import A2, E3, E4, RO, env6, run, wm  # noqa: E402

M0S = (2, 6, 12, 18, 30, 32, 34)                   # spec 1.1 item 1: the extended nested ladder
KS_TRAIN = (0, 1, 2, 4, 8, 16, 24, 32)
KS_FS = (0, 1, 2, 4, 8, 16, 24, 32)
KS_SR = (0, 1, 2, 4, 8, 16, 24)
ANY = E5.ANY                                        # ("looped_attn", "attn_K3"): anchor-aware anytime pools
OTHER = E5.OTHER + ("econ_prec",)
POOL_NAMES = ANY + OTHER
WPOOLS = E5.WPOOLS + ("econ_prec",)
FS_RAW_K24 = ("jacobi_prec", "jacobi", "looped_lin", "kc_K1", "kc_K3", "econ_prec", "econ_D3")
SR_RAW = ("looped_attn", "attn_K3", "mmd", "mix", "scaled_mix")


@dataclass
class Cfg9(E5.Cfg8):
    out: str = "/tmp/r9e1/dev"
    sr_roots: str = "18,30,32,34"      # strong-root curves (Mode A and Mode B) at these M_0; the rule uses M_0_strong
    strong_cands: str = "30,32,34"     # spec 1.1 item 1: candidates for M_0_strong, smallest first
    kl_strong: float = 0.035           # calibration KL(ref(36) || root) threshold for M_0_strong (reading 1)
    p_modeB: float = 0.5               # share of Mode B bags at M_0 >= 6 in anytime training (reading 4)
    n_pfs: int = 2000                  # calibration boards with P-FullSupport bags (reading 6)
    std_w: float = 0.5                 # jacobi_lin_T4_prec loss (gemini's amendment to reading 6): std_w x standard-bag BCE
    pfs_w: float = 0.5                 #   + pfs_w x calibration P-FullSupport BCE
    d3_w: float = 0.25                 #   + d3_w x D3 MSE (spec 1.1 item 2)


# ------------------------------------------------------------------------------------------------ new pieces
def select_strong(kls, cands, thr):
    """Spec 1.1 item 1: the smallest candidate M_0 whose calibration KL <= thr; if none, the largest candidate."""
    for M in cands:
        if kls[str(M)] <= thr:
            return int(M)
    return int(cands[-1])


def econ_prec_lambda(beta, a, b):
    """lambda(beta) = softplus(a - b beta), b >= 0 (gemini's amendment to reading 5, 19:00Z; the spec's 2 sigmoid(a - b beta)
    capped the ridge at 2)."""
    return F.softplus(a - b * beta)


def econ_prec_weights(beta, k, valid, u, a, b, gamma=2.0):
    """econ_prec_K3: w_r = u beta_r / (lambda(beta_r) + sum_b beta_b relu(<k_r,k_b>)^gamma) (reading 5: gamma = 2 as econ_kc_K3)."""
    v = valid.float()
    g = torch.relu(k @ k.transpose(1, 2)) ** gamma * v[:, None, :]
    c = (g * (beta * v)[:, None, :]).sum(2)
    return u * beta / (econ_prec_lambda(beta, a, b) + c) * v


def agg_econ_prec(l0, dl, beta, k, valid, u, a, b, gamma=2.0):
    w = econ_prec_weights(beta, k, valid, u, a, b, gamma)
    return l0 + (w[..., None] * dl).sum(1), w


class Pools9(E5.Pools8):
    """R8's pools (E5.Pools8, unchanged) plus econ_prec_K3 on the attn_L3-K3 pool's K3 key head (as econ_kc_K3)."""

    @torch.no_grad()
    def all(self, l0, lprior, M, lr, dl, beta, hbar, valid, need_w=False):
        out, W = super().all(l0, lprior, M, lr, dl, beta, hbar, valid, need_w)
        if dl.shape[1] == 0:
            out["econ_prec"] = l0.clone()
            return out, W
        k3, _ = self.any["attn_K3"][0].embed(None, dl)
        sc = self.sc
        out["econ_prec"], wep = agg_econ_prec(l0, dl, beta, k3, valid, sc["econ_prec_u"], sc["econ_prec_a"], sc["econ_prec_b"])
        if need_w:
            W["econ_prec"] = wep[..., None].expand(-1, -1, wm.NC)
        return out, W


def any_bag9(L0S, LPa, DLS, BES, REFS, M0T, KST, beta_clean, nb, lo, hi, g_, kt, pa, p_b, dev, force=None):
    """Anytime training bag (reading 4). Root strength M_0 uniform over the ladder, K uniform over KS_TRAIN. Mode A: the reads
    of the M_0 root (Delta l over l0(M_0), as R8). Mode B (probability p_b at M_0 >= 6): the standard root's reads (Delta l over
    l0(2)) added to the M_0 root. Always the anchor l0(M_0) with beta_0 = M_0 beta_clean / 6 and the target ref(M_0)."""
    b = torch.randint(lo, hi, (nb,), device=dev, generator=g_)
    Mi = torch.randint(L0S.shape[0], (nb,), device=dev, generator=g_)
    K = KST[torch.randint(len(KST), (nb,), device=dev, generator=g_)]
    u = torch.rand(nb, device=dev, generator=g_)
    if force == "A":
        mB = torch.zeros(nb, dtype=torch.bool, device=dev)
    elif force == "B":
        mB = Mi > 0
    else:
        mB = (u < p_b) & (Mi > 0)
    src = torch.where(mB, torch.zeros_like(Mi), Mi)
    idx, _ = E3.sample_bags(nb, pa, g_, dev, kt, kt)
    idx = idx.gather(1, torch.rand(nb, kt, device=dev, generator=g_).argsort(1))
    valid = torch.arange(kt, device=dev)[None] < K[:, None]
    dl = DLS[src, b].gather(1, idx[..., None].expand(nb, kt, DLS.shape[-1]))
    be = BES[src, b].gather(1, idx)
    return L0S[Mi, b], LPa[b], dl, be, valid, M0T[Mi] * beta_clean / 6.0, REFS[Mi, b], Mi, K, mB


def pfs_bag(PL0, PDL, PBE, PREF, nb, lo, hi, g_, dev, kmax=32):
    """Calibration P-FullSupport bag (reading 6): a board, a half (U_NW / U_SE stream) and a prefix K uniform over 1..kmax."""
    b = torch.randint(lo, hi, (nb,), device=dev, generator=g_)
    h = torch.randint(2, (nb,), device=dev, generator=g_)
    K = torch.randint(1, kmax + 1, (nb,), device=dev, generator=g_)
    valid = torch.arange(kmax, device=dev)[None] < K[:, None]
    return PL0[b], PDL[h, b], PBE[h, b], valid, PREF[b]


def fit_econ_prec(sampler, k3e, steps, dev, init=(1.0, math.log(math.expm1(1.0)), 0.1)):
    """(u, a, b) by Adam (lr 0.02) on BCE to the reference over D3 calibration probe bags (reading 5 as amended; R8's econ_kc_K3
    procedure). u = exp(.), a free, b = softplus(.) >= 0; init u = 1, a = log(e - 1) (so lambda(0) = 1, as econ_kc_K3's
    lambda init), b = 0.1."""
    lu = torch.tensor(math.log(init[0]), device=dev, requires_grad=True)
    a = torch.tensor(float(init[1]), device=dev, requires_grad=True)
    bl = torch.tensor(math.log(math.expm1(init[2])), device=dev, requires_grad=True)
    o = torch.optim.Adam([lu, a, bl], lr=0.02)
    for s in range(steps):
        l0b, dlb, bb, hb, valid, rb = sampler()
        out, _ = agg_econ_prec(l0b, dlb, bb, k3e(dlb), valid, lu.exp(), a, F.softplus(bl))
        loss = F.binary_cross_entropy_with_logits(out, rb)
        o.zero_grad(set_to_none=True)
        loss.backward()
        o.step()
    return float(lu.exp()), float(a), float(F.softplus(bl))


# ------------------------------------------------------------------------------------------------ calibration fit
def fit9(model, cfg, dev, log, gen):
    t0 = time.time()
    g = RO.geo(dev)
    cands = tuple(int(x) for x in cfg.strong_cands.split(","))
    C = A2.boards2_t(A2.STREAM_CAL2, cfg.n_cal, dev)
    ordC = torch.as_tensor(E5.anchor_order(A2.STREAM_CAL2, cfg.n_cal), device=dev)
    lprior = E5.prior_logits(model, C)
    ev2, l0 = E5.root(model, C, ordC, 2)
    # ---- strong-root selection on calibration boards only (spec 1.1 item 1; reading 1). ref(36): the model on all 36 readings.
    r36 = torch.sigmoid(RO.forward_chunks(model, A2.ev_from(C["s"], torch.ones_like(C["s"])), C["note"]))
    sel, sel_rev = {}, {}
    for M in cands:
        q = torch.sigmoid(E5.root(model, C, ordC, M)[1])
        sel[str(M)] = float(wm.bern_kl(r36, q).mean())
        sel_rev[str(M)] = float(wm.bern_kl(q, r36).mean())
    M_strong = select_strong(sel, cands, cfg.kl_strong)
    log(f"r9 strong-root selection (calibration boards): KL(ref36||root) {json.dumps({k: round(v, 5) for k, v in sel.items()})}; "
        f"KL(root||ref36) (reported) {json.dumps({k: round(v, 5) for k, v in sel_rev.items()})}; M_0_strong = {M_strong}")
    # ---- R8's standard pool and reference (unchanged)
    one, zero = torch.ones(cfg.n_cal, device=dev), torch.zeros(cfg.n_cal, device=dev)
    pnw, _ = RO.sample_patches(l0, cfg.pool, one, gen, cfg.gamma_voi)
    pse, _ = RO.sample_patches(l0, cfg.pool, zero, gen, cfg.gamma_voi)
    patch = torch.cat([pnw, pse], 1)
    tau = E3.draw_tau3(patch.shape, gen, dev)
    pool = E3.bag_reads3(model, C, ev2, l0, patch, tau, gen, cfg.chunk)
    h = cfg.n_ref // 2
    rnw, _ = RO.sample_patches(l0, h, one, gen, cfg.gamma_voi)
    rse, _ = RO.sample_patches(l0, cfg.n_ref - h, zero, gen, cfg.gamma_voi)
    lref, cov = A2.union_call(model, C, ev2, torch.cat([rnw, rse], 1))
    ref = torch.sigmoid(lref)
    log(f"r9 standard pool + reference: {time.time() - t0:.0f}s; reference coverage {float(cov.mean()):.4f}")
    # ---- read pools at M_0 = 6 ... 34 on the first n_any boards; M_0 = 2 reuses the standard pool
    na, pa = cfg.n_any, cfg.pool_any
    Cs = {k: v[:na] for k, v in C.items()}
    idx2 = torch.cat([torch.arange(pa, device=dev), cfg.pool + torch.arange(pa, device=dev)])
    anyp = {2: {"l0": l0[:na], "ro": {k: pool[k][:na][:, idx2] for k in ("lr", "hbar", "ent", "dis", "dl")},
                "tau": tau[:na][:, idx2], "patch": patch[:na][:, idx2]}}
    o1, z1 = torch.ones(na, device=dev), torch.zeros(na, device=dev)
    refp = torch.cat([rnw, rse], 1)[:na]
    anyp[2]["ref"] = ref[:na]
    for M in M0S[1:]:
        evM, l0M = E5.root(model, Cs, ordC[:na], M)
        pn, _ = RO.sample_patches(l0M, pa, o1, gen, cfg.gamma_voi)
        ps, _ = RO.sample_patches(l0M, pa, z1, gen, cfg.gamma_voi)
        pt = torch.cat([pn, ps], 1)
        tt = E3.draw_tau3(pt.shape, gen, dev)
        ro = E3.bag_reads3(model, Cs, evM, l0M, pt, tt, gen, cfg.chunk)
        anyp[M] = {"l0": l0M, "ro": {k: ro[k] for k in ("lr", "hbar", "ent", "dis", "dl")}, "tau": tt, "patch": pt,
                   "ref": torch.sigmoid(A2.union_call(model, Cs, evM, refp)[0])}       # ref(M_0), as R8's amended reading 3
    log(f"r9 multi-root pools (M_0 {M0S[1:]}, {na} boards x {2 * pa} reads) and references ref(M_0): {time.time() - t0:.0f}s")
    # ---- theta_trust on reads of all root strengths (train boards: standard < 0.8 n_cal, multi-root < 0.8 n_any), as R8
    ntr, ntra = int(0.8 * cfg.n_cal), int(0.8 * na)
    NP = 2 * cfg.pool

    def feats(ro, tt, pt, occ):
        nb, n = pt.shape
        flat = {k: ro[k].reshape(nb * n, *ro[k].shape[2:]) for k in ("hbar", "ent", "dis", "lr")}
        X = RO.trust_features(flat, tt.reshape(-1))
        Y = RO.es_target({"lr": flat["lr"]}, pt.reshape(-1), occ.repeat_interleave(n, 0))
        return X, Y

    Xs, Ys = feats(pool, tau, patch, C["occ"])
    Xtr, Ytr, Xho, Yho = [Xs[:ntr * NP]], [Ys[:ntr * NP]], [Xs[ntr * NP:]], [Ys[ntr * NP:]]
    for M in M0S[1:]:
        X, Y = feats(anyp[M]["ro"], anyp[M]["tau"], anyp[M]["patch"], Cs["occ"])
        Xtr.append(X[:ntra * 2 * pa])
        Ytr.append(Y[:ntra * 2 * pa])
        Xho.append(X[ntra * 2 * pa:])
        Yho.append(Y[ntra * 2 * pa:])
    Xtr, Ytr, Xho, Yho = torch.cat(Xtr), torch.cat(Ytr), torch.cat(Xho), torch.cat(Yho)
    torch.manual_seed(cfg.seed * 1000 + 321)
    trust = wm.TrustNet(Xtr.shape[1]).to(dev)
    opt = torch.optim.Adam(trust.parameters(), lr=1e-3)
    tl = []
    for s in range(cfg.trust_steps):
        ib = torch.randint(Xtr.shape[0], (1024,), device=dev, generator=gen)
        loss = F.mse_loss(trust(Xtr[ib]), Ytr[ib])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        tl.append(float(loss.detach()))
    trust.eval()
    for p in trust.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        tmon = float(F.mse_loss(torch.cat([trust(Xho[a:a + 65536]) for a in range(0, Xho.shape[0], 65536)]), Yho))
        beta = wm.beta_from(torch.cat([trust(Xs[a:a + 65536]) for a in range(0, Xs.shape[0], 65536)])).view(cfg.n_cal, NP)
        for M in M0S[1:]:
            X, _ = feats(anyp[M]["ro"], anyp[M]["tau"], anyp[M]["patch"], Cs["occ"])
            anyp[M]["beta"] = wm.beta_from(torch.cat([trust(X[a:a + 65536]) for a in range(0, X.shape[0], 65536)])).view(na, 2 * pa)
    del Xs, Ys, Xtr, Ytr, Xho, Yho
    for M in M0S[1:]:
        anyp[M]["ro"].pop("hbar", None)                                   # not needed after theta_trust (K3 keys use Delta l)
    anyp[2]["beta"] = beta[:na][:, idx2]
    bt = {str(t): float(beta[tau == t].mean()) for t in (0.3, 1.5, 2.5)}
    beta_clean = bt["0.3"]
    bM = {str(M): {str(t): float(anyp[M]["beta"][anyp[M]["tau"] == t].mean()) for t in (0.3, 1.5, 2.5)} for M in M0S}
    log(f"r9 theta_trust: train mse {np.mean(tl[-200:]):.4f} heldout {tmon:.4f}; beta by tau {bt}; by root {bM}")

    # ---- anytime pools: anchor token, M_0 over the 7 strengths, K in KS_TRAIN, Mode A / Mode B bags, BCE to ref(M_0)
    L0S = torch.stack([anyp[M]["l0"] for M in M0S])
    DLS = torch.stack([anyp[M]["ro"]["dl"] for M in M0S])
    BES = torch.stack([anyp[M]["beta"] for M in M0S])
    REFS = torch.stack([anyp[M]["ref"] for M in M0S])
    M0T = torch.tensor(M0S, device=dev, dtype=torch.float32)
    KST = torch.tensor(KS_TRAIN, device=dev)
    LPa = lprior[:na]
    abag = lambda nb, lo, hi, g_, force=None: any_bag9(L0S, LPa, DLS, BES, REFS, M0T, KST, beta_clean, nb, lo, hi, g_,
                                                        cfg.K_tr, pa, cfg.p_modeB, dev, force)

    def train_any(keyn, agg, seedoff):
        torch.manual_seed(cfg.seed * 1000 + seedoff)
        params = list(keyn.parameters()) + list(agg.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        hist = []
        for s in range(cfg.agg_steps):
            l0b, lpb, dlb, bb, valid, b0, yb, _, _, _ = abag(cfg.agg_batch, 0, ntra, gen)
            dla, ba, k, va = E5.anchor_bag(keyn, l0b, lpb, dlb, bb, valid, b0)
            ls, _ = agg(l0b, dla, ba, k, va)
            loss = F.binary_cross_entropy_with_logits(ls, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if s % 500 == 0 or s == cfg.agg_steps - 1:
                hist.append({"step": s, "bce_to_ref": float(loss.detach())})
        keyn.eval()
        agg.eval()
        for p in params:
            p.requires_grad_(False)
        return hist

    hists, anyP = {}, {}
    torch.manual_seed(cfg.seed * 1000 + 381)
    kla, la = E3.KeyNet("K3", cfg.d).to(dev), E5.LoopedAttnT3P(beta_clean, T=cfg.attn_T).to(dev)
    hists["looped_attn"] = train_any(kla, la, 382)
    anyP["looped_attn"] = (kla, la)
    log(f"r9 trained anytime looped_attn_T3; last bce {hists['looped_attn'][-1]['bce_to_ref']:.4f}; {time.time() - t0:.0f}s")
    torch.manual_seed(cfg.seed * 1000 + 391)
    ka3, a3 = E3.KeyNet("K3", cfg.d).to(dev), E5.AttnL3P(beta_clean).to(dev)
    hists["attn_K3"] = train_any(ka3, a3, 392)
    anyP["attn_K3"] = (ka3, a3)
    log(f"r9 trained anytime attn_L3-K3; last bce {hists['attn_K3'][-1]['bce_to_ref']:.4f}; {time.time() - t0:.0f}s")

    # ---- R7-style pools on the standard root (BCE to the reference), as R8: attn_L3-K1, looped_lin_T4, jacobi_lin_T4
    def bag(nb, lo, hi, g_=gen, kmin=cfg.kmin, kmax=cfg.kmax, p_fixed=None):
        b = torch.randint(lo, hi, (nb,), device=dev, generator=g_)
        idx, valid = E3.sample_bags(nb, cfg.pool, g_, dev, kmin, kmax, p_fixed)
        G = lambda x: run.gather_rows(x, b, idx)
        return l0[b], G(pool["lr"]), G(pool["dl"]), G(beta), G(pool["hbar"]), valid, ref[b]

    def train_pool(keyn, agg, seedoff):
        torch.manual_seed(cfg.seed * 1000 + seedoff)
        params = list(keyn.parameters()) + list(agg.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        hist = []
        for s in range(cfg.agg_steps):
            l0b, lrb, dlb, bb, hb, valid, rb = bag(cfg.agg_batch, 0, ntr)
            k, sal = keyn(hb, dlb, valid)
            ls, _ = agg(l0b, dlb, bb, k, valid)
            loss = F.binary_cross_entropy_with_logits(ls, rb)
            lk = E3.key_align_loss(k, sal, valid) if keyn.mode == "K4" else torch.zeros((), device=dev)
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

    torch.manual_seed(cfg.seed * 1000 + 331)
    keyK1, attnK1 = E3.KeyNet("K1", cfg.d).to(dev), wm.AttnL3().to(dev)
    hists["attn_K1"] = train_pool(keyK1, attnK1, 341)
    torch.manual_seed(cfg.seed * 1000 + 351)
    lkey = E3.KeyNet("K4", cfg.d).to(dev)
    looped = E3.LoopedLinT4(cfg.loop_T, cfg.gamma_key, 1.0 / (cfg.K * float(beta.mean()))).to(dev)
    hists["looped_lin"] = train_pool(lkey, looped, 352)
    torch.manual_seed(cfg.seed * 1000 + 361)
    jkey, jacobi = E3.KeyNet("K3", cfg.d).to(dev), E4.JacobiLinT4(cfg.loop_T, cfg.gamma_key).to(dev)
    hists["jacobi"] = train_pool(jkey, jacobi, 362)
    log(f"r9 trained attn_L3-K1, looped_lin_T4, jacobi_lin_T4; {time.time() - t0:.0f}s")

    def fit_logs(fn, inits, sampler):
        lp = torch.tensor([math.log(x) for x in inits], device=dev, requires_grad=True)
        o = torch.optim.Adam([lp], lr=0.02)
        for s in range(cfg.fit_steps):
            l0b, dlb, bb, hb, valid, rb = sampler()
            loss = F.binary_cross_entropy_with_logits(fn(l0b, dlb, bb, hb, valid, lp.exp()), rb)
            o.zero_grad(set_to_none=True)
            loss.backward()
            o.step()
        return [float(x) for x in lp.exp()]

    def std_sampler(**bk):
        def f():
            l0b, lrb, dlb, bb, hb, valid, rb = bag(cfg.agg_batch, 0, ntr, **bk)
            return l0b, dlb, bb, hb, valid, rb
        return f

    k3e = lambda dlb: ka3.embed(None, dlb)[0]
    bm = float(beta.mean())
    sc = {}
    sc["eta"] = fit_logs(lambda l0b, dlb, bb, hb, v, e: wm.agg_mmd(l0b, dlb, bb, v, e[0]), [1.0 / (cfg.K * bm)], std_sampler())[0]
    sc["u_K1"] = fit_logs(lambda l0b, dlb, bb, hb, v, e: wm.agg_kc(l0b, dlb, bb, keyK1(hb, dlb, v)[0], v, e[0]), [1.0 / bm],
                          std_sampler())[0]
    sc["u_K3"] = fit_logs(lambda l0b, dlb, bb, hb, v, e: wm.agg_kc(l0b, dlb, bb, k3e(dlb), v, e[0]), [1.0 / bm], std_sampler())[0]
    sc["econ_R7_u"], sc["econ_R7_lam"] = fit_logs(
        lambda l0b, dlb, bb, hb, v, e: E4.agg_econ(l0b, dlb, bb, k3e(dlb), v, e[0], e[1])[0], [1.0, 1.0], std_sampler())
    sc["alpha"] = fit_logs(lambda l0b, dlb, bb, hb, v, e: l0b + e[0] * (v.float()[..., None] * dlb.clamp(-8, 8)).sum(1),
                           [1.0 / cfg.K], std_sampler(kmin=cfg.K, kmax=cfg.K, p_fixed=0.5))[0]

    # ---- mixed-tier D3 calibration probes (standard root, as R8): econ_kc_K3 (u, lambda) and econ_prec_K3 (u, a, b)
    nd = cfg.n_d3
    Cd = {k: v[:nd] for k, v in C.items()}
    l0d = l0[:nd]
    astar, _ = RO.sample_patches(l0d, 1, torch.full((nd,), 0.5, device=dev), gen, cfg.gamma_voi)
    excl = g["overlap"][astar[:, 0]] > 0
    bgp, _ = RO.sample_patches(l0d, cfg.n_bg, torch.full((nd,), 0.5, device=dev), gen, cfg.gamma_voi, exclude=excl)
    bg = E3.bag_reads3(model, Cd, ev2[:nd], l0d, bgp, RO.draw_tau(bgp.shape, gen, dev), gen, cfg.chunk)
    bg["beta"] = E3.beta_of(trust, bg)
    CLD, CLB = [], []
    for tv in (0.3, 1.5, 2.5):
        cl = E3.bag_reads3(model, Cd, ev2[:nd], l0d, astar.expand(nd, cfg.n_clu), torch.full((nd, cfg.n_clu), tv, device=dev),
                           gen, cfg.chunk)
        CLD.append(cl["dl"])
        CLB.append(E3.beta_of(trust, cl))
    CLD, CLB = torch.stack(CLD), torch.stack(CLB)                        # [3, nd, 16, 36], [3, nd, 16]
    NST = torch.tensor((1, 2, 4, 8, 16), device=dev)
    ntd = int(0.8 * nd)

    def d3_sampler(lo=0, hi=ntd, g_=gen):
        nb = cfg.agg_batch
        b = torch.randint(lo, hi, (nb,), device=dev, generator=g_)
        ti = torch.randint(3, (nb,), device=dev, generator=g_)
        N = NST[torch.randint(5, (nb,), device=dev, generator=g_)]
        dlb = torch.cat([CLD[ti, b], bg["dl"][b]], 1)
        bb = torch.cat([CLB[ti, b], bg["beta"][b]], 1)
        valid = torch.cat([torch.arange(cfg.n_clu, device=dev)[None] < N[:, None],
                           torch.ones(nb, cfg.n_bg, dtype=torch.bool, device=dev)], 1)
        return l0d[b], dlb, bb, None, valid, ref[b]

    sc["econ_D3_u"], sc["econ_D3_lam"] = fit_logs(
        lambda l0b, dlb, bb, hb, v, e: E4.agg_econ(l0b, dlb, bb, k3e(dlb), v, e[0], e[1])[0], [1.0, 1.0], d3_sampler)
    sc["econ_prec_u"], sc["econ_prec_a"], sc["econ_prec_b"] = fit_econ_prec(d3_sampler, k3e, cfg.fit_steps, dev)
    log(f"r9 D3 calibration probes ({nd} boards) and scalar fits: {json.dumps({k: round(v, 5) for k, v in sc.items()})}; "
        f"{time.time() - t0:.0f}s")

    # ---- calibration P-FullSupport bags (reading 6): the test's interleaved stream on the first n_pfs calibration boards
    npf = cfg.n_pfs
    Cp = {k: v[:npf] for k, v in C.items()}
    tl6 = torch.tensor(E3.TILES, device=dev)[torch.rand(npf, 6, device=dev, generator=gen).argsort(1)]
    tiles8 = torch.cat([tl6, tl6[:, :2]], 1)
    PDL, PBE = [], []
    for p_ in (cfg.p_full, 1 - cfg.p_full):
        sk, _ = RO.sample_patches(l0[:npf], 24, torch.full((npf,), p_, device=dev), gen, cfg.gamma_voi)
        pst = E5.fs_stream(tiles8, sk)
        ro = E3.bag_reads3(model, Cp, ev2[:npf], l0[:npf], pst, torch.full(pst.shape, 0.3, device=dev), gen, cfg.chunk)
        PDL.append(ro["dl"])
        PBE.append(E3.beta_of(trust, ro))
    PDL, PBE = torch.stack(PDL), torch.stack(PBE)                        # [2, npf, 32, 36], [2, npf, 32]
    PL0, PREF = l0[:npf], ref[:npf]
    nptr = int(0.8 * npf)
    log(f"r9 calibration P-FullSupport bags ({npf} boards x 2 halves x 32 clean reads): {time.time() - t0:.0f}s")

    # ---- jacobi_lin_T4_prec (spec 1.1 item 2; reading 6 as amended by gemini, 19:00Z):
    #      std_w x BCE on R8's standard bags (mixed tiers) + pfs_w x BCE on calibration P-FullSupport bags + d3_w x MSE on D3 bags
    def train_jprec(keyn, agg, seedoff):
        torch.manual_seed(cfg.seed * 1000 + seedoff)
        params = list(keyn.parameters()) + list(agg.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        hist = []
        for s in range(cfg.agg_steps):
            l0s, _, dls, bs, _, vs, rs = bag(cfg.agg_batch, 0, ntr)
            lss, _ = agg(l0s, dls, bs, keyn.embed(None, dls)[0], vs)
            l0_ = F.binary_cross_entropy_with_logits(lss, rs)
            l0b, dlb, bb, valid, rb = pfs_bag(PL0, PDL, PBE, PREF, cfg.agg_batch, 0, nptr, gen, dev)
            ls, _ = agg(l0b, dlb, bb, keyn.embed(None, dlb)[0], valid)
            l1 = F.binary_cross_entropy_with_logits(ls, rb)
            l0d_, dld_, bd_, _, vd_, rd_ = d3_sampler()
            ld, _ = agg(l0d_, dld_, bd_, keyn.embed(None, dld_)[0], vd_)
            l2 = F.mse_loss(torch.sigmoid(ld), rd_)
            loss = cfg.std_w * l0_ + cfg.pfs_w * l1 + cfg.d3_w * l2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if s % 500 == 0 or s == cfg.agg_steps - 1:
                hist.append({"step": s, "std_bce": float(l0_.detach()), "pfs_bce": float(l1.detach()), "d3_mse": float(l2.detach())})
        keyn.eval()
        agg.eval()
        for p in params:
            p.requires_grad_(False)
        return hist

    torch.manual_seed(cfg.seed * 1000 + 365)
    pkey, jprec = E3.KeyNet("K3", cfg.d).to(dev), E5.JacobiPrecT4(cfg.loop_T, cfg.gamma_key).to(dev)
    hists["jacobi_prec"] = train_jprec(pkey, jprec, 366)
    log(f"r9 trained jacobi_lin_T4_prec ({cfg.std_w} x standard BCE + {cfg.pfs_w} x P-FullSupport BCE + {cfg.d3_w} x D3 MSE); "
        f"last {hists['jacobi_prec'][-1]}; "
        f"{time.time() - t0:.0f}s")
    P = Pools9(anyP, keyK1, attnK1, lkey, looped, jkey, jacobi, pkey, jprec, sc, beta_clean)

    # ---- held-out monitors: standard bags (BCE to the reference), anytime bags by root strength and mode, jacobi_prec terms
    mon, mon_any, mon_jp = {}, {}, {}
    g2 = torch.Generator(device=dev).manual_seed(777)
    with torch.no_grad():
        for _ in range(4):
            bi = torch.randint(ntr, cfg.n_cal, (cfg.agg_batch,), device=dev, generator=g2)
            idx, valid = E3.sample_bags(cfg.agg_batch, cfg.pool, g2, dev, cfg.kmin, cfg.kmax)
            G = lambda x: run.gather_rows(x, bi, idx)
            rb = ref[bi]
            out, _ = P.all(l0[bi], lprior[bi], 2, G(pool["lr"]), G(pool["dl"]), G(beta), G(pool["hbar"]), valid)
            for k_, v_ in out.items():
                if k_ in ANY:
                    continue
                mon.setdefault(k_, []).append(float(F.binary_cross_entropy(torch.sigmoid(v_).clamp(1e-6, 1 - 1e-6), rb)))
            for force in ("A", "B"):
                l0a, lpa, dla_, ba_, va_, b0, ya, Mi, Ka, mB = abag(cfg.agg_batch, ntra, na, g2, force)
                lr0 = F.binary_cross_entropy_with_logits(l0a, ya, reduction="none").mean(1)
                for name, (keyn, pool_) in anyP.items():
                    d_, b_, k_, v_ = E5.anchor_bag(keyn, l0a, lpa, dla_, ba_, va_, b0)
                    o_, _ = pool_(l0a, d_, b_, k_, v_)
                    lossv = F.binary_cross_entropy_with_logits(o_, ya, reduction="none").mean(1)
                    for mi, M in enumerate(M0S):
                        sel_ = Mi == mi
                        if sel_.any() and not (force == "B" and M == 2):
                            mon_any.setdefault(f"{name}_M{M}_{force}", []).append(float(lossv[sel_].mean()))
                for mi, M in enumerate(M0S):
                    sel_ = Mi == mi
                    if sel_.any() and force == "A":
                        mon_any.setdefault(f"root_M{M}", []).append(float(lr0[sel_].mean()))
            l0b, dlb, bb, valid, rb = pfs_bag(PL0, PDL, PBE, PREF, cfg.agg_batch, nptr, npf, g2, dev)
            for nm, (kn, ag) in (("jacobi_prec", (pkey, jprec)), ("jacobi", (jkey, jacobi))):
                ls, _ = ag(l0b, dlb, bb, kn.embed(None, dlb)[0], valid)
                mon_jp.setdefault(f"{nm}_pfs_bce", []).append(float(F.binary_cross_entropy_with_logits(ls, rb)))
            l0d_, dld_, bd_, _, vd_, rd_ = d3_sampler(ntd, nd, g2)
            for nm, (kn, ag) in (("jacobi_prec", (pkey, jprec)), ("jacobi", (jkey, jacobi))):
                ld, _ = ag(l0d_, dld_, bd_, kn.embed(None, dld_)[0], vd_)
                mon_jp.setdefault(f"{nm}_d3_mse", []).append(float(F.mse_loss(torch.sigmoid(ld), rd_)))
            for nm, fn in (("econ_prec", lambda: agg_econ_prec(l0d_, dld_, bd_, k3e(dld_), vd_, sc["econ_prec_u"], sc["econ_prec_a"],
                                                                 sc["econ_prec_b"])[0]),
                           ("econ_D3", lambda: E4.agg_econ(l0d_, dld_, bd_, k3e(dld_), vd_, sc["econ_D3_u"], sc["econ_D3_lam"])[0])):
                mon_jp.setdefault(f"{nm}_d3_bce", []).append(float(F.binary_cross_entropy_with_logits(fn(), rd_)))
    mon = {k_: float(np.mean(v_)) for k_, v_ in mon.items()}
    mon_any = {k_: float(np.mean(v_)) for k_, v_ in mon_any.items()}
    mon_jp = {k_: float(np.mean(v_)) for k_, v_ in mon_jp.items()}
    params = {"looped_attn": E3.nparams(kla, la), "attn_K3": E3.nparams(ka3, a3), "attn_K1": E3.nparams(keyK1, attnK1),
              "looped_lin": E3.nparams(lkey, looped), "jacobi": E3.nparams(jkey, jacobi), "jacobi_prec": E3.nparams(pkey, jprec),
              "econ_prec": 3}
    log(f"r9 pools: params {params}; held-out BCE to reference {json.dumps({k: round(v, 4) for k, v in mon.items()})}; "
        f"anytime held-out BCE to ref(M_0) {json.dumps({k: round(v, 4) for k, v in mon_any.items()})}; jacobi/econ held-out "
        f"{json.dumps({k: round(v, 5) for k, v in mon_jp.items()})}; {time.time() - t0:.0f}s")
    lam_at = lambda fn: {str(t): float(fn(torch.full((1, 1), bt[str(t)], device=dev)).mean()) for t in (0.3, 1.5, 2.5)}
    info = {"M_strong": M_strong, "strong_selection": {"kl_ref36_root": sel, "kl_root_ref36_reported": sel_rev,
                                                       "threshold": cfg.kl_strong, "candidates": list(cands)},
            "scalars": sc, "beta_by_tau": bt, "beta_clean": beta_clean, "beta_by_root": bM, "trust_train_mse": float(np.mean(tl[-200:])),
            "trust_heldout_mse": tmon, "hist": hists, "heldout_bag_bce_to_ref": mon, "heldout_any_bce_to_ref": mon_any,
            "heldout_jacobi_econ": mon_jp, "params": params, "ref_coverage_mean": float(cov.mean()),
            "jacobi": {"lambda": [float(x) for x in jacobi.lam()], "eta": [float(x) for x in jacobi.eta],
                       "P_mean": [float(x) for x in jacobi.P.mean(1)], "s_mean": float(jacobi.s.mean())},
            "jacobi_prec": {"lambda_tilde_softplus": [float(x) for x in F.softplus(jprec.lam_raw)], "a": float(jprec.a),
                            "b": float(jprec.b), "eta": [float(x) for x in jprec.eta],
                            "P_mean": [float(x) for x in jprec.P.mean(1)], "s_mean": float(jprec.s.mean()),
                            "lambda_mean_at_beta": lam_at(jprec.lam_rc)},
            "econ_prec": {"u": sc["econ_prec_u"], "a": sc["econ_prec_a"], "b": sc["econ_prec_b"],
                          "lambda_at_beta": lam_at(lambda bb_: econ_prec_lambda(bb_, sc["econ_prec_a"], sc["econ_prec_b"]))},
            "sec": time.time() - t0}
    return trust, P, info


# ------------------------------------------------------------------------------------------------ tests
def test9(model, trust, P, cfg, dev, log, gen, Pf, M_strong):
    t0 = time.time()
    n = cfg.n_test
    T = A2.boards2_t(A2.STREAM_TEST2, n, dev)
    ordT = torch.as_tensor(E5.anchor_order(A2.STREAM_TEST2, n), device=dev)
    g = RO.geo(dev)
    ev_full = A2.ev_from(T["s"], torch.ones_like(T["s"]))
    ev_std6 = A2.ev_from(T["s"], A2.cells_mask(T["std6"], dev))
    Pfull = env6.bayes_marginals(ev_full, T["note"], Pf).float()           # EVALUATION ONLY
    Pstd6 = env6.bayes_marginals(ev_std6, T["note"], Pf).float()
    kl = lambda m: float(wm.bern_kl(Pfull, m).mean())
    sig = torch.sigmoid
    pear = lambda a, b: float(np.corrcoef(a.cpu().numpy().ravel(), b.cpu().numpy().ravel())[0, 1])
    lstd = RO.forward_chunks(model, ev_std6, T["note"])
    lprior = E5.prior_logits(model, T)
    EV, L0 = {}, {}
    for M in M0S:
        EV[M], L0[M] = E5.root(model, T, ordT, M)
    R = {"n_test": n, "M_strong": M_strong, "clause1_corr_std6": pear(sig(lstd), Pstd6),
         "clause1_kl_std6": float(wm.bern_kl(Pstd6, sig(lstd)).mean()),
         "kl_root": {str(M): kl(sig(L0[M])) for M in M0S}, "kl_prior": kl(sig(lprior))}
    raw = {"pstar": run.r5(Pfull), "root_mu": {str(M): run.r5(sig(L0[M])) for M in M0S}, "anchor_order34": ordT[:, :34].tolist(),
           "occ": T["occ"].int().tolist(), "M_strong": M_strong}
    log(f"r9 roots: KL by M_0 {json.dumps({k: round(v, 5) for k, v in R['kl_root'].items()})}; clause-1 Pearson "
        f"{R['clause1_corr_std6']:.5f}; M_0_strong {M_strong}")

    # ---- E1-D0: anchor identity at K = 0 (the rule uses M_0 = 2 and M_0_strong; every M_0 is reported)
    ident = {}
    for M in M0S:
        e = torch.zeros(n, 0, 36, device=dev)
        out, _ = P.all(L0[M], lprior, M, e, e, torch.zeros(n, 0, device=dev), torch.zeros(n, 0, 128, device=dev),
                       torch.zeros(n, 0, dtype=torch.bool, device=dev))
        for a in ANY:
            ident[f"{a}_M{M}_max_abs_logit_diff"] = float((out[a] - L0[M]).abs().max())
            ident[f"{a}_M{M}_kl_diff"] = abs(kl(sig(out[a])) - kl(sig(L0[M])))
    R["identity_K0"] = ident
    log(f"r9 K=0 identity: max over pools and M_0 of |KL diff| {max(v for k, v in ident.items() if 'kl_diff' in k):.2e}, "
        f"of |logit diff| {max(v for k, v in ident.items() if 'logit' in k):.2e}")

    # ---- E1-D1.1: sparse-root (M_0 = 2) P-FullSupport prefix curve on the interleaved stream (as R8)
    tl6 = torch.tensor(E3.TILES, device=dev)[torch.rand(n, 6, device=dev, generator=gen).argsort(1)]
    tiles8 = torch.cat([tl6, tl6[:, :2]], 1)
    fs = {}
    for name, p in (("NW", cfg.p_full), ("SE", 1 - cfg.p_full)):
        sk, _ = RO.sample_patches(L0[2], 24, torch.full((n,), p, device=dev), gen, cfg.gamma_voi)
        patch = E5.fs_stream(tiles8, sk)
        ro = E3.bag_reads3(model, T, EV[2], L0[2], patch, torch.full(patch.shape, 0.3, device=dev), gen, cfg.chunk)
        ro["beta"] = E3.beta_of(trust, ro)
        fs[name] = E5.run_prefixes(P, L0[2], lprior, 2, ro, KS_FS)
        lu, _ = A2.union_call(model, T, EV[2], patch[:, :24])
        fs[name]["union24"] = lu
        raw[f"fs_{name}"] = {"patch": patch.tolist(), **{f"K{K}_{a}": run.r5(sig(fs[name][K][a])) for K in KS_FS
                                                        for a in ("looped_attn", "attn_K3", "mmd", "scaled_mix")},
                             **{f"K24_{a}": run.r5(sig(fs[name][24][a])) for a in FS_RAW_K24}}
    for K in KS_FS:
        for a in POOL_NAMES:
            mN, mS = sig(fs["NW"][K][a]), sig(fs["SE"][K][a])
            R[f"fs_kl_{a}_K{K}"] = 0.5 * (kl(mN) + kl(mS))
            R[f"fs_jsd_{a}_K{K}"] = float(wm.bern_jsd(mN, mS).mean())
    R["fs_jsd_union_K24"] = float(wm.bern_jsd(sig(fs["NW"]["union24"]), sig(fs["SE"]["union24"])).mean())
    R["fs_kl_union_K24"] = 0.5 * (kl(sig(fs["NW"]["union24"])) + kl(sig(fs["SE"]["union24"])))
    log("r9 sparse-root FS curve looped_attn KL " + " ".join(f"K{K} {R[f'fs_kl_looped_attn_K{K}']:.4f}" for K in KS_FS)
        + f"; JSD(24) {R['fs_jsd_looped_attn_K24']:.5f}; K24 KL jacobi_prec {R['fs_kl_jacobi_prec_K24']:.4f} econ_prec "
          f"{R['fs_kl_econ_prec_K24']:.4f} looped_lin {R['fs_kl_looped_lin_K24']:.4f} kc_K1 {R['fs_kl_kc_K1_K24']:.4f} kc_K3 "
          f"{R['fs_kl_kc_K3_K24']:.4f}")

    # ---- E1-D1.2: strong-root curves, Mode A (cumulative context, three-tier tau) and Mode B (bounded local reads, tau 0.3/1.5)
    SR = [int(x) for x in cfg.sr_roots.split(",")]
    assert M_strong in SR, "M_0_strong must be one of the strong-root curves"
    for M in SR:
        cur = {"A": {}, "B": {}}
        for name, p in (("NW", cfg.p_sr), ("SE", 1 - cfg.p_sr)):
            pt, _ = RO.sample_patches(L0[M], cfg.K_sr, torch.full((n,), p, device=dev), gen, cfg.gamma_voi)
            tA = E3.draw_tau3(pt.shape, gen, dev)
            roA = E3.bag_reads3(model, T, EV[M], L0[M], pt, tA, gen, cfg.chunk)          # reading 2: R8's reads from the M_0 root
            roA["beta"] = E3.beta_of(trust, roA)
            cur["A"][name] = E5.run_prefixes(P, L0[M], lprior, M, roA, KS_SR)
            tB = RO.draw_tau(pt.shape, gen, dev)
            roB = E3.bag_reads3(model, T, EV[2], L0[2], pt, tB, gen, cfg.chunk)          # reading 3: bounded reads (2 anchors + patch)
            roB["beta"] = E3.beta_of(trust, roB)
            cur["B"][name] = E5.run_prefixes(P, L0[M], lprior, M, roB, KS_SR)            # Delta l over l0(2), added to l0(M_0)
            if M == M_strong:
                for md, ro_ in (("A", roA), ("B", roB)):
                    raw[f"srS{md}_{name}"] = {"patch": pt.tolist(), "tau": ro_["tau"].tolist(),
                                              **{f"K{K}_{a}": run.r5(sig(cur[md][name][K][a])) for K in KS_SR for a in SR_RAW}}
        for md in ("A", "B"):
            tag = f"sr{md}{M}"
            for K in KS_SR:
                for a in POOL_NAMES:
                    mN, mS = sig(cur[md]["NW"][K][a]), sig(cur[md]["SE"][K][a])
                    R[f"{tag}_kl_{a}_K{K}"] = 0.5 * (kl(mN) + kl(mS))
                    R[f"{tag}_jsd_{a}_K{K}"] = float(wm.bern_jsd(mN, mS).mean())
            log(f"r9 strong-root M_0={M} mode {md}: KL(0) {R[f'{tag}_kl_looped_attn_K0']:.4f}; " + "; ".join(
                f"{a} " + " ".join(f"K{K} {R[f'{tag}_kl_{a}_K{K}']:.4f}" for K in KS_SR[1:]) for a in ("looped_attn", "mmd", "mix")))

    # ---- D3 controlled bags (standard root, as R8): W(N) and MSE
    astar, _ = RO.sample_patches(L0[2], 1, torch.full((n,), 0.5, device=dev), gen, cfg.gamma_voi)
    excl = g["overlap"][astar[:, 0]] > 0
    bgp, _ = RO.sample_patches(L0[2], cfg.n_bg, torch.full((n,), 0.5, device=dev), gen, cfg.gamma_voi, exclude=excl)
    bg = E3.bag_reads3(model, T, EV[2], L0[2], bgp, RO.draw_tau(bgp.shape, gen, dev), gen, cfg.chunk)
    bg["beta"] = E3.beta_of(trust, bg)
    pm = g["pmask"][astar[:, 0]].bool()
    ctrl = {}
    raw["controlled"] = {"astar": astar[:, 0].tolist(), "bg_patch": bgp.tolist()}
    for tv in (0.3, 1.5, 2.5):
        cp = astar.expand(n, cfg.n_clu)
        cl = E3.bag_reads3(model, T, EV[2], L0[2], cp, torch.full(cp.shape, tv, device=dev), gen, cfg.chunk)
        cl["beta"] = E3.beta_of(trust, cl)
        ro = E3.cat_ro(cl, bg)
        for N in (1, 2, 4, 8, 16):
            valid = torch.zeros(ro["patch"].shape, dtype=torch.bool, device=dev)
            valid[:, :N] = True
            valid[:, cfg.n_clu:] = True
            out, W = P.all(L0[2], lprior, 2, ro["lr"], ro["dl"], ro["beta"], ro["hbar"], valid, need_w=True)
            row = {}
            for a in WPOOLS:
                Wn = (W[a][:, :N] * pm[:, None, :].float()).sum(2) / 6.0
                row[f"W_{a}"] = float(Wn.sum(1).mean())
                if a in ("jacobi_prec", "econ_prec", "econ_D3", "econ_R7", "kc_K3", "looped_attn"):
                    raw["controlled"][f"t{tv}_N{N}_W_{a}"] = run.r5(Wn.sum(1))
            for a in POOL_NAMES:
                row[f"mse_{a}"] = float(((sig(out[a]) - Pfull) ** 2).mean())
            ctrl[f"t{tv}_N{N}"] = row
            raw["controlled"][f"t{tv}_N{N}"] = {a: run.r5(sig(out[a])) for a in ("jacobi_prec", "econ_prec", "econ_D3", "kc_K3")}
        log(f"r9 controlled tau={tv}: " + " ".join(f"N{N} W_econ_prec {ctrl[f't{tv}_N{N}']['W_econ_prec']:.3f} W_jprec "
                                                  f"{ctrl[f't{tv}_N{N}']['W_jacobi_prec']:.3f}" for N in (1, 2, 4, 8, 16)))
    R["controlled"] = ctrl
    R["sec"] = time.time() - t0
    log("r9 test done: " + json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in R.items()
                                         if not isinstance(v, (dict, list))})[:4000])
    return R, raw


def main(argv=None):
    ap = argparse.ArgumentParser(description="R9-E1 one seed")
    for f in dataclasses.fields(Cfg9):
        ap.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    cfg = Cfg9(**vars(ap.parse_args(argv)))
    os.makedirs(cfg.out, exist_ok=True)
    dev = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
    torch.set_num_threads(cfg.threads)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    log = run.Log(cfg.out)
    log("config r9e1", json.dumps(asdict(cfg)), "device", dev, torch.__version__)
    if dev.type == "cuda":   # as R8's deviation 3: torch 2.14's Triton bmm-outer-product override needs Python.h, absent on the
        try:                 # VM; deregistering the Triton overrides sends those matmuls to the standard kernels
            import torch._native.triton_utils as _tu
            _tu.deregister_op_overrides()
            log("r9 torch native Triton op overrides deregistered")
        except Exception as e:  # noqa: BLE001
            log("r9 could not deregister torch native Triton op overrides:", repr(e))
    T0 = time.time()
    Pp = env6.placements()
    Pf = torch.as_tensor(Pp, device=dev).double()
    model, meta, ck = E5.load_checked(cfg.ckpt_dir, cfg.seed, dev)
    log(f"r9 theta_real loaded from {ck['path']}: sha256 {ck['sha256'][:12]} matches its .json; {ck['n_params']} parameters")
    for p in model.parameters():
        p.requires_grad_(False)
    summary = {"cfg": asdict(cfg), "experiment": "R9-E1", "n_placements": int(Pp.shape[0]), "torch": torch.__version__,
               "device": str(dev), "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None, "checkpoint": ck,
               "theta_real_train_curve_last": (meta.get("train_curve") or [None])[-1]}
    run.dump(os.path.join(cfg.out, "summary.json"), summary)
    gen = torch.Generator(device=dev).manual_seed(cfg.seed * 1000 + 381)
    trust, Pools, info = fit9(model, cfg, dev, log, gen)
    summary["fit"] = info
    run.dump(os.path.join(cfg.out, "summary.json"), summary)
    res, raw = test9(model, trust, Pools, cfg, dev, log, gen, Pf, info["M_strong"])
    summary["test"] = res
    run.dump(os.path.join(cfg.out, "raw.json"), raw)
    summary["total_sec"] = time.time() - T0
    summary["done"] = True
    run.dump(os.path.join(cfg.out, "summary.json"), summary)
    log(f"r9 done in {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
