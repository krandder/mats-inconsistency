"""R8-E1 (E-INFO-ANYTIME): strong-root anytime monotonicity with an anchor token, precision-aware Jacobi normal equations,
mixed-tier econ_kc_K3 and the 2x2 PUCT attribution on 6x6 ShipHunt.
Spec: /home/agent/workspaces/mats/outbox/experiments-round-8-spec.md section 1 (gemini). Pre-registration: PREREG-R8-E1.md.
No world-model training: theta_real v2 is LOADED from li-net/r7e1/ckpt/s<seed>/theta_real_v2.npz after its sha256 is checked
against its .json. Imported unchanged: li-net/r7e1/e1v4.py, li-net/r6e1/e1v3.py, li-net/r5e1 (env6, wm, rollouts, run, partA2).

Per seed: load theta_real -> calibration (stream 21): the R7 standard-root pool (4,000 boards x 96 reads, tau uniform over
0.3 / 1.5 / 2.5) and K = 128 reference; read pools at root strengths M_0 = 6, 12, 18 anchor cells (first 2,000 boards x 48
reads); theta_trust on reads of all root strengths -> anytime pools with an anchor token and precision-weighted crowding
(looped_attn_T3 and attn_L3, K3 heads) trained by BCE to the model's own K = 128 reference ref(M_0) over M_0 in {2, 6, 12, 18}
and K in {0, 1, 2, 4, 8, 16, 24, 32} (gemini's amended reading 3) -> R7-style pools on the
standard root (BCE to the reference): attn_L3-K1, looped_lin_T4, jacobi_lin_T4, jacobi_lin_T4_prec; fitted scalars: mmd eta,
kc u (K1, K3), econ_kc_K3 (u, lambda) R7-style and on mixed-tier D3 calibration probes, scaled_mix alpha -> tests on the 500
R5-R7 test boards: anchor identity at K = 0, sparse-root P-FullSupport prefix curve on the interleaved 32-read stream (amended
reading 7; K = 0..32), strong-root curves (M_0 = 6, 12, 18; K = 0..24, 90/10 skewed reads with tau 50/50 over 0.3 / 1.5 as
amended reading 6, plus a three-tier version at M_0 = 18), D3 controlled bags, 2x2 PUCT (K1 vs K3 keys x R6 vs R7 score).
Exact Bayes: test metrics and the gate only.
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
R5DIR = os.environ.get("R5E1_DIR", f"{LI}/r5e1")
R6DIR = os.environ.get("R6E1_DIR", f"{LI}/r6e1")
R7DIR = os.environ.get("R7E1_DIR", f"{LI}/r7e1")
for _p in (R7DIR, R6DIR, R5DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import e1v3 as E3  # noqa: E402  (R6-E1, unchanged)
import e1v4 as E4  # noqa: E402  (R7-E1, unchanged)
import env6  # noqa: E402
import partA2 as A2  # noqa: E402
import rollouts as RO  # noqa: E402
import run  # noqa: E402
import wm  # noqa: E402

M0S = (2, 6, 12, 18)
KS_TRAIN = (0, 1, 2, 4, 8, 16, 24, 32)        # amended reading 3
KS_FS = (0, 1, 2, 4, 8, 16, 24, 32)
KS_SR = (0, 1, 2, 4, 8, 16, 24)
ANY = ("looped_attn", "attn_K3")                   # anchor-aware anytime pools
OTHER = ("mix", "scaled_mix", "mmd", "kc_K1", "kc_K3", "econ_R7", "econ_D3", "looped_lin", "jacobi", "jacobi_prec", "attn_K1")
POOL_NAMES = ANY + OTHER
WPOOLS = ("looped_attn", "attn_K3", "attn_K1", "jacobi", "jacobi_prec", "econ_R7", "econ_D3", "kc_K1", "kc_K3", "mmd")
CELLS = (("K1", "R6"), ("K1", "R7"), ("K3", "R6"), ("K3", "R7"))


@dataclass
class Cfg8(E4.Cfg7):
    out: str = "/tmp/r8e1/dev"
    ckpt_dir: str = f"{LI}/r7e1/ckpt"   # theta_real_v2.npz + .json per seed (load only)
    waves: int = 4                     # PUCT waves (K = 24)
    n_any: int = 2000                  # calibration boards with read pools at M_0 = 6, 12, 18 (M_0 = 2 reuses the standard pool)
    pool_any: int = 32                 # reads per half per board per root strength (amended reading 3)
    n_d3: int = 2000                   # calibration boards with D3 probes (econ_kc_K3 mixed-tier fit)
    K_fs: int = 32
    K_sr: int = 24
    p_sr: float = 0.9
    sr_roots: str = "6,12,18"          # strong-root curves at these M_0 (18 is the rule's; 6 and 12 descriptive)
    K_tr: int = 32                     # largest training K (amended reading 3)


# ------------------------------------------------------------------------------------------------ roots and checkpoints
def anchor_order(stream, n):
    """make_boards2's own per-board random cell permutation (its first 2 cells are the standard anchors). The root with M_0
    anchor cells sees the readings of order[:, :M_0]: nested in M_0, fixed per board, identical in every seed."""
    P = env6.placements()
    rng = np.random.default_rng([stream, 7702])
    rng.integers(len(P), size=n)
    rng.random((n, 36))
    rng.random(n)
    rng.random((n, 36))
    order = rng.random((n, 36)).argsort(1)
    assert (order[:, :2] == A2.make_boards2(stream, n)["anchors"]).all(), "anchor order does not reproduce make_boards2"
    return order


def root(model, B, order_t, M):
    ev = A2.ev_from(B["s"], A2.cells_mask(order_t[:, :M], B["s"].device))
    return ev, RO.forward_chunks(model, ev, B["note"])


def prior_logits(model, B):
    return RO.forward_chunks(model, torch.zeros_like(B["s"], dtype=torch.long), B["note"])


def load_checked(ckpt_dir, seed, dev):
    p = os.path.join(ckpt_dir, f"s{seed}", "theta_real_v2.npz")
    model, meta = E4.load_wm_npz(p, dev)
    h = E4.sha256_file(p)
    if h != meta.get("sha256"):
        raise RuntimeError(f"checkpoint {p} sha256 {h} does not match its .json {meta.get('sha256')}")
    return model, meta, {"path": p, "sha256": h, "json_sha256": meta.get("sha256"), "match": True,
                         "n_params": int(model.n_params()), "train_steps": meta.get("train_steps")}


# ------------------------------------------------------------------------------------------------ pools
def anchor_bag(keyn, l0, lprior, dl, beta, valid, beta0):
    """Prepend the anchor token r = 0: k_0 = keyn(salience(l0 - l_prior)), Delta l_0 = 0, beta_0; always valid."""
    nb = l0.shape[0]
    k0, _ = keyn.embed(None, (l0 - lprior)[:, None, :])
    k = torch.cat([k0, keyn.embed(None, dl)[0]], 1) if dl.shape[1] > 0 else k0
    dla = torch.cat([torch.zeros_like(l0)[:, None], dl], 1)
    ba = torch.cat([beta0[:, None], beta], 1)
    va = torch.cat([torch.ones(nb, 1, dtype=torch.bool, device=l0.device), valid], 1)
    return dla, ba, k, va


def crowding_prec(k, valid, beta, beta_clean, gamma=2.0):
    """Amended reading 3: c_r = sum_b v_b (beta_b / beta_clean) relu(<k_r,k_b>)^gamma, the anchor token included."""
    g = torch.relu(k @ k.transpose(1, 2)) ** gamma
    g = g * (valid.float() * beta / beta_clean)[:, None, :]
    return g.sum(2).clamp_min(1e-6)


class AttnL3P(wm.AttnL3):
    """wm.AttnL3 with precision-weighted crowding (amended reading 3); otherwise unchanged."""

    def __init__(self, beta_clean, **kw):
        super().__init__(**kw)
        self.beta_clean = float(beta_clean)

    def weights(self, dl, beta, k, valid, gamma=2.0):
        c = crowding_prec(k, valid, beta, self.beta_clean, gamma)
        dlc = dl.clamp(-8, 8)
        x = self.inp(torch.cat([k, self.Wv(dlc), beta.clamp_min(1e-6).log()[..., None], c.log()[..., None]], -1))
        pad = ~valid
        for at, ln, ff in zip(self.att, self.ln, self.ff):
            y = ln(x)
            x = x + at(y, y, y, key_padding_mask=pad, need_weights=False)[0]
            x = x + ff(x)
        return F.softplus(self.out(x)) * valid[..., None].float()


class LoopedAttnT3P(E4.LoopedAttnT3):
    """e1v4.LoopedAttnT3 with precision-weighted crowding (amended reading 3); otherwise unchanged."""

    def __init__(self, beta_clean, **kw):
        super().__init__(**kw)
        self.beta_clean = float(beta_clean)

    def weights(self, dl, beta, k, valid, gamma=2.0):
        c = crowding_prec(k, valid, beta, self.beta_clean, gamma)
        dlc = dl.clamp(-8, 8)
        x = self.inp(torch.cat([k, self.Wv(dlc), beta.clamp_min(1e-6).log()[..., None], c.log()[..., None]], -1))
        pad = ~valid
        for t in range(self.T):
            x = x + self.step_emb[t]
            y = self.ln(x)
            x = x + self.att(y, y, y, key_padding_mask=pad, need_weights=False)[0]
            x = x + self.ff(x)
        return F.softplus(self.out(x)) * valid[..., None].float()


class JacobiPrecT4(E4.JacobiLinT4):
    """jacobi_lin_T4 with a precision-aware ridge (spec 1.1 item 2): lambda_rc = softplus(lambda~_c) sigmoid(a - b beta_r)
    replaces lambda_c in the residual and in D_rc = lambda_rc + sum_b g_rb beta_b. Init: softplus(lambda~) = 2, a = b = 0
    (lambda = 1 at init, as R7's jacobi_lin_T4). 2 parameters more than jacobi_lin_T4."""

    def __init__(self, T=4, gamma=2.0):
        super().__init__(T, gamma)
        with torch.no_grad():
            self.lam_raw.fill_(math.log(math.expm1(2.0)))
        self.a = nn.Parameter(torch.zeros(()))
        self.b = nn.Parameter(torch.zeros(()))

    def lam_rc(self, beta):
        return F.softplus(self.lam_raw)[None, None, :] * torch.sigmoid(self.a - self.b * beta)[..., None]

    def parts(self, dl, beta, k, valid):
        v = valid.float()
        g = torch.relu(k @ k.transpose(1, 2)) ** self.gamma
        eye = torch.eye(k.shape[1], dtype=torch.bool, device=k.device)[None]
        g = torch.where(eye, torch.ones_like(g), g) * v[:, :, None] * v[:, None, :]
        bv = beta * v
        GB = g * bv[:, None, :]
        lam = self.lam_rc(beta)                                            # [n, K, 36]
        D = lam + GB.sum(2)[..., None]
        b = bv[..., None] * dl.clamp(-8, 8)
        return GB, lam, D, b, v

    @torch.no_grad()
    def coef(self, dl, beta, k, valid):
        GB, lam, D, b, v = self.parts(dl, beta, k, valid)
        n, K = GB.shape[:2]
        eye = torch.eye(K, device=dl.device)
        A = torch.diag_embed(lam.permute(0, 2, 1)) + GB[:, None]           # [n, 36, K, K]
        Dc = D.permute(0, 2, 1)
        vm = v[:, None, :, None]
        Q = torch.zeros(n, wm.NC, K, K, device=dl.device)
        for t in range(self.T):
            step = (self.eta[t] * self.P[t])[None, :, None] / Dc
            Q = (Q + step[..., None] * (eye - A @ Q)) * vm
        w = self.s[None, :, None] * Q.sum(2) * (beta * v)[:, None, :]
        return w.permute(0, 2, 1)


class Pools8:
    def __init__(self, anyp, keyK1, attnK1, lkey, looped, jkey, jacobi, pkey, jprec, sc, beta_clean):
        self.any = anyp                                                    # name -> (KeyNet K3, pool)
        self.keyK1, self.attnK1, self.lkey, self.looped = keyK1, attnK1, lkey, looped
        self.jkey, self.jacobi, self.pkey, self.jprec, self.sc, self.beta_clean = jkey, jacobi, pkey, jprec, sc, beta_clean

    def beta0(self, M, nb, dev):
        return torch.full((nb,), M * self.beta_clean / 6.0, device=dev)

    @torch.no_grad()
    def all(self, l0, lprior, M, lr, dl, beta, hbar, valid, need_w=False):
        out, W = {}, {}
        nb, K = dl.shape[0], dl.shape[1]
        b0 = self.beta0(M, nb, l0.device)
        for name, (keyn, pool) in self.any.items():
            dla, ba, k, va = anchor_bag(keyn, l0, lprior, dl, beta, valid, b0)
            o, w = pool(l0, dla, ba, k, va)
            out[name], W[name] = o, w[:, 1:]
        if K == 0:
            for a in OTHER:
                out[a] = l0.clone()
            return out, W
        vf = valid.float()
        sc = self.sc
        out["mix"] = wm.agg_mix(lr, valid)
        out["scaled_mix"] = l0 + sc["alpha"] * (vf[..., None] * dl.clamp(-8, 8)).sum(1)
        out["mmd"] = wm.agg_mmd(l0, dl, beta, valid, sc["eta"])
        k3, _ = self.any["attn_K3"][0].embed(None, dl)
        k1, _ = self.keyK1(hbar, dl, valid)
        out["kc_K1"] = wm.agg_kc(l0, dl, beta, k1, valid, sc["u_K1"])
        out["kc_K3"] = wm.agg_kc(l0, dl, beta, k3, valid, sc["u_K3"])
        out["econ_R7"], we7 = E4.agg_econ(l0, dl, beta, k3, valid, sc["econ_R7_u"], sc["econ_R7_lam"])
        out["econ_D3"], weD = E4.agg_econ(l0, dl, beta, k3, valid, sc["econ_D3_u"], sc["econ_D3_lam"])
        kl_, _ = self.lkey(hbar, dl, valid)
        out["looped_lin"], _ = self.looped(l0, dl, beta, kl_, valid)
        kj, _ = self.jkey(hbar, dl, valid)
        out["jacobi"], _ = self.jacobi(l0, dl, beta, kj, valid)
        kp, _ = self.pkey(hbar, dl, valid)
        out["jacobi_prec"], _ = self.jprec(l0, dl, beta, kp, valid)
        out["attn_K1"], W["attn_K1"] = self.attnK1(l0, dl, beta, k1, valid)
        if need_w:
            ex = lambda w_: w_[..., None].expand(-1, -1, wm.NC)
            W["jacobi"] = self.jacobi.coef(dl, beta, kj, valid)
            W["jacobi_prec"] = self.jprec.coef(dl, beta, kp, valid)
            W["econ_R7"], W["econ_D3"] = ex(we7), ex(weD)
            W["kc_K1"] = ex(sc["u_K1"] * beta / wm.crowding(k1, valid) * vf)
            W["kc_K3"] = ex(sc["u_K3"] * beta / wm.crowding(k3, valid) * vf)
            W["mmd"] = ex(sc["eta"] * beta * vf)
        return out, W


# ------------------------------------------------------------------------------------------------ calibration fit
def fit8(model, cfg, dev, log, gen):
    t0 = time.time()
    g = RO.geo(dev)
    C = A2.boards2_t(A2.STREAM_CAL2, cfg.n_cal, dev)
    ordC = torch.as_tensor(anchor_order(A2.STREAM_CAL2, cfg.n_cal), device=dev)
    lprior = prior_logits(model, C)
    ev2, l0 = root(model, C, ordC, 2)
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
    log(f"r8 standard pool + reference: {time.time() - t0:.0f}s; reference coverage {float(cov.mean()):.4f}")
    # read pools at root strengths M_0 = 6, 12, 18 on the first n_any boards; M_0 = 2 reuses the standard pool
    na, pa = cfg.n_any, cfg.pool_any
    Cs = {k: v[:na] for k, v in C.items()}
    idx2 = torch.cat([torch.arange(pa, device=dev), cfg.pool + torch.arange(pa, device=dev)])
    anyp = {2: {"l0": l0[:na], "ro": {k: pool[k][:na][:, idx2] for k in ("lr", "hbar", "ent", "dis", "dl")},
                "tau": tau[:na][:, idx2], "patch": patch[:na][:, idx2]}}
    o1, z1 = torch.ones(na, device=dev), torch.zeros(na, device=dev)
    for M in (6, 12, 18):
        evM, l0M = root(model, Cs, ordC[:na], M)
        pn, _ = RO.sample_patches(l0M, pa, o1, gen, cfg.gamma_voi)
        ps, _ = RO.sample_patches(l0M, pa, z1, gen, cfg.gamma_voi)
        pt = torch.cat([pn, ps], 1)
        tt = E3.draw_tau3(pt.shape, gen, dev)
        ro = E3.bag_reads3(model, Cs, evM, l0M, pt, tt, gen, cfg.chunk)
        anyp[M] = {"l0": l0M, "ro": {k: ro[k] for k in ("lr", "hbar", "ent", "dis", "dl")}, "tau": tt, "patch": pt}
    refp = torch.cat([rnw, rse], 1)[:na]
    anyp[2]["ref"] = ref[:na]
    for M in (6, 12, 18):
        evM, _ = root(model, Cs, ordC[:na], M)
        anyp[M]["ref"] = torch.sigmoid(A2.union_call(model, Cs, evM, refp)[0])    # amended reading 3: ref(M_0)
    log(f"r8 multi-root pools (M_0 6/12/18, {na} boards x {2 * pa} reads) and references ref(M_0): {time.time() - t0:.0f}s")
    # theta_trust on reads of all root strengths (train boards: standard < 0.8 n_cal, multi-root < 0.8 n_any)
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
    for M in (6, 12, 18):
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
        for M in (6, 12, 18):
            X, _ = feats(anyp[M]["ro"], anyp[M]["tau"], anyp[M]["patch"], Cs["occ"])
            anyp[M]["beta"] = wm.beta_from(torch.cat([trust(X[a:a + 65536]) for a in range(0, X.shape[0], 65536)])).view(na, 2 * pa)
    anyp[2]["beta"] = beta[:na][:, idx2]
    bt = {str(t): float(beta[tau == t].mean()) for t in (0.3, 1.5, 2.5)}
    beta_clean = bt["0.3"]
    bM = {str(M): {str(t): float(anyp[M]["beta"][anyp[M]["tau"] == t].mean()) for t in (0.3, 1.5, 2.5)} for M in M0S}
    log(f"r8 theta_trust: train mse {np.mean(tl[-200:]):.4f} heldout {tmon:.4f}; beta by tau {bt}; by root {bM}")

    # ---- anytime pools: anchor token, M_0 in {2,6,12,18}, K in {0,2,4,8,16,24}, BCE to the realized occupancy
    L0S = torch.stack([anyp[M]["l0"] for M in M0S])                     # [4, na, 36]
    DLS = torch.stack([anyp[M]["ro"]["dl"] for M in M0S])               # [4, na, 48, 36]
    BES = torch.stack([anyp[M]["beta"] for M in M0S])                   # [4, na, 48]
    M0T = torch.tensor(M0S, device=dev, dtype=torch.float32)
    KST = torch.tensor(KS_TRAIN, device=dev)
    REFS = torch.stack([anyp[M]["ref"] for M in M0S])                  # [4, na, 36]  amended reading 3
    LPa = lprior[:na]

    def any_bag(nb, lo, hi, g_=gen):
        b = torch.randint(lo, hi, (nb,), device=dev, generator=g_)
        Mi = torch.randint(len(M0S), (nb,), device=dev, generator=g_)
        K = KST[torch.randint(len(KS_TRAIN), (nb,), device=dev, generator=g_)]
        kt = cfg.K_tr
        idx, _ = E3.sample_bags(nb, pa, g_, dev, kt, kt)
        idx = idx.gather(1, torch.rand(nb, kt, device=dev, generator=g_).argsort(1))
        valid = torch.arange(kt, device=dev)[None] < K[:, None]
        dl = DLS[Mi, b].gather(1, idx[..., None].expand(nb, kt, 36))
        be = BES[Mi, b].gather(1, idx)
        return L0S[Mi, b], LPa[b], dl, be, valid, M0T[Mi] * beta_clean / 6.0, REFS[Mi, b], Mi, K

    def train_any(keyn, agg, seedoff):
        torch.manual_seed(cfg.seed * 1000 + seedoff)
        params = list(keyn.parameters()) + list(agg.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        hist = []
        for s in range(cfg.agg_steps):
            l0b, lpb, dlb, bb, valid, b0, yb, _, _ = any_bag(cfg.agg_batch, 0, ntra)
            dla, ba, k, va = anchor_bag(keyn, l0b, lpb, dlb, bb, valid, b0)
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
    kla, la = E3.KeyNet("K3", cfg.d).to(dev), LoopedAttnT3P(beta_clean, T=cfg.attn_T).to(dev)
    hists["looped_attn"] = train_any(kla, la, 382)
    anyP["looped_attn"] = (kla, la)
    log(f"r8 trained anytime looped_attn_T3; last bce {hists['looped_attn'][-1]['bce_to_ref']:.4f}; {time.time() - t0:.0f}s")
    torch.manual_seed(cfg.seed * 1000 + 391)
    ka3, a3 = E3.KeyNet("K3", cfg.d).to(dev), AttnL3P(beta_clean).to(dev)
    hists["attn_K3"] = train_any(ka3, a3, 392)
    anyP["attn_K3"] = (ka3, a3)
    log(f"r8 trained anytime attn_L3-K3; last bce {hists['attn_K3'][-1]['bce_to_ref']:.4f}; {time.time() - t0:.0f}s")

    # ---- R7-style pools on the standard root (BCE to the reference)
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
    torch.manual_seed(cfg.seed * 1000 + 365)
    pkey, jprec = E3.KeyNet("K3", cfg.d).to(dev), JacobiPrecT4(cfg.loop_T, cfg.gamma_key).to(dev)
    hists["jacobi_prec"] = train_pool(pkey, jprec, 366)
    log(f"r8 trained attn_L3-K1, looped_lin_T4, jacobi_lin_T4, jacobi_lin_T4_prec; {time.time() - t0:.0f}s")

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

    # ---- mixed-tier D3 calibration probes (standard root) and the econ_kc_K3 fit on them
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

    def d3_sampler():
        nb = cfg.agg_batch
        b = torch.randint(0, ntd, (nb,), device=dev, generator=gen)
        ti = torch.randint(3, (nb,), device=dev, generator=gen)
        N = NST[torch.randint(5, (nb,), device=dev, generator=gen)]
        dlb = torch.cat([CLD[ti, b], bg["dl"][b]], 1)
        bb = torch.cat([CLB[ti, b], bg["beta"][b]], 1)
        valid = torch.cat([torch.arange(cfg.n_clu, device=dev)[None] < N[:, None],
                           torch.ones(nb, cfg.n_bg, dtype=torch.bool, device=dev)], 1)
        return l0d[b], dlb, bb, None, valid, ref[b]

    sc["econ_D3_u"], sc["econ_D3_lam"] = fit_logs(
        lambda l0b, dlb, bb, hb, v, e: E4.agg_econ(l0b, dlb, bb, k3e(dlb), v, e[0], e[1])[0], [1.0, 1.0], d3_sampler)
    log(f"r8 D3 calibration probes ({nd} boards) and scalar fits: {json.dumps({k: round(v, 5) for k, v in sc.items()})}; "
        f"{time.time() - t0:.0f}s")
    P = Pools8(anyP, keyK1, attnK1, lkey, looped, jkey, jacobi, pkey, jprec, sc, beta_clean)
    # ---- held-out monitors: standard bags (BCE to the reference, as R7) and anytime bags (BCE to Y by root strength)
    mon, mon_any = {}, {}
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
            l0a, lpa, dla_, ba_, va_, b0, ya, Mi, Ka = any_bag(cfg.agg_batch, ntra, na, g2)
            for name, (keyn, pool_) in anyP.items():
                d_, b_, k_, v_ = anchor_bag(keyn, l0a, lpa, dla_, ba_, va_, b0)
                o_, _ = pool_(l0a, d_, b_, k_, v_)
                lossv = F.binary_cross_entropy_with_logits(o_, ya, reduction="none").mean(1)
                for mi, M in enumerate(M0S):
                    sel = Mi == mi
                    if sel.any():
                        mon_any.setdefault(f"{name}_M{M}", []).append(float(lossv[sel].mean()))
            lr0 = F.binary_cross_entropy_with_logits(l0a, ya, reduction="none").mean(1)
            for mi, M in enumerate(M0S):
                sel = Mi == mi
                if sel.any():
                    mon_any.setdefault(f"root_M{M}", []).append(float(lr0[sel].mean()))
    mon = {k_: float(np.mean(v_)) for k_, v_ in mon.items()}
    mon_any = {k_: float(np.mean(v_)) for k_, v_ in mon_any.items()}
    params = {"looped_attn": E3.nparams(kla, la), "attn_K3": E3.nparams(ka3, a3), "attn_K1": E3.nparams(keyK1, attnK1),
              "looped_lin": E3.nparams(lkey, looped), "jacobi": E3.nparams(jkey, jacobi), "jacobi_prec": E3.nparams(pkey, jprec)}
    log(f"r8 pools: params {params}; held-out BCE to reference {json.dumps({k: round(v, 4) for k, v in mon.items()})}; "
        f"anytime held-out BCE to ref(M_0) {json.dumps({k: round(v, 4) for k, v in mon_any.items()})}; {time.time() - t0:.0f}s")
    info = {"scalars": sc, "beta_by_tau": bt, "beta_clean": beta_clean, "beta_by_root": bM, "trust_train_mse": float(np.mean(tl[-200:])),
            "trust_heldout_mse": tmon, "hist": hists, "heldout_bag_bce_to_ref": mon, "heldout_any_bce_to_ref": mon_any,
            "params": params, "ref_coverage_mean": float(cov.mean()),
            "jacobi": {"lambda": [float(x) for x in jacobi.lam()], "eta": [float(x) for x in jacobi.eta],
                       "P_mean": [float(x) for x in jacobi.P.mean(1)], "s_mean": float(jacobi.s.mean())},
            "jacobi_prec": {"lambda_tilde_softplus": [float(x) for x in F.softplus(jprec.lam_raw)], "a": float(jprec.a),
                            "b": float(jprec.b), "eta": [float(x) for x in jprec.eta],
                            "P_mean": [float(x) for x in jprec.P.mean(1)], "s_mean": float(jprec.s.mean()),
                            "lambda_mean_at_beta": {str(t): float(jprec.lam_rc(torch.full((1, 1), bt[str(t)], device=dev)).mean())
                                                    for t in (0.3, 1.5, 2.5)}},
            "sec": time.time() - t0}
    return trust, P, info


# ------------------------------------------------------------------------------------------------ tests
def puct8(model, trust, P, T, ev_root, l0, lprior, pref_nw, gen, cfg, keys_mode, score_mode, waves=4):
    """2x2 PUCT cell. Wave 1: (1 + lam_U 1[half]) Var, sampled. Waves w >= 1: R6 score (1 + lam_U 1[half]) Var / (1 + crowd)^1.5
    sampled, or R7 amended score (1 + lam_U / (1 + w) 1[half]) (Var / (1 + crowd)^1.5)^2 sampled. crowd from K1 keys (attn_L3-K1's
    head on hbar) or K3 keys (attn_L3-K3's head on the salience); candidate keys from prospective reads with imputed readings.
    mu = the anytime attn_L3-K3 belief (anchor token, M_0 = 2) in every cell. Reads tau 50/50 over 0.3 / 1.5."""
    dev = l0.device
    g = RO.geo(dev)
    half = (g["in_nw"] if pref_nw else g["in_se"]).float()
    keyK3, poolK3 = P.any["attn_K3"]
    n = l0.shape[0]
    b0 = P.beta0(2, n, dev)
    mu = torch.sigmoid(l0)
    bag, picks = None, []
    for w in range(waves):
        var = (mu * (1 - mu)) @ g["pmask"].T
        if bag is None:
            crowd = torch.zeros_like(var)
        else:
            valid = torch.ones(bag["patch"].shape, dtype=torch.bool, device=dev)
            with torch.no_grad():
                hb_c, dl_c = E3.prospect(model, T, ev_root, l0, mu, cfg.chunk)
                if keys_mode == "K3":
                    ke, kc_ = keyK3.embed(None, bag["dl"])[0], keyK3.embed(None, dl_c)[0]
                else:
                    ke, kc_ = P.keyK1(bag["hbar"], bag["dl"], valid)[0], P.keyK1.embed(hb_c, dl_c)[0]
            crowd = (torch.relu(kc_ @ ke.transpose(1, 2)) ** 2).sum(-1)
        if w == 0 or score_mode == "R6":
            score = (1.0 + cfg.lam_U * half)[None] * var / (1.0 + crowd) ** cfg.alpha_nov
        else:
            score = (1.0 + (cfg.lam_U / (1.0 + w)) * half)[None] * (var / (1.0 + crowd) ** cfg.alpha_nov) ** cfg.power
        pick = torch.multinomial(score, cfg.wave_reads, replacement=False, generator=gen)
        ro = E3.bag_reads3(model, T, ev_root, l0, pick, RO.draw_tau(pick.shape, gen, dev), gen, cfg.chunk)
        ro["beta"] = E3.beta_of(trust, ro)
        bag = ro if bag is None else E3.cat_ro(bag, ro)
        picks.append(pick)
        valid = torch.ones(bag["patch"].shape, dtype=torch.bool, device=dev)
        with torch.no_grad():
            dla, ba, k, va = anchor_bag(keyK3, l0, lprior, bag["dl"], bag["beta"], valid, b0)
            mu = torch.sigmoid(poolK3(l0, dla, ba, k, va)[0])
    return bag, picks


def fs_stream(tiles8, sk):
    """Amended reading 7: 8 blocks of 4 reads; slot 4b holds tile slot b (the board's 6 partition tiles in a shared random
    order, then 2 of them again), slots 4b+1..4b+3 hold skewed reads 3b..3b+2. The first 24 reads = 6 tiles + sk[:, :18]."""
    n = sk.shape[0]
    return torch.stack([torch.cat([tiles8[:, b:b + 1], sk[:, 3 * b:3 * b + 3]], 1) for b in range(8)], 1).reshape(n, 32)


def run_prefixes(P, l0, lprior, M, ro, Ks):
    """Pool outputs on the first K reads of a bag, for every K in Ks."""
    res = {}
    for K in Ks:
        r = E4.head_ro(ro, K)
        valid = torch.ones(r["patch"].shape, dtype=torch.bool, device=l0.device)
        res[K], _ = P.all(l0, lprior, M, r["lr"], r["dl"], r["beta"], r["hbar"], valid)
    return res


def test8(model, trust, P, cfg, dev, log, gen, Pf):
    t0 = time.time()
    n = cfg.n_test
    T = A2.boards2_t(A2.STREAM_TEST2, n, dev)
    ordT = torch.as_tensor(anchor_order(A2.STREAM_TEST2, n), device=dev)
    g = RO.geo(dev)
    sector = torch.as_tensor(env6.SECTOR, device=dev)
    ev_full = A2.ev_from(T["s"], torch.ones_like(T["s"]))
    ev_std6 = A2.ev_from(T["s"], A2.cells_mask(T["std6"], dev))
    Pfull = env6.bayes_marginals(ev_full, T["note"], Pf).float()           # EVALUATION ONLY
    Pstd6 = env6.bayes_marginals(ev_std6, T["note"], Pf).float()
    kl = lambda m: float(wm.bern_kl(Pfull, m).mean())
    sig = torch.sigmoid
    pear = lambda a, b: float(np.corrcoef(a.cpu().numpy().ravel(), b.cpu().numpy().ravel())[0, 1])
    lstd = RO.forward_chunks(model, ev_std6, T["note"])
    lprior = prior_logits(model, T)
    EV, L0 = {}, {}
    for M in M0S:
        EV[M], L0[M] = root(model, T, ordT, M)
    R = {"n_test": n, "clause1_corr_std6": pear(sig(lstd), Pstd6), "clause1_kl_std6": float(wm.bern_kl(Pstd6, sig(lstd)).mean()),
         "kl_root": {str(M): kl(sig(L0[M])) for M in M0S}, "kl_prior": kl(sig(lprior))}
    raw = {"pstar": run.r5(Pfull), "root_mu": {str(M): run.r5(sig(L0[M])) for M in M0S}, "anchor_order18": ordT[:, :18].tolist(),
           "occ": T["occ"].int().tolist()}
    log(f"r8 roots: KL by M_0 {json.dumps({k: round(v, 5) for k, v in R['kl_root'].items()})}; clause-1 Pearson {R['clause1_corr_std6']:.5f}")

    # ---- E1-D0: anchor identity at K = 0
    ident = {}
    for M in (2, 18):
        e = torch.zeros(n, 0, 36, device=dev)
        out, _ = P.all(L0[M], lprior, M, e, e, torch.zeros(n, 0, device=dev), torch.zeros(n, 0, 128, device=dev),
                       torch.zeros(n, 0, dtype=torch.bool, device=dev))
        for a in ANY:
            ident[f"{a}_M{M}_max_abs_logit_diff"] = float((out[a] - L0[M]).abs().max())
            ident[f"{a}_M{M}_kl_diff"] = abs(kl(sig(out[a])) - kl(sig(L0[M])))
    R["identity_K0"] = ident
    log(f"r8 K=0 identity: {json.dumps(ident)}")

    # ---- E1-D1.1: sparse-root (M_0 = 2) P-FullSupport prefix curve on the interleaved stream (amended reading 7)
    tl6 = torch.tensor(E3.TILES, device=dev)[torch.rand(n, 6, device=dev, generator=gen).argsort(1)]   # per-board order, shared
    tiles8 = torch.cat([tl6, tl6[:, :2]], 1)                            # 8 tile slots: the 6 tiles, then 2 repeated
    fs = {}
    for name, p in (("NW", cfg.p_full), ("SE", 1 - cfg.p_full)):
        sk, _ = RO.sample_patches(L0[2], 24, torch.full((n,), p, device=dev), gen, cfg.gamma_voi)
        patch = fs_stream(tiles8, sk)
        ro = E3.bag_reads3(model, T, EV[2], L0[2], patch, torch.full(patch.shape, 0.3, device=dev), gen, cfg.chunk)
        ro["beta"] = E3.beta_of(trust, ro)
        fs[name] = run_prefixes(P, L0[2], lprior, 2, ro, KS_FS)
        lu, _ = A2.union_call(model, T, EV[2], patch[:, :24])
        fs[name]["union24"] = lu
        raw[f"fs_{name}"] = {"patch": patch.tolist(), **{f"K{K}_{a}": run.r5(sig(fs[name][K][a])) for K in KS_FS
                                                        for a in ("looped_attn", "attn_K3", "mmd", "scaled_mix")}}
    for K in KS_FS:
        for a in POOL_NAMES:
            mN, mS = sig(fs["NW"][K][a]), sig(fs["SE"][K][a])
            R[f"fs_kl_{a}_K{K}"] = 0.5 * (kl(mN) + kl(mS))
            R[f"fs_jsd_{a}_K{K}"] = float(wm.bern_jsd(mN, mS).mean())
    R["fs_jsd_union_K24"] = float(wm.bern_jsd(sig(fs["NW"]["union24"]), sig(fs["SE"]["union24"])).mean())
    log("r8 sparse-root FS curve looped_attn KL " + " ".join(f"K{K} {R[f'fs_kl_looped_attn_K{K}']:.4f}" for K in KS_FS)
        + f"; JSD(24) {R['fs_jsd_looped_attn_K24']:.5f}")

    # ---- E1-D1.2: strong-root curves (M_0 = 18; 6 and 12 descriptive): 24 VOI reads, 90/10 skew, tau uniform over 3 tiers
    srs = [(int(x), "mixed") for x in cfg.sr_roots.split(",")] + [(18, "tier3")]
    for M, tmode in srs:
        tag = f"sr{M}" if tmode == "mixed" else f"sr{M}t3"
        sr = {}
        for name, p in (("NW", cfg.p_sr), ("SE", 1 - cfg.p_sr)):
            pt, _ = RO.sample_patches(L0[M], cfg.K_sr, torch.full((n,), p, device=dev), gen, cfg.gamma_voi)
            tt = RO.draw_tau(pt.shape, gen, dev) if tmode == "mixed" else E3.draw_tau3(pt.shape, gen, dev)   # amended reading 6
            ro = E3.bag_reads3(model, T, EV[M], L0[M], pt, tt, gen, cfg.chunk)
            ro["beta"] = E3.beta_of(trust, ro)
            sr[name] = run_prefixes(P, L0[M], lprior, M, ro, KS_SR)
            if M == 18:
                raw[f"{tag}_{name}"] = {"patch": pt.tolist(), "tau": tt.tolist(),
                                       **{f"K{K}_{a}": run.r5(sig(sr[name][K][a])) for K in KS_SR
                                          for a in ("looped_attn", "attn_K3", "mmd", "mix")}}
        for K in KS_SR:
            for a in POOL_NAMES:
                mN, mS = sig(sr["NW"][K][a]), sig(sr["SE"][K][a])
                R[f"{tag}_kl_{a}_K{K}"] = 0.5 * (kl(mN) + kl(mS))
                R[f"{tag}_jsd_{a}_K{K}"] = float(wm.bern_jsd(mN, mS).mean())
        log(f"r8 strong-root M_0={M} tau {tmode}: KL(0) {R[f'{tag}_kl_looped_attn_K0']:.4f}; looped_attn " + " ".join(
            f"K{K} {R[f'{tag}_kl_looped_attn_K{K}']:.4f}" for K in KS_SR[1:]) + "; mmd " + " ".join(
            f"K{K} {R[f'{tag}_kl_mmd_K{K}']:.4f}" for K in KS_SR[1:]) + "; mix " + " ".join(
            f"K{K} {R[f'{tag}_kl_mix_K{K}']:.4f}" for K in KS_SR[1:]))

    # ---- D3 controlled bags (standard root, as R7): W(N) and MSE
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
                if a in ("jacobi_prec", "econ_D3", "econ_R7", "kc_K3", "looped_attn"):
                    raw["controlled"][f"t{tv}_N{N}_W_{a}"] = run.r5(Wn.sum(1))
            for a in POOL_NAMES:
                row[f"mse_{a}"] = float(((sig(out[a]) - Pfull) ** 2).mean())
            ctrl[f"t{tv}_N{N}"] = row
            raw["controlled"][f"t{tv}_N{N}"] = {a: run.r5(sig(out[a])) for a in ("jacobi_prec", "econ_D3", "kc_K3", "attn_K3")}
        log(f"r8 controlled tau={tv}: " + " ".join(f"N{N} W_jprec {ctrl[f't{tv}_N{N}']['W_jacobi_prec']:.3f} W_econD3 "
                                                  f"{ctrl[f't{tv}_N{N}']['W_econ_D3']:.3f}" for N in (1, 2, 4, 8, 16)))
    R["controlled"] = ctrl

    # ---- 2x2 PUCT attribution (standard root, mixed reads, 4 waves)
    for km, sm in CELLS:
        cell = f"{km}_{sm}"
        res = {}
        for name, nw in (("NW", True), ("SE", False)):
            bag, picks = puct8(model, trust, P, T, EV[2], L0[2], lprior, nw, gen, cfg, km, sm, cfg.waves)
            valid = torch.ones(bag["patch"].shape, dtype=torch.bool, device=dev)
            out, _ = P.all(L0[2], lprior, 2, bag["lr"], bag["dl"], bag["beta"], bag["hbar"], valid)
            lu, _ = A2.union_call(model, T, EV[2], bag["patch"])
            out["union"] = lu
            res[name] = out
            R[f"puct_{cell}_coverage_{name}"] = float(E3.coverage(bag["patch"], dev).mean(1).mean())
            inh = g["in_nw"] if nw else g["in_se"]
            R[f"puct_{cell}_wave1_pref_share_{name}"] = float(inh[picks[0]].float().mean())
            R[f"puct_{cell}_wave1_strict_share_{name}"] = float((sector == (0 if nw else 1))[picks[0]].float().mean())
            R[f"puct_{cell}_coverage_by_wave_{name}"] = [float(E3.coverage(torch.cat(picks[:i + 1], 1), dev).mean(1).mean())
                                                        for i in range(len(picks))]
            raw[f"puct_{cell}_{name}"] = {"patch": bag["patch"].tolist(), "attn_K3": run.r5(sig(out["attn_K3"]))}
        R[f"puct_{cell}_coverage"] = 0.5 * (R[f"puct_{cell}_coverage_NW"] + R[f"puct_{cell}_coverage_SE"])
        for a in ("attn_K3", "looped_attn", "union"):
            mN, mS = sig(res["NW"][a]), sig(res["SE"][a])
            R[f"puct_{cell}_kl_{a}"] = 0.5 * (kl(mN) + kl(mS))
            R[f"puct_{cell}_jsd_{a}"] = float(wm.bern_jsd(mN, mS).mean())
        log(f"r8 PUCT {cell}: coverage {R[f'puct_{cell}_coverage']:.4f}; attn_K3 jsd {R[f'puct_{cell}_jsd_attn_K3']:.5f} kl "
            f"{R[f'puct_{cell}_kl_attn_K3']:.4f}; wave-1 NW share {R[f'puct_{cell}_wave1_pref_share_NW']:.3f}")
    R["sec"] = time.time() - t0
    log("r8 test done: " + json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in R.items()
                                         if not isinstance(v, (dict, list))})[:4000])
    return R, raw


def main(argv=None):
    ap = argparse.ArgumentParser(description="R8-E1 one seed")
    for f in dataclasses.fields(Cfg8):
        ap.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    cfg = Cfg8(**vars(ap.parse_args(argv)))
    os.makedirs(cfg.out, exist_ok=True)
    dev = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
    torch.set_num_threads(cfg.threads)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    log = run.Log(cfg.out)
    log("config r8e1", json.dumps(asdict(cfg)), "device", dev, torch.__version__)
    if dev.type == "cuda":   # PREREG deviation 3: torch 2.14's Triton bmm-outer-product override needs Python.h, absent on the
        try:                 # VM; deregistering the Triton overrides sends those matmuls to the standard kernels, as in R7
            import torch._native.triton_utils as _tu
            _tu.deregister_op_overrides()
            log("r8 torch native Triton op overrides deregistered (deviation 3)")
        except Exception as e:  # noqa: BLE001
            log("r8 could not deregister torch native Triton op overrides:", repr(e))
    T0 = time.time()
    Pp = env6.placements()
    Pf = torch.as_tensor(Pp, device=dev).double()
    model, meta, ck = load_checked(cfg.ckpt_dir, cfg.seed, dev)
    log(f"r8 theta_real loaded from {ck['path']}: sha256 {ck['sha256'][:12]} matches its .json; {ck['n_params']} parameters")
    for p in model.parameters():
        p.requires_grad_(False)
    summary = {"cfg": asdict(cfg), "experiment": "R8-E1", "n_placements": int(Pp.shape[0]), "torch": torch.__version__,
               "device": str(dev), "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None, "checkpoint": ck,
               "theta_real_train_curve_last": (meta.get("train_curve") or [None])[-1]}
    run.dump(os.path.join(cfg.out, "summary.json"), summary)
    gen = torch.Generator(device=dev).manual_seed(cfg.seed * 1000 + 381)
    trust, Pools, info = fit8(model, cfg, dev, log, gen)
    summary["fit"] = info
    run.dump(os.path.join(cfg.out, "summary.json"), summary)
    res, raw = test8(model, trust, Pools, cfg, dev, log, gen, Pf)
    summary["test"] = res
    run.dump(os.path.join(cfg.out, "raw.json"), raw)
    summary["total_sec"] = time.time() - T0
    summary["done"] = True
    run.dump(os.path.join(cfg.out, "summary.json"), summary)
    log(f"r8 done in {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
