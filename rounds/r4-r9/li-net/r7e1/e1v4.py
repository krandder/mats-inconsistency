"""R7-E1 (E-INFO-V4): exact Jacobi-Richardson normal-equation pool, weight-tied looped attention, kc_dedup key attribution
(K1 vs K3), and coverage-complete key-space PUCT on 6x6 ShipHunt.
Spec: /home/agent/workspaces/mats/outbox/experiments-round-7-spec.md section 3 (gemini, 2026-10-17). Pre-registration:
PREREG-R7-E1.md. Builds on li-net/r6e1/e1v3.py (imported unchanged: reads, prospective reads, KeyNet, LoopedLinT4, open
search, helpers) and li-net/r5e1 (imported unchanged: env6, wm, rollouts, run, partA2).

Per seed: train theta_real v2 exactly as R5-E1 / R6-E1 (partA2.train_world_model2, 40k steps) and SAVE it as
ckpt/s<seed>/theta_real_v2.npz (+ .json metadata, reload-checked) so no later round retrains it (or load it with --load-wm)
-> calibration pools (tau uniform over 0.3 / 1.5 / 2.5) -> theta_trust -> learned pools trained by 36-cell BCE to the
model's own unskewed K = 128 union-call reference: attn_L3 with K1 / K3 / K4 keys, looped_lin_T4 (R6, K4 head),
jacobi_lin_T4 (K3 head), looped_attn_T3 (K3 head) -> fitted scalars: ataraxos_mmd eta, kc_dedup u on frozen K1 / K3 / K4
keys, econ_kc_K3 (u, lambda), scaled_mix alpha -> tests on the 500 R5/R6 test boards: P-FullSupport, P-ActivePUCT
(open_U vs 5-wave key-space PUCT with the amended Reading-9 score in waves 2+, sampled (primary) or greedy (descriptive);
mixed-noise and clean reads), key geometry, controlled bags at tau 0.3 / 1.5 / 2.5. Exact Bayes: test metrics, gate and
monitor curves only.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict

R5DIR = os.environ.get("R5E1_DIR", "/home/agent/workspaces/eurisko-study/li-net/r5e1")
R6DIR = os.environ.get("R6E1_DIR", "/home/agent/workspaces/eurisko-study/li-net/r6e1")
for _p in (R6DIR, R5DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import e1v3 as E3  # noqa: E402  (R6-E1, unchanged)
import env6  # noqa: E402
import partA2 as A2  # noqa: E402
import rollouts as RO  # noqa: E402
import run  # noqa: E402
import wm  # noqa: E402

KV = ("K1", "K3", "K4")                      # attn_L3 key variants trained in R7
KV_SEED = {"K1": 0, "K3": 2, "K4": 3}        # R6's per-variant seed offsets
POOL_NAMES = ("mix", "scaled_mix", "mmd", "kc_K1", "kc_K3", "kc_K4", "econ_K3", "looped_lin", "jacobi", "looped_attn",
              "attn_K1", "attn_K3", "attn_K4")
WPOOLS = ("attn_K1", "attn_K3", "attn_K4", "looped_attn", "jacobi", "econ_K3", "kc_K1", "kc_K3", "kc_K4", "mmd")


@dataclass
class Cfg7(E3.Cfg6):
    out: str = "/tmp/r7e1/dev"
    ckpt_dir: str = "/tmp/r7e1/ckpt"   # theta_real_v2.npz goes to <ckpt_dir>/s<seed>/
    attn_T: int = 3                    # looped_attn_T3 passes
    waves: int = 5                     # PUCT waves run; the rules use the first 4 (K = 24), wave 5 (K = 30) is descriptive
    power: float = 2.0                 # waves 2+: sample in proportion to score ** power
    K30: int = 30
    load_wm: str = ""                  # a theta_real_v2.npz to load instead of training (recovery or reuse)


# ------------------------------------------------------------------------------------------------ theta_real checkpoints
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def save_wm_npz(model, path, meta):
    """np.savez_compressed of the state_dict (float32), written atomically, plus <path without .npz>.json metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **sd)
    os.replace(tmp, path)
    meta = dict(meta, file=os.path.basename(path), sha256=sha256_file(path), bytes=os.path.getsize(path), n_arrays=len(sd),
                arch={"d": model.d, "layers": model.layers, "heads": model.heads, "ff": model.ff},
                n_params=int(sum(p.numel() for p in model.parameters())),
                loader="li-net/r7e1/e1v4.py: load_wm_npz(path, device) -> (wm.WorldModel, meta)")
    run.dump(path[:-4] + ".json", meta)
    return meta


def load_wm_npz(path, dev):
    jp = path[:-4] + ".json"
    meta = json.load(open(jp)) if os.path.exists(jp) else {}
    a = meta.get("arch", {"d": 128, "layers": 4, "heads": 4, "ff": 384})
    model = wm.WorldModel(a["d"], a["layers"], a["heads"], a["ff"]).to(dev)
    with np.load(path) as z:
        sd = {k: torch.from_numpy(z[k]) for k in z.files}
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, meta


# ------------------------------------------------------------------------------------------------ pools
class JacobiLinT4(nn.Module):
    """Exact Jacobi-preconditioned normal equations (spec 3.1): lambda_c x_rc + sum_b g_rb beta_b x_bc = b_rc = beta_r Dl_rc,
    D_rc = lambda_c + sum_b g_rb beta_b divides the whole residual; x^(t+1) = x^(t) + eta_t P_tc (b - A x^(t)) / D, x^(0) = 0,
    T steps; l* = l0 + s_c sum_r x_rc^(T). g_rb = relu(<k_r,k_b>)^gamma v_r v_b, g_rr = v_r (v = validity mask)."""

    def __init__(self, T=4, gamma=2.0):
        super().__init__()
        self.T, self.gamma = T, gamma
        self.lam_raw = nn.Parameter(torch.full((wm.NC,), math.log(math.expm1(1.0))))   # softplus -> lambda_c = 1
        self.eta = nn.Parameter(torch.ones(T))
        self.P = nn.Parameter(torch.ones(T, wm.NC))
        self.s = nn.Parameter(torch.ones(wm.NC))

    def lam(self):
        return F.softplus(self.lam_raw)

    def parts(self, dl, beta, k, valid):
        v = valid.float()
        g = torch.relu(k @ k.transpose(1, 2)) ** self.gamma
        eye = torch.eye(k.shape[1], dtype=torch.bool, device=k.device)[None]
        g = torch.where(eye, torch.ones_like(g), g) * v[:, :, None] * v[:, None, :]
        bv = beta * v
        GB = g * bv[:, None, :]                                            # g_rb beta_b
        lam = self.lam()
        D = lam[None, None, :] + GB.sum(2)[..., None]                      # [n, K, 36]
        b = bv[..., None] * dl.clamp(-8, 8)
        return GB, lam, D, b, v

    def forward(self, l0, dl, beta, k, valid):
        GB, lam, D, b, v = self.parts(dl, beta, k, valid)
        x = torch.zeros_like(b)
        for t in range(self.T):
            res = b - (lam * x + GB @ x)
            x = (x + self.eta[t] * self.P[t] * res / D) * v[..., None]
        return l0 + self.s * x.sum(1), x

    @torch.no_grad()
    def coef(self, dl, beta, k, valid):
        """Per-read effective weights w [n, K, 36] with l* - l0 = sum_r w_r * clamp(Dl_r, -8, 8) exactly (keys and beta fixed):
        x^(T) = Q b per cell, w_bc = s_c beta_b v_b sum_r Q_c[r, b]."""
        GB, lam, D, b, v = self.parts(dl, beta, k, valid)
        n, K = GB.shape[:2]
        eye = torch.eye(K, device=dl.device)
        A = lam[None, :, None, None] * eye + GB[:, None]                   # [n, 36, K, K]
        Dc = D.permute(0, 2, 1)                                            # [n, 36, K]
        vm = v[:, None, :, None]
        Q = torch.zeros(n, wm.NC, K, K, device=dl.device)
        for t in range(self.T):
            step = (self.eta[t] * self.P[t])[None, :, None] / Dc
            Q = (Q + step[..., None] * (eye - A @ Q)) * vm
        w = self.s[None, :, None] * Q.sum(2) * (beta * v)[:, None, :]
        return w.permute(0, 2, 1)


class LoopedAttnT3(nn.Module):
    """wm.AttnL3 with ONE shared (MultiheadAttention, LayerNorm, FFN) block applied T = 3 times, a learned step embedding
    added before each pass; same inputs [k, W_v Dl, log beta, log c] and output head (per-read per-cell weights)."""

    def __init__(self, kdim=16, vdim=16, d=64, T=3, heads=4, init_w=1.0 / 24):
        super().__init__()
        self.T = T
        self.Wv = nn.Linear(wm.NC, vdim)
        self.inp = nn.Linear(kdim + vdim + 2, d)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.step_emb = nn.Parameter(torch.randn(T, d) * 0.02)
        self.out = nn.Linear(d, wm.NC)
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, math.log(math.expm1(init_w)))

    def weights(self, dl, beta, k, valid, gamma=2.0):
        c = wm.crowding(k, valid, gamma)
        dlc = dl.clamp(-8, 8)
        x = self.inp(torch.cat([k, self.Wv(dlc), beta.clamp_min(1e-6).log()[..., None], c.log()[..., None]], -1))
        pad = ~valid
        for t in range(self.T):
            x = x + self.step_emb[t]
            y = self.ln(x)
            x = x + self.att(y, y, y, key_padding_mask=pad, need_weights=False)[0]
            x = x + self.ff(x)
        return F.softplus(self.out(x)) * valid[..., None].float()

    def forward(self, l0, dl, beta, k, valid):
        w = self.weights(dl, beta, k, valid)
        return l0 + (w * dl.clamp(-8, 8)).sum(1), w


def econ_weights(beta, k, valid, u, lam, gamma=2.0):
    """econ_kc_K3 (1-step precision crowding): w_r = u beta_r / (lambda + sum_b beta_b g_rb), g_rb = relu(<k_r,k_b>)^gamma."""
    v = valid.float()
    g = torch.relu(k @ k.transpose(1, 2)) ** gamma * v[:, None, :]
    c = (g * (beta * v)[:, None, :]).sum(2)
    return u * beta / (lam + c) * v


def agg_econ(l0, dl, beta, k, valid, u, lam, gamma=2.0):
    w = econ_weights(beta, k, valid, u, lam, gamma)
    return l0 + (w[..., None] * dl).sum(1), w


class Pools7:
    def __init__(self, keys, attn, lkey, looped, jkey, jacobi, akey, lattn, sc):
        self.keys, self.attn, self.lkey, self.looped = keys, attn, lkey, looped
        self.jkey, self.jacobi, self.akey, self.lattn, self.sc = jkey, jacobi, akey, lattn, sc

    @torch.no_grad()
    def all(self, l0, lr, dl, beta, hbar, valid, need_w=False):
        out, W, K = {}, {}, {}
        vf = valid.float()
        out["mix"] = wm.agg_mix(lr, valid)
        out["scaled_mix"] = l0 + self.sc["alpha"] * (vf[..., None] * dl.clamp(-8, 8)).sum(1)
        out["mmd"] = wm.agg_mmd(l0, dl, beta, valid, self.sc["eta"])
        for v in KV:
            k, _ = self.keys[v](hbar, dl, valid)
            la, w = self.attn[v](l0, dl, beta, k, valid)
            out[f"attn_{v}"], W[f"attn_{v}"], K[v] = la, w, k
            out[f"kc_{v}"] = wm.agg_kc(l0, dl, beta, k, valid, self.sc[f"u_{v}"])
        out["econ_K3"], we = agg_econ(l0, dl, beta, K["K3"], valid, self.sc["econ_u"], self.sc["econ_lam"])
        kl_, _ = self.lkey(hbar, dl, valid)
        out["looped_lin"], _ = self.looped(l0, dl, beta, kl_, valid)
        kj, _ = self.jkey(hbar, dl, valid)
        out["jacobi"], _ = self.jacobi(l0, dl, beta, kj, valid)
        ka, _ = self.akey(hbar, dl, valid)
        out["looped_attn"], W["looped_attn"] = self.lattn(l0, dl, beta, ka, valid)
        K["looped_lin"], K["jacobi"], K["looped_attn"] = kl_, kj, ka
        if need_w:
            ex = lambda w_: w_[..., None].expand(-1, -1, wm.NC)
            W["jacobi"] = self.jacobi.coef(dl, beta, kj, valid)
            W["econ_K3"] = ex(we)
            for v in KV:
                W[f"kc_{v}"] = ex(self.sc[f"u_{v}"] * beta / wm.crowding(K[v], valid) * vf)
            W["mmd"] = ex(self.sc["eta"] * beta * vf)
        return out, W, K


# ------------------------------------------------------------------------------------------------ calibration fit
def fit7(model, cfg, dev, log, gen):
    t0 = time.time()
    C = A2.boards2_t(A2.STREAM_CAL2, cfg.n_cal, dev)
    ev_root = A2.ev_from(C["s"], A2.cells_mask(C["anchors"], dev))
    l0 = RO.forward_chunks(model, ev_root, C["note"])
    one, zero = torch.ones(cfg.n_cal, device=dev), torch.zeros(cfg.n_cal, device=dev)
    pnw, _ = RO.sample_patches(l0, cfg.pool, one, gen, cfg.gamma_voi)
    pse, _ = RO.sample_patches(l0, cfg.pool, zero, gen, cfg.gamma_voi)
    patch = torch.cat([pnw, pse], 1)
    tau = E3.draw_tau3(patch.shape, gen, dev)
    pool = E3.bag_reads3(model, C, ev_root, l0, patch, tau, gen, cfg.chunk)
    h = cfg.n_ref // 2
    rnw, _ = RO.sample_patches(l0, h, one, gen, cfg.gamma_voi)
    rse, _ = RO.sample_patches(l0, cfg.n_ref - h, zero, gen, cfg.gamma_voi)
    lref, cov = A2.union_call(model, C, ev_root, torch.cat([rnw, rse], 1))
    ref = torch.sigmoid(lref)
    log(f"r7 cal pools + reference: {time.time() - t0:.0f}s; reference coverage {float(cov.mean()):.4f}")
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
    log(f"r7 theta_trust: train mse {np.mean(tl[-200:]):.4f} heldout {tmon:.4f}; beta by tau {bt}")

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

    keys, attn, hists = {}, {}, {}
    for v in KV:
        i = KV_SEED[v]
        torch.manual_seed(cfg.seed * 1000 + 331 + i)
        keys[v] = E3.KeyNet(v, cfg.d).to(dev)
        attn[v] = wm.AttnL3().to(dev)
        hists[f"attn_{v}"] = train_pool(keys[v], attn[v], 341 + i)
        log(f"r7 trained attn_L3 with {v} keys; last bce {hists[f'attn_{v}'][-1]['bce']:.4f}; {time.time() - t0:.0f}s")
    bm = float(beta.mean())
    torch.manual_seed(cfg.seed * 1000 + 351)
    lkey = E3.KeyNet("K4", cfg.d).to(dev)
    looped = E3.LoopedLinT4(cfg.loop_T, cfg.gamma_key, 1.0 / (cfg.K * bm)).to(dev)
    hists["looped_lin"] = train_pool(lkey, looped, 352)
    log(f"r7 trained looped_lin_T4 (R6); last bce {hists['looped_lin'][-1]['bce']:.4f}; {time.time() - t0:.0f}s")
    torch.manual_seed(cfg.seed * 1000 + 361)
    jkey = E3.KeyNet("K3", cfg.d).to(dev)
    jacobi = JacobiLinT4(cfg.loop_T, cfg.gamma_key).to(dev)
    hists["jacobi"] = train_pool(jkey, jacobi, 362)
    log(f"r7 trained jacobi_lin_T4; last bce {hists['jacobi'][-1]['bce']:.4f}; {time.time() - t0:.0f}s")
    torch.manual_seed(cfg.seed * 1000 + 371)
    akey = E3.KeyNet("K3", cfg.d).to(dev)
    lattn = LoopedAttnT3(T=cfg.attn_T).to(dev)
    hists["looped_attn"] = train_pool(akey, lattn, 372)
    log(f"r7 trained looped_attn_T3; last bce {hists['looped_attn'][-1]['bce']:.4f}; {time.time() - t0:.0f}s")

    def fit_logs(fn, inits, **bk):
        lp = torch.tensor([math.log(x) for x in inits], device=dev, requires_grad=True)
        o = torch.optim.Adam([lp], lr=0.02)
        for s in range(cfg.fit_steps):
            l0b, lrb, dlb, bb, hb, valid, rb = bag(cfg.agg_batch, 0, ntr, **bk)
            loss = F.binary_cross_entropy_with_logits(fn(l0b, dlb, bb, hb, valid, lp.exp()), rb)
            o.zero_grad(set_to_none=True)
            loss.backward()
            o.step()
        return [float(x) for x in lp.exp()]

    sc = {}
    sc["eta"] = fit_logs(lambda l0b, dlb, bb, hb, v, e: wm.agg_mmd(l0b, dlb, bb, v, e[0]), [1.0 / (cfg.K * bm)])[0]
    for v in KV:
        sc[f"u_{v}"] = fit_logs(lambda l0b, dlb, bb, hb, vv, e, v=v: wm.agg_kc(l0b, dlb, bb, keys[v](hb, dlb, vv)[0], vv, e[0]),
                                [1.0 / bm])[0]
    sc["econ_u"], sc["econ_lam"] = fit_logs(
        lambda l0b, dlb, bb, hb, vv, e: agg_econ(l0b, dlb, bb, keys["K3"](hb, dlb, vv)[0], vv, e[0], e[1])[0], [1.0, 1.0])
    sc["alpha"] = fit_logs(lambda l0b, dlb, bb, hb, v, e: l0b + e[0] * (v.float()[..., None] * dlb.clamp(-8, 8)).sum(1),
                           [1.0 / cfg.K], kmin=cfg.K, kmax=cfg.K, p_fixed=0.5)[0]
    P = Pools7(keys, attn, lkey, looped, jkey, jacobi, akey, lattn, sc)
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
    params = {f"attn_{v}": E3.nparams(keys[v], attn[v]) for v in KV}
    params.update(looped_lin=E3.nparams(lkey, looped), jacobi=E3.nparams(jkey, jacobi), looped_attn=E3.nparams(akey, lattn),
                  attn_core=E3.nparams(attn["K3"]), looped_attn_core=E3.nparams(lattn), jacobi_core=E3.nparams(jacobi),
                  key_head_K3=E3.nparams(jkey))
    log(f"r7 pools: scalars {json.dumps({k: round(v, 5) for k, v in sc.items()})}; params {params}; held-out bag BCE to "
        "reference " + " ".join(f"{k_} {v_:.4f}" for k_, v_ in mon.items()) + f"; {time.time() - t0:.0f}s")
    info = {"scalars": sc, "beta_by_tau": bt, "trust_train_mse": float(np.mean(tl[-200:])), "trust_heldout_mse": tmon,
            "hist": hists, "heldout_bag_bce_to_ref": mon, "params": params, "ref_coverage_mean": float(cov.mean()),
            "jacobi": {"lambda": [float(x) for x in jacobi.lam()], "eta": [float(x) for x in jacobi.eta],
                       "P_mean": [float(x) for x in jacobi.P.mean(1)], "s": [float(x) for x in jacobi.s]},
            "looped_lin": {"eta": [float(x) for x in looped.eta], "s": float(looped.log_s.exp())},
            "looped_attn_step_emb_norm": [float(x) for x in lattn.step_emb.norm(dim=1)], "sec": time.time() - t0}
    return trust, P, info


# ------------------------------------------------------------------------------------------------ test protocols
def head_ro(ro, k):
    return {key: (v[:, :k] if torch.is_tensor(v) and v.dim() >= 2 and v.shape[1] >= k else v) for key, v in ro.items()}


def merge(P, l0, ro, need_w=False):
    valid = torch.ones(ro["patch"].shape, dtype=torch.bool, device=l0.device)
    return P.all(l0, ro["lr"], ro["dl"], ro["beta"], ro["hbar"], valid, need_w)


def puct_search7(model, trust, P, T, ev_root, l0, pref_nw, gen, cfg, mode="power", tau_mode="mixed", waves=None):
    """Key-space PUCT (spec 3.1 item 3 with amended Reading 9, experiments-round-7-amendments.md Part 2):
    wave 1 (w = 0): score = (1 + lam_U 1[p in half(U)]) Var_mu(p), 6 distinct patches sampled in proportion to it (as R6);
    waves w >= 1: score_w = (1 + lam_U / (1 + w) 1[p in half(U)]) (Var_mu(p) / (1 + crowd(p))^alpha_nov) ** cfg.power,
    mode 'power' samples 6 distinct patches in proportion to score_w (primary), mode 'greedy' takes its top 6 (descriptive).
    crowd(p) = sum over earlier waves' reads r' of relu(<k_p,k_r'>)^2 with K3 keys (attn_L3-K3's head on the salience;
    candidate keys from prospective reads whose readings are imputed from mu, never the board's real readings);
    mu = attn_L3-K3's merged belief of the reads so far (the root in wave 1).
    tau_mode 'mixed': tau 50/50 over 0.3 / 1.5 per read; 'clean': every read at tau 0.3."""
    dev = l0.device
    g = RO.geo(dev)
    half = g["in_nw"] if pref_nw else g["in_se"]
    keyn, attn = P.keys["K3"], P.attn["K3"]
    mu = torch.sigmoid(l0)
    bag, picks = None, []
    for w in range(waves or cfg.waves):
        var = (mu * (1 - mu)) @ g["pmask"].T                                  # [n, 40]
        if bag is None:
            crowd = torch.zeros_like(var)
        else:
            valid = torch.ones(bag["patch"].shape, dtype=torch.bool, device=dev)
            with torch.no_grad():
                ke, _ = keyn(bag["hbar"], bag["dl"], valid)
                _, dl_c = E3.prospect(model, T, ev_root, l0, mu, cfg.chunk)
                kc_, _ = keyn.embed(None, dl_c)                              # K3: a function of the salience only
            crowd = (torch.relu(kc_ @ ke.transpose(1, 2)) ** 2).sum(-1)       # [n, 40]
        if w == 0:
            score = (1.0 + cfg.lam_U * half.float())[None] * var / (1.0 + crowd) ** cfg.alpha_nov   # crowd = 0: pref * var
            pick = torch.multinomial(score, cfg.wave_reads, replacement=False, generator=gen)
        else:
            pref_w = 1.0 + (cfg.lam_U / (1.0 + w)) * half.float()
            score = pref_w[None] * (var / (1.0 + crowd) ** cfg.alpha_nov) ** cfg.power
            if mode == "power":
                pick = torch.multinomial(score, cfg.wave_reads, replacement=False, generator=gen)
            elif mode == "greedy":
                pick = torch.topk(score, cfg.wave_reads, dim=1).indices
            else:
                raise ValueError(mode)
        tau = RO.draw_tau(pick.shape, gen, dev) if tau_mode == "mixed" else torch.full(pick.shape, 0.3, device=dev)
        ro = E3.bag_reads3(model, T, ev_root, l0, pick, tau, gen, cfg.chunk)
        ro["beta"] = E3.beta_of(trust, ro)
        bag = ro if bag is None else E3.cat_ro(bag, ro)
        picks.append(pick)
        valid = torch.ones(bag["patch"].shape, dtype=torch.bool, device=dev)
        with torch.no_grad():
            k, _ = keyn(bag["hbar"], bag["dl"], valid)
            la, _ = attn(l0, bag["dl"], bag["beta"], k, valid)
        mu = torch.sigmoid(la)
    return bag, picks


def test7(model, trust, P, cfg, dev, log, gen, Pf):
    t0 = time.time()
    n = cfg.n_test
    T = A2.boards2_t(A2.STREAM_TEST2, n, dev)
    g = RO.geo(dev)
    sector = torch.as_tensor(env6.SECTOR, device=dev)
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
    sig = torch.sigmoid
    R = {"n_test": n, "clause1_corr_std6": pear(sig(lstd), Pstd6),
         "clause1_kl_std6": float(wm.bern_kl(Pstd6, sig(lstd)).mean()), "kl_p0": kl(mu0)}
    raw = {"pstar": run.r5(Pfull), "mu0": run.r5(mu0), "occ": T["occ"].int().tolist(), "anchors": T["anchors"].tolist()}

    # ---- P-FullSupport: 6 partition tiles + 18 clean reads skewed 90/10 (as R6-E1)
    fs = {}
    tiles = torch.tensor(E3.TILES, device=dev)[None].expand(n, -1)
    for name, p in (("NW", cfg.p_full), ("SE", 1 - cfg.p_full)):
        sk, half = RO.sample_patches(l0, cfg.K - len(E3.TILES), torch.full((n,), p, device=dev), gen, cfg.gamma_voi)
        patch = torch.cat([tiles, sk], 1)
        ro = E3.bag_reads3(model, T, ev_root, l0, patch, torch.full(patch.shape, 0.3, device=dev), gen, cfg.chunk)
        ro["beta"] = E3.beta_of(trust, ro)
        out, _, _ = merge(P, l0, ro)
        lu, cov = A2.union_call(model, T, ev_root, patch)
        out["union"] = lu
        fs[name] = out
        R[f"fs_frac_pref_{name}"] = float((half if name == "NW" else ~half).float().mean())
        R[f"fs_coverage_{name}"] = float(cov.mean())
        raw[f"fs_{name}"] = {"patch": patch.tolist(), **{a: run.r5(sig(out[a])) for a in list(POOL_NAMES) + ["union"]}}
    for a in list(POOL_NAMES) + ["union"]:
        mN, mS = sig(fs["NW"][a]), sig(fs["SE"][a])
        R[f"fs_kl_{a}"] = 0.5 * (kl(mN) + kl(mS))
        R[f"fs_jsd_{a}"] = float(wm.bern_jsd(mN, mS).mean())
    log("r7 P-FullSupport: " + " ".join(f"{a} kl {R[f'fs_kl_{a}']:.4f} jsd {R[f'fs_jsd_{a}']:.5f}" for a in
                                        ("scaled_mix", "mmd", "kc_K1", "kc_K3", "jacobi", "looped_lin", "looped_attn",
                                         "attn_K3", "union")))

    # ---- P-ActivePUCT under mixed-noise (tau 50/50 0.3 / 1.5) and clean (tau 0.3) reads: open_U (K = 30, first 24 = K 24),
    #      puct power (5 waves: first 4 = K 24), puct greedy (4 waves, descriptive)
    RAWP = ("attn_K3", "scaled_mix", "mmd", "kc_K1", "kc_K3")
    open_keys = {}
    for st in ("mixed", "clean"):
        tv = None if st == "mixed" else 0.3
        res = {}
        for name, nw in (("NW", True), ("SE", False)):
            op30 = E3.open_search(model, trust, T, ev_root, l0, cfg.p_open if nw else 1 - cfg.p_open, gen, cfg, K=cfg.K30, tau=tv)
            pb5, waves = puct_search7(model, trust, P, T, ev_root, l0, nw, gen, cfg, "power", st, cfg.waves)
            gb, gwaves = puct_search7(model, trust, P, T, ev_root, l0, nw, gen, cfg, "greedy", st, 4)
            conds = {"open": head_ro(op30, cfg.K), "puct": head_ro(pb5, cfg.K), "greedy": gb, "open30": op30, "puct30": pb5}
            res[name] = {}
            for kind, ro in conds.items():
                o, _, Kd = merge(P, l0, ro)
                lu, _ = A2.union_call(model, T, ev_root, ro["patch"])
                o["union"] = lu
                res[name][kind] = o
                R[f"ap_{st}_coverage_{kind}_{name}"] = float(E3.coverage(ro["patch"], dev).mean(1).mean())
                if st == "mixed" and kind == "open":
                    open_keys[name] = (Kd, ro["patch"])
            inh = g["in_nw"] if nw else g["in_se"]
            strict = sector == (0 if nw else 1)
            for kind, wv in (("puct", waves), ("greedy", gwaves)):
                R[f"ap_{st}_{kind}_wave1_pref_share_{name}"] = float(inh[wv[0]].float().mean())
                R[f"ap_{st}_{kind}_wave1_strict_share_{name}"] = float(strict[wv[0]].float().mean())
                R[f"ap_{st}_{kind}_wave_pref_share_{name}"] = [float(inh[w_].float().mean()) for w_ in wv]
                R[f"ap_{st}_{kind}_coverage_by_wave_{name}"] = [
                    float(E3.coverage(torch.cat(wv[:i + 1], 1), dev).mean(1).mean()) for i in range(len(wv))]
            R[f"ap_{st}_open_pref_share_{name}"] = float(inh[op30["patch"][:, :cfg.K]].float().mean())
            raw[f"ap_{st}_{name}"] = {"open_patch": op30["patch"].tolist(), "puct_patch": pb5["patch"].tolist(),
                                      "greedy_patch": gb["patch"].tolist(),
                                      **{f"{kind}_{a}": run.r5(sig(res[name][kind][a]))
                                         for kind in ("open", "puct") for a in RAWP + ("union",)},
                                      **{f"{kind}_attn_K3": run.r5(sig(res[name][kind]["attn_K3"]))
                                         for kind in ("greedy", "open30", "puct30")}}
        for kind in ("open", "puct", "greedy", "open30", "puct30"):
            R[f"ap_{st}_coverage_{kind}"] = 0.5 * (R[f"ap_{st}_coverage_{kind}_NW"] + R[f"ap_{st}_coverage_{kind}_SE"])
            for a in list(POOL_NAMES) + ["union"]:
                mN, mS = sig(res["NW"][kind][a]), sig(res["SE"][kind][a])
                R[f"ap_{st}_{kind}_kl_{a}"] = 0.5 * (kl(mN) + kl(mS))
                R[f"ap_{st}_{kind}_jsd_{a}"] = float(wm.bern_jsd(mN, mS).mean())
        log(f"r7 P-ActivePUCT {st}: coverage open {R[f'ap_{st}_coverage_open']:.4f} puct {R[f'ap_{st}_coverage_puct']:.4f} "
            f"greedy {R[f'ap_{st}_coverage_greedy']:.4f} puct30 {R[f'ap_{st}_coverage_puct30']:.4f}; attn_K3 jsd open "
            f"{R[f'ap_{st}_open_jsd_attn_K3']:.5f} puct {R[f'ap_{st}_puct_jsd_attn_K3']:.5f}; kl open "
            f"{R[f'ap_{st}_open_kl_attn_K3']:.4f} puct {R[f'ap_{st}_puct_kl_attn_K3']:.4f}; union jsd open "
            f"{R[f'ap_{st}_open_jsd_union']:.5f} puct {R[f'ap_{st}_puct_jsd_union']:.5f}; wave-1 NW share "
            f"{R[f'ap_{st}_puct_wave1_pref_share_NW']:.3f}")

    # ---- key geometry (descriptive): within-bag pairs of the mixed open_U NW / SE bags (first 24 reads) and an unskewed bag
    un = E3.open_search(model, trust, T, ev_root, l0, 0.5, gen, cfg)
    _, _, un_K = merge(P, l0, un)
    keysets = [open_keys["NW"], open_keys["SE"], (un_K, un["patch"])]
    for v in ("K1", "K3", "K4", "jacobi", "looped_attn", "looped_lin"):
        cs_l, ov_l, sm_l, ac_l = [], [], [], []
        for Kd, pt in keysets:
            cs, ov, sm, ac = E3.key_stats(Kd[v], pt, dev)
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
    raw["key_bags"] = {"patch": [pt.tolist() for _, pt in keysets], "keys_K3": [run.r5(Kd["K3"], 4) for Kd, _ in keysets]}
    log("r7 keys: " + " ".join(f"{v} r {R[f'key_{v}_pearson']:.3f} gap {R[f'key_{v}_gap']:.3f}" for v in
                               ("K1", "K3", "K4", "jacobi", "looped_attn", "looped_lin")))

    # ---- controlled bags: N in {1,2,4,8,16} reads of one patch at tau 0.3 / 1.5 / 2.5 + 8 shared background reads (as R6)
    astar, _ = RO.sample_patches(l0, 1, torch.full((n,), 0.5, device=dev), gen, cfg.gamma_voi)
    excl = g["overlap"][astar[:, 0]] > 0
    bgp, _ = RO.sample_patches(l0, cfg.n_bg, torch.full((n,), 0.5, device=dev), gen, cfg.gamma_voi, exclude=excl)
    bg = E3.bag_reads3(model, T, ev_root, l0, bgp, RO.draw_tau(bgp.shape, gen, dev), gen, cfg.chunk)
    bg["beta"] = E3.beta_of(trust, bg)
    pm = g["pmask"][astar[:, 0]].bool()
    ctrl = {}
    raw["controlled"] = {"astar": astar[:, 0].tolist(), "bg_patch": bgp.tolist()}
    RAWC = ("attn_K3", "jacobi", "mmd", "kc_K1", "kc_K3")
    for tv in (0.3, 1.5, 2.5):
        cp = astar.expand(n, cfg.n_clu)
        cl = E3.bag_reads3(model, T, ev_root, l0, cp, torch.full(cp.shape, tv, device=dev), gen, cfg.chunk)
        cl["beta"] = E3.beta_of(trust, cl)
        ro = E3.cat_ro(cl, bg)
        for N in (1, 2, 4, 8, 16):
            valid = torch.zeros(ro["patch"].shape, dtype=torch.bool, device=dev)
            valid[:, :N] = True
            valid[:, cfg.n_clu:] = True
            out, W, _ = P.all(l0, ro["lr"], ro["dl"], ro["beta"], ro["hbar"], valid, need_w=True)
            row = {}
            for a in WPOOLS:
                Wn = (W[a][:, :N] * pm[:, None, :].float()).sum(2) / 6.0
                row[f"W_{a}"] = float(Wn.sum(1).mean())
                if a in ("attn_K3", "jacobi"):
                    raw["controlled"][f"t{tv}_N{N}_W_{a}"] = run.r5(Wn.sum(1))
            for a in POOL_NAMES:
                row[f"mse_{a}"] = float(((sig(out[a]) - Pfull) ** 2).mean())
            ctrl[f"t{tv}_N{N}"] = row
            raw["controlled"][f"t{tv}_N{N}"] = {a: run.r5(sig(out[a])) for a in RAWC}
        log(f"r7 controlled tau={tv}: " + " ".join(f"N{N} W_attnK3 {ctrl[f't{tv}_N{N}']['W_attn_K3']:.3f} W_jacobi "
                                                  f"{ctrl[f't{tv}_N{N}']['W_jacobi']:.3f}" for N in (1, 2, 4, 8, 16)))
    R["controlled"] = ctrl
    R["sec"] = time.time() - t0
    log("r7 test done: " + json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in R.items()
                                         if not isinstance(v, (dict, list))}))
    return R, raw


def main(argv=None):
    ap = argparse.ArgumentParser(description="R7-E1 one seed")
    for f in dataclasses.fields(Cfg7):
        ap.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    cfg = Cfg7(**vars(ap.parse_args(argv)))
    os.makedirs(cfg.out, exist_ok=True)
    dev = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
    torch.set_num_threads(cfg.threads)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    log = run.Log(cfg.out)
    log("config r7e1", json.dumps(asdict(cfg)), "device", dev, torch.__version__)
    T0 = time.time()
    P = env6.placements()
    Pt = torch.as_tensor(P, device=dev)
    Pf = Pt.double()
    summary = {"cfg": asdict(cfg), "experiment": "R7-E1", "n_placements": int(P.shape[0]), "torch": torch.__version__,
               "device": str(dev), "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None}
    ck = os.path.join(cfg.ckpt_dir, f"s{cfg.seed}", "theta_real_v2.npz")
    if cfg.load_wm:
        model, meta = load_wm_npz(cfg.load_wm, dev)
        curve, tsec = meta.get("train_curve"), meta.get("train_sec")
        summary["loaded_theta_real"] = {"path": cfg.load_wm, "sha256": sha256_file(cfg.load_wm), "meta_sha256": meta.get("sha256")}
        log("r7 theta_real LOADED from", cfg.load_wm, "sha256", summary["loaded_theta_real"]["sha256"])
    else:
        model, curve, tsec = A2.train_world_model2(cfg, dev, log, Pt, Pf)
        meta = save_wm_npz(model, ck, {"experiment": "R7-E1", "seed": cfg.seed, "train_steps": cfg.train_steps,
                                       "batch": cfg.batch, "lr": cfg.lr, "train_sec": tsec, "train_curve": curve,
                                       "torch": torch.__version__, "device": str(dev),
                                       "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                       "trainer": "li-net/r5e1/partA2.py train_world_model2 (R5-E1 Part A v2)"})
        m2, _ = load_wm_npz(ck, dev)
        Tb = A2.boards2_t(A2.STREAM_TEST2, min(200, cfg.n_test), dev)
        evr = A2.ev_from(Tb["s"], A2.cells_mask(Tb["anchors"], dev))
        with torch.no_grad():
            d_ = float((RO.forward_chunks(model, evr, Tb["note"]) - RO.forward_chunks(m2, evr, Tb["note"])).abs().max())
        summary["ckpt"] = {"path": ck, "sha256": meta["sha256"], "bytes": meta["bytes"], "reload_max_abs_logit_diff": d_}
        log(f"r7 theta_real saved to {ck} ({meta['bytes']} bytes, sha256 {meta['sha256'][:12]}); reload max |logit diff| {d_}")
        del m2
    summary.update(theta_real_params=model.n_params(), train_sec=tsec, train_curve=curve)
    run.dump(os.path.join(cfg.out, "summary.json"), summary)
    for p in model.parameters():
        p.requires_grad_(False)
    gen = torch.Generator(device=dev).manual_seed(cfg.seed * 1000 + 371)
    trust, Pools, info = fit7(model, cfg, dev, log, gen)
    summary["fit"] = info
    run.dump(os.path.join(cfg.out, "summary.json"), summary)
    res, raw = test7(model, trust, Pools, cfg, dev, log, gen, Pf)
    summary["test"] = res
    run.dump(os.path.join(cfg.out, "raw.json"), raw)
    summary["total_sec"] = time.time() - T0
    summary["done"] = True
    run.dump(os.path.join(cfg.out, "summary.json"), summary)
    log(f"r7 done in {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
