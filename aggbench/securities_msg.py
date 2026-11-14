"""Securities as the message (learnability plan step 7, outbox/learnability-review-2026-10-22.md section 4 and 5.7):
the experimental link between objective (1) (listed securities / the log-likelihood span, `analysis/gsigma.py`) and
objective (2) (learned compression on parity matching the uncompressed transformer).

Task (`SPECS`): "lr" = the leak-reduced primary parity benchmark on main, `Z9Spec(readers_random=True,
readers_alpha=1.0, readers_per_bit=6)` (13 agents x 2 readings, per-episode bit weights ~ Dirichlet(1); prior KL 0.270,
count code 0.2595: leak-reduced, not leak-free); "rr3" = readers random with 3 per bit (7 agents; prior 0.308, count
code 0.256). World omega = 6 bits (4 parity bits, 2 distractors), uniform prior; Y = XOR of the parity bits.

Message. Each agent sends ONE trade: (security id, quantised amount), id_bits + amount_bits = b, so the id is paid for
inside the b-bit budget. Securities are functions phi: omega -> R^S, S = 2^id_bits. The amount is FSQ-quantised to
2^amount_bits levels in [-1, 1] times a learned per-security scale (amount_bits = 0: a fixed learned amount).
Decoder. The trades are summed in natural-parameter space over the listed securities, eta = sum_i a_i e_{s_i}, and
read out either by the market itself ("market": q(omega) ~ pi(omega) exp(eta . phi(omega)), P(Y = 1) = q(Y = 1); it
knows the public world model pi and Y(omega), never the per-episode structure) or by a learned MLP on eta ("mlp").
With bit securities the market readout is the exact parity readout P(Y = 1) = (1 - prod_j (1 - 2 p_j)) / 2.

Arms (`ARMS`). Unless marked "free", the trade is tied to the agent's forecast of phi(omega) (see `build`).
  oracle_closed       closed form: each agent's trade on the bit securities at the mean-accuracy LLR (b = 6: 41
                      distinct trade vectors) or with accuracy in 2 levels (b = 8); the review's reference is 0.0147 at
                      5.49 bits on fixed readers (0.013 on rr3, 0.006 on lr)
  oraclephi_*         phi fixed to bit securities (single bits, then signed pairs and triples of parity-bit spins, all
                      in span(bits)) [structure-informed: the securities are given]; Y labels only, or with
                      settlement (agents observe the bits at settlement: latent supervision)
  settle_*            phi learned, trained with SETTLEMENT: phi(omega) is observed at settlement and every agent pays a
                      local squared loss on its forecast of phi(omega) [latent supervision]; plus the Y loss
  settledict_*        as settle_*, but phi is standardised, not whitened (columns may be non-orthogonal, like the
                      oracle baskets), and each security's payoff function is moved only by the agents that traded it
  settlefree_market   as settle_market, but id and amount come from their own heads (settlement only shapes the trunk)
  yonly_*             phi learned, Y labels only (the same network as settle_*, no settlement loss)
  ysec_*              one security, Y itself (id_bits = 0, all b bits on the amount); its settlement is the local Y loss
Learned phi is a (64, S) table, centred and whitened under pi (orthonormal columns), so it cannot collapse to a
constant or to S copies of one function. The encoder reads its own token with a separate projection + LayerNorm per
input part (readings, own keys) (the gate's input-scale fix). Ids use straight-through Gumbel-softmax; eval is argmax.

Reported per run (`train_arm`): dev KL to Bayes (nats, best-val checkpoint on a separate validation split, fresh
episodes every step), symbol entropy (bits per agent; id and amount parts), distinct symbols, and for the securities:
exact R(phi) = additive-pool regret of agents trading their exact posteriors on span(1, phi) (moment matching, exact
enumeration on gsigma Z9-p2 cells with readers_random, nats), principal angles between span(phi) and the parity-bit span
and the true log-likelihood span (gsigma.span_svd), and R for a random phi of the same size (degeneracy control).

    python3 -m aggbench.securities_msg --spec lr --arms settle_market yonly_market --bits 2 4 6 8 --seeds 0 1 2 --out-dir runs/x
"""
import copy
import json
import math
import os
import sys

import numpy as np

from .parity_learn import (DEV_SEED, DIST_IDS, NV, VAL_SEED, Z9Spec, _entropy_bits, _kl, bayes_logit, prior_logit,
                           tokens_from)

W = 2 ** NV
BITS = (np.arange(W)[:, None] >> np.arange(NV)) & 1          # (64, 6): bit j of omega = (omega >> j) & 1
LN2 = math.log(2.0)
ID_AMOUNT_BITS = {1: (0, 1), 2: (1, 1), 3: (1, 2), 4: (2, 2), 5: (3, 2), 6: (4, 2), 7: (4, 3), 8: (5, 3)}
SPECS = {   # name -> (sampler spec, zoo overrides for the gsigma cells)
    "lr": (Z9Spec(readers_random=True, readers_alpha=1.0, readers_per_bit=6),
           dict(readers_random=True, readers_alpha=1.0, readers_per_bit=6)),     # leak-reduced primary (main)
    "rr3": (Z9Spec(readers_random=True), dict(readers_random=True)),            # readers random, 3 per bit
}
SPEC = SPECS["lr"][0]


def target(k=4):
    return BITS[:, :k].sum(1) % 2


# ----------------------------------------------------------------------------- data

def sample_world(spec, n, rng):
    """parity_learn.sample with the world state kept: the same draws in the same order (tests check equality), plus
    w (n,) = sum_j bit_j 2^j and the per-reader var, acc, read arrays."""
    R = spec.n_readers
    if spec.readers_random and getattr(spec, "readers_alpha", 0) > 0:
        wts = rng.dirichlet(np.full(spec.k, spec.readers_alpha), n)
        u = rng.random((n, spec.k * spec.readers_per_bit, 1))
        par = (u > np.cumsum(wts, 1)[:, None, :]).sum(-1).clip(0, spec.k - 1)
    elif spec.readers_random:
        par = rng.integers(0, spec.k, (n, spec.k * spec.readers_per_bit))
    else:
        par = np.broadcast_to(np.repeat(np.arange(spec.k), spec.readers_per_bit), (n, spec.k * spec.readers_per_bit))
    dist = np.asarray(DIST_IDS)[rng.integers(0, len(DIST_IDS), (n, spec.n_dist_agents))]
    var = np.concatenate([par, dist], 1)
    var = np.take_along_axis(var, np.argsort(rng.random((n, R)), 1), 1)
    acc = rng.uniform(*spec.acc, (n, R))
    bits = (rng.random((n, NV)) < spec.bit_prob).astype(np.int64)
    truth = np.take_along_axis(bits, var, 1)
    read = np.where(rng.random((n, R)) < acc, truth, 1 - truth)
    y = bits[:, :spec.k].sum(1) % 2
    tok, keys = tokens_from(var, acc, read, spec.p)
    return {"tokens": tok, "keys": keys, "y": y.astype(np.float32), "w": (bits << np.arange(NV)).sum(1),
            "bayes": bayes_logit(var, acc, read, spec.bit_prob, spec.k), "prior_logit": prior_logit(spec),
            "var": var, "acc": acc, "read": read}


# ----------------------------------------------------------------------------- securities

def basket_securities(S, k=4):
    """S securities in span(parity-bit spins): single bits, then signed pairs s_j +- s_j', then signed triples
    (4 + 12 + 16 = 32 for k = 4). (64, S)."""
    s = 2.0 * BITS[:, :k] - 1
    cols = [s[:, j] for j in range(k)]
    for j in range(k):
        for l in range(j + 1, k):
            cols += [s[:, j] + s[:, l], s[:, j] - s[:, l]]
    for j in range(k):
        for l in range(j + 1, k):
            for m in range(l + 1, k):
                cols += [s[:, j] + s[:, l] + s[:, m], s[:, j] + s[:, l] - s[:, m], s[:, j] - s[:, l] + s[:, m],
                         s[:, j] - s[:, l] - s[:, m]]
    if S > len(cols):
        raise ValueError(f"at most {len(cols)} basket securities")
    return np.stack(cols[:S], 1)


def y_security(k=4):
    return (2.0 * target(k) - 1)[:, None]


def market_logit_np(eta, phi, k=4):
    """log-odds of Y = 1 under q(omega) ~ pi exp(eta . phi) (uniform pi). eta (n, S), phi (64, S)."""
    s = eta @ phi.T
    y = target(k)
    return np.logaddexp.reduce(s[:, y == 1], 1) - np.logaddexp.reduce(s[:, y == 0], 1)


def oracle_closed(d, spec, acc_levels=1):
    """Closed-form oracle bit securities: each reading of parity bit j trades (2o - 1) logit(abar) on the indicator
    of b_j (abar = E[acc | level]; accuracies uniform, so the level mid-point), summed, exact parity readout. Returns
    forecast log-odds, per-agent trade-vector symbols (for entropy) and the count of possible trade vectors."""
    lo, hi = spec.acc
    edges = np.linspace(lo, hi, acc_levels + 1)
    lev = np.clip(np.digitize(d["acc"], edges[1:-1]), 0, acc_levels - 1)
    mid = (edges[:-1] + edges[1:]) / 2
    amt = (2 * d["read"] - 1) * np.log(mid / (1 - mid))[lev]
    n, R = d["var"].shape
    k = spec.k
    eta = np.zeros((n, k))
    for j in range(k):
        eta[:, j] = (amt * (d["var"] == j)).sum(1)
    ind = BITS[:, :k].astype(float)
    f = market_logit_np(eta, ind, k)
    # per-agent trade vector (decoder needs only this): rounded vector -> symbol id
    N = R // spec.p
    vec = np.zeros((n, N, k))
    for j in range(k):
        vec[..., j] = (amt * (d["var"] == j)).reshape(n, N, spec.p).sum(-1)
    key = np.round(vec, 6).reshape(-1, k)
    _, sym = np.unique(key, axis=0, return_inverse=True)
    # count of possible trade vectors: per reading token in {none} u {(j, o, level)}
    toks = [None] + [(j, o, l) for j in range(k) for o in (0, 1) for l in range(acc_levels)]
    possible = set()
    a = np.log(mid / (1 - mid))
    for t1 in toks:
        for t2 in toks:
            v = np.zeros(k)
            for t in (t1, t2):
                if t is not None:
                    v[t[0]] += (2 * t[1] - 1) * a[t[2]]
            possible.add(tuple(np.round(v, 6)))
    return f, sym.reshape(n, N), len(possible)


# ----------------------------------------------------------------------------- exact R(phi), principal angles

def _orth(phi, prior):
    """Orthonormal basis (sqrt(pi)-weighted coordinates) of span of the pi-centred columns of phi."""
    B = (phi - prior @ phi) * np.sqrt(prior)[:, None]
    if B.shape[1] == 0:
        return B
    U, s, _ = np.linalg.svd(B, full_matrices=False)
    return U[:, s > 1e-8 * max(s.max(), 1e-300)]


def span_regret(cell, phi, n_mc=20000, seed=0):
    """Exact additive-pool regret R(phi) in NATS on a gsigma cell: every agent moves the market from pi to its own
    exact posterior's moments on span(1, phi) (gsigma.mm_trade: the LMSR trade, KL projection), the trades are
    summed, and R = E KL(p(Y | O) || q(Y | O)). phi (W, S) real-valued; only its span matters. Exact enumeration up
    to gsigma.EXACT_LIMIT observation tuples, else Monte Carlo over n_mc draws (common random numbers via seed)."""
    from .analysis import gsigma as gs
    Q = _orth(phi, cell.prior)
    basis = Q / np.sqrt(cell.prior)[:, None]                   # pi-orthonormal, pi-centred functions of w
    V = []
    for i in range(cell.N):
        post, po = gs.agent_posteriors(cell, i)
        th = np.zeros((len(po), basis.shape[1]))
        for o in np.flatnonzero(po > 0):
            th[o] = gs.mm_trade(cell.prior, basis, post[o] @ basis)
        V.append(th @ basis.T)
    ymat = gs._onehot(cell.target)
    logpi = gs._log(cell.prior)
    vals, ws = [], []
    cls = gs.identity_classes(cell)
    X = gs.sample_obs(cell, n_mc, np.random.default_rng(seed))[1] if gs._needs_mc(cell, cls) else None
    for codes, lj, w in gs.tuples(cell, cls, X):
        p = gs._py_given(lj, ymat)
        s = logpi[None] + sum(V[i][codes[:, i]] for i in range(cell.N))
        vals.append(gs._kl(p, gs._py_given(s, ymat)))
        ws.append(w)
    return float((np.concatenate(vals) * np.concatenate(ws)).sum()) * LN2


def angles(phi, ref, prior=None):
    """Principal angles (degrees, largest first) of span(ref) inside span(phi) (both pi-centred), and the share of
    ref's energy outside span(phi) (mean sin^2). All angles 0 iff span(ref) lies in span(phi)."""
    prior = np.full(W, 1.0 / W) if prior is None else prior
    Qp, Qr = _orth(phi, prior), _orth(ref, prior)
    if Qp.shape[1] == 0:
        cos = np.zeros(Qr.shape[1])
    else:
        cos = np.linalg.svd(Qr.T @ Qp, compute_uv=False)
        cos = np.r_[cos, np.zeros(max(0, Qr.shape[1] - len(cos)))][:Qr.shape[1]]
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return {"angles_deg": sorted(ang.tolist(), reverse=True), "energy_outside": float(np.mean(1 - np.clip(cos, 0, 1) ** 2))}


def securities_report(phi, n_struct=6, seed0=0, with_random=True, overrides=None):
    """R(phi) and principal angles averaged over n_struct gsigma Z9-p2 structures (zoo overrides of the spec)."""
    overrides = SPECS["lr"][1] if overrides is None else overrides
    from .analysis import gsigma as gs
    out = {"R_nats": [], "angles_loglik": [], "energy_outside_loglik": []}
    rand = np.random.default_rng(1234).standard_normal(phi.shape) if with_random else None
    if with_random:
        out["R_random_phi_nats"] = []
    for s in range(seed0, seed0 + n_struct):
        cell = gs.z9p2_cell(s, **overrides)
        out["R_nats"].append(span_regret(cell, phi))
        if with_random:
            out["R_random_phi_nats"].append(span_regret(cell, rand))
        sv = gs.span_svd(cell)
        V = sv["V"][:, :sv["d_omega"]] / np.sqrt(cell.prior)[:, None]
        a = angles(phi, V, cell.prior)
        out["angles_loglik"].append(a["angles_deg"])
        out["energy_outside_loglik"].append(a["energy_outside"])
    out["R_mean_nats"] = float(np.mean(out["R_nats"]))
    if with_random:
        out["R_random_phi_mean_nats"] = float(np.mean(out["R_random_phi_nats"]))
    out["energy_outside_loglik_mean"] = float(np.mean(out["energy_outside_loglik"]))
    par = angles(phi, 2.0 * BITS[:, :4] - 1)
    out["angles_parity_bits"] = par["angles_deg"]
    out["energy_outside_parity_bits"] = par["energy_outside"]
    out["angles_Y"] = angles(phi, y_security())["angles_deg"]
    return out


# ----------------------------------------------------------------------------- model

def build(S, id_bits, amount_bits, phi_mode, readout, k=4, width=128, F_obs=2, F_key=14, phi_init=None,
          trade_mode="forecast", power_norm=False):
    """trade_mode "forecast": the agent's trade IS its forecast of phi(omega) (whitened securities: the LMSR trade from
    the prior is about the forecast itself), on the security where the forecast departs most from the prior price
    (id = argmax_s |f_s|, no gradient), amount = FSQ(tanh(f_s / 2)) x a learned per-security scale. "free": id from
    a separate head (straight-through Gumbel) and amount from a separate head; the forecast head is then only an
    auxiliary output (settlement shapes the shared trunk)."""
    import torch
    from torch import nn
    L = 2 ** amount_bits

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.p_obs = nn.Sequential(nn.Linear(F_obs, width), nn.LayerNorm(width))
            self.p_key = nn.Sequential(nn.Linear(F_key, width), nn.LayerNorm(width))
            self.trunk = nn.Sequential(nn.GELU(), nn.Linear(width, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
            self.id_head = nn.Linear(width, S) if S > 1 else None
            self.amt_head = nn.Linear(width, S)
            self.fc_head = nn.Linear(width, S)                    # settlement forecast of phi(omega)
            self.log_scale = nn.Parameter(torch.full((S,), math.log(2.0) if trade_mode == "forecast" else 0.0))
            y = torch.tensor(target(k), dtype=torch.bool)
            self.register_buffer("y1", y)
            if phi_mode in ("learned", "learned_std"):
                self.phi_raw = nn.Parameter(torch.randn(W, S) if phi_init is None else torch.tensor(phi_init).float())
            else:
                self.register_buffer("phi_raw", torch.tensor(phi_init).float())
            self.mlp = (nn.Sequential(nn.Linear(S, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 1))
                        if readout == "mlp" else None)
            # (2026-10-22, opus-mats) fixed power: the forecast that sets the trade is standardised per security over
            # the batch (BatchNorm, no affine) before the id argmax and the amount FSQ, so it cannot shrink to ~0
            self.fc_norm = nn.BatchNorm1d(S, affine=False) if power_norm else None
            # masked-agent pretraining head: the held-out agent's readings from the OTHER agents' summed trades and
            # the held-out agent's own key (training-only query)
            self.pre_head = nn.Sequential(nn.Linear(S + F_key, 128), nn.GELU(), nn.Linear(128, F_obs))

        def phi(self):
            P = self.phi_raw - self.phi_raw.mean(0, keepdim=True)
            if phi_mode == "learned":                           # whitened: orthonormal under uniform pi
                Q, R = torch.linalg.qr(P / math.sqrt(W))
                return Q * torch.sign(torch.diagonal(R) + 1e-12)[None] * math.sqrt(W)
            return P / P.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)

        def encode(self, x, tau=1.0, hard=False):
            h = self.trunk(self.p_obs(x[..., :F_obs]) + self.p_key(x[..., F_obs:]))
            fc = self.fc_head(h)
            ft = fc
            if self.fc_norm is not None:
                sh = fc.shape
                ft = self.fc_norm(fc.reshape(-1, S)).reshape(sh)
            if trade_mode == "forecast":
                idx = ft.detach().abs().argmax(-1)
                one = torch.nn.functional.one_hot(idx, S).float()
                u = torch.tanh(ft / 2)
            elif S > 1:
                lg = self.id_head(h)
                if hard:
                    idx = lg.argmax(-1)
                    one = torch.nn.functional.one_hot(idx, S).float()
                else:
                    g = -torch.log(-torch.log(torch.rand_like(lg).clamp(1e-9, 1 - 1e-9)))
                    soft = torch.softmax((lg + g) / tau, -1)
                    idx = soft.argmax(-1)
                    one = torch.nn.functional.one_hot(idx, S).float() + soft - soft.detach()
            else:
                idx = torch.zeros(h.shape[:-1], dtype=torch.long, device=h.device)
                one = torch.ones(*h.shape[:-1], 1, device=h.device)
            if trade_mode != "forecast":
                u = torch.tanh(self.amt_head(h))
            if L > 1:
                li = torch.round((u + 1) / 2 * (L - 1))
                a = u + (li / (L - 1) * 2 - 1 - u).detach()
            else:
                li = torch.zeros_like(u)
                a = torch.ones_like(u)
            trade = one * a * torch.exp(self.log_scale)
            lev = li.gather(-1, idx[..., None])[..., 0].long()
            return trade, idx * L + lev, idx, lev, fc

        def read(self, eta):
            if self.mlp is not None:
                return self.mlp(eta)[..., 0]
            s = eta @ self.phi().T
            neg = torch.tensor(-1e30, device=s.device)
            return torch.logsumexp(torch.where(self.y1, s, neg), -1) - torch.logsumexp(torch.where(self.y1, neg, s), -1)

        def forward(self, x, tau=1.0, hard=False):
            trade, sym, idx, lev, fc = self.encode(x, tau, hard)
            return self.read(trade.sum(1)), sym, idx, lev, fc

    return Net()


# ----------------------------------------------------------------------------- arms

SETTLE = "latent supervision (settlement of phi(omega)) + Y"
ARMS = {
    "oraclephi_market": dict(phi="oracle", settle=0.0, readout="market", trade="forecast",
                             supervision="Y labels only; oracle securities given [structure-informed]"),
    "oraclephi_settle_market": dict(phi="oracle", settle=1.0, readout="market", trade="forecast",
                                    supervision="latent supervision (settlement of the bits) + Y; oracle securities"),
    "settle_market": dict(phi="learned", settle=1.0, readout="market", trade="forecast", supervision=SETTLE),
    "settle_mlp": dict(phi="learned", settle=1.0, readout="mlp", trade="forecast", supervision=SETTLE),
    "settledict_market": dict(phi="learned_std", settle=1.0, settle_on="traded", readout="market", trade="forecast",
                              supervision=SETTLE + "; standardised (not whitened) phi, each security settled by its traders"),
    "settledict_mlp": dict(phi="learned_std", settle=1.0, settle_on="traded", readout="mlp", trade="forecast",
                           supervision=SETTLE + "; standardised (not whitened) phi, each security settled by its traders"),
    "settlefree_market": dict(phi="learned", settle=1.0, readout="market", trade="free",
                              supervision=SETTLE + "; trades not tied to the forecast"),
    "yonly_market": dict(phi="learned", settle=0.0, readout="market", trade="forecast", supervision="Y labels only"),
    "yonly_mlp": dict(phi="learned", settle=0.0, readout="mlp", trade="forecast", supervision="Y labels only"),
    "ysec_market": dict(phi="Y", settle=1.0, readout="market", trade="forecast",
                        supervision="Y labels only (settlement of the Y security is Y)"),
    "ysec_mlp": dict(phi="Y", settle=1.0, readout="mlp", trade="forecast",
                     supervision="Y labels only (settlement of the Y security is Y)"),
}


def train_arm(arm, b, seed=0, spec_name="lr", steps=8000, batch=512, lr=1e-3, eval_every=200, n_val=2000, n_dev=4000,
              tau=(1.0, 0.5), device=None, n_boot=2000, n_struct=6, threads=None, phi_lr_mult=10.0, pretrain=0,
              power_norm=False):
    import torch
    from .compression import Channel
    from .metrics import summarize_values
    if threads:
        torch.set_num_threads(threads)
    cfg = ARMS[arm]
    spec, overrides = SPECS[spec_name]
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if cfg["phi"] == "Y":
        id_bits, amount_bits = 0, b
    else:
        id_bits, amount_bits = ID_AMOUNT_BITS[b]
    S = 2 ** id_bits
    phi_init = {"oracle": lambda: basket_securities(S, spec.k), "Y": lambda: y_security(spec.k),
                "learned": lambda: None, "learned_std": lambda: None}[cfg["phi"]]()
    phi_mode = {"learned": "learned", "learned_std": "learned_std"}.get(cfg["phi"], "fixed")
    net = build(S, id_bits, amount_bits, phi_mode, cfg["readout"], spec.k,
                phi_init=phi_init, trade_mode=cfg["trade"], power_norm=power_norm).to(device)
    val = sample_world(spec, n_val, np.random.default_rng(VAL_SEED))
    dev = sample_world(spec, n_dev, np.random.default_rng(DEV_SEED))
    rng = np.random.default_rng([seed, 77])
    tt = lambda a: torch.tensor(a, device=device)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    l0 = prior_logit(spec)
    phi_p = [p_ for n_, p_ in net.named_parameters() if n_ == "phi_raw"]          # a (64, S) table: its own lr
    rest = [p_ for n_, p_ in net.named_parameters() if n_ != "phi_raw"]
    groups = [{"params": rest, "lr": lr}] + ([{"params": phi_p, "lr": lr * phi_lr_mult}] if phi_p else [])
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=steps)
    rec = {"arm": arm, "b": b, "seed": seed, "id_bits": id_bits, "amount_bits": amount_bits, "S": S,
           "phi": cfg["phi"], "readout": cfg["readout"], "trade": cfg["trade"], "settle": cfg["settle"], "supervision": cfg["supervision"],
           "spec_name": spec_name, "spec": spec.__dict__, "steps": steps, "batch": batch, "lr": lr,
           "phi_lr": lr * phi_lr_mult if phi_mode != "fixed" else None, "settle_on": cfg.get("settle_on", "all"), "curve": [],
           "pretrain": pretrain, "power_norm": power_norm,
           "pretrain_query": "held-out agent's own key (training only)" if pretrain else None}

    if pretrain:            # masked-agent prediction through the trades, no Y (opus-mats, 2026-10-22)
        pre_params = [p_ for n_, p_ in net.named_parameters() if n_ not in ("phi_raw",)]
        popt = torch.optim.AdamW(pre_params, lr=lr, weight_decay=0.0)
        psched = torch.optim.lr_scheduler.OneCycleLR(popt, max_lr=lr, total_steps=pretrain)
        prng = np.random.default_rng([seed, 78])
        for t in range(pretrain):
            d = sample_world(spec, batch, prng)
            x = tt(d["tokens"])
            trade = net.encode(x, tau[0])[0]
            n_, N_ = x.shape[:2]
            ar = torch.arange(n_, device=x.device)
            idx = torch.randint(0, N_, (n_,), device=x.device)
            eta_o = trade.sum(1) - trade[ar, idx]
            pl = net.pre_head(torch.cat([eta_o, x[ar, idx, 2:]], -1))
            ploss = bce(pl, x[ar, idx, :2])
            popt.zero_grad()
            ploss.backward()
            torch.nn.utils.clip_grad_norm_(pre_params, 1.0)
            popt.step()
            psched.step()
            if t % eval_every == 0 or t == pretrain - 1:
                rec["curve"].append({"phase": "pretrain", "step": t, "pre_loss": float(ploss.detach())})

    def predict(d):
        net.eval()
        out, syms = [], []
        with torch.no_grad():
            for s in range(0, len(d["y"]), 2048):
                lg, sym, idx, lev, _ = net(tt(d["tokens"][s:s + 2048]), hard=True)
                out.append(lg.cpu().numpy())
                syms.append(np.stack([sym.cpu().numpy(), idx.cpu().numpy(), lev.cpu().numpy()], -1))
        net.train()
        return l0 + np.concatenate(out).astype(float), np.concatenate(syms)

    best = (math.inf, None, -1)
    for t in range(steps):
        d = sample_world(spec, batch, rng)
        x, y = tt(d["tokens"]), tt(d["y"])
        tau_t = tau[0] + (tau[1] - tau[0]) * t / max(steps - 1, 1)
        lg, _, idx, _, fc = net(x, tau_t)
        loss_y = bce(lg + l0, y)
        loss = loss_y
        loss_s = None
        if cfg["settle"] > 0:
            target_phi = net.phi()[tt(d["w"])][:, None, :]              # (B, 1, S): phi(omega), observed at settlement
            if cfg.get("settle_on") == "traded":
                # forecasts are settled on every security (phi held fixed); each security's payoff function moves
                # only with the agents that traded it (a dictionary: securities specialise to traders' evidence)
                one = torch.nn.functional.one_hot(idx, S).float()
                loss_s = ((fc - target_phi.detach()) ** 2).mean() + \
                    (((fc.detach() - target_phi) ** 2) * one).sum(-1).mean()
            else:
                loss_s = ((fc - target_phi) ** 2).mean()
            loss = loss + cfg["settle"] * loss_s
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if t % eval_every == 0 or t == steps - 1:
            v = float(_kl(predict(val)[0], val["bayes"]).mean())
            tr = float(_kl(lg.detach().cpu().numpy().astype(float) + l0, d["bayes"]).mean())
            with torch.no_grad():
                out_par = angles(net.phi().cpu().numpy().astype(float), 2.0 * BITS[:, :spec.k] - 1)["energy_outside"]
            rec["curve"].append({"step": t, "val_kl": v, "train_kl_soft": tr, "parity_energy_outside": out_par,
                                 "settle_mse": None if loss_s is None else float(loss_s.detach())})
            if v < best[0]:
                best = (v, copy.deepcopy(net.state_dict()), t)
    net.load_state_dict(best[1])
    rec["best_val_kl"], rec["best_step"] = best[0], best[2]
    f, syms = predict(dev)
    kl = _kl(f, dev["bayes"])
    rec["regret"] = summarize_values(kl, n_boot)
    rec["prior_regret"] = float(_kl(np.full(len(kl), l0), dev["bayes"]).mean())
    Channel(b).send(syms[..., 0].astype(np.int64))                     # the message fits in b bits
    rec["symbol_entropy_bits"] = _entropy_bits(syms[..., 0].ravel())
    rec["id_entropy_bits"] = _entropy_bits(syms[..., 1].ravel())
    rec["amount_entropy_bits"] = _entropy_bits(syms[..., 2].ravel())
    rec["distinct_symbols"] = int(len(np.unique(syms[..., 0])))
    rec["distinct_ids"] = int(len(np.unique(syms[..., 1])))
    with torch.no_grad():
        phi = net.phi().cpu().numpy().astype(float)
    rec["phi"] = phi.round(5).tolist()
    rec["securities"] = securities_report(phi, n_struct, overrides=overrides)
    print(f"[{arm}] b={b} seed={seed}: dev KL {rec['regret']['mean']:.4f} (best val {best[0]:.4f} @ {best[2]}) "
          f"H={rec['symbol_entropy_bits']:.2f}b R={rec['securities']['R_mean_nats']:.4f} "
          f"out(par)={rec['securities']['energy_outside_parity_bits']:.2f}", file=sys.stderr, flush=True)
    return rec


def closed_forms(spec_name="lr", n_dev=4000, n_struct=6, n_boot=2000):
    """oracle_closed at b = 6 (mean-accuracy LLR) and b = 8 (2 accuracy levels); R and angles of the bit, Y and
    basket securities (exact, no learning)."""
    from .metrics import summarize_values
    spec, overrides = SPECS[spec_name]
    dev = sample_world(spec, n_dev, np.random.default_rng(DEV_SEED))
    rows = []
    for b, levels in ((6, 1), (8, 2)):
        f, sym, possible = oracle_closed(dev, spec, levels)
        rows.append({"arm": "oracle_closed", "b": b, "acc_levels": levels, "possible_trade_vectors": possible,
                     "fixed_length_bits": int(math.ceil(math.log2(possible))),
                     "regret": summarize_values(_kl(f, dev["bayes"]), n_boot),
                     "symbol_entropy_bits": _entropy_bits(sym.ravel()),
                     "supervision": "closed form, oracle bit securities"})
    sec = {"bits b1..b4": 2.0 * BITS[:, :4] - 1, "all six bits": 2.0 * BITS - 1, "Y": y_security()}
    for S in (2, 4, 16, 32):
        sec[f"baskets S={S}"] = basket_securities(S)
    secrows = {name: securities_report(phi, n_struct, with_random=False, overrides=overrides) for name, phi in sec.items()}
    return {"spec_name": spec_name, "closed": rows, "securities": secrows,
            "prior_regret": float(_kl(np.full(n_dev, prior_logit(spec)), dev["bayes"]).mean())}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--bits", nargs="+", type=int, default=[2, 4, 6, 8])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--n-struct", type=int, default=6)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--spec", default="lr", choices=list(SPECS))
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--closed", action="store_true", help="also compute the closed forms")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="securities_msg")
    ap.add_argument("--pretrain", type=int, default=0, help="masked-agent pretraining steps (no Y)")
    ap.add_argument("--power-norm", action="store_true", help="standardise the trade-setting forecast (fixed power)")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    out = {"schema": "aggbench.securities-msg/v1", "results": []}
    try:
        from .results import harness_info
        out["harness"] = harness_info()
    except Exception:   # noqa: BLE001
        pass
    path = os.path.join(a.out_dir, f"{a.tag}.json")
    if a.closed:
        out["closed_forms"] = closed_forms(a.spec, n_struct=a.n_struct)
    for arm in ([] if a.no_train else a.arms):
        for b in a.bits:
            for seed in a.seeds:
                out["results"].append(train_arm(arm, b, seed, a.spec, steps=a.steps, n_struct=a.n_struct, threads=a.threads,
                                              device=a.device, pretrain=a.pretrain, power_norm=a.power_norm))
                with open(path, "w") as f:
                    json.dump(out, f)
    with open(path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
