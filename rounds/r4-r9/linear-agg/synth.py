"""Experiment A generator (linear-agg, 2026-10-16): synthetic correlated-subspace rollouts + active VoI probes.

gemini (Argon) spec, outbox/gemini-linear-transformer-candidates-2026-10-15.md Sec. 3, Experiment A:
  hidden w in R^16: 12 core chain variables (prior N(0, 1)) + 4 high-variance "noisy TV" variables (prior N(0, 3^2)).
  K = 20 rollouts r, each observing v_r = <k_r, w> + noise, noise variance sigma_tau^2 * m_r * (9 if bad branch else 1):
    12 hub rollouts: 4 near-duplicate keys around each of 3 hub directions (blocks 0-3, 4-7, 8-11), so the Gram is
       ill-conditioned (median kappa of the core block ~ 50, checked by `python3 synth.py check`);
    6 bridge links (e_i - e_j)/sqrt 2 along the chain; 2 noisy-TV rollouts on the distractor variables.
  depth m_r ~ U{1..6} (known), episode noise sigma_tau ~ U(0.1, 3.0) (unknown), each rollout a bad branch w.p. 0.2 (unknown).
  query k_q = (e_0 + e_11)/sqrt 2 (the two chain ends); target y_q = <k_q, w> (noiseless).
  8 probe candidates: 2 missing bridge links (e_0 - e_1, e_10 - e_11), 3 fresh hub copies (redundant), 3 noisy-TV keys.
  A probe is observed like a depth-1, clean rollout.
Exact oracle (scoring / reference only): Gaussian posterior with the true prior, the true per-rollout noise variances
(sigma_tau, m_r and the bad-branch flags), and the exact expected variance reduction of y_q from each probe.
"""
from __future__ import annotations

import sys

import numpy as np

D, K, NCORE = 16, 20, 12
PRIOR_VAR = np.array([1.0] * 12 + [9.0] * 4)
SIG_LO, SIG_HI = 0.1, 3.0
P_BAD, BAD_VAR = 0.2, 9.0
HUB_EPS = 0.35
HUBW = {0: [1.0, 0.5, 0.3, 0.2], 1: [0.4, 1.0, 1.0, 0.4], 2: [0.2, 0.3, 0.5, 1.0]}      # weights over each block's 4 variables
LINKS = [(1, 2), (2, 3), (3, 4), (7, 8), (8, 9), (9, 10)]
TV_ROLL = [(12, 13), (14, 15)]
BRIDGE_CAND = [(0, 1), (10, 11)]
TV_CAND = [[12], [14], [13, 15]]
CAND_TYPE = np.array([0, 0, 1, 1, 1, 2, 2, 2])          # 0 = missing bridge link, 1 = redundant hub copy, 2 = noisy TV
ROLL_TYPE = np.array([1] * 12 + [0] * 6 + [2] * 2)       # 1 = hub, 0 = link, 2 = noisy TV
KQ = np.zeros(D)
KQ[0] = KQ[11] = 1 / np.sqrt(2)


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def hub_keys(n, rng, copies):
    """[n, 3*copies, D] unit keys: `copies` near-duplicates of each hub direction (perturbed inside the hub's block)."""
    out = np.zeros((n, 3 * copies, D))
    for h in range(3):
        blk = np.arange(4 * h, 4 * h + 4)
        a = np.zeros(D)
        a[blk] = HUBW[h]
        a = _unit(a)
        for c in range(copies):
            xi = np.zeros((n, D))
            xi[:, blk] = rng.standard_normal((n, 4)) * HUB_EPS
            out[:, h * copies + c] = _unit(a[None] + xi)
    return out


def pair_key(i, j, sign=-1.0):
    k = np.zeros(D)
    k[i], k[j] = 1.0, sign
    return _unit(k)


def episodes(n, seed, sigma=None):
    """Draw n episodes. sigma: None -> U(0.1, 3.0) per episode, else a fixed value. Returns a dict of arrays."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n, D)) * np.sqrt(PRIOR_VAR)
    keys = np.concatenate([hub_keys(n, rng, 4),
                           np.broadcast_to(np.stack([pair_key(i, j) for i, j in LINKS]), (n, len(LINKS), D)),
                           np.broadcast_to(np.stack([pair_key(i, j, 1.0) for i, j in TV_ROLL]), (n, len(TV_ROLL), D))], 1)
    m = rng.integers(1, 7, (n, K)).astype(float)
    bad = rng.random((n, K)) < P_BAD
    sig = rng.uniform(SIG_LO, SIG_HI, n) if sigma is None else np.full(n, float(sigma))
    nvar = sig[:, None] ** 2 * m * np.where(bad, BAD_VAR, 1.0)
    v = np.einsum("nkd,nd->nk", keys, w) + rng.standard_normal((n, K)) * np.sqrt(nvar)
    cand = np.concatenate([np.broadcast_to(np.stack([pair_key(i, j) for i, j in BRIDGE_CAND]), (n, 2, D)),
                           hub_keys(n, rng, 1),
                           np.broadcast_to(np.stack([_unit(np.bincount(c, minlength=D).astype(float)) if len(c) == 1
                                                     else pair_key(c[0], c[1]) for c in TV_CAND]), (n, 3, D))], 1)
    yq = w @ KQ
    return {"w": w, "keys": keys.copy(), "m": m, "bad": bad, "sigma": sig, "nvar": nvar, "v": v, "cand": cand.copy(),
            "yq": yq}


def oracle(ep):
    """Exact posterior mean of y_q, its posterior variance, the posterior covariance S, and per-candidate exact VoI
    (expected reduction of Var(y_q) from one clean depth-1 probe: (k_q' S k_z)^2 / (k_z' S k_z + sigma^2))."""
    keys, v, nvar = ep["keys"], ep["v"], ep["nvar"]
    prec = np.einsum("nkd,nk,nke->nde", keys, 1.0 / nvar, keys) + np.diag(1.0 / PRIOR_VAR)[None]
    S = np.linalg.inv(prec)
    b = np.einsum("nkd,nk->nd", keys, v / nvar)
    mu = np.einsum("nde,ne->nd", S, b)
    yhat = mu @ KQ
    var = np.einsum("d,nde,e->n", KQ, S, KQ)
    Skq = S @ KQ
    c = ep["cand"]
    num = np.einsum("nzd,nd->nz", c, Skq) ** 2
    den = np.einsum("nzd,nde,nze->nz", c, S, c) + ep["sigma"][:, None] ** 2
    return {"yhat": yhat, "var": var, "S": S, "voi": num / den, "pred_var_cand": den}


def pair_value(ep, S, picks):
    """Exact reduction of Var(y_q) from adding the two picked candidates (clean depth-1 probes) jointly. picks [n, 2]."""
    n = S.shape[0]
    C = ep["cand"][np.arange(n)[:, None], picks]                    # [n, 2, D]
    s2 = ep["sigma"] ** 2
    A = np.einsum("nzd,nde,nye->nzy", C, S, C) + s2[:, None, None] * np.eye(2)[None]
    g = np.einsum("nzd,nde,e->nz", C, S, KQ)
    return np.einsum("nz,nzy,ny->n", g, np.linalg.inv(A), g)


def adarr(ep):
    """Paper's AdaRR on the depth-whitened tokens: OLS (pseudo-inverse) residual variance / (n - rank) as sigma^2,
    then ridge (X'X + sigma^2 I)^-1 X'y. Returns yhat [n]."""
    X = ep["keys"] / np.sqrt(ep["m"])[..., None]
    y = ep["v"] / np.sqrt(ep["m"])
    out = np.zeros(len(y))
    for i in range(len(y)):
        wls, res, rank, sv = np.linalg.lstsq(X[i], y[i], rcond=None)
        rss = float(((y[i] - X[i] @ wls) ** 2).sum())
        s2 = rss / max(K - rank, 1)
        wr = np.linalg.solve(X[i].T @ X[i] + s2 * np.eye(D), X[i].T @ y[i])
        out[i] = wr @ KQ
    return out


def kappa_core(ep, whiten=True):
    X = ep["keys"][:, :, :NCORE]
    if whiten:
        X = X / np.sqrt(ep["m"])[..., None]
    ev = np.linalg.eigvalsh(np.einsum("nkd,nke->nde", X, X))
    return ev[:, -1] / ev[:, 0]


if __name__ == "__main__":
    if (sys.argv[1] if len(sys.argv) > 1 else "check") == "check":
        ep = episodes(20000, 1)
        kc, kw = kappa_core(ep, False), kappa_core(ep, True)
        print(f"kappa(core Gram), unit keys: median {np.median(kc):.1f} [p10 {np.percentile(kc, 10):.1f}, p90 {np.percentile(kc, 90):.1f}];"
              f" depth-whitened: median {np.median(kw):.1f}")
        for s in (0.1, 0.5, 1.0, 2.0, 3.0):
            e = episodes(5000, 2, sigma=s)
            o = oracle(e)
            top = np.argsort(-o["voi"], 1)[:, :2]
            both = (np.sort(top, 1) == np.array([0, 1])).all(1).mean()
            cos = np.abs(np.einsum("nzd,d->nz", e["cand"], KQ))
            topc = np.argsort(-cos, 1)[:, :2]
            bothc = (np.sort(topc, 1) == np.array([0, 1])).all(1).mean()
            topu = np.argsort(-o["pred_var_cand"], 1)[:, :2]
            tvu = (CAND_TYPE[topu] == 2).mean()
            best = np.max(np.stack([pair_value(e, o["S"], np.tile(np.array([a, b]), (len(e["yq"]), 1)))
                                    for a in range(8) for b in range(a + 1, 8)], 1), 1)
            vb = pair_value(e, o["S"], np.tile(np.array([0, 1]), (len(e["yq"]), 1)))
            mse_o = np.mean((o["yhat"] - e["yq"]) ** 2)
            mse_a = np.mean((adarr(e) - e["yq"]) ** 2)
            print(f"sigma {s}: oracle MSE {mse_o:.4f} (mean post var {o['var'].mean():.4f}), AdaRR MSE {mse_a:.4f}, prior var 1.0;"
                  f" oracle-VoI top2 = both bridges {both:.3f}; raw-cosine top2 = both bridges {bothc:.3f};"
                  f" uncertainty picks TV {tvu:.3f}; bridge pair value / best pair {np.mean(vb / best):.3f};"
                  f" mean VoI by type {[round(float(o['voi'][:, CAND_TYPE == t].mean()), 4) for t in range(3)]}")


# ------------------------------------------------------------------------------------------------ torch port (training batches)
def episodes_torch(n, gen, dev, dtype):
    """Same distribution as `episodes` (sigma ~ U(0.1, 3)), drawn with a torch generator on `dev`; returns the training
    fields only: keys [n,20,16], m [n,20], v [n,20], cand [n,8,16], yq [n]."""
    import torch
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=dev)
    w = torch.randn(n, D, generator=gen, device=dev, dtype=dtype) * t(np.sqrt(PRIOR_VAR))
    hubs = []
    for copies in (4, 1):
        out = torch.zeros(n, 3 * copies, D, device=dev, dtype=dtype)
        for h in range(3):
            a = np.zeros(D)
            a[4 * h:4 * h + 4] = HUBW[h]
            a = t(a / np.linalg.norm(a))
            for c in range(copies):
                xi = torch.zeros(n, D, device=dev, dtype=dtype)
                xi[:, 4 * h:4 * h + 4] = torch.randn(n, 4, generator=gen, device=dev, dtype=dtype) * HUB_EPS
                k = a[None] + xi
                out[:, h * copies + c] = k / k.norm(dim=-1, keepdim=True)
        hubs.append(out)
    fixed = t(np.stack([pair_key(i, j) for i, j in LINKS] + [pair_key(i, j, 1.0) for i, j in TV_ROLL]))
    keys = torch.cat([hubs[0], fixed[None].expand(n, -1, -1)], 1)
    m = torch.randint(1, 7, (n, K), generator=gen, device=dev).to(dtype)
    bad = torch.rand(n, K, generator=gen, device=dev, dtype=dtype) < P_BAD
    sig = SIG_LO + (SIG_HI - SIG_LO) * torch.rand(n, generator=gen, device=dev, dtype=dtype)
    nvar = sig[:, None] ** 2 * m * torch.where(bad, torch.full_like(m, BAD_VAR), torch.ones_like(m))
    v = torch.einsum("nkd,nd->nk", keys, w) + torch.randn(n, K, generator=gen, device=dev, dtype=dtype) * nvar.sqrt()
    bc = t(np.stack([pair_key(i, j) for i, j in BRIDGE_CAND]))
    tvc = t(np.stack([_unit(np.bincount(c, minlength=D).astype(float)) if len(c) == 1 else pair_key(c[0], c[1])
                      for c in TV_CAND]))
    cand = torch.cat([bc[None].expand(n, -1, -1), hubs[1], tvc[None].expand(n, -1, -1)], 1)
    return {"keys": keys, "m": m, "v": v, "cand": cand, "yq": w @ t(KQ)}
