"""Generalized splitting (Botev-Kroese) for P(y > t), black-box: uses only forward simulation of the noisy network.

xi = (x, eps_1..eps_L) is standard Gaussian; y(xi) = w.h_L. MCMC: pCN moves xi' = sqrt(1-b^2) xi + b Z (leave N(0, I)
invariant), accepted iff y stays above the current level.

Unbiasedness needs fixed levels AND kernels that do not depend on the run's own particles. gs_run() therefore has two modes:
a pilot (adaptive levels and/or pCN scale b adapted towards acceptance 0.3) and a frozen run (fixed levels, b[k][m] taken
from an independent pilot for level k and move m). Since review fix 2 (2026-10-29) every reference value uses the frozen
mode; splitting() without bs keeps the old adaptive-b behaviour (used only by the estimator comparison and pilots).
"""
import math
import numpy as np
import torch

DT = torch.float64


def output(net, XI):
    h = XI[:, 0]
    for l in range(1, net.L + 1):
        h = net.f(l, h) + net.sigma * XI[:, l]
    return h @ net.w


def gs_run(score, draw, N, gen, t, levels=None, bs=None, rho=0.1, n_moves=5, max_levels=60, b0=0.5, stops=()):
    """Generic fixed-effort splitting for P(score(Z) > t), Z = draw(n) ~ N(0, I) (any trailing shape).
    levels None: adaptive (1 - rho) quantiles, never stepping over a value in `stops` (so every stop becomes a level).
    bs None: pCN scale adapted towards acceptance 0.3 (pilot mode); else bs[k][m] is the frozen scale of move m after level k.
    Returns dict(levels, logfrac (log of the surviving fraction at each level), bs (scales used), n_eval)."""
    Z = draw(N); Y = score(Z); nev = N
    used, logfr, bsu = [], [], []
    b = b0; k = 0
    stops = sorted(float(s) for s in stops if s < t)
    while True:
        if levels is None:
            g = float(torch.quantile(Y, 1 - rho))
            prev = used[-1] if used else -float('inf')
            nxt = [s for s in stops if s > prev + 1e-12]
            if nxt and g >= nxt[0]:
                g = nxt[0]
            if g >= t or k >= max_levels:
                g = t
        else:
            g = levels[k] if k < len(levels) else t
        above = Y > g
        frac = float(above.to(DT).mean())
        used.append(g)
        if frac == 0:
            logfr.append(-float('inf')); break
        logfr.append(math.log(frac))
        if g >= t:
            break
        idx = torch.nonzero(above)[:, 0]
        pick = idx[torch.randint(len(idx), (N,), generator=gen)]
        Z, Y = Z[pick].clone(), Y[pick].clone()
        row = []
        for m in range(n_moves):
            bk = bs[k][m] if bs is not None else b
            row.append(bk)
            ZP = math.sqrt(1 - bk * bk) * Z + bk * draw(N); YP = score(ZP); nev += N
            acc = YP > g
            Z = torch.where(acc.view(-1, *([1] * (Z.dim() - 1))), ZP, Z); Y = torch.where(acc, YP, Y)
            if bs is None:
                b = min(0.95, max(0.02, b * math.exp(float(acc.to(DT).mean()) - 0.3)))
        bsu.append(row)
        k += 1
    return dict(levels=used, logfrac=logfr, bs=bsu, n_eval=nev)


def frozen_estimates(score, draw, targets, gen, N, R, n_moves=10, levels=None, pilot_N=None, rho=0.1, max_levels=60):
    """Unbiased fixed-level, frozen-kernel splitting estimates of P(score > t) for every t in `targets`.
    Pilot (independent random numbers): adaptive levels with every target as a stop (unless `levels` is given) and
    adaptive pCN scales; then R runs with those levels and the pilot's scales frozen. One run to the largest target gives
    all targets (each target is a level; the product of the surviving fractions up to it is unbiased for its probability).
    Returns dict(levels, bs, per_target={t: dict(logp, rel_se, logs, half_bits, max_share)}, n_eval, pilot_n_eval)."""
    tmax = max(targets)
    pN = pilot_N or N
    if levels is None:
        pil = gs_run(score, draw, pN, gen, tmax, rho=rho, n_moves=n_moves, max_levels=max_levels, stops=targets)
        lev = pil['levels']
    else:
        lev = [x for x in levels if x < tmax - 1e-12] + [tmax]
        pil = gs_run(score, draw, pN, gen, tmax, levels=lev, n_moves=n_moves)
    bs = pil['bs']
    if len(pil['logfrac']) < len(lev) or not np.isfinite(pil['logfrac'][-1]):
        # pilot died before the last level: reuse its last scales for the remaining levels
        while len(bs) < len(lev) - 1:
            bs.append(list(bs[-1]) if bs else [0.5] * n_moves)
    lev_inner = lev[:-1]
    runs = []; nev = 0
    for _ in range(R):
        o = gs_run(score, draw, N, gen, tmax, levels=lev_inner, bs=bs, n_moves=n_moves)
        cum = np.cumsum(o['logfrac']); nev += o['n_eval']
        cum = np.concatenate([cum, np.full(len(lev) - len(cum), -np.inf)])
        runs.append(cum)
    runs = np.array(runs)                                       # (R, n_levels): log estimate of P(score > level j)
    out = {}
    for t in targets:
        j = int(np.argmin(np.abs(np.array(lev) - t)))
        assert abs(lev[j] - t) < 1e-9, (lev[j], t)
        out[t] = summarize(runs[:, j])
    return dict(levels=lev, bs=bs, per_target=out, n_eval=nev, pilot_n_eval=pil['n_eval'])


def summarize(v):
    """Mean of R unbiased estimates given as logs; relative SE; split-half difference (bits); largest single share."""
    v = np.asarray(v, dtype=float); R = len(v)
    if not np.isfinite(v).any():
        return dict(logp=-float('inf'), rel_se=float('inf'), logs=v.tolist(), half_bits=float('nan'), max_share=1.0, R=R)
    m = v[np.isfinite(v)].max(); w = np.exp(v - m)
    lp = float(np.log(w.mean()) + m)
    rel_se = float(w.std(ddof=1) / w.mean() / math.sqrt(R)) if R > 1 else float('inf')
    h = R // 2
    a, b = w[:h].mean(), w[h:].mean()
    half = float((np.log(a) - np.log(b)) / math.log(2)) if a > 0 and b > 0 else float('inf')
    return dict(logp=lp, rel_se=rel_se, logs=v.tolist(), half_bits=half, max_share=float(w.max() / w.sum()), R=R)


def net_score_draw(net, gen):
    d, L = net.d, net.L
    return (lambda Z: output(net, Z)), (lambda n: torch.randn(n, L + 1, d, generator=gen, dtype=DT))


def splitting(net, t, N, gen, levels=None, rho=0.1, n_moves=5, max_levels=60, bs=None):
    """Returns (log p_hat nats, levels used, FLOPs). Adaptive pCN scale unless bs is given (see gs_run)."""
    d, L = net.d, net.L
    per = L * (2 * d * d + 12 * d)
    draw = lambda n: torch.randn(n, L + 1, d, generator=gen, dtype=DT)
    o = gs_run(lambda Z: output(net, Z), draw, N, gen, t, levels=levels, bs=bs, rho=rho, n_moves=n_moves,
               max_levels=max_levels)
    lp = float(np.sum(o['logfrac']))
    return lp, o['levels'], o['n_eval'] * per


def level_curve(net, N, gen, rho=0.1, n_levels=20, n_moves=5):
    """Adaptive run: levels gamma_k with P(y > gamma_k) ~ rho^k (k = 1..n_levels)."""
    _, lev, fl = splitting(net, float('inf'), N, gen, rho=rho, n_moves=n_moves, max_levels=n_levels)
    return lev[:-1] if lev and lev[-1] == float('inf') else lev, fl
