"""Generalized splitting (Botev-Kroese) over the Gaussian latent, adapted from work/cm-interp-R/rare_gs.py (copied, not
imported; the original is not edited).

P(score(Z) > t), Z ~ N(0, I) of any trailing shape. MCMC: pCN moves Z' = sqrt(1 - b^2) Z + b G, which leave N(0, I)
invariant, accepted iff the score stays above the current level. Unbiasedness needs fixed levels and kernels that do not
depend on the run's own particles, so there are two modes: a pilot (adaptive (1 - rho)-quantile levels and pCN scale
adapted towards acceptance 0.3 on its own particles; biased, used only for tuning and threshold setting) and a frozen
run (fixed levels, b[k][m] from an independent pilot). One frozen run to the deepest level estimates every level.
"""
import math
import numpy as np
import torch

DT = torch.float64


def _where(acc, ZP, Z):
    return torch.where(acc.view(-1, *([1] * (Z.dim() - 1))), ZP, Z)


def gs_run(score, draw, N, gen, t, levels=None, bs=None, rho=0.1, n_moves=5, max_levels=80, b0=0.5, stops=(),
           keep=False, acc_target=0.3):
    """Fixed-effort splitting for P(score(Z) > t).
    levels None: adaptive (1 - rho) quantiles, never stepping over a value in `stops`; else the given inner levels then t.
    bs None: pCN scale adapted towards acceptance acc_target (pilot); else bs[k][m] is the frozen scale of move m after
    level k. Returns dict(levels, logfrac, bs, acc, n_eval[, Z, Y])."""
    Z = draw(N); Y = score(Z); nev = N
    used, logfr, bsu, accs = [], [], [], []
    b = b0; k = 0
    stops = sorted(float(s) for s in stops if s < t)
    while True:
        if levels is None:
            g = float(torch.quantile(Y.double(), 1 - rho))
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
        pick = idx[torch.randint(len(idx), (N,), generator=gen, device=idx.device)]
        Z, Y = Z[pick].clone(), Y[pick].clone()
        row, arow = [], []
        for m in range(n_moves):
            bk = bs[k][m] if bs is not None else b
            row.append(bk)
            ZP = math.sqrt(1 - bk * bk) * Z + bk * draw(N); YP = score(ZP); nev += N
            acc = YP > g
            a = float(acc.to(DT).mean()); arow.append(a)
            Z = _where(acc, ZP, Z); Y = torch.where(acc, YP, Y)
            if bs is None:
                b = min(0.95, max(0.02, b * math.exp(a - acc_target)))
        bsu.append(row); accs.append(arow)
        k += 1
    out = dict(levels=used, logfrac=logfr, bs=bsu, acc=accs, n_eval=nev)
    if keep:
        out['Z'] = Z; out['Y'] = Y
    return out


def pad_bs(bs, n_levels, n_moves):
    bs = [list(r) for r in bs]
    while len(bs) < n_levels - 1:
        bs.append(list(bs[-1]) if bs else [0.5] * n_moves)
    return bs


def frozen_runs(score, draw, levels, bs, gen, N, R, n_moves):
    """R independent frozen runs over the fixed `levels` (last = deepest target). Returns (R, n_levels) log estimates
    of P(score > levels[j]) and the number of score evaluations."""
    runs = []; nev = 0
    for _ in range(R):
        o = gs_run(score, draw, N, gen, levels[-1], levels=levels[:-1], bs=bs, n_moves=n_moves)
        cum = np.cumsum(o['logfrac']); nev += o['n_eval']
        cum = np.concatenate([cum, np.full(len(levels) - len(cum), -np.inf)])
        runs.append(cum)
    return np.array(runs), nev


def summarize(v):
    """Mean of R unbiased estimates given as logs; relative SE; split-half difference (bits); largest single share."""
    v = np.asarray(v, dtype=float); R = len(v)
    if not np.isfinite(v).any():
        return dict(logp=-float('inf'), rel_se=float('inf'), half_bits=float('nan'), max_share=1.0, R=R)
    m = v[np.isfinite(v)].max(); w = np.exp(v - m)
    lp = float(np.log(w.mean()) + m)
    rel_se = float(w.std(ddof=1) / w.mean() / math.sqrt(R)) if R > 1 else float('inf')
    h = R // 2
    a, b = w[:h].mean(), w[h:].mean()
    half = float((np.log(a) - np.log(b)) / math.log(2)) if a > 0 and b > 0 else float('inf')
    return dict(logp=lp, rel_se=rel_se, half_bits=half, max_share=float(w.max() / w.sum()), R=R)


def ess_move(score, draw, Z, Y, g, gen, max_tries=60):
    """One elliptical-slice move for every particle, targeting N(0, I) restricted to {score > g} (Murray, Adams and
    MacKay 2010, with the indicator as likelihood). No tuning parameter. Returns (Z, Y, n_eval, n_capped); a particle
    that has not found an accepted point after max_tries shrinkage steps stays where it is (counted in n_capped)."""
    N = Z.shape[0]; dev = Z.device
    nu = draw(N)
    th = torch.rand(N, generator=gen, device=dev, dtype=torch.float64) * (2 * math.pi)
    lo = th - 2 * math.pi; hi = th.clone()
    act = torch.arange(N, device=dev)
    nev = 0
    for _ in range(max_tries):
        if len(act) == 0:
            break
        t = th[act]
        c = torch.cos(t).to(Z.dtype).view(-1, *([1] * (Z.dim() - 1))); s = torch.sin(t).to(Z.dtype).view(-1, *([1] * (Z.dim() - 1)))
        ZP = Z[act] * c + nu[act] * s
        YP = score(ZP); nev += len(act)
        ok = YP > g
        ia = act[ok]
        Z[ia] = ZP[ok]; Y[ia] = YP[ok]
        rej = act[~ok]
        tr = th[rej]
        neg = tr < 0
        lo[rej] = torch.where(neg, tr, lo[rej]); hi[rej] = torch.where(neg, hi[rej], tr)
        th[rej] = lo[rej] + torch.rand(len(rej), generator=gen, device=dev, dtype=torch.float64) * (hi[rej] - lo[rej])
        act = rej
    return Z, Y, nev, len(act)


def gs_run_ess(score, draw, N, gen, t, levels=None, rho=0.1, n_moves=3, max_levels=80, stops=(), keep=False,
               max_tries=60):
    """Fixed-effort splitting with elliptical-slice moves (no kernel tuning, so with fixed levels it is unbiased
    without a tuning pilot). levels None: adaptive (pilot / threshold setting only)."""
    Z = draw(N); Y = score(Z); nev = N
    used, logfr, capped = [], [], 0
    k = 0
    stops = sorted(float(s) for s in stops if s < t)
    while True:
        if levels is None:
            g = float(torch.quantile(Y.double(), 1 - rho))
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
        pick = idx[torch.randint(len(idx), (N,), generator=gen, device=idx.device)]
        Z, Y = Z[pick].clone(), Y[pick].clone()
        for m in range(n_moves):
            Z, Y, ne, nc = ess_move(score, draw, Z, Y, g, gen, max_tries)
            nev += ne; capped += nc
        k += 1
    out = dict(levels=used, logfrac=logfr, n_eval=nev, capped=capped)
    if keep:
        out['Z'] = Z; out['Y'] = Y
    return out


def gs_readout(score, draw, N, gen, levels, bs, n_moves, readout):
    """Frozen fixed-level splitting on `score` with a readout of other events (unbiased, Feynman-Kac): for any h,
    E[h(Z)] = sum_k P(S > g_k) E[h(Z) 1{S <= g_(k+1)} | S > g_k] (g_0 = -inf, g_(K+1) = +inf), each term estimated by the
    product of surviving fractions times the mean over the level-k particles (after their moves).
    readout(Z, Y, g_next) -> (E,) tensor: mean over particles of h_e(Z) 1{Y <= g_next}.
    Returns (estimates (E,) float64 tensor, cumulative log fractions, n_eval)."""
    Z = draw(N); Y = score(Z); nev = N
    logP = 0.0; est = None; cum = []
    K = len(levels)
    for k in range(K + 1):
        g_next = levels[k] if k < K else float('inf')
        band = readout(Z, Y, g_next).double()
        est = band * math.exp(logP) if est is None else est + band * math.exp(logP)
        if k == K:
            break
        above = Y > g_next
        frac = float(above.to(DT).mean())
        if frac == 0:
            break
        logP += math.log(frac); cum.append(logP)
        idx = torch.nonzero(above)[:, 0]
        pick = idx[torch.randint(len(idx), (N,), generator=gen, device=idx.device)]
        Z, Y = Z[pick].clone(), Y[pick].clone()
        for m in range(n_moves):
            bk = bs[k][m]
            ZP = math.sqrt(1 - bk * bk) * Z + bk * draw(N); YP = score(ZP); nev += N
            acc = YP > g_next
            Z = _where(acc, ZP, Z); Y = torch.where(acc, YP, Y)
    return est, cum, nev
