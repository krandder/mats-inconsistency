"""Batched generalized splitting for the GPU: B independent particle systems at once (same net; one score and one set
of levels per system). Same algorithm as gs.py: pCN moves Z' = sqrt(1 - b^2) Z + b G accepted iff the score stays
above the current level. pilot(): adaptive (1 - rho)-quantile levels that never step over a system's stops and pCN
scales adapted on the system's own particles (biased; used only to set thresholds, levels and frozen scales).
frozen(): fixed levels and frozen scales from an independent pilot (unbiased), optionally with a readout of other events.

Latent of system b, particle j: Z[b, j] of shape (L, n) = (x, eps_1..eps_{L-1}). Default score: z_L,idx[b].
"""
import math
import torch


def neuron_scores(net, idx):
    W = None

    def f(Z):
        nonlocal W
        if W is None:
            W = net.W[-1][idx]                                    # (B, n)
        B, N = Z.shape[:2]
        h = net.hidden(Z.reshape(B * N, *Z.shape[2:])).reshape(B, N, -1)
        return (h * W[:, None, :]).sum(-1)
    return f


def draw(net, B, N, gen):
    return torch.randn(B, N, getattr(net, 'latent_L', None) or net.L, net.n, generator=gen, device=net.device,
                       dtype=net.dtype)


def _select(Z, Y, above, gen):
    w = above.float()
    w[w.sum(1) == 0] = 1.0                                         # dead systems: keep something, ignored later
    pick = torch.multinomial(w, above.shape[1], replacement=True, generator=gen)
    Z = torch.gather(Z, 1, pick[:, :, None, None].expand(-1, -1, *Z.shape[2:])).contiguous()
    return Z, torch.gather(Y, 1, pick)


def _moves(net, sf, Z, Y, g, gen, n_moves, bs_k=None, b=None, acc_target=0.3, record=None):
    """n_moves pCN moves per system at level g (B,). bs_k (B, n_moves) frozen scales, or b (B,) adapted in place."""
    B, N = Y.shape
    for m in range(n_moves):
        bk = bs_k[:, m] if bs_k is not None else b
        if record is not None:
            record.append(bk.clone())
        c1 = torch.sqrt(1 - bk * bk).to(Z.dtype)[:, None, None, None]
        ZP = draw(net, B, N, gen).mul_(bk.to(Z.dtype)[:, None, None, None]).addcmul_(Z, c1)
        YP = sf(ZP)
        acc = YP.double() > g[:, None]
        Z.copy_(torch.where(acc[:, :, None, None], ZP, Z)); Y = torch.where(acc, YP, Y)
        del ZP
        if bs_k is None:
            a = acc.double().mean(1)
            b = torch.clamp(b * torch.exp(a - acc_target), 0.02, 0.95)
    return Z, Y, b


def pilot(net, sf, B, N, gen, stops, rho=0.1, n_moves=5, max_levels=60, b0=0.5, acc_target=0.3):
    """Adaptive pilot for B systems. stops[b]: increasing thresholds (last = deepest target of system b).
    Returns dict(levels: list of per-system level lists (each ends at its last stop, or at max_levels / death),
    bs: list of per-system lists of per-level move scales, logfrac: per-system lists, n_eval)."""
    dev = net.device
    st = [sorted(float(v) for v in s) for s in stops]
    Z = draw(net, B, N, gen); Y = sf(Z); nev = B * N
    b = torch.full((B,), b0, dtype=torch.float64, device=dev)
    levels = [[] for _ in range(B)]; bsl = [[] for _ in range(B)]; lfr = [[] for _ in range(B)]
    active = [True] * B
    for k in range(max_levels):
        q = torch.quantile(Y.double(), 1 - rho, dim=1).cpu().tolist()
        g = []
        for bb in range(B):
            if not active[bb]:
                g.append(levels[bb][-1] if levels[bb] else -float('inf')); continue
            prev = levels[bb][-1] if levels[bb] else -float('inf')
            nxt = [s for s in st[bb] if s > prev + 1e-12]
            gb = q[bb]
            if nxt and gb >= nxt[0]:
                gb = nxt[0]
            if k == max_levels - 1 and nxt:
                gb = nxt[-1]
            g.append(gb)
        gt = torch.tensor(g, dtype=torch.float64, device=dev)
        above = Y.double() > gt[:, None]
        frac = above.double().mean(1).cpu().tolist()
        for bb in range(B):
            if not active[bb]:
                continue
            levels[bb].append(g[bb]); lfr[bb].append(math.log(frac[bb]) if frac[bb] > 0 else -float('inf'))
            if frac[bb] == 0 or abs(g[bb] - st[bb][-1]) < 1e-12:
                active[bb] = False
        if not any(active):
            break
        Z, Y = _select(Z, Y, above, gen)
        rec = []
        Z, Y, b = _moves(net, sf, Z, Y, gt, gen, n_moves, b=b, acc_target=acc_target, record=rec)
        nev += B * N * n_moves
        for bb in range(B):
            if active[bb]:
                bsl[bb].append([float(r[bb]) for r in rec])
    return dict(levels=levels, bs=bsl, logfrac=lfr, n_eval=nev)


def pad(levels, bs, n_moves):
    """Per-system lists -> (B, K) levels and (B, K - 1, n_moves) scales, padding by repeating the last level (a repeated
    level has surviving fraction 1, so it changes no estimate) and the last scale row."""
    K = max(len(l) for l in levels)
    Lv = torch.zeros(len(levels), K, dtype=torch.float64)
    Bs = torch.full((len(levels), max(K - 1, 1), n_moves), 0.3, dtype=torch.float64)
    for i, (l, s) in enumerate(zip(levels, bs)):
        Lv[i, :len(l)] = torch.tensor(l, dtype=torch.float64); Lv[i, len(l):] = l[-1]
        rows = list(s) + [s[-1] if s else [0.3] * n_moves] * (K - 1 - len(s))
        if K > 1:
            Bs[i] = torch.tensor(rows[:K - 1], dtype=torch.float64)
    return Lv, Bs


def frozen(net, sf, N, gen, levels, bs, n_moves=5, readout=None):
    """Fixed-level frozen-scale splitting for B = levels.shape[0] systems.
    Returns dict(logp: (B, K) cumulative log estimates of P(score > levels[b, k]) (-inf once a system dies),
    n_eval, est: (B, E) readout estimates if readout). readout(Z, Y, g) -> (B, E): mean over a system's particles
    of h_e(Z) 1{Y <= g}; the estimator sum_k P(S > g_(k-1)) * mean_(particles at level k-1)[h 1{S <= g_k}] + last term
    is unbiased for E[h(Z)] (Feynman-Kac; g_(-1) = -inf)."""
    dev = net.device
    B, K = levels.shape
    lev = levels.to(dev); bsd = bs.to(dev)
    Z = draw(net, B, N, gen); Y = sf(Z); nev = B * N
    cum = torch.zeros(B, dtype=torch.float64, device=dev)
    logp = torch.full((B, K), -float('inf'), dtype=torch.float64, device=dev)
    est = None
    for k in range(K):
        g = lev[:, k]
        if readout is not None:
            band = readout(Z, Y, g).double() * torch.exp(cum)[:, None]
            est = band if est is None else est + band
        above = Y.double() > g[:, None]
        frac = above.double().mean(1)
        cum = cum + torch.log(frac)                                # -inf once a system dies; stays -inf
        logp[:, k] = cum
        if k == K - 1:
            break
        Z, Y = _select(Z, Y, above, gen)
        Z, Y, _ = _moves(net, sf, Z, Y, g, gen, n_moves, bs_k=bsd[:, k])
        nev += B * N * n_moves
    if readout is not None:
        Z, Y = _select(Z, Y, Y.double() > lev[:, -1][:, None], gen)
        Z, Y, _ = _moves(net, sf, Z, Y, lev[:, -1], gen, n_moves, bs_k=bsd[:, -1])
        nev += B * N * n_moves
        inf = torch.full((B,), float('inf'), dtype=torch.float64, device=dev)
        fin = readout(Z, Y, inf).double() * torch.exp(cum)[:, None]
        fin[~torch.isfinite(cum)] = 0
        est = est + fin
    out = dict(logp=logp.cpu(), n_eval=nev)
    if readout is not None:
        out['est'] = est.cpu()
    return out
