"""On-device (torch) sampler and input builder: the same worlds as sfworld.sample, for fast training on a GPU.
Tested against the numpy sampler by test_sfgpu.py (same Bayes NLL distribution)."""
import math
import torch
import sfworld as W

VLEV_T = torch.tensor(W.VLEV, dtype=torch.float32)
DS = 16


def pairs_partition_t(n, N, a1, a2, gen, device):
    """Same sequential algorithm as sfworld.sample_pairs_partition, written without host syncs (masks instead of
    boolean indexing; always N iterations): the lowest free source is single with prob a1 f(r-1) / f(r), else it
    pairs with a uniformly chosen other free source."""
    f = torch.tensor(W.pair_counts(N, a1, a2), dtype=torch.float64, device=device)
    free = torch.ones(n, N, dtype=torch.bool, device=device)
    lab = torch.full((n, N), -1, dtype=torch.long, device=device)
    cols = torch.arange(N, device=device)[None]
    for _ in range(N):
        has = free.any(1)
        i = torch.argmax(free.int(), 1)
        r = free.sum(1)
        ps = a1 * f[(r - 1).clamp_min(0)] / f[r.clamp_min(1)]
        single = (torch.rand(n, generator=gen, device=device, dtype=torch.float64) < ps) | (r <= 1)
        oh_i = (cols == i[:, None]) & has[:, None]
        lab = torch.where(oh_i, i[:, None], lab)
        free = free & ~oh_i
        sc = torch.rand(n, N, generator=gen, device=device).masked_fill(~free, -1.0)
        t = torch.argmax(sc, 1)
        oh_t = (cols == t[:, None]) & (has & ~single)[:, None] & free
        lab = torch.where(oh_t, i[:, None], lab)
        free = free & ~oh_t
    return lab


def sample_t(n, N, gen, device, mode='indep', rho=0.0, gsize=1, a1=1.0, a2=1.0):
    vidx = torch.randint(0, 3, (n, N), generator=gen, device=device)
    v = VLEV_T.to(device)[vidx]
    y = torch.randn(n, W.Q, generator=gen, device=device)
    eps = torch.randn(n, N, W.Q, generator=gen, device=device)
    if mode == 'indep':
        blk = torch.arange(N, device=device)[None].expand(n, N); r = 0.0
    elif mode == 'groups':
        blk = (torch.arange(N, device=device) // gsize)[None].expand(n, N); r = rho
    elif mode == 'pairs':
        blk = pairs_partition_t(n, N, a1, a2, gen, device); r = rho
    else:
        raise ValueError(mode)
    u = torch.randn(n, N, W.Q, generator=gen, device=device)
    shared = torch.gather(u, 1, blk[:, :, None].expand(n, N, W.Q))
    e = v.sqrt()[:, :, None] * (math.sqrt(r) * shared + math.sqrt(1 - r) * eps)
    o = y[:, None, :] + e
    grp = blk if mode == 'groups' else torch.zeros_like(blk)
    return dict(y=y, o=o, v=v, blk=blk, grp=grp, N=N, mode=mode, rho=rho, gsize=gsize)


def src_features_t(d):
    n, N = d['o'].shape[:2]
    yl = d['y'][:, None, :W.QL].expand(n, N, W.QL)
    return torch.cat([d['o'][:, :, :W.QL], yl, d['o'][:, :, W.QL:W.QL + 1]], 2)


def build_inputs_t(d, arch, gen, device, want_group=False):
    n, N = d['o'].shape[:2]
    xs = src_features_t(d)
    ge = None
    if want_group and d['mode'] == 'groups':
        G = N // d['gsize']
        gemb = torch.randn(n, G, DS, generator=gen, device=device) / 4
        ge = torch.gather(gemb, 1, d['grp'][:, :, None].expand(n, N, DS))
    if arch in ('tree', 'hier', 'tree2'):
        return dict(xs=xs)
    if arch == 'set':
        return dict(xs=torch.cat([xs, ge], 2) if ge is not None else xs)
    if arch == 'tok':
        Q, QL = W.Q, W.QL
        semb = torch.randn(n, N, DS, generator=gen, device=device) / 4
        qemb = torch.randn(n, Q, DS, generator=gen, device=device) / 4
        T = N * Q + 1
        F = 2 * DS + 4 + (DS if ge is not None else 0)
        Xs = torch.zeros(n, N, Q, F, device=device)
        Xs[:, :, :, :DS] = semb[:, :, None, :]
        Xs[:, :, :, DS:2 * DS] = qemb[:, None, :, :]
        Xs[:, :, :, 2 * DS] = d['o']
        Xs[:, :, :QL, 2 * DS + 1] = d['y'][:, None, :QL]
        Xs[:, :, QL, 2 * DS + 2] = 1.0
        if ge is not None:
            Xs[:, :, :, 2 * DS + 4:] = ge[:, :, None, :]
        qtok = torch.zeros(n, 1, F, device=device)
        qtok[:, 0, DS:2 * DS] = qemb[:, QL]
        qtok[:, 0, 2 * DS + 3] = 1.0
        X = torch.cat([Xs.reshape(n, N * Q, F), qtok], 1)
        src = torch.cat([torch.arange(N, device=device).repeat_interleave(Q), torch.tensor([-1], device=device)])
        qq = torch.cat([torch.arange(Q, device=device).repeat(N), torch.tensor([-1], device=device)])
        gs = d['gsize'] if d['mode'] == 'groups' else 1
        grp = torch.cat([(torch.arange(N, device=device) // gs).repeat_interleave(Q), torch.tensor([-1], device=device)])
        return dict(tok=X, src=src[None].expand(n, T), qq=qq[None].expand(n, T), grp=grp[None].expand(n, T))
    raise ValueError(arch)
