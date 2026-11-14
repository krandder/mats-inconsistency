"""W2b fast path: the world2_market generator, tokens and exact Bayes, batched in torch (GPU or CPU).

gen(E, cfg)          -> batch dict of tensors (same generative model and policy as world2_market.sample_batch;
                        cfg adds curriculum caps and stepping-stone switches)
tokens(b, T, items)  -> X (E, T', FEAT), mask, query position(s); identical layout to world2_market.pack
bayes(b)             -> per-item posterior mixture moments (E, Km) and the query mixture (w, m, v)
to_episodes(b)       -> list of world2_market.Episode (for tests against the numpy oracle)

Episode layout in a batch (position t < N_e): events are indexed by the position of the token that created them
(root[t]); a non-relay token creates its own (visible) event; a relay token either re-quotes an earlier event or
creates a hidden event (measured outside the context) and quotes it.
"""
import math
import numpy as np
import torch
import world2_market as W

SPOT, DIFF, BASKET = W.SPOT, W.DIFF, W.BASKET
MAXI = W.MAXI

BASE = dict(nmin=8, nmax=128, Kcap=10 ** 9, Scap=10 ** 9, Umax=W.U_MAX, p_relay=W.TRAIN['p_relay'],
            alpha=W.TRAIN['alpha'], p_hidden=W.TRAIN['p_hidden'], p_noSPOT=W.TRAIN['p_noSPOT_query'],
            rK=W.TRAIN['rK'], rS=W.TRAIN['rS'], types=(1, 1, 1), tau=W.TAU,
            p_rich=0.0, rich_relay=(0.3, 0.6), rich_gamma=4.0, p_sparse=0.5, sparse_spot=0.15)
RICH = dict(p_rich=0.6)     # W2c shared-evidence-rich training mix (PREREG-W2, W2c amendment)

CURRICULUM = [  # (start fraction, overrides)
    (0.00, dict(nmin=8, nmax=24, Kcap=4, Scap=4, Umax=1, p_relay=(0.0, 0.0))),
    (0.10, dict(nmin=8, nmax=48, Kcap=8, Scap=6, Umax=4, p_relay=(0.0, 0.15))),
    (0.25, dict(nmin=8, nmax=96, Umax=8, p_relay=(0.0, 0.3))),
    (0.40, dict()),
]
STEPPING = dict(types=(1, 1, 0), Umax=0, p_relay=(0.0, 0.0))     # SPOT+DIFF, known noise, no relays


def stage_cfg(frac, base=None, curriculum=True):
    c = dict(BASE, **(base or {}))
    if not curriculum:
        return c
    st = [o for f, o in CURRICULUM if frac >= f][-1]
    c.update(st)
    if base:                                     # stepping-stone switches override the curriculum
        for k in ('types', 'Umax', 'p_relay', 'tau'):
            if k in base:
                c[k] = base[k]
    return c


def _u(shape, dev):
    return torch.rand(shape, device=dev, dtype=torch.float64)


def _dirichlet(alpha, valid):
    g = torch._standard_gamma(alpha[:, None].expand(valid.shape).contiguous()) * valid
    g = torch.where(valid, g.clamp_min(1e-30), torch.zeros_like(g))
    return g / g.sum(1, keepdim=True)


def gen(E, cfg, dev='cpu', N=None):
    """E episodes. N: fixed quote count (int) or None (log-uniform in [nmin, nmax] as harness sample_train)."""
    c = cfg
    f64 = torch.float64
    if N is None:
        Ne = torch.exp(torch.log(torch.tensor(float(c['nmin']))) + _u(E, dev) *
                       (math.log(c['nmax'] + 1) - math.log(c['nmin']))).long().clamp(c['nmin'], c['nmax'])
    else:
        Ne = torch.full((E,), int(N), device=dev, dtype=torch.long)
    rk = c['rK'][0] + _u(E, dev) * (c['rK'][1] - c['rK'][0])
    rs = c['rS'][0] + _u(E, dev) * (c['rS'][1] - c['rS'][0])
    Ke = torch.round(Ne / rk).long().clamp_min(2).clamp_max(max(2, c['Kcap']))
    Se = torch.round(Ne / rs).long().clamp_min(2).clamp_max(max(2, c['Scap']))
    Ue = (_u(E, dev) * (torch.clamp(Se, max=c['Umax']) + 1)).long()
    Nm, Km, Sm = int(Ne.max()), int(Ke.max()), int(Se.max())
    ar = lambda n: torch.arange(n, device=dev)
    vk = ar(Km)[None] < Ke[:, None]; vs = ar(Sm)[None] < Se[:, None]; vt = ar(Nm)[None] < Ne[:, None]
    la = lambda: torch.exp(math.log(c['alpha'][0]) + _u(E, dev) * (math.log(c['alpha'][1]) - math.log(c['alpha'][0])))
    ws = _dirichlet(la(), vs); wi = _dirichlet(la(), vk)
    tmask = torch.tensor(c['types'], device=dev, dtype=f64)
    wt = (0.1 + 0.7 * _dirichlet(torch.ones(E, device=dev, dtype=f64), torch.ones(E, 3, device=dev, dtype=torch.bool))) * tmask
    rich = _u(E, dev) < c.get('p_rich', 0.0)
    sparse = rich & (_u(E, dev) < c.get('p_sparse', 0.5))
    wt = torch.where(sparse[:, None], wt * torch.tensor([c.get('sparse_spot', 0.15), 1.0, 1.0], dtype=f64, device=dev), wt)
    wt = wt / wt.sum(1, keepdim=True)
    p_relay = c['p_relay'][0] + _u(E, dev) * (c['p_relay'][1] - c['p_relay'][0])
    rr = c.get('rich_relay', (0.3, 0.6))
    p_relay = torch.where(rich, rr[0] + _u(E, dev) * (rr[1] - rr[0]), p_relay)
    relay = (_u((E, Nm), dev) < p_relay[:, None]) & vt
    relay[:, 0] = False
    hidden_new = relay & (_u((E, Nm), dev) < c['p_hidden'])
    creates = (~relay | hidden_new) & vt
    cnt = torch.cumsum(creates.long(), 1) - creates.long()
    order = torch.sort((~creates).long(), dim=1, stable=True).indices
    gam = torch.where(rich, torch.full((E,), float(c.get('rich_gamma', 4.0)), dtype=f64, device=dev),
                      torch.ones(E, dtype=f64, device=dev))
    pick = (_u((E, Nm), dev) ** gam[:, None] * cnt.clamp_min(1)).long().clamp_max((cnt - 1).clamp_min(0))
    root = torch.where(creates, ar(Nm)[None].expand(E, Nm), torch.gather(order, 1, pick))
    osrc = torch.multinomial(ws.float(), Nm, replacement=True)
    typ = torch.multinomial(wt.float(), Nm, replacement=True)
    k4 = min(MAXI, Km)
    g = torch.log(torch.where(vk, wi, torch.full_like(wi, 1e-300)))[:, None, :] - torch.log(-torch.log(_u((E, Nm, Km), dev)))
    top = torch.topk(g, k4, dim=2).indices
    if k4 < MAXI:
        top = torch.cat([top, torch.zeros(E, Nm, MAXI - k4, dtype=torch.long, device=dev)], 2)
    mb = 2 + (_u((E, Nm), dev) * (torch.clamp(Ke, max=MAXI)[:, None] - 1)).long()
    m = torch.where(typ == SPOT, 1, torch.where(typ == DIFF, 2, mb))
    slot = ar(MAXI)[None, None]
    items = torch.where(slot < m[..., None], top, -1)
    coef = torch.where(slot < m[..., None], 1.0 / m[..., None].to(f64), torch.zeros((), dtype=f64, device=dev))
    dcoef = torch.tensor([1.0, -1.0, 0.0, 0.0], dtype=f64, device=dev)
    coef = torch.where((typ == DIFF)[..., None], dcoef.expand_as(coef), coef)
    cb = torch.where(typ == DIFF, 0.0, 1.0).to(f64)
    src = torch.multinomial(ws.float(), Nm, replacement=True)
    o_root = torch.gather(osrc, 1, root)
    for _ in range(3):
        bad = relay & (src == o_root)
        if not bad.any():
            break
        src = torch.where(bad, torch.multinomial(ws.float(), Nm, replacement=True), src)
    bad = relay & (src == o_root)
    alt = (o_root + 1 + (_u((E, Nm), dev) * (Se[:, None] - 1)).long()) % Se[:, None]
    src = torch.where(bad, alt, src)
    src = torch.where(relay, src, osrc)
    v = torch.randn(E, Km, device=dev, dtype=f64)
    b = c['tau'] * torch.randn(E, Sm, device=dev, dtype=f64)
    sig = torch.exp(math.log(W.SIG_KNOWN[0]) + _u((E, Sm), dev) * (math.log(W.SIG_KNOWN[1]) - math.log(W.SIG_KNOWN[0])))
    rkk = torch.argsort(torch.argsort(torch.where(vs, _u((E, Sm), dev), torch.full((E, Sm), 2.0, device=dev, dtype=f64)), 1), 1)
    unk = rkk < Ue[:, None]
    sig = torch.where(unk, torch.where(_u((E, Sm), dev) < 0.5, W.SIG_L, W.SIG_H), sig)
    zev = torch.randn(E, Nm, device=dev, dtype=f64); delta = W.RHO * torch.randn(E, Nm, device=dev, dtype=f64)
    # values
    it_r = torch.gather(items, 1, root[..., None].expand(E, Nm, MAXI))
    co_r = torch.gather(coef, 1, root[..., None].expand(E, Nm, MAXI))
    cb_r = torch.gather(cb, 1, root)
    av = (co_r * torch.gather(v, 1, it_r.clamp_min(0).view(E, -1)).view(E, Nm, MAXI) * (it_r >= 0)).sum(2)
    eps_ev = torch.gather(sig, 1, osrc) * zev
    y = av + cb_r * torch.gather(b, 1, src) + torch.gather(eps_ev, 1, root) + relay * delta
    y = torch.where(vt, y, torch.zeros_like(y))
    # query
    styp = torch.gather(typ, 1, root)
    sp = vt & (styp == SPOT)
    spot = torch.zeros(E, Km, dtype=torch.long, device=dev).scatter_add_(1, it_r[..., 0].clamp_min(0), sp.long()) > 0
    nospot = vk & ~spot
    sc = torch.where(nospot, _u((E, Km), dev), torch.full((E, Km), -1.0, device=dev, dtype=f64))
    q_ns = sc.argmax(1)
    q_any = (_u(E, dev) * Ke).long().clamp_max(Ke - 1)
    query = torch.where(nospot.any(1) & (_u(E, dev) < c['p_noSPOT']), q_ns, q_any)
    if c.get('p_rich', 0.0) > 0:     # rich episodes: often query an item of the most-relayed measurement
        nrel = torch.zeros(E, Nm, dtype=torch.long, device=dev).scatter_add_(1, root, relay.long())
        top_root = nrel.argmax(1)
        q_rel = items[torch.arange(E, device=dev), top_root, 0].clamp_min(0)
        query = torch.where(rich & (nrel.max(1).values > 0) & (_u(E, dev) < 0.7), q_rel, query)
    iv = torch.randn(E, Km, W.D_ITEM, device=dev) / math.sqrt(W.D_ITEM)
    sv = torch.randn(E, Sm, W.D_SRC, device=dev) / math.sqrt(W.D_SRC)
    ev = torch.randn(E, Nm, W.D_EV, device=dev) / math.sqrt(W.D_EV)
    return dict(rich=rich, N=Ne, K=Ke, S=Se, U=Ue, vt=vt, vk=vk, vs=vs, relay=relay, creates=creates, root=root, osrc=osrc,
                typ=typ, items=items, coef=coef, cb=cb, src=src, v=v, b=b, sig=sig, unk=unk, y=y, query=query,
                iv=iv, sv=sv, ev=ev, tau=c['tau'])


# ------------------------------------------------------------------------------------------------ tokens
def tokens(b, T=None, item_queries=False):
    """X (E, T', FEAT) float32. Context in positions < N_e, padding up to T-1, the query token last (T-1).
    item_queries: also append one query token per item slot (positions T-1 .. T-1+Km-1, before the main query
    which stays last); they get mask False, so they are never keys (harness SetPFN semantics).
    Returns X, mask, item_pos (or None)."""
    E, Nm = b['y'].shape
    dev = b['y'].device
    T = T or Nm + 1
    Km = b['v'].shape[1]
    n_extra = Km if item_queries else 0
    X = torch.zeros(E, T + n_extra, W.FEAT, device=dev)
    root = b['root']
    typ = torch.gather(b['typ'], 1, root)
    it = torch.gather(b['items'], 1, root[..., None].expand(E, Nm, MAXI))
    co = torch.gather(b['coef'], 1, root[..., None].expand(E, Nm, MAXI)).float()
    vt = b['vt'].float()[..., None]
    X[:, :Nm, :4] = torch.nn.functional.one_hot(typ, 4).float() * vt
    X[:, :Nm, W.F_Y] = b['y'].float() * b['vt']
    X[:, :Nm, W.F_RELAY] = b['relay'].float()
    ivg = torch.gather(b['iv'], 1, it.clamp_min(0).view(E, -1)[..., None].expand(E, Nm * MAXI, W.D_ITEM)).view(E, Nm, MAXI, W.D_ITEM)
    X[:, :Nm, W.F_FUN:W.F_SRC] = (co[..., None] * ivg * (it >= 0)[..., None]).sum(2) * vt
    ls = torch.where(b['unk'], torch.zeros_like(b['sig']), torch.log(b['sig'])).float()
    unkf = b['unk'].float()

    def srcblock(s, off):
        X[:, :Nm, off:off + W.D_SRC] = torch.gather(b['sv'], 1, s[..., None].expand(E, Nm, W.D_SRC)) * vt
        X[:, :Nm, off + W.D_SRC] = torch.gather(ls, 1, s) * b['vt']
        X[:, :Nm, off + W.D_SRC + 1] = torch.gather(unkf, 1, s) * b['vt']
    srcblock(b['src'], W.F_SRC)
    srcblock(torch.gather(b['osrc'], 1, root), W.F_OSRC)
    X[:, :Nm, W.F_EV:] = torch.gather(b['ev'], 1, root[..., None].expand(E, Nm, W.D_EV)) * vt
    mask = torch.zeros(E, T + n_extra, dtype=torch.bool, device=dev)
    mask[:, :Nm] = b['vt']
    item_pos = None
    if item_queries:
        p0 = T - 1
        X[:, p0:p0 + Km, 3] = 1.0
        X[:, p0:p0 + Km, W.F_FUN:W.F_SRC] = b['iv']
        item_pos = torch.arange(p0, p0 + Km, device=dev)
    qi = T - 1 + n_extra
    X[:, qi, 3] = 1.0
    X[:, qi, W.F_FUN:W.F_SRC] = torch.gather(b['iv'], 1, b['query'][:, None, None].expand(E, 1, W.D_ITEM))[:, 0]
    mask[:, qi] = True
    return X, mask, item_pos


# ------------------------------------------------------------------------------------------------ exact Bayes
def _system(b):
    """Transformed rows (E, Nm, D) over latents [v (Km), b (Sm), u (Hm hidden events)], observations, noise and
    prior specs. Noise var of a row = c0 + level var of unknown slot uj (if uj >= 0); same for priors (pj)."""
    f64 = torch.float64
    E, Nm = b['y'].shape
    dev = b['y'].device
    Km, Sm = b['v'].shape[1], b['b'].shape[1]
    ar = lambda n: torch.arange(n, device=dev)
    relay, root, vt = b['relay'], b['root'], b['vt']
    hid_ev = b['creates'] & relay                                  # positions that created hidden events
    hslot = torch.cumsum(hid_ev.long(), 1) - 1
    Hm = int(hid_ev.sum(1).max()) if hid_ev.any() else 0
    D = Km + Sm + Hm
    root_hidden = torch.gather(relay, 1, root)                     # root created by a relay token
    orig = ~relay & vt
    rv = relay & ~root_hidden & vt                                  # relay of a visible root
    rh = relay & root_hidden & vt                                   # relay of a hidden root
    it = torch.gather(b['items'], 1, root[..., None].expand(E, Nm, MAXI))
    co = torch.gather(b['coef'], 1, root[..., None].expand(E, Nm, MAXI))
    cbr = torch.gather(b['cb'], 1, root)
    o = torch.gather(b['osrc'], 1, root)
    R = torch.zeros(E, Nm, D, dtype=f64, device=dev)
    use_a = (orig | rh)
    R[:, :, :Km].scatter_add_(2, it.clamp_min(0), co * (it >= 0) * use_a[..., None])
    bsel = torch.zeros(E, Nm, Sm, dtype=f64, device=dev)
    bsel.scatter_add_(2, b['src'][..., None], (cbr * (orig | rh | rv))[..., None])
    bsel.scatter_add_(2, o[..., None], (-cbr * rv)[..., None])
    R[:, :, Km:Km + Sm] = bsel
    if Hm:
        hs = torch.gather(hslot, 1, root).clamp_min(0)
        R[:, :, Km + Sm:].scatter_add_(2, hs[..., None], rh[..., None].to(f64))
    yr = torch.gather(b['y'], 1, root)
    obs = torch.where(rv, b['y'] - yr, b['y'])
    obs = torch.where(vt, obs, torch.zeros_like(obs))
    uslot = torch.where(b['unk'], torch.cumsum(b['unk'].long(), 1) - 1,
                        torch.full((E, Sm), -1, dtype=torch.long, device=dev))
    s2 = b['sig'] ** 2
    src_unk = torch.gather(uslot, 1, b['src'])
    c0 = torch.where(orig, torch.where(src_unk >= 0, torch.zeros_like(s2[:, :1]).expand(E, Nm), torch.gather(s2, 1, b['src'])),
                     torch.full((E, Nm), W.RHO ** 2, dtype=f64, device=dev))
    c0 = torch.where(vt, c0, torch.ones_like(c0))
    uj = torch.where(orig, src_unk, torch.full_like(src_unk, -1))
    tau2 = max(b['tau'] ** 2, 1e-8)
    p0 = torch.cat([torch.ones(E, Km, dtype=f64, device=dev), torch.full((E, Sm), tau2, dtype=f64, device=dev),
                    torch.ones(E, Hm, dtype=f64, device=dev)], 1)
    pj = torch.full((E, D), -1, dtype=torch.long, device=dev)
    if Hm:
        # hidden event at position t -> slot hslot[t]; prior var of its eps = sigma^2 of its origin source
        e_idx, t_idx = torch.nonzero(hid_ev, as_tuple=True)
        sl = hslot[e_idx, t_idx]
        os_ = b['osrc'][e_idx, t_idx]
        u_unk = uslot[e_idx, os_]
        p0[e_idx, Km + Sm + sl] = torch.where(u_unk >= 0, torch.zeros_like(s2[e_idx, os_]), s2[e_idx, os_])
        pj[e_idx, Km + Sm + sl] = u_unk
    return R, obs, c0, uj, p0, pj, D


def bayes(b, chunk_elems=1 << 26):
    """Exact posterior. Returns item_mean, item_var (E, Km) (mixture moments; padded items: prior 0/1),
    and the query mixture (w, m, v) padded to 256 components (w = 0 on padding)."""
    f64 = torch.float64
    R, obs, c0, uj, p0, pj, D = _system(b)
    E, Nm = obs.shape
    dev = obs.device
    Km = b['v'].shape[1]
    s2lev = torch.tensor([W.SIG_L ** 2, W.SIG_H ** 2], dtype=f64, device=dev)
    imean = torch.zeros(E, Km, dtype=f64, device=dev); ivar = torch.ones(E, Km, dtype=f64, device=dev)
    qw = torch.zeros(E, 256, dtype=f64, device=dev); qm = torch.zeros(E, 256, dtype=f64, device=dev)
    qv = torch.ones(E, 256, dtype=f64, device=dev)
    Ucount = b['unk'].sum(1)
    for u in sorted(set(Ucount.tolist())):
        idx_all = torch.nonzero(Ucount == u, as_tuple=True)[0]
        Hh = 1 << u
        per = max(1, chunk_elems // (Hh * D * D))
        lev = torch.tensor([[(h >> (u - 1 - j)) & 1 for j in range(u)] for h in range(Hh)], device=dev,
                           dtype=torch.long).view(Hh, u)
        for a0 in range(0, len(idx_all), per):
            idx = idx_all[a0:a0 + per]
            Rg, yg, cg, ujg, p0g, pjg = R[idx], obs[idx], c0[idx], uj[idx], p0[idx], pj[idx]
            Eg = len(idx)
            Wk = torch.where(ujg < 0, 1.0 / cg, torch.zeros_like(cg))
            Lam = torch.einsum('etd,et,etf->edf', Rg, Wk, Rg)[:, None].expand(Eg, Hh, D, D).clone()
            eta = torch.einsum('etd,et->ed', Rg, Wk * yg)[:, None].expand(Eg, Hh, D).clone()
            yWy = (Wk * yg ** 2).sum(1)[:, None].expand(Eg, Hh).clone()
            lR = torch.where(ujg < 0, torch.log(cg), torch.zeros_like(cg)).sum(1)[:, None].expand(Eg, Hh).clone()
            for j in range(u):
                sel = (ujg == j).to(f64)
                for l in (0, 1):
                    w = sel / (cg + s2lev[l])
                    A = torch.einsum('etd,et,etf->edf', Rg, w, Rg)
                    e_ = torch.einsum('etd,et->ed', Rg, w * yg)
                    hm = (lev[:, j] == l).to(f64)                                   # (Hh,)
                    Lam += A[:, None] * hm[None, :, None, None]
                    eta += e_[:, None] * hm[None, :, None]
                    yWy += (w * yg ** 2).sum(1)[:, None] * hm[None]
                    lR += (sel * torch.log(cg + s2lev[l])).sum(1)[:, None] * hm[None]
            P = p0g[:, None].expand(Eg, Hh, D).clone()
            for j in range(u):
                add = s2lev[lev[:, j]]                                              # (Hh,)
                P += (pjg == j).to(f64)[:, None, :] * add[None, :, None]
            Lam = Lam + torch.diag_embed(1.0 / P)
            L = torch.linalg.cholesky(Lam)
            mu = torch.cholesky_solve(eta[..., None], L)[..., 0]
            eye = torch.eye(D, Km, dtype=f64, device=dev).expand(Eg, Hh, D, Km).contiguous()
            Z = torch.linalg.solve_triangular(L, eye, upper=False)
            var = (Z ** 2).sum(2)                                                   # (Eg, Hh, Km)
            logdet = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
            logml = -0.5 * (lR + torch.log(P).sum(-1) + logdet + yWy - (eta * mu).sum(-1))
            w = torch.softmax(logml, 1)                                             # (Eg, Hh)
            m_it = mu[..., :Km]
            mean = (w[..., None] * m_it).sum(1)
            imean[idx] = mean
            ivar[idx] = (w[..., None] * (var + m_it ** 2)).sum(1) - mean ** 2
            q = b['query'][idx]
            qw[idx, :Hh] = w
            qm[idx, :Hh] = torch.gather(m_it, 2, q[:, None, None].expand(Eg, Hh, 1))[..., 0]
            qv[idx, :Hh] = torch.gather(var, 2, q[:, None, None].expand(Eg, Hh, 1))[..., 0]
    return imean, ivar, (qw, qm, qv)


# ------------------------------------------------------------------------------------------------ conversion
def to_episodes(b):
    """Batch -> list of world2_market.Episode (numpy), for tests against the reference implementation."""
    out = []
    g = {k: (v.cpu().numpy() if torch.is_tensor(v) else v) for k, v in b.items()}
    for i in range(len(g['N'])):
        n, K, S = int(g['N'][i]), int(g['K'][i]), int(g['S'][i])
        cpos = np.flatnonzero(g['creates'][i, :n])
        emap = -np.ones(n, np.int64); emap[cpos] = np.arange(len(cpos))
        os_ = g['osrc'][i, cpos]
        ep = W.Episode(K, S, g['v'][i, :K], g['b'][i, :S], g['sig'][i, :S], g['unk'][i, :S], os_, g['typ'][i, cpos],
                       g['items'][i, cpos], g['coef'][i, cpos], g['cb'][i, cpos], np.zeros(len(cpos)),
                       g['src'][i, :n], emap[g['root'][i, :n]], g['relay'][i, :n], np.zeros(n), int(g['query'][i]),
                       item_vec=g['iv'][i, :K], src_vec=g['sv'][i, :S], ev_vec=g['ev'][i, cpos])
        ep.y = g['y'][i, :n].copy()
        out.append(ep)
    return out
