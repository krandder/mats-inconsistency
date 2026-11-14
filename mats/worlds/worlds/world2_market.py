"""W2 market world (DESIGN-v1.md, section W2; cruxes C4 and C7).

Generative model (one episode)
  K items      v_k ~ N(0, 1)
  S sources    bias b_s ~ N(0, TAU^2); noise sd sigma_s.
               Known-noise sources: sigma_s ~ logU[SIG_KNOWN], shown on every token that names the source.
               Unknown-noise sources (U <= 8 per episode): sigma_s in {SIG_L, SIG_H} with prior 1/2 each; tokens show
               only an "unknown" flag. This keeps exact Bayes at <= 2^8 Gaussian solves however large S is.
  Measurements (events). Each event e has a linear functional a_e over items, a bias coefficient c_e, an
  origin source o(e) and its own noise eps_e ~ N(0, sigma_{o(e)}^2). Quote types:
    SPOT    a = e_k,                 c = 1      (v_k + b_s + eps)
    DIFF    a = e_k - e_j,           c = 0      (v_k - v_j + eps; bias-free: an additive source bias cancels in
                                                 a spread. Documented choice: DIFFs calibrate nothing about b_s.)
    BASKET  a = mean of 2..4 items,  c = 1      (mean v + b_s + eps)
  Quotes (tokens):
    original  y = a_e.v + c_e b_{o(e)} + eps_e                       (the event's own source)
    relay     y = a_e.v + c_e b_{s'} + eps_e + delta,  delta ~ N(0, RHO^2), s' != o(e)
              A relay re-quotes event e's measurement through the relayer's own bias, so it shares eps_e with the
              original and with every other relay of e. The token carries the root event id (observed provenance)
              and the origin source. The original may be in the context (visible) or not (hidden: measured
              outside the context; only relays of it appear).
  Policy (exogenous, ignorable): per-episode Dirichlet weights over sources and items with log-uniform
  concentration, Dirichlet type mix, relay rate, hidden-root rate, uniform choice of the root among existing
  events. Choices depend only on keys and independent randomness, never on v, b, sigma.
  Query: an item q (with prob 1/2 drawn among items with no SPOT quote, else uniform); target v_q.
  Ids: item, source and event ids are random unit vectors (R^64, R^32, R^32) drawn fresh per episode, so no id
  carries information across episodes; everything about a source must be inferred in context.

Exact Bayes: latent z = (v, b, u) where u holds eps_e of hidden events that have relays. A relay of a visible
original is replaced by its difference to the original (unit-Jacobian transform, same for every hypothesis):
y_r - y_e = c_e (b_{s'} - b_{o(e)}) + delta. After this transform all observation noises are independent, so each
noise hypothesis h (levels of the unknown sources that matter) is one sparse linear-Gaussian solve; the predictive
for v_q is the mixture over h with weights p(h | data). References are the same solver with one assumption broken:
  prior          N(0, 1)
  freq_only      unknown sources get the prior-mean variance (no in-context inference of noise level); its weights
                 depend only on how often each source quotes what, and on the known sigmas.
  bias_naive     no shared source biases: each quote's bias terms are treated as independent extra noise
                 (TAU^2 per term), i.e. no in-context bias calibration; mixture over h.
  prov_naive     relays treated as independent fresh quotes (noise sigma_o^2 + RHO^2); mixture over h.
  noise_oracle   true sigma of the unknown sources (a cheating lower bound, can beat Bayes).
  bayes          full mixture.
"""
import itertools, math
import numpy as np
import torch

SPOT, DIFF, BASKET = 0, 1, 2
TAU = 0.5
SIG_L, SIG_H = 0.2, 1.2
SIG_KNOWN = (0.15, 1.0)
RHO = 0.1
MAXI = 4                       # max items per quote
U_MAX = 8
D_ITEM, D_SRC, D_EV = 64, 32, 32     # random unit id vectors, fresh per episode
TRAIN = dict(N=(8, 128), rK=(6.0, 12.0), rS=(6.0, 16.0), alpha=(0.2, 5.0), p_relay=(0.0, 0.3), p_hidden=0.4,
             p_noSPOT_query=0.5)


# ------------------------------------------------------------------------------------------------ episode
class Episode:
    """Arrays for one episode. Events E (measurements), quotes n (tokens). Quote t refers to event q_ev[t]."""

    def __init__(self, K, S, v, b, sig, unk, ev_src, ev_typ, ev_items, ev_coef, ev_cb, ev_eps,
                 q_src, q_ev, q_relay, q_delta, query, item_vec=None, src_vec=None, ev_vec=None, rng=None, meta=None):
        self.K, self.S = K, S
        self.v, self.b, self.sig, self.unk = v, b, sig, unk.astype(bool)
        self.ev_src, self.ev_typ, self.ev_items, self.ev_coef, self.ev_cb = ev_src, ev_typ, ev_items, ev_coef, ev_cb
        self.ev_eps = ev_eps
        self.q_src, self.q_ev, self.q_relay, self.q_delta = q_src, q_ev, q_relay.astype(bool), q_delta
        self.query = int(query)
        E = len(ev_src)
        self.ev_vis = np.zeros(E, bool)
        self.ev_vis[q_ev[~self.q_relay]] = True
        rng = rng if rng is not None else np.random.default_rng(0)
        self.item_vec = item_vec if item_vec is not None else _unit(rng, (K, D_ITEM))
        self.src_vec = src_vec if src_vec is not None else _unit(rng, (S, D_SRC))
        self.ev_vec = ev_vec if ev_vec is not None else _unit(rng, (E, D_EV))
        self.meta = meta or {}
        self.y = self.values()

    @property
    def n(self):
        return len(self.q_src)

    def values(self, v=None, b=None):
        v = self.v if v is None else v
        b = self.b if b is None else b
        e = self.q_ev
        it = self.ev_items[e]
        av = (self.ev_coef[e] * np.where(it >= 0, v[np.maximum(it, 0)], 0.0)).sum(1)
        return av + self.ev_cb[e] * b[self.q_src] + self.ev_eps[e] + self.q_relay * self.q_delta

    def h_true(self):
        return {int(s): int(self.sig[s] > 0.5 * (SIG_L + SIG_H)) for s in np.flatnonzero(self.unk)}


def _cat(rng, cum, shape):
    """Categorical draws from per-episode cumulative weights cum (E, C) -> (E, *shape)."""
    u = rng.random((cum.shape[0],) + shape)
    idx = (u[..., None] > cum.reshape((cum.shape[0],) + (1,) * len(shape) + (cum.shape[1],))).sum(-1)
    return np.minimum(idx, cum.shape[1] - 1)


def _dirichlet(rng, alpha, valid):
    g = rng.gamma(np.broadcast_to(alpha[:, None], valid.shape), 1.0) * valid
    g = np.where(valid, np.maximum(g, 1e-30), 0.0)
    return g / g.sum(1, keepdims=True)


def sample_batch(rng, E, N, cfg=None, K=None, S=None, U=None):
    """E episodes from the generator (vectorised structure draws). N: int or (E,) quote counts. K, S, U: optional
    overrides (int). Returns a list of Episode objects. Deterministic given rng."""
    c = dict(TRAIN, **(cfg or {}))
    Ne = np.broadcast_to(np.asarray(N, np.int64), (E,)).copy()
    Ke = np.maximum(2, np.round(Ne / rng.uniform(*c['rK'], E))).astype(np.int64) if K is None else np.full(E, K)
    Se = np.maximum(2, np.round(Ne / rng.uniform(*c['rS'], E))).astype(np.int64) if S is None else np.full(E, S)
    Ue = rng.integers(0, np.minimum(U_MAX, Se) + 1) if U is None else np.minimum(np.full(E, U), Se)
    Nm, Km, Sm = int(Ne.max()), int(Ke.max()), int(Se.max())
    vk = np.arange(Km)[None] < Ke[:, None]; vs = np.arange(Sm)[None] < Se[:, None]
    vt = np.arange(Nm)[None] < Ne[:, None]
    la = lambda: np.exp(rng.uniform(np.log(c['alpha'][0]), np.log(c['alpha'][1]), E))
    ws = _dirichlet(rng, la(), vs); wi = _dirichlet(rng, la(), vk)
    wt = 0.1 + 0.7 * rng.dirichlet(np.ones(3), E); wt /= wt.sum(1, keepdims=True)
    p_relay = rng.uniform(*c['p_relay'], E)
    relay = (rng.random((E, Nm)) < p_relay[:, None]) & vt
    relay[:, 0] = False
    hidden_new = relay & (rng.random((E, Nm)) < c['p_hidden'])
    creates = (~relay | hidden_new) & vt
    cnt = np.cumsum(creates, 1) - creates                    # events created before t
    order = np.argsort(~creates, 1, kind='stable')           # creating positions first, in order
    pick = (rng.random((E, Nm)) * np.maximum(cnt, 1)).astype(np.int64)
    root = np.where(creates, np.arange(Nm)[None], np.take_along_axis(order, pick, 1))
    # event attributes (drawn at every position; used where creates)
    osrc = _cat(rng, np.cumsum(ws, 1), (Nm,))
    typ = _cat(rng, np.cumsum(wt, 1), (Nm,))
    g = np.log(np.where(vk, wi, 1e-300))[:, None, :] - np.log(-np.log(rng.random((E, Nm, Km))))
    top = np.argsort(-g, 2)[:, :, :MAXI]
    if top.shape[2] < MAXI:
        top = np.concatenate([top, np.zeros((E, Nm, MAXI - top.shape[2]), np.int64)], 2)
    mb = rng.integers(2, np.minimum(MAXI, Ke)[:, None] + 1, (E, Nm))
    m = np.where(typ == SPOT, 1, np.where(typ == DIFF, 2, mb))
    slot = np.arange(MAXI)[None, None]
    items = np.where(slot < m[..., None], top, -1)
    coef = np.where(slot < m[..., None], 1.0 / m[..., None], 0.0)
    coef = np.where(typ[..., None] == DIFF, np.where(slot == 0, 1.0, np.where(slot == 1, -1.0, 0.0)), coef)
    cb = np.where(typ == DIFF, 0.0, 1.0)
    # token sources: originals = event source; relays: a different source
    src = _cat(rng, np.cumsum(ws, 1), (Nm,))
    o_root = np.take_along_axis(osrc, root, 1)
    for _ in range(3):
        bad = relay & (src == o_root)
        if not bad.any():
            break
        src = np.where(bad, _cat(rng, np.cumsum(ws, 1), (Nm,)), src)
    bad = relay & (src == o_root)
    src = np.where(bad, (o_root + 1 + (rng.random((E, Nm)) * (Se[:, None] - 1)).astype(np.int64)) % Se[:, None], src)
    src = np.where(relay, src, osrc)
    # hidden values
    v = rng.standard_normal((E, Km)); b = TAU * rng.standard_normal((E, Sm))
    sig = np.exp(rng.uniform(np.log(SIG_KNOWN[0]), np.log(SIG_KNOWN[1]), (E, Sm)))
    rk = np.argsort(np.argsort(np.where(vs, rng.random((E, Sm)), 2.0), 1), 1)
    unk = rk < Ue[:, None]
    sig = np.where(unk, np.where(rng.random((E, Sm)) < 0.5, SIG_L, SIG_H), sig)
    zev = rng.standard_normal((E, Nm)); delta = RHO * rng.standard_normal((E, Nm))
    # query
    styp = np.take_along_axis(typ, root, 1)
    sit = np.take_along_axis(items[:, :, 0], root, 1)
    spot = np.zeros((E, Km), bool)
    ee, tt = np.nonzero(vt & (styp == SPOT))
    spot[ee, sit[ee, tt]] = True
    nospot = vk & ~spot
    rq = rng.random(E); rq2 = rng.random(E)
    nn_ = nospot.sum(1)
    q_any = np.minimum((rq2 * Ke).astype(np.int64), Ke - 1)
    q_ns = np.array([np.flatnonzero(nospot[i])[min(int(rq2[i] * nn_[i]), max(nn_[i] - 1, 0))] if nn_[i] else 0
                     for i in range(E)])
    query = np.where((nn_ > 0) & (rq < c['p_noSPOT_query']), q_ns, q_any)
    # ids
    iv = _unit(rng, (E, Km, D_ITEM)); sv = _unit(rng, (E, Sm, D_SRC)); ev = _unit(rng, (E, Nm, D_EV))
    out = []
    for i in range(E):
        n = int(Ne[i]); K_, S_ = int(Ke[i]), int(Se[i])
        cpos = np.flatnonzero(creates[i, :n])
        emap = -np.ones(n, np.int64); emap[cpos] = np.arange(len(cpos))
        os_ = osrc[i, cpos]
        out.append(Episode(K_, S_, v[i, :K_], b[i, :S_], sig[i, :S_], unk[i, :S_], os_, typ[i, cpos],
                           items[i, cpos], coef[i, cpos], cb[i, cpos], sig[i, os_] * zev[i, cpos],
                           src[i, :n], emap[root[i, :n]], relay[i, :n], delta[i, :n], int(query[i]),
                           item_vec=iv[i, :K_], src_vec=sv[i, :S_], ev_vec=ev[i, cpos],
                           meta=dict(N=n, U=int(Ue[i]), p_relay=float(p_relay[i]))))
    return out


def _unit(rng, shape):
    """Random id vectors with i.i.d. N(0, 1/d) entries (norm ~1)."""
    return rng.standard_normal(shape, dtype=np.float32) * np.float32(1 / math.sqrt(shape[-1]))


def sample_episode(rng, N=None, K=None, S=None, U=None, cfg=None):
    """One episode from the training-family generator (N drawn from cfg['N'] if None)."""
    c = dict(TRAIN, **(cfg or {}))
    if N is None:
        N = int(rng.integers(c['N'][0], c['N'][1] + 1))
    return sample_batch(rng, 1, N, cfg, K, S, U)[0]


def sample_adaptive(rng, N, K, S, U, mode):
    """Test-only generator whose choices depend on the data. mode='ignorable': the next quoted item is the one
    whose latest observed quote value was highest (depends on recorded observations only). mode='leaky': the
    item is chosen with weight exp(2 v_k) and the query with exp(4 v_k) (depends on hidden values: NOT ignorable, a negative
    control)."""
    v = rng.standard_normal(K); b = TAU * rng.standard_normal(S)
    sig = np.exp(rng.uniform(np.log(SIG_KNOWN[0]), np.log(SIG_KNOWN[1]), S))
    unk = np.zeros(S, bool); unk[rng.choice(S, U, replace=False)] = True
    sig[unk] = np.where(rng.random(U) < 0.5, SIG_L, SIG_H)
    last = np.full(K, np.inf)                              # unquoted items first
    ev_src, ev_items, ev_coef, ev_cb, ev_eps, ev_typ = [], [], [], [], [], []
    for t in range(N):
        s = int(rng.integers(0, S))
        if mode == 'ignorable':
            k = int(np.argmax(last + 1e-9 * rng.random(K)))
        else:
            p = np.exp(2 * v); k = int(rng.choice(K, p=p / p.sum()))
        if rng.random() < 0.3 and K >= 2:
            j = int(rng.choice([x for x in range(K) if x != k]))
            it = np.array([k, j, -1, -1]); co = np.array([1.0, -1.0, 0, 0]); cb = 0.0; ty = DIFF
        else:
            it = np.array([k, -1, -1, -1]); co = np.array([1.0, 0, 0, 0]); cb = 1.0; ty = SPOT
        eps = sig[s] * rng.standard_normal()
        y = co[0] * v[k] + (co[1] * v[it[1]] if it[1] >= 0 else 0) + cb * b[s] + eps
        last[k] = y
        ev_src.append(s); ev_items.append(it); ev_coef.append(co); ev_cb.append(cb); ev_eps.append(eps)
        ev_typ.append(ty)
    E = len(ev_src)
    if mode == 'ignorable':
        q = int(rng.integers(0, K))
    else:                                                  # leaky query choice too
        p = np.exp(4 * v); q = int(rng.choice(K, p=p / p.sum()))
    return Episode(K, S, v, b, sig, unk, np.array(ev_src), np.array(ev_typ), np.array(ev_items), np.array(ev_coef),
                   np.array(ev_cb), np.array(ev_eps), np.array(ev_src), np.arange(E), np.zeros(E, bool),
                   np.zeros(E), q, rng=rng)


# ------------------------------------------------------------------------------------------------ oracle
def _system(ep, mode):
    """Rows of the transformed linear-Gaussian system.
    Returns H (n, D), y (n,), noise (c0 (n,), uj (n,)), prior (p0 (D,), pj (D,)), unknown-source list, q index.
    Noise var of row r under hypothesis h = c0[r] + (sig2_h[uj[r]] if uj[r] >= 0 else 0); same for priors."""
    K, S = ep.K, ep.S
    tau2 = 0.0 if mode == 'bias_naive' else TAU ** 2
    unk_list = list(np.flatnonzero(ep.unk))
    if mode in ('freq_only', 'noise_oracle'):
        unk_list = []
    uidx = {s: i for i, s in enumerate(unk_list)}

    def sig2(s):            # (c0, uj)
        if s in uidx:
            return 0.0, uidx[s]
        if ep.unk[s] and mode == 'freq_only':
            return 0.5 * (SIG_L ** 2 + SIG_H ** 2), -1
        return float(ep.sig[s] ** 2), -1

    nb = S if tau2 > 0 else 0
    hidden_with_relay = sorted(set(ep.q_ev[ep.q_relay]) - set(np.flatnonzero(ep.ev_vis))) \
        if mode != 'prov_naive' else []
    hid = {e: K + nb + i for i, e in enumerate(hidden_with_relay)}
    D = K + nb + len(hid)
    orig_row = {}
    for t in np.flatnonzero(~ep.q_relay):
        orig_row[int(ep.q_ev[t])] = int(t)
    rows, ys, c0, uj = [], [], [], []
    for t in range(ep.n):
        e = int(ep.q_ev[t]); s = int(ep.q_src[t]); o = int(ep.ev_src[e]); cb = ep.ev_cb[e]
        h = np.zeros(D)
        hb = np.zeros(S)                         # bias coefficients of this row
        if not ep.q_relay[t]:
            _add_a(h, ep, e); hb[s] += cb
            a, j = sig2(s); yv = ep.y[t]
        elif mode == 'prov_naive':
            _add_a(h, ep, e); hb[s] += cb
            a, j = sig2(o); a += RHO ** 2; yv = ep.y[t]
        elif ep.ev_vis[e]:
            hb[s] += cb; hb[o] -= cb
            a, j = RHO ** 2, -1; yv = ep.y[t] - ep.y[orig_row[e]]
        else:
            _add_a(h, ep, e); hb[s] += cb
            h[hid[e]] += 1.0
            a, j = RHO ** 2, -1; yv = ep.y[t]
        if nb:
            h[K:K + S] += hb
        if mode == 'bias_naive':                 # biases as independent per-quote noise
            a += TAU ** 2 * float((hb ** 2).sum())
        rows.append(h); ys.append(yv); c0.append(a); uj.append(j)
    p0 = np.zeros(D); pj = -np.ones(D, np.int64)
    p0[:K] = 1.0
    p0[K:K + nb] = tau2
    for e, i in hid.items():
        p0[i], pj[i] = sig2(int(ep.ev_src[e]))
    H = np.array(rows).reshape(ep.n, D)
    return H, np.array(ys), np.array(c0), np.array(uj, np.int64), p0, pj, len(unk_list), ep.query


def _add_a(h, ep, e):
    for k, c in zip(ep.ev_items[e], ep.ev_coef[e]):
        if k >= 0:
            h[k] += c


def predictive(ep, mode='bayes', chunk=64, device='cpu'):
    """Mixture predictive for v_q: (w, m, s) arrays over noise hypotheses (sum w = 1)."""
    if mode == 'prior':
        return np.ones(1), np.zeros(1), np.ones(1)
    if mode == 'noise_oracle':
        ep2 = _copy_known(ep)
        return predictive(ep2, 'bayes', chunk, device)
    H, y, c0, uj, p0, pj, U, q = _system(ep, mode)
    # only unknown sources that touch a row or a prior matter
    used = sorted(set(uj[uj >= 0]) | set(pj[pj >= 0]))
    remap = {u: i for i, u in enumerate(used)}
    Uu = len(used)
    levels = np.array(list(itertools.product((0, 1), repeat=Uu)), np.int64).reshape(2 ** Uu, Uu)   # (nh, Uu)
    s2lev = np.array([SIG_L ** 2, SIG_H ** 2])
    T = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)
    Ht, yt = T(H), T(y)
    D = H.shape[1]
    # base (known) parts
    kn = uj < 0
    Wk = 1.0 / c0[kn]
    A0 = T(H[kn].T @ (Wk[:, None] * H[kn]))
    e0 = T(H[kn].T @ (Wk * y[kn]))
    yWy0 = float((Wk * y[kn] ** 2).sum()); lR0 = float(np.log(c0[kn]).sum())
    Aj, ej, yWyj, lRj = [], [], [], []
    for u in used:
        r = uj == u
        A_l, e_l, q_l, l_l = [], [], [], []
        for lv in (0, 1):
            W = 1.0 / (c0[r] + s2lev[lv])
            A_l.append(H[r].T @ (W[:, None] * H[r])); e_l.append(H[r].T @ (W * y[r]))
            q_l.append((W * y[r] ** 2).sum()); l_l.append(-np.log(W).sum())
        Aj.append(T(np.stack(A_l))); ej.append(T(np.stack(e_l))); yWyj.append(np.array(q_l)); lRj.append(np.array(l_l))
    # priors per hypothesis
    P = np.tile(p0, (len(levels), 1))
    for i in range(D):
        if pj[i] >= 0:
            P[:, i] = p0[i] + s2lev[levels[:, remap[pj[i]]]]
    logml, mq, vq = [], [], []
    for a in range(0, len(levels), chunk):
        lv = levels[a:a + chunk]; nh = len(lv)
        Lam = A0.expand(nh, D, D).clone(); eta = e0.expand(nh, D).clone()
        yWy = np.full(nh, yWy0); lR = np.full(nh, lR0)
        for i in range(Uu):
            li = torch.as_tensor(lv[:, i], device=device)
            Lam += Aj[i][li]; eta += ej[i][li]
            yWy += yWyj[i][lv[:, i]]; lR += lRj[i][lv[:, i]]
        Pc = T(P[a:a + nh])
        Lam += torch.diag_embed(1.0 / Pc)
        L = torch.linalg.cholesky(Lam)
        mu = torch.cholesky_solve(eta[..., None], L)[..., 0]
        eq = torch.zeros(nh, D, 1, dtype=torch.float64, device=device); eq[:, q, 0] = 1.0
        z = torch.linalg.solve_triangular(L, eq, upper=False)
        vq.append((z ** 2).sum((1, 2)).cpu().numpy()); mq.append(mu[:, q].cpu().numpy())
        logdetL = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1).cpu().numpy()
        quad = (eta * mu).sum(-1).cpu().numpy()
        logml.append(-0.5 * (lR + torch.log(Pc).sum(-1).cpu().numpy() + logdetL + yWy - quad))
    logml = np.concatenate(logml); mq = np.concatenate(mq); vq = np.concatenate(vq)
    w = np.exp(logml - logml.max()); w /= w.sum()
    return w, mq, np.sqrt(vq)


def _copy_known(ep):
    import copy
    ep2 = copy.copy(ep)
    ep2.unk = np.zeros(ep.S, bool)
    return ep2


def h_posterior(ep):
    """Posterior over the noise level of each unknown source that matters: dict s -> P(high)."""
    _, _, _, uj, _, pj, _, _ = _system(ep, 'bayes')
    unk_list = list(np.flatnonzero(ep.unk))
    w, _, _ = predictive(ep, 'bayes')
    used = sorted(set(uj[uj >= 0]) | set(pj[pj >= 0]))
    levels = np.array(list(itertools.product((0, 1), repeat=len(used))), np.int64).reshape(2 ** len(used), len(used))
    return {int(unk_list[u]): float((w * levels[:, i]).sum()) for i, u in enumerate(used)}


REFS = ('prior', 'freq_only', 'bias_naive', 'prov_naive', 'noise_oracle', 'bayes')


def references(ep, device='cpu'):
    return {r: predictive(ep, r, device=device) for r in REFS}


# ------------------------------------------------------------------------------------------------ metrics
def mix_logpdf(x, w, m, s):
    x = np.asarray(x, float)[..., None]
    lp = np.log(w + 1e-300) - 0.5 * np.log(2 * np.pi) - np.log(s) - 0.5 * ((x - m) / s) ** 2
    mx = lp.max(-1, keepdims=True)
    return (mx + np.log(np.exp(lp - mx).sum(-1, keepdims=True)))[..., 0]


def mix_cdf(x, w, m, s):
    from math import erf
    return float(sum(wi * 0.5 * (1 + erf((x - mi) / (si * math.sqrt(2)))) for wi, mi, si in zip(w, m, s)))


def kl_mix(p, qmix, n=4001):
    """KL(p || q) for 1-D Gaussian mixtures by quadrature on a grid covering p."""
    w, m, s = p
    lo = (m - 10 * s).min(); hi = (m + 10 * s).max()
    x = np.linspace(lo, hi, n)
    lp = mix_logpdf(x, *p); lq = mix_logpdf(x, *qmix)
    pp = np.exp(lp)
    return float(np.trapezoid(pp * (lp - lq), x))


def gauss(m, s):
    return np.ones(1), np.array([float(m)]), np.array([float(s)])


def moment_match(p):
    w, m, s = p
    mu = (w * m).sum()
    return gauss(mu, math.sqrt((w * (s ** 2 + m ** 2)).sum() - mu ** 2))


# ------------------------------------------------------------------------------------------------ builder
class Builder:
    """Explicit episode construction for diagnostic sets. Sources/items are integers; values are drawn by
    realize(); noise draws can be shared across paired episodes through the `z` dicts."""

    def __init__(self, K, S):
        self.K, self.S = K, S
        self.ev = []        # (src, typ, items, coef, cb)
        self.qs = []        # (src, ev, relay)

    def event(self, s, typ, items):
        it = -np.ones(MAXI, np.int64); co = np.zeros(MAXI)
        it[:len(items)] = items
        if typ == SPOT:
            co[0] = 1.0; cb = 1.0
        elif typ == DIFF:
            co[:2] = (1.0, -1.0); cb = 0.0
        else:
            co[:len(items)] = 1.0 / len(items); cb = 1.0
        self.ev.append((s, typ, it, co, cb))
        return len(self.ev) - 1

    def quote(self, s, typ, items):
        e = self.event(s, typ, items)
        self.qs.append((s, e, False))
        return e

    def relay(self, s, e):
        self.qs.append((s, e, True))

    def build(self, v, b, sig, unk, eps_z, delta_z, query, rng, perm=None, ids=None):
        """eps_z: standard-normal draw per event (scaled by the origin's sigma); delta_z per quote."""
        ev_src = np.array([e[0] for e in self.ev], np.int64)
        qs = self.qs if perm is None else [self.qs[i] for i in perm]
        dz = delta_z if perm is None else delta_z[perm]
        ids = ids or {}
        return Episode(self.K, self.S, v, b, sig, unk, ev_src, np.array([e[1] for e in self.ev]),
                       np.array([e[2] for e in self.ev]), np.array([e[3] for e in self.ev]),
                       np.array([e[4] for e in self.ev]), sig[ev_src] * eps_z[:len(self.ev)],
                       np.array([q[0] for q in qs], np.int64), np.array([q[1] for q in qs], np.int64),
                       np.array([q[2] for q in qs], bool), RHO * dz, query, rng=rng, **ids)


def _background(B, rng, K0, S0, items, srcs, n, psrc=None):
    """n quotes on `items` by `srcs` (probabilities psrc, default uniform): SPOT/DIFF/BASKET 0.5/0.3/0.2."""
    for _ in range(n):
        s = int(rng.choice(srcs, p=psrc)); t = int(rng.choice(3, p=[0.5, 0.3, 0.2]))
        if t == SPOT:
            B.quote(s, SPOT, [int(rng.choice(items))])
        elif t == DIFF:
            B.quote(s, DIFF, list(rng.choice(items, 2, replace=False)))
        else:
            m = int(rng.integers(2, 5)); B.quote(s, BASKET, list(rng.choice(items, m, replace=False)))


def c4_pair(rng, n_bg=80, M=12, min_delta=0.3, max_tries=50, scale=96):
    """C4 frequency-matched calibration pair. Query q gets one SPOT from A and one from B and nothing else; A and B
    are unknown-noise sources each quoting other items exactly M times (SPOT); a pool of known sources measures the
    other items. Member 0: A low-noise, B high-noise; member 1: swapped. Identical structure, ids, query readings,
    and noise draws (z); only the scale of A's and B's other-item noise differs. Kept only if
    KL(Bayes_0 || Bayes_1) >= min_delta (pre-registered 0.3 nats). Returns (ep0, ep1, delta, tries)."""
    for tries in range(1, max_tries + 1):
        K = max(8, int(scale / rng.uniform(6, 12))); S = max(6, int(scale / rng.uniform(8, 16)))
        q, A, Bs = K - 1, S - 2, S - 1
        items = np.arange(K - 1); known = np.arange(S - 2)
        Bd = Builder(K, S)
        _background(Bd, rng, K, S, items, known, n_bg)
        for s in (A, Bs):
            for _ in range(M):
                Bd.quote(s, SPOT, [int(rng.choice(items))])
        eA = Bd.quote(A, SPOT, [q]); eB = Bd.quote(Bs, SPOT, [q])
        v = rng.standard_normal(K); b = TAU * rng.standard_normal(S)
        sig = np.exp(rng.uniform(np.log(SIG_KNOWN[0]), np.log(SIG_KNOWN[1]), S))
        unk = np.zeros(S, bool); unk[[A, Bs]] = True
        E = len(Bd.ev); z = rng.standard_normal(E); dz = rng.standard_normal(len(Bd.qs))
        perm = rng.permutation(len(Bd.qs))
        ids = dict(item_vec=_unit(rng, (K, D_ITEM)), src_vec=_unit(rng, (S, D_SRC)), ev_vec=_unit(rng, (E, D_EV)))
        eps = []
        for lowA in (True, False):
            sg = sig.copy(); sg[A], sg[Bs] = (SIG_L, SIG_H) if lowA else (SIG_H, SIG_L)
            ep = Bd.build(v, b, sg, unk, z, dz, q, rng, perm, ids)
            # the two query readings are identical across the pair (drawn as in member 0)
            zz = ep.ev_eps.copy()
            zz[eA] = SIG_L * z[eA]; zz[eB] = SIG_H * z[eB]
            ep.ev_eps = zz; ep.y = ep.values()
            eps.append(ep)
        delta = kl_mix(predictive(eps[0]), predictive(eps[1]))
        if delta >= min_delta:
            for i, ep in enumerate(eps):
                ep.meta = dict(diag='C4', member=i, delta=delta, A=A, B=Bs)
            return eps[0], eps[1], delta, tries
    raise RuntimeError('no pair reached min_delta')


def c7_set(rng, n_bg=90, sig_anchor=0.4, sig_path=0.05, scale=96):
    """C7 sets sharing truth, background and noise draws. Background: items 0..K0-1 quoted by precise known
    calibrators plus s0, s1, s2 (20% of the background each, so their biases are calibrated). Diagnostic subgraph
    on fresh items: roots r1..r4, mids m1..m4, query q. Anchor source s0 (sigma_anchor), path source sp
    (sigma_path; DIFFs are bias-free). Path i: DIFF(m_i - root_i), DIFF(q - m_i) -- every path has length 2.
      D   diamond, 2 paths from one root:  SPOT(r1) by s0; paths 1,2 from r1; 1 filler SPOT(bg item) by s0.
      T   two-anchor:                      SPOT(r1), SPOT(r2) by s0; path i from r_i.
      D4  diamond, 4 paths from one root:  SPOT(r1); paths 1..4 from r1; 3 fillers.
      T4  four-anchor:                     SPOT(r1..r4); path i from r_i.
      DR  D + relay of SPOT(r1) by s1 (shares the anchor's noise; carries the root event id), no filler.
      DI  D + independent SPOT(r1) by s1 (noise variance sigma_anchor^2 + RHO^2, matched to the relay), no filler.
      HR  two relays (by s1, s2) of a hidden SPOT(r1) by s0, plus paths 1,2 from r1 (no visible root).
      HI  two independent SPOT(r1) by s1, s2 (variance matched) plus paths 1,2 from r1.
      HR4 / HI4  as HR / HI with four relays (s1, s2, s1, s2) / four independent SPOTs.
    Pairs: D/T and D4/T4 (topological dedup), DR/DI, HR/HI, HR4/HI4 (provenance dedup); within a pair the token count,
    sources, path lengths and edge noises match."""
    K0, S0 = max(10, scale // 10), max(6, scale // 16)
    R = [K0, K0 + 1, K0 + 2, K0 + 3]; Mi = [K0 + 4, K0 + 5, K0 + 6, K0 + 7]; q = K0 + 8
    K, S = K0 + 9, S0 + 4
    s0, s1, s2, sp = S0, S0 + 1, S0 + 2, S0 + 3
    v = rng.standard_normal(K); b = TAU * rng.standard_normal(S)
    sig = np.exp(rng.uniform(np.log(SIG_KNOWN[0]), np.log(0.4), S))       # precise calibrators
    sig[s0] = sig_anchor; sig[sp] = sig_path
    sig[s1] = sig[s2] = math.sqrt(sig_anchor ** 2 + RHO ** 2)     # independent anchors matched to a relay
    unk = np.zeros(S, bool)
    bg_seed = int(rng.integers(1 << 31))
    ids = dict(item_vec=_unit(rng, (K, D_ITEM)), src_vec=_unit(rng, (S, D_SRC)))
    nz = max(1000, 2 * n_bg + 40)
    evv = _unit(rng, (nz, D_EV))
    zbg = rng.standard_normal(nz); dzbg = rng.standard_normal(nz)
    out = {}

    def make(name, sub):
        Bd = Builder(K, S)
        r_ = np.random.default_rng(bg_seed)
        _background(Bd, r_, K, S, np.arange(K0), np.r_[np.arange(S0), s0, s1, s2], n_bg,
                    np.r_[np.full(S0, 0.4 / S0), 0.2, 0.2, 0.2])
        sub(Bd)
        E = len(Bd.ev)
        z = zbg[:E].copy(); dz = dzbg[:len(Bd.qs)].copy()
        perm = np.random.default_rng(bg_seed + 1).permutation(len(Bd.qs))
        ep = Bd.build(v, b, sig, unk, z, dz, q, rng, perm, dict(ids, ev_vec=evv[:E]))
        ep.meta = dict(diag='C7', arm=name)
        out[name] = ep

    def paths(Bd, roots):
        for i, r in enumerate(roots):
            Bd.quote(sp, DIFF, [Mi[i], r]); Bd.quote(sp, DIFF, [q, Mi[i]])

    def fill(Bd, k):
        for j in range(k):
            Bd.quote(s0, SPOT, [j % K0])

    arms = {
        'D': lambda B: (B.quote(s0, SPOT, [R[0]]), paths(B, [R[0]] * 2), fill(B, 1)),
        'T': lambda B: (B.quote(s0, SPOT, [R[0]]), paths(B, R[:2]), B.quote(s0, SPOT, [R[1]])),
        'D4': lambda B: (B.quote(s0, SPOT, [R[0]]), paths(B, [R[0]] * 4), fill(B, 3)),
        'T4': lambda B: (B.quote(s0, SPOT, [R[0]]), paths(B, R), [B.quote(s0, SPOT, [r]) for r in R[1:]]),
        'DR': lambda B: (B.relay(s1, B.quote(s0, SPOT, [R[0]])), paths(B, [R[0]] * 2)),
        'DI': lambda B: (B.quote(s0, SPOT, [R[0]]), paths(B, [R[0]] * 2), B.quote(s1, SPOT, [R[0]])),
        'HR': lambda B: _hr(B, s0, s1, s2, R[0], paths),
        'HI': lambda B: (paths(B, [R[0]] * 2), B.quote(s1, SPOT, [R[0]]), B.quote(s2, SPOT, [R[0]])),
        'HR4': lambda B: _hr(B, s0, s1, s2, R[0], paths, 4),
        'HI4': lambda B: (paths(B, [R[0]] * 2), [B.quote((s1, s2)[j % 2], SPOT, [R[0]]) for j in range(4)]),
    }
    for name in C7_ARMS:
        make(name, arms[name])
    return out


def _hr(B, s0, s1, s2, r, paths, k=2):
    e = B.event(s0, SPOT, [r]); paths(B, [r] * 2)
    for j in range(k):
        B.relay((s1, s2)[j % 2], e)


def naive_double_count(ep_D):
    """Reference for a diamond arm (D or D4): the root SPOT is counted once per path, as non-deduplicating message
    passing does. Built by giving every path after the first its own root item whose SPOT repeats the root's
    reading with independent noise (same source, same noise level)."""
    import copy
    ep = copy.deepcopy(ep_D)
    K = ep.K
    K0 = K - 9
    r = K0; mids = [K0 + 4, K0 + 5, K0 + 6, K0 + 7]
    root_q = [t for t in range(ep.n) if not ep.q_relay[t] and ep.ev_typ[ep.q_ev[t]] == SPOT
              and ep.ev_items[ep.q_ev[t], 0] == r][0]
    e_root = ep.q_ev[root_q]
    path_ev = [e for e in range(len(ep.ev_src)) if ep.ev_typ[e] == DIFF and ep.ev_items[e, 1] == r
               and ep.ev_items[e, 0] in mids]
    for j, e in enumerate(path_ev[1:]):
        kn = ep.K
        ep.K += 1
        ep.v = np.r_[ep.v, ep.v[r]]
        ep.item_vec = np.vstack([ep.item_vec, _unit(np.random.default_rng(j), (1, D_ITEM))])
        ep.ev_items[e, 1] = kn
        ep.ev_src = np.r_[ep.ev_src, ep.ev_src[e_root]]
        ep.ev_typ = np.r_[ep.ev_typ, SPOT]
        ep.ev_items = np.vstack([ep.ev_items, [kn, -1, -1, -1]])
        ep.ev_coef = np.vstack([ep.ev_coef, [1, 0, 0, 0]])
        ep.ev_cb = np.r_[ep.ev_cb, 1.0]
        ep.ev_eps = np.r_[ep.ev_eps, ep.ev_eps[e_root]]
        ep.ev_vis = np.r_[ep.ev_vis, True]
        ep.ev_vec = np.vstack([ep.ev_vec, _unit(np.random.default_rng(100 + j), (1, D_EV))])
        ep.q_src = np.r_[ep.q_src, ep.q_src[root_q]]
        ep.q_ev = np.r_[ep.q_ev, len(ep.ev_src) - 1]
        ep.q_relay = np.r_[ep.q_relay, False]
        ep.q_delta = np.r_[ep.q_delta, 0.0]
    ep.y = ep.values()
    return ep


# ------------------------------------------------------------------------------------------------ tokens
F_TYPE, F_Y, F_RELAY = 0, 4, 5
F_FUN = 6                                   # functional vector sum_k a_k id_k (query token: id of q)
F_SRC = F_FUN + D_ITEM                      # quoting source id, log sigma (0 if unknown), unknown flag
F_OSRC = F_SRC + D_SRC + 2                  # origin source of the measurement (= quoting source for originals)
F_EV = F_OSRC + D_SRC + 2                   # root event id (shared by an original and all its relays)
FEAT = F_EV + D_EV


def pack(eps, T=None):
    """Episodes -> X (E, T, FEAT) float32 with the query token last, mask (E, T), target (E,).
    Token: type one-hot (SPOT, DIFF, BASKET, QUERY), y, relay flag, functional vector, source block, origin-source
    block, root-event id. No positions."""
    E = len(eps)
    T = T or max(e.n for e in eps) + 1
    X = np.zeros((E, T, FEAT), np.float32); M = np.zeros((E, T), bool); Y = np.zeros(E)
    for i, ep in enumerate(eps):
        n, e = ep.n, ep.q_ev
        ls = np.where(ep.unk, 0.0, np.log(ep.sig))
        X[i, np.arange(n), ep.ev_typ[e]] = 1.0
        X[i, :n, F_Y] = ep.y; X[i, :n, F_RELAY] = ep.q_relay
        it = ep.ev_items[e]
        X[i, :n, F_FUN:F_SRC] = np.einsum('nj,njd->nd', ep.ev_coef[e],
                                          ep.item_vec[np.maximum(it, 0)] * (it >= 0)[..., None])
        X[i, :n, F_SRC:F_SRC + D_SRC] = ep.src_vec[ep.q_src]
        X[i, :n, F_SRC + D_SRC] = ls[ep.q_src]; X[i, :n, F_SRC + D_SRC + 1] = ep.unk[ep.q_src]
        o = ep.ev_src[e]
        X[i, :n, F_OSRC:F_OSRC + D_SRC] = ep.src_vec[o]
        X[i, :n, F_OSRC + D_SRC] = ls[o]; X[i, :n, F_OSRC + D_SRC + 1] = ep.unk[o]
        X[i, :n, F_EV:F_EV + D_EV] = ep.ev_vec[e]
        X[i, T - 1, 3] = 1.0; X[i, T - 1, F_FUN:F_SRC] = ep.item_vec[ep.query]
        M[i, :n] = True; M[i, T - 1] = True
        Y[i] = ep.v[ep.query]
    return X, M, Y


# ------------------------------------------------------------------------------------------------ harness world
POLICIES = {'dir': {}, 'conc': dict(alpha=(0.03, 0.1)), 'relay': dict(p_relay=(0.5, 0.6))}
DIAG = ('c4', 'c7')
C7_ARMS = ('D', 'T', 'D4', 'T4', 'DR', 'DI', 'HR', 'HI', 'HR4', 'HI4')
C7_PAIRS = (('D', 'T'), ('D4', 'T4'), ('DR', 'DI'), ('HR', 'HI'), ('HR4', 'HI4'))


def c4_batch(rng, E, N, M=12):
    """E episodes = E//2 frequency-matched pairs (consecutive members), ~N quotes each."""
    out = []
    for p in range(max(1, E // 2)):
        a, b, delta, tries = c4_pair(rng, n_bg=max(20, N - 2 * M - 2), M=M, scale=N)
        for ep in (a, b):
            ep.meta.update(pair=p, tries=tries)
        out += [a, b]
    return out


def c7_batch(rng, E, N):
    """E episodes = E//10 sets of the ten C7 arms (consecutive), ~N quotes each."""
    out = []
    for p in range(max(1, E // len(C7_ARMS))):
        d = c7_set(rng, n_bg=max(20, N - 8), scale=N)
        for i, k in enumerate(C7_ARMS):
            d[k].meta.update(set=p, arm=i)
            out.append(d[k])
    return out


class MarketWorld:
    target_kind = 'real'
    name = 'world2_market'

    def __init__(self, cfg=None):
        c = dict(train_policies=['dir'], heldout=['conc', 'relay'])
        c.update(cfg or {})
        self.cfg = c
        self.feat_dim = FEAT
        self.train_policies = list(c['train_policies']); self.heldout = list(c['heldout'])
        self.test_policies = self.train_policies + [p for p in self.heldout if p not in self.train_policies]

    def _wrap(self, eps, policy, T=None):
        X, M, Y = pack(eps, T)
        return dict(X=X, mask=M, target=Y, eps=np.array(eps, dtype=object), policy=np.array([policy] * len(eps)))

    def sample(self, rng, E, N, policy, n=None):
        """policy in POLICIES (generator family) or 'c4' / 'c7' (diagnostic sets; E rounded down to whole
        pairs / sets, N = approximate quotes per episode)."""
        if policy == 'c4':
            return self._wrap(c4_batch(rng, E, N), policy)
        if policy == 'c7':
            return self._wrap(c7_batch(rng, E, N), policy)
        return self._wrap(sample_batch(rng, E, N if n is None else n, POLICIES[policy]), policy, T=N + 1)

    def sample_train(self, rng, E, nmin, nmax):
        """Training batch: policy 'dir', N_i log-uniform in [nmin, nmax]."""
        n = np.exp(rng.uniform(np.log(nmin), np.log(nmax + 1), E)).astype(np.int64).clip(nmin, nmax)
        return self._wrap(sample_batch(rng, E, n, POLICIES['dir']), 'dir', T=nmax + 1)

    @staticmethod
    def _stack(preds):
        k = max(len(p[0]) for p in preds); E = len(preds)
        w = np.zeros((E, k)); m = np.zeros((E, k)); v = np.ones((E, k))
        for i, (a, b, s) in enumerate(preds):
            w[i, :len(a)] = a; m[i, :len(a)] = b; v[i, :len(a)] = s ** 2
        return w, m, v

    def exact_bayes(self, ep):
        return self._stack([predictive(e, 'bayes') for e in ep['eps']])

    def references(self, ep):
        return {r: self._stack([predictive(e, r) for e in ep['eps']]) for r in REFS}

    def meta(self, ep):
        rows = []
        for e in ep['eps']:
            qtouch = np.array([(e.ev_items[x] == e.query).any() for x in e.q_ev])
            qspot = qtouch & (e.ev_typ[e.q_ev] == SPOT)
            hid_rel = e.q_relay & ~e.ev_vis[e.q_ev]
            md = e.meta
            rows.append((e.n, e.K, e.S, int(e.unk.sum()), int(e.q_relay.sum()), int(hid_rel.sum()), int(qspot.sum()),
                         int(qtouch.sum()), md.get('pair', -1), md.get('member', -1), md.get('set', -1),
                         md.get('arm', -1), md.get('delta', np.nan)))
        a = np.array(rows, float)
        names = ('n', 'K', 'S', 'U', 'n_relay', 'n_hidden_relay', 'q_nspot', 'q_ntouch', 'pair', 'member', 'set',
                 'arm', 'delta')
        d = {k: a[:, i] for i, k in enumerate(names)}
        d['policy'] = ep['policy']
        return d

    splits = {'all': lambda m: np.ones(len(m['n']), bool),
              'q_noSPOT': lambda m: m['q_nspot'] == 0,
              'unk': lambda m: m['U'] > 0,
              'relay': lambda m: m['n_relay'] > 0,
              'hidden_relay': lambda m: m['n_hidden_relay'] > 0}
    abilities = [('source inference (freq_only->bayes)', 'freq_only', 'bayes', 'unk'),
                 ('provenance dedup (prov_naive->bayes)', 'prov_naive', 'bayes', 'relay'),
                 ('bias tracking (bias_naive->bayes)', 'bias_naive', 'bayes', 'all'),
                 ('any (prior->bayes)', 'prior', 'bayes', 'all')]


def make_world(cfg=None):
    return MarketWorld(cfg)
