"""Tiny-PFN plumbing for round 1 (METHOD §2, §11): checkpoints, cut-and-resume forward with per-component caches,
the exact conditional sampler x' ~ p(x | c), token features phi (q0 / q0U / candidate library), and the G quadrature
G = KL(p_B(.|c) || f(h)) with the Bayes grid shared within a cell."""
import json, math, os, sys
import numpy as np
import torch

TP = '/home/agent/workspaces/mats/work/tiny-pfn'
TPI = '/home/agent/workspaces/mats/work/tiny-pfn-interp'
sys.path.insert(0, TP)
import world as W  # noqa: E402
import pfn as P    # noqa: E402

torch.set_num_threads(1)
S, Q, QL = W.S, W.Q, W.QL
T = S * Q + 1
QI = T - 1
DS = 16
LAB = [s * Q + j for s in range(S) for j in range(QL)]
QQ = [s * Q + QL for s in range(S)]
QT = [QI]
GROUPS = {'LAB': LAB, 'QQ': QQ, 'QT': QT}
TOK_SRC = np.array([t // Q for t in range(T - 1)] + [-1])
TOK_Q = np.array([t % Q for t in range(T - 1)] + [QL])
VC = W.VLEV[W.COMB]
PREC = 1 + (1 / VC).sum(1)

MODELS = {f'2L_s{s}': f'{TP}/runs/w32_h2_L2_mix4_ws_s{s}' for s in range(5)}
MODELS.update({f'3L_s{s}': f'{TPI}/heads/hruns/L3_ws_s{s}' for s in range(10, 15)})
MODELS['plat_s3'] = f'{TP}/runs/w32_h2_L2_mix4_s3'


def load(name):
    rd = MODELS[name]
    meta = json.load(open(os.path.join(rd, 'meta.json')))
    m = P.build(dict(width=meta['width'], heads=meta['heads'], depth=meta['depth'], attn_only=meta['attn_only'],
                     ds=meta['ds'], mixk=meta['mixk']))
    m.load_state_dict(torch.load(os.path.join(rd, 'model.pt'), map_location='cpu'))
    m.eval()
    ev = json.load(open(os.path.join(rd, 'eval.json')))
    return m, meta, ev


# ----------------------------------------------------------------------------- world
def sample_episodes(n, seed):
    rng = np.random.default_rng(seed)
    d = W.sample_np(n, rng)
    d['semb'] = (rng.standard_normal((n, S, DS)) / 4).astype(np.float64)
    d['qemb'] = (rng.standard_normal((n, Q, DS)) / 4).astype(np.float64)
    return d


def cond_draws(d, M, seed):
    """Exact conditional sampler: M draws per episode of x' ~ p(x | c), c = (query readings, labelled SS per source).
    Resamples IDs, residual directions at fixed SS_s and labelled y. Returns a flat episode dict of n*M draws
    (cell-major: draw j of cell i at i*M + j)."""
    rng = np.random.default_rng(seed)
    n = len(d['y'])
    rep = lambda a: np.repeat(a, M, axis=0)
    o, y = rep(d['o']).copy(), rep(d['y']).copy()
    N = n * M
    R = o[:, :, :QL] - y[:, None, :QL]
    ss = (R ** 2).sum(2, keepdims=True)
    u = rng.standard_normal((N, S, QL)); u /= np.linalg.norm(u, axis=2, keepdims=True)
    y[:, :QL] = rng.standard_normal((N, QL))
    o[:, :, :QL] = y[:, None, :QL] + np.sqrt(ss) * u
    return dict(o=o, y=y, v=rep(d['v']), vidx=rep(d['vidx']), semb=rng.standard_normal((N, S, DS)) / 4,
                qemb=rng.standard_normal((N, Q, DS)) / 4, cell=np.repeat(np.arange(n), M))


def tokens(d):
    X, m = W.assemble(d['semb'], d['qemb'], d['o'], d['y'])
    return X, m


# ----------------------------------------------------------------------------- Bayes-side quantities (functions of c)
def blind_weights(d):
    """(n,27) weights of the reference that keeps Q and only the unordered multiset U of the SS (binding paper eq. 2)."""
    R = d['o'][:, :, :QL] - d['y'][:, None, :QL]
    ss = (R ** 2).sum(2)
    ostar = d['o'][:, :, QL]
    quadq = np.einsum('ns,pst,nt->np', ostar, W.SIGQI, ostar)
    logq = -0.5 * quadq - 0.5 * W.LOGDETQ[None]
    import itertools
    lls = []
    for perm in itertools.permutations(range(S)):
        ll = 0
        for s in range(S):
            v = VC[:, s][None]
            ll = ll - 0.5 * ss[:, perm[s]][:, None] / v - 0.5 * QL * np.log(v)
        lls.append(ll)
    lls = np.stack(lls, 0)
    mx = lls.max(0)
    lperm = mx + np.log(np.exp(lls - mx).mean(0))
    lw = logq + lperm
    lw -= lw.max(1, keepdims=True)
    w = np.exp(lw)
    return w / w.sum(1, keepdims=True)


def cell_stats(d):
    """Per-episode Bayes-side quantities; all are functions of c (plus Q for the blind reference)."""
    out, nll, mom, wb = W.references(d)
    means, vars_, _ = out['bayes']
    wl = out['lab'][2]
    R = d['o'][:, :, :QL] - d['y'][:, None, :QL]
    ss = (R ** 2).sum(2)
    ll = -0.5 * ss[:, :, None] / W.VLEV - 0.5 * QL * np.log(2 * np.pi * W.VLEV)
    pl = np.exp(ll - ll.max(2, keepdims=True)); pl /= pl.sum(2, keepdims=True)
    pb = np.stack([np.stack([wb[:, W.COMB[:, s] == j].sum(1) for j in range(3)], 1) for s in range(S)], 1)
    El = (pl / W.VLEV).sum(2); Eb = (pb / W.VLEV).sum(2)
    beta = np.einsum('nc,cs->ns', wb, (1 / VC) / PREC[:, None])
    ostar = d['o'][:, :, QL]
    mb, vb = mom['bayes']; ml, vl = mom['lab']
    bcv = ((wb * (means - mb[:, None]) ** 2).sum(1)) / vb                     # between-component variance fraction
    ent = -(wb * np.log(np.maximum(wb, 1e-300))).sum(1)
    spread = ostar.max(1) - ostar.min(1)
    coup = (wb * (np.log(np.maximum(wb, 1e-300)) - np.log(np.maximum(wl, 1e-300)))).sum(1)
    wbl = blind_weights(d)
    mbl, vbl = W.mix_moments(means, vars_, wbl)
    return dict(comp_means=means, comp_vars=vars_, w_bayes=wb, ss=ss, lss=np.log(ss), pl=pl, pb=pb, El=El, Eb=Eb,
                beta=beta, ostar=ostar, mb=mb, lvb=np.log(vb), ml=ml, lvl=np.log(vl), bcv=bcv, ent=ent,
                spread=spread, coup=coup, mbl=mbl, lvbl=np.log(vbl), nll_bayes=nll['bayes'])


def take(cs, idx):
    return {k: (v if k == 'comp_vars' else v[idx]) for k, v in cs.items()}


# ----------------------------------------------------------------------------- features
def _cyc(a, s):
    """values of the two other sources, cyclic order: (n,S,...) -> (n, 2, ...) for source s."""
    return np.stack([a[:, (s + 1) % S], a[:, (s + 2) % S]], 1)


def tilted_pools(d, c1=1.6, c2=-0.4):
    R = d['o'][:, :, :QL] - d['y'][:, None, :QL]                               # (n,S,4)
    out = []
    for sg in (1, -1):
        lg = sg * c1 * R + c2 * R ** 2
        a = np.exp(lg - lg.max(2, keepdims=True)); a /= a.sum(2, keepdims=True)
        out += [(a * R).sum(2), (a * R ** 2).sum(2)]
    return np.stack(out + [R.mean(2), (R ** 3).mean(2)], 2)                    # (n,S,6)


FEATURE_SETS = ('Rmarg', 'Rin', 'q0', 'q0noID', 'q0U')
CANDIDATES = {'LAB': ('r2', 'pool', 'cross', 'idgram', 'episode', 'quadlat'),
              'QQ': ('pool', 'cross', 'idgram', 'episode', 'quadlat'),
              'QT': ('betaQ', 'idgram', 'episode', 'quadlat')}


def token_features(d, cs, ttype, fset, extra=()):
    """phi for every token of type ttype: returns (n, n_tok, k) array. fset in FEATURE_SETS; extra = candidate
    groups (CANDIDATES) appended. All features are functions of x (never of the true v)."""
    n = len(d['y'])
    idx = GROUPS[ttype]
    rows = []
    for t in idx:
        s, j = TOK_SRC[t], TOK_Q[t]
        f = []
        if fset != 'Rmarg':
            ids = [] if fset == 'q0noID' else ([d['semb'][:, s], d['qemb'][:, j]] if ttype != 'QT' else [d['qemb'][:, QL]])
            f += ids
            if ttype == 'LAB':
                f += [d['o'][:, s, j:j + 1], d['y'][:, j:j + 1]]
            elif ttype == 'QQ':
                f += [d['o'][:, s, QL:QL + 1]]
        if fset in ('q0', 'q0noID'):
            if ttype in ('LAB', 'QQ'):
                f += [cs['lss'][:, s:s + 1], cs['pl'][:, s], cs['El'][:, s:s + 1], cs['beta'][:, s:s + 1],
                      cs['ostar'][:, s:s + 1], (cs['ostar'][:, s] * cs['El'][:, s])[:, None]]
            else:
                f += [cs['ostar'].mean(1, keepdims=True), cs['mb'][:, None], cs['lvb'][:, None], cs['ml'][:, None],
                      cs['lvl'][:, None], (cs['beta'] * cs['ostar']).sum(1, keepdims=True)]
        if fset == 'q0U':
            f += [np.sort(cs['lss'], 1), cs['ostar'].mean(1, keepdims=True), cs['mbl'][:, None], cs['lvbl'][:, None]]
            if ttype == 'QQ' or ttype == 'LAB':
                f += [cs['ostar'][:, s:s + 1]]
        for g in extra:
            f += _cand(d, cs, ttype, g, s, j)
        rows.append(np.concatenate(f, 1) if f else np.zeros((n, 0)))
    return np.stack(rows, 1)


def _cand(d, cs, ttype, g, s, j):
    if g == 'r2':
        r = d['o'][:, s, j] - d['y'][:, j]
        return [np.stack([r ** 2, np.abs(r), r ** 3], 1)]
    if g == 'pool':
        if ttype == 'QT':
            P_ = tilted_pools(d); return [P_.mean(1)]
        return [tilted_pools(d)[:, s]]
    if g == 'cross':
        return [_cyc(cs['ostar'], s), _cyc(cs['lss'], s), cs['pb'][:, s], (cs['Eb'] - cs['El'])[:, s:s + 1]]
    if g == 'cross_o':                                   # post hoc split of 'cross': other sources' readings / SS
        return [_cyc(cs['ostar'], s), _cyc(cs['lss'], s)]
    if g == 'cross_p':                                   # post hoc split of 'cross': query-coupled own posterior
        return [cs['pb'][:, s], (cs['Eb'] - cs['El'])[:, s:s + 1]]
    if g == 'meansq':                                    # post hoc split of 'quadlat' at QT: squared predictive means
        bo = (cs['beta'] * cs['ostar']).sum(1)
        return [np.stack([cs['mb'] ** 2, bo ** 2, cs['mb'] * bo, cs['ml'] ** 2, cs['ostar'].mean(1) ** 2], 1)]
    if g == 'idgram':
        se, qe = d['semb'], d['qemb']
        if ttype == 'QT':
            qs = qe[:, QL]
            return [np.stack([(qs * qs).sum(1), sum((qs * se[:, k]).sum(1) for k in range(S)),
                              sum((qs * qe[:, k]).sum(1) for k in range(QL))], 1)]
        jj = QL if ttype == 'QQ' else j
        a = se[:, s]
        return [np.stack([(a * a).sum(1), (qe[:, jj] ** 2).sum(1), (a * se[:, (s + 1) % S]).sum(1),
                          (a * se[:, (s + 2) % S]).sum(1), (a * qe[:, QL]).sum(1), (qe[:, jj] * qe[:, QL]).sum(1)], 1)]
    if g == 'episode':
        return [np.stack([cs['bcv'], cs['ent'], cs['spread'], cs['coup'], cs['mb'], cs['lvb']], 1)]
    if g == 'betaQ':
        b, o = cs['beta'], cs['ostar']
        return [np.stack([(b * b).sum(1), (b * o * o).sum(1), (b * o).sum(1) ** 2, np.sort(b, 1)[:, 0],
                          np.sort(b, 1)[:, 2], (cs['Eb'] * o).sum(1) / cs['Eb'].sum(1)], 1)]
    if g == 'quadlat':
        if ttype == 'QT':
            L = np.stack([cs['ostar'].mean(1), cs['mb'], cs['lvb'], cs['ml'], cs['lvl'], (cs['beta'] * cs['ostar']).sum(1)], 1)
        else:
            L = np.stack([cs['lss'][:, s], cs['El'][:, s], cs['beta'][:, s], cs['ostar'][:, s]] +
                         [cs['pl'][:, s, k] for k in range(2)], 1)
        iu = np.triu_indices(L.shape[1])
        return [(L[:, :, None] * L[:, None, :])[:, iu[0], iu[1]]]
    raise ValueError(g)


# ----------------------------------------------------------------------------- network: cuts and components
def _allow(valid):
    B = valid.shape[0]
    ctx = valid.clone(); ctx[:, QI] = False
    allow = ctx[:, None, :].expand(B, T, T).clone(); allow[:, QI, QI] = True
    eye = torch.eye(T, dtype=torch.bool)[None]
    return allow | (eye & ~valid[:, :, None])


def _attn_heads(blk, x, allow):
    """returns per-head contributions (B,H,T,d) after the output projection (no bias) and the bias."""
    B, Tn, d = x.shape
    q, k, v = blk.qkv(blk.ln1(x)).split(d, -1)
    dh = d // blk.h
    sp = lambda t: t.view(B, Tn, blk.h, dh).transpose(1, 2)
    q, k, v = sp(q), sp(k), sp(v)
    s = (q @ k.transpose(-1, -2) / math.sqrt(dh)).masked_fill(~allow[:, None], float('-inf'))
    a = s.softmax(-1)
    z = a @ v                                                                   # (B,H,T,dh)
    Wo = blk.o.weight                                                           # (d, H*dh)
    contrib = torch.stack([z[:, h] @ Wo[:, h * dh:(h + 1) * dh].T for h in range(blk.h)], 1)
    return contrib, blk.o.bias


def cuts_of(model):
    L = len(model.blocks)
    return [('post', k) for k in range(L + 1)] + [('attn', k) for k in range(1, L + 1)]


def run_real(model, X, valid, want_cuts, comp_cuts=None, bs=4000):
    """Real forward. Returns head (mu, logs, logits) and dict cut -> (N,T,d) float32 states for want_cuts,
    and comp[cut] -> (N,T,C,d) per-component decomposition of the state at ('post',k) cuts:
    [state before block k (+ attention bias), head_0, ..., head_{H-1}, mlp_k]."""
    if not isinstance(want_cuts, dict):
        want_cuts = {c: None for c in want_cuts}
    comp_cuts = comp_cuts or {}
    if not isinstance(comp_cuts, dict):
        comp_cuts = {c: None for c in comp_cuts}
    sel = lambda a, ix: a if ix is None else a[:, ix]
    heads, st, comp = [], {c: [] for c in want_cuts}, {c: [] for c in comp_cuts}
    with torch.no_grad():
        for i in range(0, len(X), bs):
            Xb, vb = torch.from_numpy(X[i:i + bs]), torch.from_numpy(valid[i:i + bs])
            al = _allow(vb)
            x = model.inp(Xb)
            if ('post', 0) in st: st[('post', 0)].append(sel(x, want_cuts[('post', 0)]).numpy())
            for k, blk in enumerate(model.blocks, 1):
                contrib, bias = _attn_heads(blk, x, al)
                prev = x + bias
                x = prev + contrib.sum(1)
                if ('attn', k) in st: st[('attn', k)].append(sel(x, want_cuts[('attn', k)]).numpy())
                m = blk.mlp(blk.ln2(x))
                x = x + m
                if ('post', k) in st: st[('post', k)].append(sel(x, want_cuts[('post', k)]).numpy())
                if ('post', k) in comp:
                    comp[('post', k)].append(sel(torch.cat([prev[:, :, None], contrib.transpose(1, 2), m[:, :, None]], 2), comp_cuts[('post', k)]).numpy())
            heads.append(_head(model, x))
    head = tuple(np.concatenate([h[j] for h in heads]) for j in range(3))
    return head, {c: np.concatenate(v) for c, v in st.items()}, {c: np.concatenate(v) for c, v in comp.items()}


def _head(model, x):
    y = model.out(model.ln(x[:, QI])).double().numpy()
    K = model.mixk
    return (y[:, K:2 * K], np.clip(y[:, 2 * K:], -6, 3), y[:, :K])


def forward_from(model, h, cut, valid, bs=4000):
    """Resume the real network from state h (N,T,d) at cut. ('attn',k): before mlp_k."""
    kind, k = cut
    L = len(model.blocks)
    out = []
    with torch.no_grad():
        for i in range(0, len(h), bs):
            x = torch.from_numpy(np.ascontiguousarray(h[i:i + bs], dtype=np.float32))
            vb = torch.from_numpy(valid[i:i + bs]); al = _allow(vb)
            start = k + 1
            if kind == 'attn':
                blk = model.blocks[k - 1]
                x = x + blk.mlp(blk.ln2(x))
            for kk in range(start, L + 1):
                x, _ = model.blocks[kk - 1](x, al)
            out.append(_head(model, x))
    return tuple(np.concatenate([o[j] for o in out]) for j in range(3))


def forward_qt(model, hqt, cut):
    """Final-layer cuts: only the query token is read. hqt (N, d) or (N,1,d)."""
    x = torch.from_numpy(np.ascontiguousarray(hqt.reshape(len(hqt), -1), dtype=np.float32))
    with torch.no_grad():
        if cut[0] == 'attn':
            blk = model.blocks[cut[1] - 1]
            x = x + blk.mlp(blk.ln2(x))
        y = model.out(model.ln(x)).double().numpy()
    K = model.mixk
    return (y[:, K:2 * K], np.clip(y[:, 2 * K:], -6, 3), y[:, :K])


def push_block(model, h, k_to, valid, bs=4000):
    """Run real block k_to on full states h (N,T,d) (state after block k_to - 1); returns (N,T,d)."""
    out = []
    with torch.no_grad():
        for i in range(0, len(h), bs):
            x = torch.from_numpy(np.ascontiguousarray(h[i:i + bs], dtype=np.float32))
            al = _allow(torch.from_numpy(valid[i:i + bs]))
            x, _ = model.blocks[k_to - 1](x, al)
            out.append(x.numpy())
    return np.concatenate(out)


def head_vec(head):
    mu, ls, lg = head
    return np.concatenate([mu, ls, lg - lg.mean(1, keepdims=True)], 1)


# ----------------------------------------------------------------------------- quotient coordinates
def needs_norm(model, cut, ttype):
    """Context tokens at the second-to-last post-block cut are read only through LayerNorm by the last block's
    attention: invariant to shift and positive scale. Everything else: shift only."""
    L = len(model.blocks)
    return cut == ('post', L - 1) and ttype != 'QT'


def to_q(h, norm):
    h = h - h.mean(-1, keepdims=True)
    if norm:
        h = h / (h.std(-1, keepdims=True) + 1e-8)
    return h


# ----------------------------------------------------------------------------- G quadrature
NGRID = 201


class GQuad:
    """G = KL(p_B(.|c) || model) per draw; grid = Bayes support +-9 sd, shared by all draws of a cell."""

    def __init__(self, cs):
        means, vars_, w = cs['comp_means'], cs['comp_vars'], cs['w_bayes']
        sd = np.sqrt(vars_)
        lo = (means - 9 * sd).min(1); hi = (means + 9 * sd).max(1)
        self.g = lo[:, None] + (hi - lo)[:, None] * np.linspace(0, 1, NGRID)[None]
        lp = np.log(np.maximum(w, 1e-300))[:, None, :] - 0.5 * np.log(2 * np.pi * vars_) \
            - 0.5 * (self.g[:, :, None] - means[:, None]) ** 2 / vars_
        self.lp = np.logaddexp.reduce(lp, axis=2)
        dx = (hi - lo) / (NGRID - 1)
        wt = np.ones(NGRID); wt[0] = wt[-1] = 0.5
        self.pw = np.exp(self.lp) * wt[None] * dx[:, None]                     # trapezoid weights * p_B
        self.plp = (self.pw * self.lp).sum(1)

    def __call__(self, head, cell, chunk=20000):
        mu, ls, lg = head
        lgn = lg - lg.max(1, keepdims=True); lgn = lgn - np.log(np.exp(lgn).sum(1, keepdims=True))
        out = np.empty(len(mu))
        for i in range(0, len(mu), chunk):
            sl = slice(i, i + chunk); c = cell[sl]
            g = self.g[c]
            lq = lgn[sl][:, None, :] - ls[sl][:, None, :] - 0.5 * math.log(2 * math.pi) \
                - 0.5 * (g[:, :, None] - mu[sl][:, None, :]) ** 2 / np.exp(2 * ls[sl])[:, None, :]
            lq = np.logaddexp.reduce(lq, axis=2)
            out[sl] = self.plp[c] - (self.pw[c] * lq).sum(1)
        return out
