"""Round 2 (PREREG amendment 4): information-matched, on-manifold oracles.

For an evaluation draw x' and a token t, an oracle variant V draws a fresh world episode x''_t ~ p(x | psi_V,t(x'))
and the site's token t is replaced by the network's REAL state h_t(x'') at the cut (rest of x' real). Nothing is
fitted and no state is off the manifold. Per-token draws are independent given x', which is the factorisation q0
assumes (tokens independent given their own phi_t). Variant J draws one x'' for the whole site.

World (tiny-pfn/world.py): y_j ~ N(0,1) iid (j < 4 labelled, j = 4 the hidden query y*); v_s iid uniform on
{0.1, 0.5, 2}; o_sj = y_j + e_sj, e_sj ~ N(0, v_s); IDs semb (3x16), qemb (5x16) iid N(0, 1/16).
c = (Q, S): Q_s = o_s4, S_s = sum_j<4 (o_sj - y_j)^2. Given (Q, S): y_lab ~ N(0, I), the residual directions
r_s / |r_s| are uniform on S^3, IDs are iid - all independent of (v, y*). Every sampler below is exact except the
beta match of O_q0 (nearest neighbour among K exact candidates; its error is returned and reported).

Conditioning sets psi_V,t for a token t of source s (LAB t = (s, j); QQ_s; QT has no source):
  raw_t = the token's own inputs: IDs (semb_s, qemb_j) and values (LAB: o_sj, y_j; QQ: Q_s; QT: qemb_4 only)
  lat   : c only (no token raw: IDs fresh)                                   "Bayes latents only"
  own   : raw_t + own-source (Q_s, S_s)                                      (QT: = c)
  q0    : own + beta_s (q0's exact information: q0 reads IDs_t, raw_t, Q_s, functions of S_s, beta_s)  (QT: = c)
  c     : raw_t + c
  U     : raw_t + (Q, unordered S); the SS-to-source assignment is drawn from its exact posterior
  q0ID  : q0 + all IDs (for the order check of the cross-source and ID steps)        (QT: = cID)
  cID   : raw_t + c + all IDs
  cIDr  : cID + own source's full residual vector + y_lab                   (QT: = cID)
  J     : (whole site, one draw) c + all IDs; residual directions and y_lab resampled jointly
"""
import itertools, math
import numpy as np
import torch
from . import tp

W = tp.W
S, Q, QL, DS = tp.S, tp.Q, tp.QL, tp.DS
VLEV = W.VLEV
VC = W.VC                                   # (27, 3)
PREC = W.PREC                               # (27,)
BCOEF = (1.0 / VC) / PREC[:, None]          # beta_s = sum_k w_k BCOEF[k, s]
PERMS = np.array(list(itertools.permutations(range(S))))   # (6, 3)
VARIANTS = ('lat', 'own', 'q0', 'q0ID', 'c', 'U', 'cID', 'cIDr')


# ----------------------------------------------------------------------------- Bayes helpers
IVC = (1.0 / VC).T                          # (3, 27)
LCONST = -2.0 * np.log(VC).sum(1) - 0.5 * W.LOGDETQ


def bayes_logw(Qc, Sc):
    """(M, 27) unnormalised log posterior over the 27 variance combinations given c = (Qc, Sc) (uniform prior).
    Labelled part: sum_s -S_s/(2 v_s) - 2 log v_s. Query part: N(Q; 0, diag(v) + 11^T), whose inverse is
    D^-1 - D^-1 1 1^T D^-1 / PREC (Sherman-Morrison), so everything is two (M,3)x(3,27) products."""
    a = Qc @ IVC
    quad = (Qc * Qc) @ IVC - a * a / PREC[None]
    return -0.5 * (Sc @ IVC) - 0.5 * quad + LCONST[None]


def _bayes_logw_ref(Qc, Sc):
    llc = (-0.5 * Sc[:, None, :] / VC[None] - 2.0 * np.log(VC)[None]).sum(2)
    quad = np.einsum('ms,kst,mt->mk', Qc, W.SIGQI, Qc)
    return llc - 0.5 * quad - 0.5 * W.LOGDETQ[None]


def beta_of(Qc, Sc, chunk=4096):
    """(M, 3) Bayes precision weights beta_s(c) (identical to tp.cell_stats['beta'])."""
    out = np.empty(Qc.shape)
    for i in range(0, len(Qc), chunk):
        lw = bayes_logw(Qc[i:i + chunk], Sc[i:i + chunk])
        lw -= lw.max(1, keepdims=True)
        w = np.exp(lw); w /= w.sum(1, keepdims=True)
        out[i:i + chunk] = w @ BCOEF
    return out


def log_evidence(Qc, Sc, chunk=4096):
    """(M,) log p(Q, S) up to a constant common to all c (logsumexp of bayes_logw over the 27 combinations)."""
    out = np.empty(len(Qc))
    for i in range(0, len(Qc), chunk):
        lw = bayes_logw(Qc[i:i + chunk], Sc[i:i + chunk])
        m = lw.max(1, keepdims=True)
        out[i:i + chunk] = m[:, 0] + np.log(np.exp(lw - m).sum(1))
    return out


def cvars(d):
    R = d['o'][:, :, :QL] - d['y'][:, None, :QL]
    return d['o'][:, :, QL].copy(), (R ** 2).sum(2), R


# ----------------------------------------------------------------------------- building blocks
def _unit(rng, shape):
    u = rng.standard_normal(shape)
    return u / np.linalg.norm(u, axis=-1, keepdims=True)


def build(Qc, Sc, y, semb, qemb, rng, keep_coord=None, keep_vec=None):
    """Assemble x'' from c'' = (Qc, Sc), labels y (N, Q) and IDs. Residual vectors have |r_s|^2 = Sc_s and uniform
    direction, except: keep_coord = (s, j, r) fixes coordinate j of source s at r (rest uniform on the sphere of
    radius sqrt(Sc_s - r^2)); keep_vec = (s, R_s) fixes source s's whole residual vector (|R_s|^2 must equal Sc_s)."""
    N = len(Qc)
    R = np.sqrt(Sc)[:, :, None] * _unit(rng, (N, S, QL))
    if keep_coord is not None:
        s, j, r = keep_coord
        rest = [k for k in range(QL) if k != j]
        rad = np.sqrt(np.maximum(Sc[:, s] - r ** 2, 0.0))
        tmp = np.empty((N, QL))
        tmp[:, rest] = rad[:, None] * _unit(rng, (N, QL - 1))
        tmp[:, j] = r
        R[:, s] = tmp
    if keep_vec is not None:
        s, Rs = keep_vec
        R[:, s] = Rs
    o = np.empty((N, S, Q))
    o[:, :, :QL] = y[:, None, :QL] + R
    o[:, :, QL] = Qc
    return dict(o=o, y=y, semb=semb, qemb=qemb)


def _ids(d, rng, keep):
    """keep in {'none', 'own', 'all'}; own needs (s, j) handled by the caller."""
    N = len(d['y'])
    if keep == 'all':
        return d['semb'].copy(), d['qemb'].copy()
    return rng.standard_normal((N, S, DS)) / 4, rng.standard_normal((N, Q, DS)) / 4


def _post_own_ystar(Qs, Ss, rng, K):
    """K exact draws of y* | (Q_s, S_s) for one source: v_s ~ p(v | Q_s, S_s), y* ~ N(Q_s/(1+v), v/(1+v)).
    Q_s ~ N(0, 1 + v) and S_s/v ~ chi2_4 are independent given v."""
    lv = (-0.5 * Ss[:, None] / VLEV - 2 * np.log(VLEV) - 0.5 * Qs[:, None] ** 2 / (1 + VLEV) - 0.5 * np.log(1 + VLEV))
    lv -= lv.max(1, keepdims=True)
    p = np.exp(lv); p /= p.sum(1, keepdims=True)
    cdf = np.cumsum(p, 1)
    u = rng.random((len(Qs), K))
    vi = (u[:, :, None] > cdf[:, None, :]).sum(2)
    vi = np.minimum(vi, len(VLEV) - 1)
    v = VLEV[vi]
    return Qs[:, None] / (1 + v) + np.sqrt(v / (1 + v)) * rng.standard_normal((len(Qs), K))


def _other_sources(ystar, rng):
    """Given y* (N, K): the two other sources from the prior: v ~ U{levels}, Q = y* + N(0, v), S = v * chi2_4."""
    N, K = ystar.shape
    v = VLEV[rng.integers(0, len(VLEV), (N, K, S - 1))]
    Qo = ystar[:, :, None] + np.sqrt(v) * rng.standard_normal((N, K, S - 1))
    So = v * (rng.standard_normal((N, K, S - 1, QL)) ** 2).sum(3)
    return Qo, So


def own_candidates(Qx, Sx, s, rng, K):
    """K exact candidates of the full c'' given source s's (Q_s, S_s): returns Qc, Sc (N, K, 3)."""
    N = len(Qx)
    ystar = _post_own_ystar(Qx[:, s], Sx[:, s], rng, K)
    Qo, So = _other_sources(ystar, rng)
    others = [k for k in range(S) if k != s]
    Qc = np.empty((N, K, S)); Sc = np.empty((N, K, S))
    Qc[:, :, s] = Qx[:, s][:, None]; Sc[:, :, s] = Sx[:, s][:, None]
    Qc[:, :, others] = Qo; Sc[:, :, others] = So
    return Qc, Sc


def lab_coord_logpdf(r, Ssrc):
    """log density of one labelled residual coordinate r given its source's SS (uniform on S^3 of radius sqrt(S)):
    f(r | S) = (2/pi) sqrt(1 - r^2/S) / sqrt(S) on |r| < sqrt(S)."""
    z = 1.0 - r ** 2 / Ssrc
    out = math.log(2 / math.pi) + 0.5 * np.log(np.maximum(z, 1e-300)) - 0.5 * np.log(Ssrc)
    return np.where(z > 0, out, -np.inf)


def unordered_assign(Qx, Sx, rng, lab=None):
    """Draw S'' = S o pi with pi ~ p(pi | Q, U) (U = multiset of S). lab = (s, r): the token's own residual coordinate r
    of source s is also conditioned on (it constrains S''_s). Returns S'' (N, 3) and the posterior prob of pi = id."""
    N = len(Qx)
    lps = []
    for pi in PERMS:
        Sp = Sx[:, pi]
        lp = log_evidence(Qx, Sp)
        if lab is not None:
            s, r = lab
            lp = lp + lab_coord_logpdf(r, Sp[:, s])
        lps.append(lp)
    lps = np.stack(lps, 1)
    lps -= lps.max(1, keepdims=True)
    p = np.exp(lps); p /= p.sum(1, keepdims=True)
    cdf = np.cumsum(p, 1)
    k = np.minimum((rng.random(N)[:, None] > cdf).sum(1), len(PERMS) - 1)
    Snew = np.take_along_axis(Sx, PERMS[k], 1)
    return Snew, p[:, 0]


# ----------------------------------------------------------------------------- the per-token samplers
def draw(d, t, variant, rng, K=32, beta_x=None, want_own=False):
    """One x''_t per draw of d for token t (t = tp.QI for the query token), variant in VARIANTS.
    Returns (x'' dict, info dict). For the beta-matched variants ('q0', 'q0ID') info['beta_err'] is the matching error
    and, if want_own, info['own'] is the unmatched candidate 0, an exact 'own' draw sharing labels, IDs and the own
    residual coordinate with x''."""
    N = len(d['y'])
    Qx, Sx, Rx = cvars(d)
    is_qt = t == tp.QI
    s, j = (int(tp.TOK_SRC[t]), int(tp.TOK_Q[t])) if not is_qt else (-1, QL)
    is_lab = (not is_qt) and j < QL
    if is_qt and variant in ('own', 'q0'):
        variant = 'c'                                  # q0 at QT reads 6 permutation-invariant functions of c
    if is_qt and variant in ('cIDr', 'q0ID'):
        variant = 'cID'                                # no own source
    info = {}
    all_ids = variant in ('cID', 'cIDr', 'q0ID')
    semb, qemb = _ids(d, rng, 'all' if all_ids else 'none')
    if variant != 'lat' and not all_ids:
        if not is_qt:
            semb[:, s] = d['semb'][:, s]
        qemb[:, j] = d['qemb'][:, j]
    y = rng.standard_normal((N, Q)); y[:, QL] = 0.0
    if variant == 'cIDr':
        y[:, :QL] = d['y'][:, :QL]
    elif is_lab and variant != 'lat':
        y[:, j] = d['y'][:, j]
    keep_coord = (s, j, Rx[:, s, j]) if (is_lab and variant not in ('lat', 'cIDr')) else None
    keep_vec = (s, Rx[:, s]) if (variant == 'cIDr' and not is_qt) else None
    if variant in ('lat', 'c', 'cID', 'cIDr'):
        Qc, Sc = Qx, Sx
    elif variant == 'U':
        Sc, pid = unordered_assign(Qx, Sx, rng, lab=(s, Rx[:, s, j]) if is_lab else None)
        Qc = Qx
        info['p_identity'] = pid
    elif variant == 'own':
        Qk, Sk = own_candidates(Qx, Sx, s, rng, 1)
        Qc, Sc = Qk[:, 0], Sk[:, 0]
    elif variant in ('q0', 'q0ID'):
        Qk, Sk = own_candidates(Qx, Sx, s, rng, K)
        bt = beta_x if beta_x is not None else beta_of(Qx, Sx)[:, s]
        bk = beta_of(Qk.reshape(-1, S), Sk.reshape(-1, S))[:, s].reshape(N, K)
        pick = np.abs(bk - bt[:, None]).argmin(1)
        Qc = Qk[np.arange(N), pick]; Sc = Sk[np.arange(N), pick]
        info['beta_err'] = bk[np.arange(N), pick] - bt
        info['beta_sd_cand'] = bk.std(1)
        if want_own:
            info['own'] = build(Qk[:, 0], Sk[:, 0], y.copy(), semb.copy(), qemb.copy(), rng, keep_coord=keep_coord)
    else:
        raise ValueError(variant)
    return build(Qc, Sc, y, semb, qemb, rng, keep_coord=keep_coord, keep_vec=keep_vec), info


def draw_joint(d, rng):
    """J: one x'' per draw for the whole site, c and all IDs kept; y_lab and residual directions resampled."""
    Qx, Sx, _ = cvars(d)
    N = len(d['y'])
    y = rng.standard_normal((N, Q)); y[:, QL] = 0.0
    return build(Qx, Sx, y, d['semb'].copy(), d['qemb'].copy(), rng)


# ----------------------------------------------------------------------------- network: states of one token at several cuts
def forward_token(model, X, V, tok, cuts, bs=4000):
    """Real forward of episodes X; returns {cut: (N, d) float32 state of token tok}. cuts: ('post', k) / ('attn', k)."""
    L = len(model.blocks)
    need = max(k for _, k in cuts)
    out = {c: [] for c in cuts}
    with torch.no_grad():
        for i in range(0, len(X), bs):
            Xb, vb = torch.from_numpy(X[i:i + bs]), torch.from_numpy(V[i:i + bs])
            al = tp._allow(vb)
            x = model.inp(Xb)
            if ('post', 0) in out:
                out[('post', 0)].append(x[:, tok].numpy())
            for k, blk in enumerate(model.blocks, 1):
                if k > need:
                    break
                B_, T_, dm = x.shape
                q, kk, v = blk.qkv(blk.ln1(x)).split(dm, -1)
                sp = lambda z: z.view(B_, T_, blk.h, dm // blk.h).transpose(1, 2)
                q, kk, v = sp(q), sp(kk), sp(v)
                sc = (q @ kk.transpose(-1, -2) / math.sqrt(dm // blk.h)).masked_fill(~al[:, None], float('-inf'))
                x = x + blk.o((sc.softmax(-1) @ v).transpose(1, 2).reshape(B_, T_, dm))
                if ('attn', k) in out:
                    out[('attn', k)].append(x[:, tok].numpy())
                x = x + blk.mlp(blk.ln2(x))
                if ('post', k) in out:
                    out[('post', k)].append(x[:, tok].numpy())
    return {c: np.concatenate(v) for c, v in out.items()}
