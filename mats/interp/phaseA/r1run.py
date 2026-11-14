"""Round 1 on one tiny-PFN checkpoint (PREREG.md). usage: python r1run.py MODEL [--quick]
Writes r1/MODEL.json. Steps: (1) data + real forward on fit / eval-A / eval-B splits; (2) per-site rung ladder
(Rmarg, Rin, q0, q0U, q0noID) with direct Delta / Delta_rms on A; (3) output-space ratio for q0 at every site:
L(alpha), L(0)-L(1), Cov_q(G,w), ESS, T1/T2; (4) localisation on A: component (head) knock-out/in, relative-Fisher
subspace knock-out/in vs random; (5) naming from where w is large -> q1 per token type, accepted on fresh B with DL;
(6) cross-layer consistency tier i/ii on B."""
import json, math, os, sys, time
import numpy as np

sys.path.insert(0, '/home/agent/workspaces/mats/work/cm-interp')
from lib import tp                                  # noqa: E402
from lib import r1est as E                          # noqa: E402
from cmlib.ratio import fit_ratio                   # noqa: E402
from cmlib import dl as DL                          # noqa: E402
from cmlib.stats import auc as AUC                  # noqa: E402

NAME = sys.argv[1]
QUICK = '--quick' in sys.argv
NFIT, NC, M = (4000, 800, 8) if QUICK else (20000, 6000, 8)
REPS = 2
OUT = f'/home/agent/workspaces/mats/work/cm-interp/r1/{NAME}{"_quick" if QUICK else ""}.json'
LIB_SIZE = 48
T0 = time.time()
log = lambda *a: print(f'[{NAME} {time.time() - T0:6.0f}s]', *a, flush=True)

model, meta, ev = tp.load(NAME)
L = meta['depth']
MULTI = [('post', k) for k in range(1, L)]
FINAL = [('attn', L), ('post', L)]
want = {c: None for c in MULTI}
want.update({c: [tp.QI] for c in FINAL})
comp_cuts = {('post', k): None for k in range(1, L)}
comp_cuts[('post', L)] = [tp.QI]
cname = lambda c: f"{'h' if c[0] == 'post' else 'a'}{c[1]}"
res = dict(model=NAME, meta={k: meta[k] for k in ('cfg', 'seed', 'depth', 'params')}, eval_kl=ev['best']['kl'],
           sizes=dict(NFIT=NFIT, NC=NC, M=M, REPS=REPS))


def prep(seed, n, M_, comps):
    d0 = tp.sample_episodes(n, seed)
    cs0 = tp.cell_stats(d0)
    if M_:
        d = tp.cond_draws(d0, M_, seed + 1)
    else:
        d = d0; d['cell'] = np.arange(n)
    X, V = tp.tokens(d)
    head, st, comp = tp.run_real(model, X, V, want, comp_cuts if comps else None)
    GQ = tp.GQuad(cs0)
    Gp = GQ(head, d['cell']).reshape(n, max(M_, 1))
    return dict(d=d, cs=tp.take(cs0, d['cell']), cs0=cs0, V=V, head=head, st=st, comp=comp, GQ=GQ, cell=d['cell'], Gp=Gp,
                n=n, M=max(M_, 1))


F = prep(60000, NFIT, 0, True)
A = prep(61000, NC, M, True)
B = prep(63000, NC, M, False)
KLm = float(A['Gp'].mean())
res['kl_model_A'] = KLm; res['kl_model_B'] = float(B['Gp'].mean())
log('data ready; KL_model A %.5f B %.5f eval.json %.5f' % (KLm, res['kl_model_B'], ev['best']['kl']))


def tok_idx(cut, t):
    return [0] if cut in FINAL else tp.GROUPS[t]


def real_states(P, cut, t):
    return tp.to_q(P['st'][cut][:, tok_idx(cut, t)].astype(np.float64), tp.needs_norm(model, cut, t))


def types_at(cut):
    return ['QT'] if cut in FINAL else ['LAB', 'QQ', 'QT']


def run_down(P, cut, h):
    if cut in FINAL:
        return tp.forward_qt(model, h, cut)
    return tp.forward_from(model, h, cut, P['V'])


def G_of(P, head):
    return P['GQ'](head, P['cell']).reshape(P['n'], P['M'])


def site_eval(P, cut, repl, return_head=False):
    """repl: dict ttype -> (n, nt, d) replacement states (quotient coordinates)."""
    h = P['st'][cut].copy()
    for t, Hs in repl.items():
        h[:, tok_idx(cut, t)] = Hs
    head = run_down(P, cut, h)
    G = G_of(P, head)
    return (G, head) if return_head else G


def site_types(cut, site):
    return types_at(cut) if site == 'ALL' else [site]


SITES = [(c, s) for c in MULTI for s in ('LAB', 'QQ', 'QT', 'ALL')] + [(c, 'QT') for c in FINAL]
sname = lambda c, s: f'{cname(c)}-{s}'

# ----------------------------------------------------------------------------- (2) fits + ladder
Q = {}


def fit_q(cut, t, fset, extra=(), hetero=False, lam=None, P=None):
    P = P or F
    q = E.QTok(t, fset, extra, norm=tp.needs_norm(model, cut, t), hetero=hetero, lam=lam)
    return q.fit(P['d'], P['cs'], real_states(P, cut, t))


RUNGS = ['Rmarg', 'Rin', 'q0', 'q0U', 'q0noID']
for cut in MULTI + FINAL:
    for t in types_at(cut):
        for r in RUNGS:
            if r == 'q0noID' and cut in FINAL:
                continue
            Q[(cut, t, r)] = fit_q(cut, t, r)
log('fitted ladder q families')

# V1 at the embedding: R_in is exact (embedding is linear in the raw inputs)
v1 = {}
Hemb = tp.run_real(model, *tp.tokens(F['d']), {('post', 0): None})[1][('post', 0)]
for t in ('LAB', 'QQ', 'QT'):
    Ht = tp.to_q(Hemb[:, tp.GROUPS[t]].astype(np.float64), False)
    qe = E.QTok(t, 'Rin').fit(F['d'], F['cs'], Ht)
    mu, covs, _ = qe.mean_cov(F['d'], F['cs'])
    v1[t] = dict(max_abs_resid=float(np.abs(Ht - mu).max()), trace_cov=float(sum(np.trace(C) for C in covs.values())))
res['V1'] = v1
del Hemb

direct = {}
GQ_A = {}
rng = np.random.default_rng(7)
for cut, site in SITES:
    for r in RUNGS:
        if r == 'q0noID' and (cut in FINAL):
            continue
        repl = {t: Q[(cut, t, r)].sample(A['d'], A['cs'], rng) for t in site_types(cut, site)}
        G, head = site_eval(A, cut, repl, True)
        direct[(sname(cut, site), r)] = E.direct(G, A['Gp'])
        if r == 'q0':
            GQ_A[(cut, site)] = (G, head)
    log('ladder', sname(cut, site), {r: round(direct[(sname(cut, site), r)]['delta'], 5) for r in RUNGS if (sname(cut, site), r) in direct})
res['direct_A'] = {f'{k[0]}|{k[1]}': v for k, v in direct.items()}
json.dump(res, open(OUT, 'w'), default=float)

# ----------------------------------------------------------------------------- (3) output-space ratio for q0
def zc(P):
    cs = P['cs']
    return np.concatenate([cs['ostar'], cs['lss'], cs['mb'][:, None], cs['lvb'][:, None]], 1)


ZA = zc(A)
HVp = tp.head_vec(A['head'])
ratio = {}
RM_OUT = {}
for cut, site in SITES:
    G, head = GQ_A[(cut, site)]
    Fp = np.concatenate([HVp, ZA], 1); Fq = np.concatenate([tp.head_vec(head), ZA], 1)
    reps = E.crossfit_cells(Fp, Fq, A['cell'], A['n'], reps=REPS, seed=11)
    blocks, diags = [], []
    for lwp, lwq, dg, _ in reps:
        blocks.append(E.ratio_block(A['Gp'], G, lwp.reshape(A['n'], M), lwq.reshape(A['n'], M)))
        diags.append(dg)
    out = E.combine_reps(blocks)
    out['diag'] = {k: float(np.mean([d[k] for d in diags])) for k in diags[0]}
    out['L_kelvin_q'] = np.mean([b['L_kelvin_q'] for b in blocks], 0).tolist()
    dr = direct[(sname(cut, site), 'q0')]
    out['direct_delta'] = dr['delta']; out['direct_se'] = dr['se']
    # T1: bridge endpoints vs direct means
    tol = lambda se: max(2 * se, 0.05 * abs(dr['delta']))
    out['T1_L0'] = bool(abs(out['L'][0] - dr['Lq']) <= tol(math.hypot(out['L_se'][0], dr['se'])))
    out['T1_L1'] = bool(abs(out['L'][-1] - dr['Lp']) <= tol(math.hypot(out['L_se'][-1], dr['se'])))
    out['T2'] = bool(abs(out['Eq_w'] - 1) <= 0.05 and abs(out['Ep_invw'] - 1) <= 0.05)
    out['V5'] = bool(abs(-out['integral'] - dr['delta']) <= 2 * math.hypot(out['integral_se'], dr['se']) + 0.05 * abs(dr['delta']))
    ratio[sname(cut, site)] = out
    # keep out-of-fold log w (rep 0) of real draws for naming / regions
    GQ_A[(cut, site)] = (G, head, reps[0][0].reshape(A['n'], M), reps[0][1].reshape(A['n'], M))
    RM_OUT[(cut, site)] = reps[0][3][0][0]
    log('ratio', sname(cut, site), 'auc %.3f gap %.5f cov %.5f direct %.5f ess %.3f' % (
        out['diag']['auc'], out['gap'], out['cov_plain'], dr['delta'], out['ess_q_frac']))
res['ratio_A'] = ratio
json.dump(res, open(OUT, 'w'), default=float)

# regions: per-cell gap by quintile of episode statistics (q0, every site) and by decile of out-of-fold log w_out
regions = {}
for cut, site in SITES:
    G, head, lwp, lwq = GQ_A[(cut, site)]
    dcell = (G - A['Gp']).mean(1)
    rr = {}
    for stat in ('bcv', 'spread', 'coup', 'ent'):
        v = A['cs0'][stat]; edges = np.quantile(v, [.2, .4, .6, .8]); b = np.digitize(v, edges)
        rr[stat] = [float(dcell[b == k].sum() / dcell.sum()) if dcell.sum() != 0 else 0 for k in range(5)]
    lwc = lwp.mean(1); edges = np.quantile(lwc, np.linspace(.1, .9, 9)); b = np.digitize(lwc, edges)
    rr['lw_out_decile_share'] = [float(dcell[b == k].sum() / dcell.sum()) for k in range(10)]
    regions[sname(cut, site)] = rr
res['regions_A'] = regions

# ----------------------------------------------------------------------------- (4) localisation
loc = {}
# (a) components: [prev, head_0..H-1, mlp]
H_ = meta['heads']
COMP = ['prev'] + [f'head{h}' for h in range(H_)] + ['mlp']


def comp_q(P, cut, t):
    C = P['comp'][cut]
    C = C[:, tok_idx(cut, t)] if cut not in FINAL else C
    C = C.astype(np.float64)
    tot = C.sum(2)
    s = tot.std(-1, keepdims=True) + 1e-8 if tp.needs_norm(model, cut, t) else 1.0
    Cn = (C - C.mean(-1, keepdims=True)) / (s[:, :, None] if np.ndim(s) else 1.0)
    return Cn                                                              # (n, nt, C, d)


stackQ = {}
for k in range(1, L + 1):
    cut = ('post', k)
    for t in types_at(cut):
        q0 = Q[(cut, t, 'q0')]
        Cn = comp_q(F, cut, t)
        n, nt, nc, dm = Cn.shape
        qs = E.QTok(t, 'q0', norm=False, lam=q0.m.lam)
        qs.m = E.LinearGaussian(cov_type='full', lambdas=(q0.m.lam,), shrink=1e-5)
        Phi = q0.phi(F['d'], F['cs']).reshape(n * nt, -1)
        qs.m.fit(Cn.reshape(n * nt, nc * dm), Phi)
        stackQ[(cut, t)] = qs
compA = {}
for k in range(1, L + 1):
    cut = ('post', k)
    for t in types_at(cut):
        compA[(cut, t)] = comp_q(A, cut, t)


def comp_replace(cut, t, ci, mode, rng):
    qs, q0 = stackQ[(cut, t)], Q[(cut, t, 'q0')]
    Cn = compA[(cut, t)]
    n, nt, nc, dm = Cn.shape
    Phi = q0.phi(A['d'], A['cs']).reshape(n * nt, -1)
    flat = Cn.reshape(n * nt, nc * dm)
    if ci is None:
        Hs = qs.m.sample(Phi, rng=rng)
    else:
        Adims = np.arange(ci * dm, (ci + 1) * dm)
        Hs = qs.m.subset_sample(flat, Adims, Phi, rng=rng, mode=mode)
    return tp.to_q(Hs.reshape(n, nt, nc, dm).sum(2), tp.needs_norm(model, cut, t))


rng = np.random.default_rng(17)
for k in range(1, L + 1):
    cut = ('post', k)
    tys = types_at(cut)
    G = site_eval(A, cut, {t: comp_replace(cut, t, None, None, rng) for t in tys})
    out = {'stack_all': E.direct(G, A['Gp'])}
    for ci, cn in enumerate(COMP):
        for mode in ('knockout', 'knockin'):
            G = site_eval(A, cut, {t: comp_replace(cut, t, ci, mode, rng) for t in tys})
            out[f'{cn}|{mode}|ALL'] = E.direct(G, A['Gp'])
        if len(tys) > 1:
            for t in tys:
                G = site_eval(A, cut, {t: comp_replace(cut, t, ci, 'knockout', rng)})
                out[f'{cn}|knockout|{t}'] = E.direct(G, A['Gp'])
    loc[f'components_{cname(cut)}'] = out
    log('components', cname(cut), {kk: round(v['delta'], 5) for kk, v in out.items() if kk.endswith('ALL') or kk == 'stack_all'})


# (b) relative-Fisher subspaces from an h-space classifier per token type: input [h_t, zeta_t]
def zeta(P, cut, t):
    q0 = Q[(cut, t, 'q0')]
    F0 = q0.phi(P['d'], P['cs'])
    Fc = tp.token_features(P['d'], P['cs'], t, 'Rmarg', tp.CANDIDATES[t])
    return np.concatenate([F0, Fc], 2)


def hspace(cut, t, rng, nmax=40000):
    q0 = Q[(cut, t, 'q0')]
    Hr = real_states(A, cut, t)
    Hq = q0.sample(A['d'], A['cs'], rng)
    Z = zeta(A, cut, t)
    n, nt, dm = Hr.shape
    sel = rng.permutation(n * nt)[:nmax]
    Xp = np.concatenate([Hr.reshape(n * nt, dm), Z.reshape(n * nt, -1)], 1)[sel]
    Xq = np.concatenate([Hq.reshape(n * nt, dm), Z.reshape(n * nt, -1)], 1)[sel]
    rm, dg = fit_ratio(Xp, Xq, family='mlp', seed=5)
    import torch
    x = torch.tensor((Xp - rm.clf.mu) / rm.clf.sd, dtype=torch.float32, requires_grad=True)
    rm.clf.net(x).sum().backward()
    g = x.grad.numpy()[:, :dm] / rm.clf.sd[:dm] * rm.a
    one = np.ones((dm, 1)) / math.sqrt(dm)
    Pm = np.eye(dm) - one @ one.T
    Fm = Pm @ (g.T @ g / len(g)) @ Pm
    ev, U = np.linalg.eigh(Fm)
    U = U[:, ::-1][:, :4]; ev = ev[::-1]
    lw_real = rm.log_w(Xp)
    return dict(U=U, ev=ev[:8].tolist(), auc=dg['auc'], kl_plugin=dg['kl_plugin'], dv=dg['dv'], sel=sel, lw=lw_real)


def pullback_U(cut, t, rng, ndraw=6000):
    """G-relevant directions (amendment 1): top-4 eigenvectors of E_p[g g^T], g = gradient w.r.t. the token's quotient
    coordinates of the output-space log w_out(f(h), c) (classifier of the site's q0 ratio), on real draws."""
    import torch
    rm = RM_OUT[(cut, t)]
    sel = rng.permutation(A['n'] * M)[:ndraw]
    ix = tok_idx(cut, t)
    h = torch.tensor(A['st'][cut][sel], dtype=torch.float32, requires_grad=True)
    if cut in FINAL:
        x = h[:, 0]
        if cut[0] == 'attn':
            blk = model.blocks[cut[1] - 1]; x = x + blk.mlp(blk.ln2(x))
    else:
        al = tp._allow(torch.from_numpy(A['V'][sel])); x = h
        for kk in range(cut[1] + 1, L + 1):
            x, _ = model.blocks[kk - 1](x, al)
        x = x[:, tp.QI]
    y = model.out(model.ln(x)); K = model.mixk
    mu, ls, lg = y[:, K:2 * K], y[:, 2 * K:].clamp(-6, 3), y[:, :K]
    hv = torch.cat([mu, ls, lg - lg.mean(1, keepdim=True)], 1)
    feat = torch.cat([hv, torch.tensor(ZA[sel], dtype=torch.float32)], 1)
    xs = (feat - torch.tensor(rm.clf.mu, dtype=torch.float32)) / torch.tensor(rm.clf.sd, dtype=torch.float32)
    (rm.a * rm.clf.net(xs)).sum().backward()
    g = h.grad.numpy()[:, ix].astype(np.float64)
    if tp.needs_norm(model, cut, t):
        raw = A['st'][cut][sel][:, ix].astype(np.float64)
        g = g * (raw - raw.mean(-1, keepdims=True)).std(-1, keepdims=True)
    g = g.reshape(-1, g.shape[-1]); dm = g.shape[1]
    one = np.ones((dm, 1)) / math.sqrt(dm); Pm = np.eye(dm) - one @ one.T
    Fm = Pm @ (g.T @ g / len(g)) @ Pm
    ev, U = np.linalg.eigh(Fm)
    return U[:, ::-1][:, :4], ev[::-1][:8].tolist()


HS = {}
for cut in MULTI + FINAL:
    for t in types_at(cut):
        hs = hspace(cut, t, rng)
        hs['U_h'] = hs['U']
        hs['U'], hs['ev_out'] = pullback_U(cut, t, rng)
        HS[(cut, t)] = hs
        q0 = Q[(cut, t, 'q0')]
        Hr = real_states(A, cut, t)
        dm = Hr.shape[-1]
        Ur = np.linalg.qr(np.random.default_rng(3).standard_normal((dm, 4)) - 0)[0]
        one = np.ones((dm, 1)) / math.sqrt(dm); Ur = np.linalg.qr(Ur - one @ (one.T @ Ur))[0]
        o = dict(auc=hs['auc'], kl_plugin=hs['kl_plugin'], dv=hs['dv'], fisher_ev_h=hs['ev'], fisher_ev_out=hs['ev_out'],
                 cos_Uh_Uout=float(np.linalg.svd(hs['U_h'].T @ hs['U'])[1].mean()))
        for lab_, U in (('Uout4', hs['U']), ('Uh4', hs['U_h']), ('rand4', Ur)):
            for mode in ('knockout', 'knockin'):
                Hs = E.subspace_knock(q0, A['d'], A['cs'], Hr, U, rng, mode)
                o[f'{lab_}|{mode}'] = E.direct(site_eval(A, cut, {t: Hs}), A['Gp'])
        loc[f'subspace_{sname(cut, t)}'] = o
        log('subspace', sname(cut, t), 'auc %.3f' % hs['auc'], {kk: round(v['delta'], 5) for kk, v in o.items() if isinstance(v, dict)})
res['localisation_A'] = loc
json.dump(res, open(OUT, 'w'), default=float)

# ----------------------------------------------------------------------------- (5) naming -> q1
naming, Q1 = {}, {}
for cut in MULTI + FINAL:
    for t in types_at(cut):
        q0 = Q[(cut, t, 'q0')]
        hs = HS[(cut, t)]
        Hr = real_states(A, cut, t)
        base = q0.logpdf_proj(A['d'], A['cs'], Hr, hs['U'])
        base_h = q0.logpdf_proj(A['d'], A['cs'], Hr, hs['U_h'])
        # where w is large: top decile of the out-of-fold output-space log w_out of the real draws -> library features of
        # their tokens of this type, z-scored against all draws
        Z = zeta(A, cut, t)
        nt_ = Z.shape[1]
        lw_draw = GQ_A[(cut, t)][2].ravel()
        Z = Z.reshape(-1, Z.shape[-1]); lw_tok = np.repeat(lw_draw, nt_)
        top = lw_tok >= np.quantile(lw_tok, 0.9)
        zs = (Z[top].mean(0) - Z.mean(0)) / (Z.std(0) + 1e-9)
        nf0 = q0.phi(A['d'], A['cs']).shape[-1]
        names = ['phi%d' % i for i in range(nf0)]
        for g in tp.CANDIDATES[t]:
            k_ = tp.token_features(A['d'], A['cs'], t, 'Rmarg', (g,)).shape[-1]
            names += [f'{g}{i}' for i in range(k_)]
        order = np.argsort(-np.abs(zs))[:8]
        gains = {}
        cands = list(tp.CANDIDATES[t]) + ['hetero']
        for g in cands:
            q1 = fit_q(cut, t, 'q0', () if g == 'hetero' else (g,), hetero=(g == 'hetero'))
            lp = q1.logpdf_proj(A['d'], A['cs'], Hr, hs['U'])
            lph = q1.logpdf_proj(A['d'], A['cs'], Hr, hs['U_h'])
            gains[g] = dict(gain_U=float((lp - base).mean()), gain_U_se=float((lp - base).std(ddof=1) / math.sqrt(len(lp))),
                            gain_Uh=float((lph - base_h).mean()),
                            n_extra=int(q1.m.n_params() - q0.m.n_params()))
            Q[(cut, t, 'add:' + g)] = q1
        best = max(cands, key=lambda g: gains[g]['gain_U'])
        # tie rule: within 10% of the best gain, prefer fewer extra parameters
        close = [g for g in cands if gains[g]['gain_U'] >= 0.9 * gains[best]['gain_U']]
        best = min(close, key=lambda g: gains[g]['n_extra'])
        Q1[(cut, t)] = Q[(cut, t, 'add:' + best)]
        best_h = max(cands, key=lambda g: gains[g]['gain_Uh'])
        naming[sname(cut, t)] = dict(gains=gains, chosen=best, chosen_under_Uh=best_h,
                                     top_decile_shift=[(names[i], float(zs[i])) for i in order])
        log('naming', sname(cut, t), 'chosen', best, {g: round(v['gain_U'], 4) for g, v in gains.items()})
res['naming_A'] = naming

# ----------------------------------------------------------------------------- fresh B: q0 vs q1 (+ add-one attribution), DL
fresh = {}
rngB = np.random.default_rng(23)
GB = {}
for cut, site in SITES:
    tys = site_types(cut, site)
    outs = {}
    for lab_, getq in (('q0', lambda t: Q[(cut, t, 'q0')]), ('q1', lambda t: Q1[(cut, t)])):
        G = site_eval(B, cut, {t: getq(t).sample(B['d'], B['cs'], rngB) for t in tys})
        outs[lab_] = E.direct(G, B['Gp']); GB[(cut, site, lab_)] = G
    if site != 'ALL':
        for g in list(tp.CANDIDATES[site]) + ['hetero']:
            G = site_eval(B, cut, {site: Q[(cut, site, 'add:' + g)].sample(B['d'], B['cs'], rngB)})
            outs['add:' + g] = E.direct(G, B['Gp'])
    dd, se = E.rms_diff(GB[(cut, site, 'q1')] - B['Gp'], GB[(cut, site, 'q0')] - B['Gp'])
    outs['rms_q1_minus_q0'] = dd; outs['rms_q1_minus_q0_se'] = se
    outs['accepted'] = bool(-dd >= max(0.1 * outs['q0']['rms'], 3 * se))
    fresh[sname(cut, site)] = outs
    log('fresh', sname(cut, site), 'q0 rms %.5f q1 rms %.5f diff %.5f+-%.5f' % (outs['q0']['rms'], outs['q1']['rms'], dd, se))
res['fresh_B'] = fresh
json.dump(res, open(OUT, 'w'), default=float)

# description length per (cut, type) for q0 and q1: struct bits + quantised param bits (tolerance 0.02 KL_model on
# Delta with common random numbers on a 1500-cell subset of B) and BIC
NS = min(1500, B['n'])
subB = dict(B)
sel_draw = np.arange(NS * M)
dlres = {}


def dl_of(q, cut, t):
    sub_d = {k: (v[sel_draw] if isinstance(v, np.ndarray) and len(v) == B['n'] * M else v) for k, v in B['d'].items()}
    sub_cs = {k: (v if k == 'comp_vars' else v[sel_draw]) for k, v in B['cs'].items()}
    st = B['st'][cut][sel_draw]
    Vs = B['V'][sel_draw]

    def ev_(qq):
        Hs = qq.sample(sub_d, sub_cs, np.random.default_rng(99))
        h = st.copy(); h[:, tok_idx(cut, t)] = Hs
        head = tp.forward_qt(model, h, cut) if cut in FINAL else tp.forward_from(model, h, cut, Vs)
        return float(B['GQ'](head, B['cell'][sel_draw]).mean())

    class Wrap:
        def __init__(s, qq): s.qq = qq

        def params(s): return s.qq.m.params()

        def set_params(s, P_): s.qq.m.set_params(P_)

        def sample(s, *a): return s.qq.sample(*a)
    import copy
    wq = Wrap(copy.deepcopy(q))
    pb = DL.param_bits(wq, lambda w: ev_(w.qq), tol=0.02 * KLm, bits=range(2, 17), N_fit=NFIT * len(tok_idx(cut, t)))
    sb = DL.struct_bits(n_features=q.n_struct_groups(), library_size=LIB_SIZE, map_type=True, noise_type=True,
                        mixture_K=None)
    if q.hetero:
        sb += math.log2(8)
    return dict(struct=sb, param=pb['bits'], b_star=pb['b_star'], n_eff=pb['n_eff'], bic=pb.get('bic_bits'),
                total=sb + (pb['bits'] or float('nan')))


for cut in MULTI + FINAL:
    for t in types_at(cut):
        dlres[sname(cut, t)] = {'q0': dl_of(Q[(cut, t, 'q0')], cut, t), 'q1': dl_of(Q1[(cut, t)], cut, t)}
    log('DL', cname(cut), {t: (round(dlres[sname(cut, t)]['q0']['total']), round(dlres[sname(cut, t)]['q1']['total'])) for t in types_at(cut)})
nn_params = sum(p.numel() for n_, p in model.named_parameters() if n_.startswith('blocks.'))
res['DL'] = dlres
res['DL_net_block_bits_fp16'] = {f'block{k}': 16 * sum(p.numel() for p in model.blocks[k - 1].parameters()) for k in range(1, L + 1)}
json.dump(res, open(OUT, 'w'), default=float)

# ----------------------------------------------------------------------------- (6) cross-layer consistency
cons = {}
rngC = np.random.default_rng(31)
for k in range(1, L):
    cut_l, cut_n = ('post', k), ('post', k + 1)
    acc_all = fresh[sname(cut_l, 'ALL')]['accepted']
    ql = {t: (Q1[(cut_l, t)] if acc_all else Q[(cut_l, t, 'q0')]) for t in types_at(cut_l)}
    tys_n = types_at(cut_n)
    # push the layer-l description through the real block k+1 on the fit split
    hF = F['st'][cut_l].copy()
    for t in types_at(cut_l):
        hF[:, tp.GROUPS[t]] = ql[t].sample(F['d'], F['cs'], rngC)
    pushedF = tp.push_block(model, hF, k + 1, F['V'])
    hB = B['st'][cut_l].copy()
    for t in types_at(cut_l):
        hB[:, tp.GROUPS[t]] = ql[t].sample(B['d'], B['cs'], rngC)
    pushedB = tp.push_block(model, hB, k + 1, B['V'])

    def states_n(Hfull, t):
        ix = tok_idx(cut_n, t) if cut_n not in FINAL else [tp.QI]
        return tp.to_q(Hfull[:, ix].astype(np.float64), tp.needs_norm(model, cut_n, t))
    # tier i: Delta_l(q_l) = Delta_{l+1}(F#q_l), sample by sample
    head_l = tp.forward_from(model, hB, cut_l, B['V'])
    head_n = tp.forward_qt(model, pushedB[:, [tp.QI]], cut_n) if cut_n in FINAL else tp.forward_from(model, pushedB, cut_n, B['V'])
    Gl, Gn = G_of(B, head_l), G_of(B, head_n)
    out = dict(tier_i_max_abs=float(np.abs(Gl - Gn).max()), q_l=('q1' if acc_all else 'q0'))
    out['delta_push'] = E.direct(Gn, B['Gp'])
    # tier ii
    qt = {t: E.QTok(t, 'q0', norm=tp.needs_norm(model, cut_n, t)).fit(F['d'], F['cs'], states_n(pushedF, t)) for t in tys_n}
    qf = {t: Q[(cut_n, t, 'q0')] for t in tys_n}
    cut_eval = cut_n
    Gt = site_eval(B, cut_eval, {t: qt[t].sample(B['d'], B['cs'], rngC) for t in tys_n})
    Gf = site_eval(B, cut_eval, {t: qf[t].sample(B['d'], B['cs'], rngC) for t in tys_n})
    out['delta_tilde'] = E.direct(Gt, B['Gp']); out['delta_fit'] = E.direct(Gf, B['Gp'])
    gfit, gse = E.rms_diff(Gt - B['Gp'], Gf - B['Gp'])
    # refit spread: half-sample refits of both families
    sp_r, sp_d = [], []
    for r in range(3):
        hi = np.random.default_rng(40 + r).permutation(F['n'])[:F['n'] // 2]
        sub = lambda P_: ({kk: (v[hi] if isinstance(v, np.ndarray) and len(v) == F['n'] else v) for kk, v in P_['d'].items()},
                          {kk: (v if kk == 'comp_vars' else v[hi]) for kk, v in P_['cs'].items()})
        dsub, cssub = sub(F)
        qt2 = {t: E.QTok(t, 'q0', norm=tp.needs_norm(model, cut_n, t)).fit(dsub, cssub, states_n(pushedF, t)[hi]) for t in tys_n}
        Hreal = {t: (real_states(F, cut_n, t)[hi]) for t in tys_n}
        qf2 = {t: E.QTok(t, 'q0', norm=tp.needs_norm(model, cut_n, t)).fit(dsub, cssub, Hreal[t]) for t in tys_n}
        Gt2 = site_eval(B, cut_eval, {t: qt2[t].sample(B['d'], B['cs'], rngC) for t in tys_n})
        Gf2 = site_eval(B, cut_eval, {t: qf2[t].sample(B['d'], B['cs'], rngC) for t in tys_n})
        sp_r.append(E.rms_diff(Gt2 - B['Gp'], Gf2 - B['Gp'])[0]); sp_d.append(float((Gt2 - Gf2).mean()))
    out['gamma_fit'] = gfit
    out['gamma_fit_se'] = float(math.hypot(gse, np.std(sp_r, ddof=1)))
    out['gamma_proj'] = out['delta_tilde']['delta'] - out['delta_push']['delta']
    gd = (Gt - Gf).mean(1)
    out['gamma_delta'] = float(gd.mean()); out['gamma_delta_se'] = float(math.hypot(gd.std(ddof=1) / math.sqrt(len(gd)), np.std(sp_d, ddof=1)))
    # AUC(q~ vs q^fit) per token type, classifier on [h, phi]
    aucs = {}
    for t in tys_n:
        Ht = qt[t].sample(B['d'], B['cs'], rngC); Hf = qf[t].sample(B['d'], B['cs'], rngC)
        Ph = qf[t].phi(B['d'], B['cs'])
        n_, nt_, dm_ = Ht.shape
        s_ = rngC.permutation(n_ * nt_)[:40000]
        Xa = np.concatenate([Ht.reshape(-1, dm_), Ph.reshape(n_ * nt_, -1)], 1)[s_]
        Xb = np.concatenate([Hf.reshape(-1, dm_), Ph.reshape(n_ * nt_, -1)], 1)[s_]
        _, dg = fit_ratio(Xa, Xb, family='mlp', seed=9)
        aucs[t] = dg['auc']
    out['auc'] = aucs
    tolc = 0.1 * KLm
    out['pass_method'] = bool(gfit <= max(2 * out['gamma_fit_se'], tolc) and max(aucs.values()) <= 0.55)
    out['pass_two_sided'] = bool(abs(gfit) <= max(2 * out['gamma_fit_se'], tolc) and abs(out['gamma_delta']) <= max(2 * out['gamma_delta_se'], tolc)
                                 and max(aucs.values()) <= 0.55)
    cons[f'{cname(cut_l)}->{cname(cut_n)}'] = out
    log('consistency', cname(cut_l), '->', cname(cut_n), 'tier_i %.1e gamma_fit %.5f+-%.5f gamma_delta %.5f auc %s' % (
        out['tier_i_max_abs'], gfit, out['gamma_fit_se'], out['gamma_delta'], {t: round(a, 3) for t, a in aucs.items()}))
res['consistency_B'] = cons
res['seconds'] = time.time() - T0
json.dump(res, open(OUT, 'w'), default=float)
log('done')
