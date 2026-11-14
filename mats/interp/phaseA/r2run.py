"""Round 2 on one tiny-PFN checkpoint (PREREG.md amendment 4): information-matched on-manifold oracles.
usage: python r2run.py MODEL [--quick]
Writes r2/MODEL.json (summary, checks) and r2/MODEL_D.npz (per draw G_V - G_p, shape (cells, M), every site/variant).
Steps: (1) fit split (seed 60000, as round 1) + fresh eval E (cells 70000, draws 70001); (2) q descriptions (q0, q0x,
q1, q0_LN); (3) J (joint control); (4) per-token oracles lat, q0ID, c, U, cID, cIDr; (5) K = 8 beta-matched draws:
O_q0 (draw 0), O_own (its unmatched candidate), M_q0 / M_q0mean (conditional mean +- Gaussian noise, Sigma_w from 2,000
fit episodes); (6) joint real-vs-stitched classifier AUC; (7) whole-input JS binding measure."""
import json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/cm-interp')
from lib import tp                       # noqa: E402
from lib import oracle as O              # noqa: E402
from lib import r1est as E               # noqa: E402

torch.set_num_threads(1)
NAME = sys.argv[1]
QUICK = '--quick' in sys.argv
NFIT, NC, M = (4000, 300, 8) if QUICK else (20000, 6000, 8)
NW = 200 if QUICK else 2000          # fit episodes for Sigma_w
KM = 8                               # beta-matched draws per token for the conditional mean
KCAND = 32                           # candidates for the beta match
NCLF = 150 if QUICK else 3000        # cells for the joint classifier
NJS = 400 if QUICK else 10000        # episodes for the JS binding measure
OUTD = '/home/agent/workspaces/mats/work/cm-interp/r2'
os.makedirs(OUTD, exist_ok=True)
SFX = '_quick' if QUICK else ''
OUT, OUTNPZ = f'{OUTD}/{NAME}{SFX}.json', f'{OUTD}/{NAME}{SFX}_D.npz'
T0 = time.time()
log = lambda *a: print(f'[{NAME} {time.time() - T0:6.0f}s]', *a, flush=True)

model, meta, ev = tp.load(NAME)
L = meta['depth']
MULTI = [('post', k) for k in range(1, L)]
FINAL = [('attn', L), ('post', L)]
FPOST = ('post', L)
ALLCUTS = MULTI + FINAL
want = {c: None for c in MULTI}
want.update({c: [tp.QI] for c in FINAL})
cname = lambda c: f"{'h' if c[0] == 'post' else 'a'}{c[1]}"
sname = lambda c, s: f'{cname(c)}-{s}'
SITES = [(c, s) for c in MULTI for s in ('LAB', 'QQ', 'QT', 'ALL')] + [(c, 'QT') for c in FINAL]
MAINCUT = ('post', 1) if L == 2 else ('post', 2)
R1 = json.load(open(f'/home/agent/workspaces/mats/work/cm-interp/r1/{NAME}.json'))
res = dict(model=NAME, depth=L, eval_kl=ev['best']['kl'], sizes=dict(NFIT=NFIT, NC=NC, M=M, NW=NW, KM=KM, KCAND=KCAND,
                                                                         NCLF=NCLF, NJS=NJS), checks={}, direct={}, timing={})
D = {}                                                     # f'{site}|{variant}' -> (NC, M) float32


def save():
    res['seconds'] = time.time() - T0
    json.dump(res, open(OUT, 'w'), default=float)
    np.savez_compressed(OUTNPZ, **{k: v.astype(np.float32) for k, v in D.items()}, Gp=E_['Gp'].astype(np.float32))


# ----------------------------------------------------------------------------- data
def prep(seed, n, M_, with_G=True):
    d0 = tp.sample_episodes(n, seed)
    cs0 = tp.cell_stats(d0)
    if M_:
        d = tp.cond_draws(d0, M_, seed + 1)
    else:
        d = d0; d['cell'] = np.arange(n)
    X, V = tp.tokens(d)
    head, st, _ = tp.run_real(model, X, V, want)
    GQ = tp.GQuad(cs0) if with_G else None
    Gp = GQ(head, d['cell'], chunk=1000).reshape(n, max(M_, 1)) if with_G else None
    return dict(d=d, cs=tp.take(cs0, d['cell']), cs0=cs0, V=V, head=head, st=st, GQ=GQ, cell=d['cell'], Gp=Gp, n=n,
                M=max(M_, 1))


F = prep(60000, NFIT, 0, with_G=False)
E_ = prep(70000, NC, M)
KLm = float(E_['Gp'].mean())
res['kl_model_E'] = KLm
log('data ready; KL_model(E) %.5f eval.json %.5f' % (KLm, ev['best']['kl']))


def tok_idx(cut, t):
    return [0] if cut in FINAL else tp.GROUPS[t]


def types_at(cut):
    return ['QT'] if cut in FINAL else ['LAB', 'QQ', 'QT']


def site_types(cut, site):
    return types_at(cut) if site == 'ALL' else [site]


def norm_of(cut, t):
    return tp.needs_norm(model, cut, t)


def real_states(P, cut, t, norm=None):
    nm = norm_of(cut, t) if norm is None else norm
    return tp.to_q(P['st'][cut][:, tok_idx(cut, t)].astype(np.float64), nm)


def run_down(cut, h):
    return tp.forward_qt(model, h, cut) if cut in FINAL else tp.forward_from(model, h, cut, E_['V'], bs=1000)


def eval_site(cut, site, repl):
    h = E_['st'][cut].copy()
    for t, Hs in repl.items():
        h[:, tok_idx(cut, t)] = Hs
    return E_['GQ'](run_down(cut, h), E_['cell'], chunk=1000).reshape(NC, M)


def record(cut, site, var, G):
    key = f'{sname(cut, site)}|{var}'
    Dv = G - E_['Gp']
    D[key] = Dv
    r = E.direct(G, E_['Gp'], B=400)
    dc = Dv.mean(1)
    lo_, hi_ = np.quantile(dc, [0.01, 0.99])
    r['trim1'] = float(dc[(dc >= lo_) & (dc <= hi_)].mean())
    res['direct'][key] = r
    return r


# ----------------------------------------------------------------------------- (2) q descriptions
chosen = {k: v['chosen'] for k, v in R1['naming_A'].items()}
res['q1_chosen'] = chosen
QD = {}
for cut in ALLCUTS:
    for t in types_at(cut):
        nm = norm_of(cut, t)
        Hf = real_states(F, cut, t)
        QD[(cut, t, 'q0')] = E.QTok(t, 'q0', norm=nm).fit(F['d'], F['cs'], Hf)
        g = chosen[sname(cut, t)]
        QD[(cut, t, 'q1')] = E.QTok(t, 'q0', () if g == 'hetero' else (g,), norm=nm, hetero=(g == 'hetero')).fit(F['d'], F['cs'], Hf)
        if t in ('LAB', 'QQ'):
            QD[(cut, t, 'q0x')] = E.QTok(t, 'q0', ('cross',), norm=nm).fit(F['d'], F['cs'], Hf)
        if cut == FPOST:
            QD[(cut, t, 'q0_LN')] = E.QTok(t, 'q0', norm=True).fit(F['d'], F['cs'], real_states(F, cut, t, True))
log('fitted q descriptions')
rng = np.random.default_rng(70050)
QSAMP = {}                                                 # keep q0 samples at the main cut for the classifier
for cut, site in SITES:
    tys = site_types(cut, site)
    for var in ('q0', 'q1', 'q0x', 'q0_LN'):
        if var == 'q0x' and site == 'QT':
            continue
        if var == 'q0_LN' and cut != FPOST:
            continue
        repl = {}
        for t in tys:
            q = QD.get((cut, t, var)) or QD[(cut, t, 'q0')]     # q0x at ALL: QT keeps q0
            repl[t] = q.sample(E_['d'], E_['cs'], rng)
            if var == 'q0' and cut == MAINCUT and site == 'ALL':
                QSAMP[t] = repl[t][:NCLF * M].astype(np.float32)
        record(cut, site, var, eval_site(cut, site, repl))
    log('q', sname(cut, site), {v: round(res['direct'][f'{sname(cut, site)}|{v}']['delta'], 4)
                                 for v in ('q0', 'q1', 'q0x', 'q0_LN') if f'{sname(cut, site)}|{v}' in res['direct']})
save()

# ----------------------------------------------------------------------------- (3) J: joint control (one x'' per draw)
t1 = time.time()
xj = O.draw_joint(E_['d'], np.random.default_rng(70099))
Xj, Vj = tp.tokens(xj)
_, stJ, _ = tp.run_real(model, Xj, Vj, want)
for cut, site in SITES:
    repl = {t: stJ[cut][:, tok_idx(cut, t)] for t in site_types(cut, site)}
    record(cut, site, 'J', eval_site(cut, site, repl))
STJ_MAIN = stJ[MAINCUT][:NCLF * M].copy()
del stJ, Xj, Vj, xj
res['timing']['J'] = time.time() - t1
log('J', {sname(c, s): round(res['direct'][f'{sname(c, s)}|J']['delta'], 5) for c, s in SITES})
save()


# ----------------------------------------------------------------------------- (4) per-token oracles
def oracle_states(var, seed, final_too, d=None, K=KCAND, want_own=False):
    """Per-token oracle states: {cut: (N, T or 1, dm) float32}; plus beta errors, p_identity, and the 'own' companion."""
    d = E_['d'] if d is None else d
    N = len(d['y'])
    dm = meta['width']
    out = {c: np.zeros((N, tp.T, dm), np.float32) for c in MULTI}
    out.update({c: np.zeros((N, 1, dm), np.float32) for c in FINAL if final_too})
    own = {c: np.zeros_like(v) for c, v in out.items()} if want_own else None
    info = dict(beta_err=[], p_identity=[])
    bx = O.beta_of(*O.cvars(d)[:2])
    for t in range(tp.T):
        rng_t = np.random.default_rng([seed, t])
        cuts = (MULTI + FINAL) if (t == tp.QI and final_too) else MULTI
        if not cuts:
            continue
        x2, inf = O.draw(d, t, var, rng_t, K=K, beta_x=None if t == tp.QI else bx[:, tp.TOK_SRC[t]], want_own=want_own)
        X, V = tp.tokens(x2)
        st = O.forward_token(model, X, V, t, cuts, bs=1000)
        for c in cuts:
            if c in FINAL:
                out[c][:, 0] = st[c]
            else:
                out[c][:, t] = st[c]
        if 'beta_err' in inf:
            info['beta_err'].append(inf['beta_err'])
        if 'p_identity' in inf:
            info['p_identity'].append(inf['p_identity'])
        if want_own:
            if 'own' in inf:
                X, V = tp.tokens(inf['own'])
                sto = O.forward_token(model, X, V, t, cuts, bs=1000)
            else:                                       # QT: 'own' = 'c' = this draw
                sto = st
            for c in cuts:
                if c in FINAL:
                    own[c][:, 0] = sto[c]
                else:
                    own[c][:, t] = sto[c]
    return out, own, info


def eval_oracle(var, states, sites):
    """records under 'O_' + var (oracle variants never share a name with a q description)."""
    for cut, site in sites:
        repl = {t: states[cut][:, tok_idx(cut, t)] for t in site_types(cut, site)}
        record(cut, site, 'O_' + var, eval_site(cut, site, repl))


MULTI_SITES = [(c, s) for c, s in SITES if c in MULTI]
NONQT_MULTI = [(c, s) for c, s in MULTI_SITES if s != 'QT']
FINAL_SITES = [(c, s) for c, s in SITES if c in FINAL]
ORACLE_PLAN = [  # variant, seed, evaluate at final cuts, sites at multi cuts (QT-only sites skipped where QT is an alias)
    ('lat', 70101, False, MULTI_SITES), ('c', 70102, True, MULTI_SITES), ('U', 70103, True, MULTI_SITES),
    ('cID', 70104, False, MULTI_SITES), ('cIDr', 70105, False, NONQT_MULTI), ('q0ID', 70106, False, NONQT_MULTI)]
CLF_STATES = {}
for var, seed, fin, msites in ORACLE_PLAN:
    t1 = time.time()
    st, _, info = oracle_states(var, seed, fin)
    eval_oracle(var, st, msites + (FINAL_SITES if fin else []))
    if var in ('c', 'cID', 'cIDr'):
        CLF_STATES[var] = st[MAINCUT][:NCLF * M].copy()
    if info['p_identity']:
        pid = np.concatenate(info['p_identity'])
        res['checks'][f'U_p_identity'] = dict(mean=float(pid.mean()), q10=float(np.quantile(pid, .1)), q90=float(np.quantile(pid, .9)))
    if info['beta_err']:
        be = np.abs(np.concatenate(info['beta_err']))
        res['checks'][f'beta_err_{var}'] = dict(median=float(np.median(be)), q99=float(np.quantile(be, .99)))
    del st
    res['timing'][var] = time.time() - t1
    log('oracle', var, {sname(c, s): round(res['direct'][f'{sname(c, s)}|O_{var}']['delta'], 4)
                        for c, s in msites + (FINAL_SITES if fin else []) if f'{sname(c, s)}|O_{var}' in res['direct']})
    save()


# ----------------------------------------------------------------------------- (5) beta-matched draws: O_q0, O_own, M_q0
def qcoord(cut, t, h, ln=False):
    """quotient coordinates used by q0 for token type t at cut (LN gauge if ln)."""
    return tp.to_q(np.asarray(h, np.float64), True if ln else norm_of(cut, t))


def within_cov(Fsub, var='q0', final_too=True, seed=70200):
    """Sigma_w per (cut, type[, 'LN']) from K draws per token on the fit subset."""
    acc = {}
    draws = [oracle_states(var, seed + k, final_too, d=Fsub)[0] for k in range(KM)]
    for cut in (ALLCUTS if final_too else MULTI):
        for t in types_at(cut):
            for ln in ((False, True) if cut == FPOST else (False,)):
                H = np.stack([qcoord(cut, t, dr[cut][:, tok_idx(cut, t)], ln) for dr in draws], 0)   # (K, N, nt, dm)
                Hm = H.mean(0, keepdims=True)
                R_ = (H - Hm).reshape(KM, -1, H.shape[-1])
                C = np.einsum('kni,knj->ij', R_, R_) / (R_.shape[1] * (KM - 1))
                acc[(cut, t, ln)] = 0.5 * (C + C.T)
    return acc


t1 = time.time()
Fsub = {k: (v[:NW] if isinstance(v, np.ndarray) and len(v) == NFIT else v) for k, v in F['d'].items()}
SIGW = within_cov(Fsub)
res['checks']['sigma_w_trace'] = {f'{sname(c, t)}{"_LN" if ln else ""}': float(np.trace(C)) for (c, t, ln), C in SIGW.items()}
res['checks']['q0_resid_trace'] = {sname(c, t): float(sum(np.trace(C) for C in QD[(c, t, 'q0')].mean_cov(F['d'], F['cs'])[1].values()))
                                   for c in ALLCUTS for t in types_at(c)}
log('Sigma_w done (%.0fs)' % (time.time() - t1), res['checks']['sigma_w_trace'])
t1 = time.time()
MEAN = {}
for k in range(KM):
    st, own, info = oracle_states('q0', 70300 + k, True, want_own=(k == 0))
    for cut in ALLCUTS:
        for t in types_at(cut):
            for ln in ((False, True) if cut == FPOST else (False,)):
                key = (cut, t, ln)
                MEAN[key] = MEAN.get(key, 0) + qcoord(cut, t, st[cut][:, tok_idx(cut, t)], ln) / KM
    if k == 0:
        be = np.abs(np.concatenate(info['beta_err']))
        res['checks']['beta_err_q0'] = dict(median=float(np.median(be)), q99=float(np.quantile(be, .99)),
                                            q90=float(np.quantile(be, .9)))
        eval_oracle('q0', st, NONQT_MULTI)                         # QT: O_q0 = O_c (alias, evaluated above)
        eval_oracle('own', own, NONQT_MULTI)
        CLF_STATES['q0'] = st[MAINCUT][:NCLF * M].copy()
        log('oracle q0/own', {sname(c, s): (round(res['direct'][f'{sname(c, s)}|O_q0']['delta'], 4),
                                            round(res['direct'][f'{sname(c, s)}|O_own']['delta'], 4)) for c, s in NONQT_MULTI})
    del st, own
res['timing']['q0_K'] = time.time() - t1
rngM = np.random.default_rng(70400)
for cut, site in SITES:
    for var, ln, noise in (('M_q0', False, True), ('M_q0mean', False, False), ('M_q0_LN', True, True), ('M_q0mean_LN', True, False)):
        if ln and cut != FPOST:
            continue
        repl = {}
        for t in site_types(cut, site):
            mu = MEAN[(cut, t, ln)]
            if noise:
                C = SIGW[(cut, t, ln)] * (1 - 1 / KM)
                Lc = np.linalg.cholesky(C + 1e-9 * np.trace(C) / len(C) * np.eye(len(C)))
                mu = mu + (rngM.standard_normal(mu.shape) @ Lc.T)
            repl[t] = tp.to_q(mu, True if ln else norm_of(cut, t))
        record(cut, site, var, eval_site(cut, site, repl))
    log('M', sname(cut, site), {v: round(res['direct'][f'{sname(cut, site)}|{v}']['delta'], 4)
                                 for v in ('q0', 'M_q0', 'M_q0mean', 'M_q0_LN') if f'{sname(cut, site)}|{v}' in res['direct']})
del MEAN
save()

# ----------------------------------------------------------------------------- (5b) amendment 5: M_c at c-information
t1 = time.time()
SIGWc = within_cov(Fsub, 'c', False, 70500)
res['checks']['sigma_w_c_trace'] = {sname(c, t): float(np.trace(C)) for (c, t, ln), C in SIGWc.items()}
MEANc = {}
for k in range(KM):
    st, _, _ = oracle_states('c', 70600 + k, False)
    for cut in MULTI:
        for t in types_at(cut):
            MEANc[(cut, t)] = MEANc.get((cut, t), 0) + qcoord(cut, t, st[cut][:, tok_idx(cut, t)]) / KM
    del st
for cut, site in NONQT_MULTI:
    for var, noise in (('M_c', True), ('M_cmean', False)):
        repl = {}
        for t in site_types(cut, site):
            mu = MEANc[(cut, t)]
            if noise:
                C = SIGWc[(cut, t, False)] * (1 - 1 / KM)
                Lc = np.linalg.cholesky(C + 1e-9 * np.trace(C) / len(C) * np.eye(len(C)))
                mu = mu + (rngM.standard_normal(mu.shape) @ Lc.T)
            repl[t] = tp.to_q(mu, norm_of(cut, t))
        record(cut, site, var, eval_site(cut, site, repl))
    log('M_c', sname(cut, site), {v: round(res['direct'][f'{sname(cut, site)}|{v}']['delta'], 4)
                                   for v in ('q0x', 'O_c', 'M_c', 'M_cmean') if f'{sname(cut, site)}|{v}' in res['direct']})
del MEANc
res['timing']['M_c'] = time.time() - t1
save()

# ----------------------------------------------------------------------------- (6) joint real-vs-stitched classifier
t1 = time.time()
ncl = NCLF * M
real_layer = E_['st'][MAINCUT][:ncl].astype(np.float64)
cell_cl = E_['cell'][:ncl]


def layer_q(Hl):
    """flattened quotient coordinates of a whole layer (N, T, dm) at MAINCUT."""
    out = np.empty(Hl.shape)
    for t in ('LAB', 'QQ', 'QT'):
        ix = tp.GROUPS[t]
        out[:, ix] = tp.to_q(np.asarray(Hl[:, ix], np.float64), norm_of(MAINCUT, t))
    return out.reshape(len(Hl), -1)


Fp = layer_q(real_layer)
clf = {}
for site in ('QQ', 'ALL'):
    for var in ('q0', 'O_q0', 'O_c', 'O_cID', 'O_cIDr', 'J'):
        Hs = real_layer.copy()
        for t in site_types(MAINCUT, site):
            ix = tp.GROUPS[t]
            if var == 'q0':
                Hs[:, ix] = QSAMP[t]
            elif var == 'J':
                Hs[:, ix] = STJ_MAIN[:, ix]
            else:
                Hs[:, ix] = CLF_STATES[var[2:]][:, ix]
        reps = E.crossfit_cells(Fp, layer_q(Hs), cell_cl, NCLF, reps=1, seed=91)
        dg = reps[0][2]
        clf[f'{sname(MAINCUT, site)}|{var}'] = dict(auc=dg['auc'], auc_se=dg['auc_se'], bce=dg['bce'])
        log('joint classifier', sname(MAINCUT, site), var, 'AUC %.4f +- %.4f' % (dg['auc'], dg['auc_se']))
res['joint_classifier'] = clf
res['timing']['classifier'] = time.time() - t1
del CLF_STATES, STJ_MAIN, QSAMP
save()

# ----------------------------------------------------------------------------- (7) whole-input JS binding measure
t1 = time.time()


def mix_logpdf(g, means, sds, logw):
    """g (n, G); means/sds/logw (n, K) -> log mixture density (n, G)."""
    z = (g[:, :, None] - means[:, None, :]) / sds[:, None, :]
    lp = logw[:, None, :] - np.log(sds)[:, None, :] - 0.5 * math.log(2 * math.pi) - 0.5 * z * z
    m = lp.max(2, keepdims=True)
    return m[:, :, 0] + np.log(np.exp(lp - m).sum(2))


def js_measure(ngrid, n, seed, chunk=25, n_blind=500):
    d = tp.sample_episodes(n, seed)
    Qx, Sx, Rx = O.cvars(d)
    lev = np.stack([O.log_evidence(Qx, Sx[:, pi]) for pi in O.PERMS], 1)
    w = np.exp(lev - lev.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
    nets, bay, bli = [], [], []
    bmeans = np.einsum('ks,ns->nk', 1.0 / O.VC, Qx) / O.PREC[None]
    bsd = np.broadcast_to(np.sqrt(1.0 / O.PREC)[None], bmeans.shape)
    for pi in O.PERMS:
        o = d['o'].copy()
        o[:, :, :tp.QL] = d['y'][:, None, :tp.QL] + Rx * np.sqrt(Sx[:, pi] / Sx)[:, :, None]
        xp = dict(o=o, y=d['y'], semb=d['semb'], qemb=d['qemb'])
        X, V = tp.tokens(xp)
        mu, ls, lg = tp.run_real(model, X, V, {})[0]
        nets.append((mu, np.exp(ls), lg - np.logaddexp.reduce(lg, axis=1, keepdims=True)))
        lw = O.bayes_logw(Qx, Sx[:, pi]); lw = lw - np.logaddexp.reduce(lw, axis=1, keepdims=True)
        bay.append((bmeans, bsd, lw))
        bw = tp.blind_weights(dict(o=o, y=d['y']))
        bli.append((bmeans, bsd, np.log(np.maximum(bw, 1e-300))))
    out = {}
    for lab, comps in (('net', nets), ('bayes', bay), ('blind', bli)):
        lo = np.min([(m - 9 * s).min(1) for m, s, _ in comps], 0)
        hi = np.max([(m + 9 * s).max(1) for m, s, _ in comps], 0)
        nn = min(n, n_blind) if lab == 'blind' else n
        js = np.empty(nn); H_ = np.empty((nn, len(O.PERMS)))
        for i in range(0, nn, chunk):
            sl = slice(i, i + chunk)
            g = lo[sl, None] + (hi - lo)[sl, None] * np.linspace(0, 1, ngrid)[None]
            dx = ((hi - lo)[sl] / (ngrid - 1))[:, None]
            tw = np.ones(ngrid); tw[0] = tw[-1] = 0.5
            fs = []
            for p_, (m, s, lw) in enumerate(comps):
                lf = mix_logpdf(g, m[sl], s[sl], lw[sl])
                f = np.exp(lf)
                H_[sl, p_] = -(f * lf * tw[None]).sum(1) * dx[:, 0]
                fs.append(f)
            fbar = sum(w[sl, p_][:, None] * fs[p_] for p_ in range(len(fs)))
            Hbar = -(fbar * np.log(np.maximum(fbar, 1e-300)) * tw[None]).sum(1) * dx[:, 0]
            js[sl] = Hbar - (w[sl] * H_[sl]).sum(1)
        out[lab] = js
    return out, w


js, wperm = js_measure(801, NJS, 72000)
js_lo, _ = js_measure(801, min(NJS, 300), 72100, n_blind=50)      # grid check: same 300 episodes, two grids
js_hi, _ = js_measure(4001, min(NJS, 300), 72100, n_blind=50)
rngB = np.random.default_rng(5)
bs_idx = rngB.integers(0, NJS, (1000, NJS))
ratio_bs = js['net'][bs_idx].mean(1) / js['bayes'][bs_idx].mean(1)
top = js['bayes'] >= np.median(js['bayes'])
res['js_binding'] = dict(js_net=float(js['net'].mean()), js_bayes=float(js['bayes'].mean()), js_blind_max=float(np.abs(js['blind']).max()),
                         B_JS=float(js['net'].mean() / js['bayes'].mean()), B_JS_lo=float(np.quantile(ratio_bs, .025)),
                         B_JS_hi=float(np.quantile(ratio_bs, .975)), B_JS_tophalf=float(js['net'][top].mean() / js['bayes'][top].mean()),
                         corr_net_bayes=float(np.corrcoef(js['net'], js['bayes'])[0, 1]),
                         grid_check_bayes_rel=float(np.abs(js_hi['bayes'] - js_lo['bayes']).max() / js_hi['bayes'].mean()),
                         grid_check_net_rel=float(np.abs(js_hi['net'] - js_lo['net']).max() / max(js_hi['net'].mean(), 1e-12)),
                         perm_weight_entropy=float(-(wperm * np.log(np.maximum(wperm, 1e-300))).sum(1).mean()))
res['timing']['js'] = time.time() - t1
log('JS binding', res['js_binding'])
save()
log('done')
