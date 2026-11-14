"""B2 (PREREG Amendment 3): budget of the model's KL to Bayes into named mechanisms.
usage: python budget.py RUN_DIR TAG OUT_JSON
"""
import itertools, json, math, os, sys
import numpy as np
import torch
import torch.nn as nn
from locate import TP, W, P, NTR, QI, Q, QL, S, targets
from steps import run, QQ

torch.set_num_threads(1)
K = 16
NUIS = ('id', 'rot', 'y')


def split_ids(X, ds):
    semb = np.stack([X[:, s * Q, :ds] for s in range(S)], 1)
    qemb = np.stack([X[:, q, ds:2 * ds] for q in range(Q)], 1)
    return semb, qemb


def draw(rng, which, o, y, semb, qemb, ds):
    n = len(y)
    o, y = o.copy(), y.copy()
    if 'y' in which:
        R = o[:, :, :QL] - y[:, None, :QL]
        y[:, :QL] = rng.standard_normal((n, QL))
        o[:, :, :QL] = y[:, None, :QL] + R
    if 'rot' in which:
        R = o[:, :, :QL] - y[:, None, :QL]
        ss = (R ** 2).sum(2, keepdims=True)
        u = rng.standard_normal((n, S, QL)); u /= np.linalg.norm(u, axis=2, keepdims=True)
        o[:, :, :QL] = y[:, None, :QL] + np.sqrt(ss) * u
    if 'id' in which:
        semb = rng.standard_normal((n, S, ds)) / 4; qemb = rng.standard_normal((n, Q, ds)) / 4
    X, m = W.assemble(semb, qemb, o, y)
    return X, m, o, y


def logsm(lg):
    lg = lg - lg.max(1, keepdims=True)
    return lg - np.log(np.exp(lg).sum(1, keepdims=True))


def mixture(heads):
    return (np.concatenate([h[0] for h in heads], 1), np.concatenate([h[1] for h in heads], 1),
            np.concatenate([logsm(h[2]) - math.log(len(heads)) for h in heads], 1))


def beta_from(w):
    VC = W.VLEV[W.COMB]; PREC = 1 + (1 / VC).sum(1)
    return np.einsum('nc,cs->ns', w, (1 / VC) / PREC[:, None])


def zfeat(kind, o, y):
    R = o[:, :, :QL] - y[:, None, :QL]
    lss = np.log((R ** 2).sum(2)); os_ = o[:, :, QL]
    if kind == 'suff':
        f = [lss, os_]
    else:
        f = [lss, R.mean(2), (R ** 3).mean(2), os_, np.repeat(y[:, :QL].mean(1, keepdims=True), S, 1)]
    return np.stack(f, 2)                                                    # (n, S, k)


def fit_g(Z, Y, seed=0):
    torch.manual_seed(seed)
    Zt = torch.from_numpy(Z.astype(np.float32)); Yt = torch.from_numpy(Y.astype(np.float32))
    zm, zs = Zt.mean(0), Zt.std(0) + 1e-6
    ym, ys = Yt.mean(0), Yt.std(0) + 1e-6
    net = nn.Sequential(nn.Linear(Z.shape[1], 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, Y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(3000):
        loss = ((net((Zt - zm) / zs) - (Yt - ym) / ys) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    def g(Zn):
        with torch.no_grad():
            return (net((torch.from_numpy(Zn.astype(np.float32)) - zm) / zs) * ys + ym).numpy()
    return g


def main():
    run_dir, tag, outp = sys.argv[1:4]
    rd = os.path.join(TP, run_dir)
    meta = json.load(open(os.path.join(rd, 'meta.json')))
    ds = meta['ds']
    D = dict(np.load(os.path.join(TP, 'runs', 'testset.npz' if ds == W.DS else f'testset_ds{ds}.npz')))
    model = P.build(meta); model.load_state_dict(torch.load(os.path.join(rd, 'model.pt')))
    n = len(D['y']); tr = np.arange(NTR); te = np.arange(NTR, n)
    X, m = D['X'], D['valid']
    semb, qemb = split_ids(X, ds)
    Tk = {k: D[k][te] for k in ('comp_means', 'w_bayes')}
    Tk['comp_vars'] = D['comp_vars']
    ys = D['ystar'][te]
    res = dict(tag=tag)

    mb_, vb_ = D['mean_bayes'][te], D['var_bayes'][te]

    def moment_fix(h, var_too):
        mu, ls, lg = h
        mm, vm = P.head_moments(h)
        if not var_too:
            return (mu + (mb_ - mm)[:, None], ls, lg)
        c = np.sqrt(vb_ / vm)
        return (mb_[:, None] + (mu - mm[:, None]) * c[:, None], ls + np.log(c)[:, None], lg)

    def score(h, name, store=None, split=False):
        if split:
            res.setdefault('moment_split', {})[name] = dict(kl_meanfix=float(P.kl_grid(Tk, moment_fix(h, False), chunk=200).mean()),
                                                            kl_mvfix=float(P.kl_grid(Tk, moment_fix(h, True), chunk=200).mean()))
        kl = P.kl_grid(Tk, h, chunk=200)
        nll = P.head_nll_np(h, ys)
        mu, _ = P.head_moments(h)
        res.setdefault('kl', {})[name] = dict(kl=float(kl.mean()), nll=float(nll.mean()), gap=float((nll - D['nll_bayes'][te]).mean()),
                                              mse_mean=float(((mu - D['mean_bayes'][te]) ** 2).mean()))
        if store is not None: store[name] = kl.astype(np.float32)
        print(tag, name, round(float(kl.mean()), 5), flush=True)

    perep = {}
    # ---------------- Part 2 fits (layer-2 weights) and Part 3 fits (MLP1 g) on all episodes' caches
    head, C = run(model, X, m, cache=True)
    A1 = C['attn1'][:, :, QI][:, :, QQ]                                     # (n, H, 3)
    wq = A1 / A1.sum(2, keepdims=True)
    src = targets(D, P.head_moments(head)[0])[0]
    beta = beta_from(D['w_bayes']); beta_lab = beta_from(D['w_lab'])
    spread = D['o'][:, :, QL].max(1) - D['o'][:, :, QL].min(1)
    coup = (D['w_bayes'] * (np.log(np.maximum(D['w_bayes'], 1e-300)) - np.log(np.maximum(D['w_lab'], 1e-300)))).sum(1)
    cen = lambda a: a - a.mean(-1, keepdims=True)
    p2 = {}
    a_fit = {}
    for h in range(wq.shape[1]):
        lw = cen(np.log(wq[:, h]))
        for nm, b in (('beta', beta), ('beta_lab', beta_lab)):
            lb = cen(np.log(b))
            a = float((lw[tr] * lb[tr]).sum() / (lb[tr] ** 2).sum())
            a_fit[(h, nm)] = a
            resid = lw[te] - a * lb[te]
            r2 = 1 - (resid ** 2).sum() / (lw[te] ** 2).sum()
            # systematic residual: in which direction relative to beta is the model off? residual on the most-trusted
            # source (largest beta) by quintile of spread and coupling
            top = b[te].argmax(1)
            rt = resid[np.arange(len(te)), top]
            qs = lambda v: [float(rt[(v >= lo) & (v <= hi)].mean()) for lo, hi in zip(np.quantile(v, [0, .2, .4, .6, .8]), np.quantile(v, [.2, .4, .6, .8, 1]))]
            p2[f'H{h}_{nm}'] = dict(slope=a, r2=float(r2), resid_top_by_spread_q=qs(spread[te]), resid_top_by_coupling_q=qs(coup[te]),
                                   resid_top_mean=float(rt.mean()))
    res['part2'] = p2
    # residual of the model's log-weights vs beta as a function of the source's own SS rank and its query reading
    # deviation from the other two (does the model under/over-trust outlying readings?)
    ostar = D['o'][:, :, QL]
    dev = np.abs(ostar - (ostar.sum(1, keepdims=True) - ostar) / 2)          # |o_s - mean of the other two|
    for h in range(wq.shape[1]):
        lw = cen(np.log(wq[:, h])); lb = cen(np.log(beta))
        resid = (lw - a_fit[(h, 'beta')] * lb)[te]
        dq = dev[te]; edges = np.quantile(dq, [0, .2, .4, .6, .8, 1])
        res['part2'][f'H{h}_beta']['resid_by_outlyingness_q'] = [float(resid[(dq >= lo) & (dq <= hi)].mean()) for lo, hi in zip(edges[:-1], edges[1:])]
        res['part2'][f'H{h}_beta']['corr_resid_outlyingness'] = float(np.corrcoef(resid.ravel(), dq.ravel())[0, 1])
        lss = np.log(D['ss_resid'][te])
        res['part2'][f'H{h}_beta']['corr_resid_logSS'] = float(np.corrcoef(resid.ravel(), lss.ravel())[0, 1])

    mo = C['mlp_out0'][:, QQ]                                               # (n, 3, d)
    gs = {}
    for kind in ('suff', 'faith'):
        Z = zfeat(kind, D['o'], D['y'])
        g = fit_g(Z[tr].reshape(-1, Z.shape[-1]), mo[tr].reshape(-1, mo.shape[-1]))
        pred = g(Z[te].reshape(-1, Z.shape[-1])).reshape(len(te), S, -1)
        r2 = 1 - ((mo[te] - pred) ** 2).sum() / ((mo[te] - mo[te].reshape(-1, mo.shape[-1]).mean(0)) ** 2).sum()
        gs[kind] = g
        res.setdefault('part3_fit_r2', {})[kind] = float(r2)

    # ---------------- hooks (parametrised by the draw's o, y; beta is invariant)
    def hook_a(nm, heads_=None):
        bb = (beta if nm == 'beta' else beta_lab)[te]
        def f(a, sl):
            a = a.clone(); idx = np.arange(sl.start, sl.stop)
            row = a[:, :, QI]
            M = row[:, :, QQ].sum(-1, keepdim=True)
            for h in (range(a.shape[1]) if heads_ is None else heads_):
                lw = a_fit[(h, nm)] * np.log(bb[idx]); w = np.exp(lw - lw.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
                row[:, h, QQ] = M[:, h] * torch.from_numpy(w.astype(np.float32))
            a[:, :, QI] = row
            return a
        return {('attn', 1): f}

    def hook_b(kind, o, y):
        Z = zfeat(kind, o, y)
        G = gs[kind](Z.reshape(-1, Z.shape[-1])).reshape(len(o), S, -1).astype(np.float32)
        def f(x, sl):
            x = x.clone(); x[:, QQ] = torch.from_numpy(G[sl.start:sl.stop]); return x
        return {('mlp_out', 0): f}

    o0, y0 = D['o'][te], D['y'][te]
    s0, q0 = semb[te], qemb[te]
    Xt, mt = X[te], m[te]

    def avg_run(which, hookf=None, seed=0):
        rng = np.random.default_rng(1000 + seed)
        heads = []
        for k in range(K):
            Xd, md, od, yd = draw(rng, which, o0, y0, s0, q0, ds)
            if k == 0 and which:
                dd = dict(o=od, y=yd, v=D['v'][te], vidx=D['vidx'][te])
                _, _, mom, _ = W.references(dd)
                res.setdefault('bayes_invariance_maxabs', {})['+'.join(which)] = float(max(
                    np.abs(mom['bayes'][0] - D['mean_bayes'][te]).max(), np.abs(mom['bayes'][1] - D['var_bayes'][te]).max()))
            heads.append(run(model, Xd, md, hk=hookf(od, yd) if hookf else None))
        return mixture(heads)

    # Part 1
    for r_ in range(4):
        for sub in itertools.combinations(NUIS, r_):
            name = 'N:' + ('+'.join(sub) if sub else 'none')
            h = run(model, Xt, mt) if not sub else avg_run(sub)
            score(h, name, perep, split=name in ('N:none', 'N:id+rot+y'))
    # Part 4: mechanism idealisations alone and with all nuisances averaged
    mech = {'a_beta': lambda o, y: hook_a('beta'), 'a_beta_lab': lambda o, y: hook_a('beta_lab'),
            'b_suff': lambda o, y: hook_b('suff', o, y), 'b_faith': lambda o, y: hook_b('faith', o, y),
            'a_beta+b_suff': lambda o, y: {**hook_a('beta'), **hook_b('suff', o, y)},
            'a_beta_H0': lambda o, y: hook_a('beta', [0]), 'a_beta_H1': lambda o, y: hook_a('beta', [1])}
    for nm, hf in mech.items():
        score(run(model, Xt, mt, hk=hf(o0, y0)), nm, perep)
        score(avg_run(NUIS, hf), nm + '|N:all', perep)
    # Shapley over the 3 nuisances
    klv = {k: v['kl'] for k, v in res['kl'].items()}
    def v_of(sub):
        return klv['N:' + ('+'.join([x for x in NUIS if x in sub]) if sub else 'none')]
    sh = {}
    for x in NUIS:
        others = [y for y in NUIS if y != x]; tot = 0.0
        for r_ in range(3):
            for sub in itertools.combinations(others, r_):
                wgt = math.factorial(len(sub)) * math.factorial(2 - len(sub)) / math.factorial(3)
                tot += wgt * (v_of(set(sub)) - v_of(set(sub) | {x}))
        sh[x] = tot
    res['shapley_kl_removed'] = sh
    res['a_fit'] = {f'{h}_{nm}': v for (h, nm), v in a_fit.items()}
    np.savez_compressed(outp.replace('.json', '_perep.npz'), **{k.replace('|', '__').replace(':', '_').replace('+', 'p'): v for k, v in perep.items()})
    json.dump(res, open(outp, 'w'))


if __name__ == '__main__':
    main()
