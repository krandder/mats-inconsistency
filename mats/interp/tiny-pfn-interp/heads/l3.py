"""PREREG Amendment 6: what the third layer computes (probes, readout weights, replacements).
usage: python l3.py RUN_DIR TAG OUT_JSON
"""
import json, os, sys
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from locate import TP, W, NTR, QI, Q, QL, S, targets, probe_set, per_source_to_tokens, lab_idx, qq_idx
from steps import run
from budget import beta_from
sys.path.insert(0, HERE)
import pfn2 as P

torch.set_num_threads(1)
QQ = qq_idx(); LI = lab_idx()


def sites(model, X, m, C):
    with torch.no_grad():
        emb = np.concatenate([model.inp(torch.from_numpy(X[i:i + 2000])).numpy() for i in range(0, len(X), 2000)])
    out = {'emb': emb}
    prev = emb
    for l in range(len(model.blocks)):
        out[f'attn{l + 1}'] = prev + C[f'attn_out{l}']
        out[f'mlp{l + 1}'] = C[f'resid{l}']
        prev = C[f'resid{l}']
    return out


def main():
    rd, tag, outp = sys.argv[1:4]
    meta = json.load(open(os.path.join(rd, 'meta.json')))
    D = dict(np.load(os.path.join(TP, 'runs', 'testset.npz')))
    X, m = D['X'], D['valid']
    n = len(X); tr = np.arange(NTR); te = np.arange(NTR, n)
    trained = P.build(meta); trained.load_state_dict(torch.load(os.path.join(rd, 'model.pt'), map_location='cpu'))
    torch.manual_seed(meta['seed']); untrained = P.build(meta)
    perm = np.random.default_rng(0).permutation(NTR)
    L = len(trained.blocks)
    res = dict(tag=tag, depth=L)
    beta = beta_from(D['w_bayes']); beta_lab = beta_from(D['w_lab'])
    coup = (D['w_bayes'] * (np.log(np.maximum(D['w_bayes'], 1e-300)) - np.log(np.maximum(D['w_lab'], 1e-300)))).sum(1)
    topq = coup[te] >= np.quantile(coup[te], 0.8)
    store = {}
    for nm, model in (('trained', trained), ('untrained', untrained)):
        head, C = run(model, X, m, cache=True)
        A = sites(model, X, m, C)
        src = targets(D, P.head_moments(head)[0])[0]
        tgt_src = dict(E_lab=src['E_lab'], E_bayes=src['E_bayes'], p_bayes=src['p_bayes'], beta=src['beta'],
                       dE=src['E_bayes'] - src['E_lab'], dp=src['p_bayes'] - src['p_lab'])
        tgt_qt = dict(dmean=D['mean_bayes'] - D['mean_a'], dmean_coup=D['mean_bayes'] - D['mean_lab'], logvar=np.log(D['var_bayes']))
        pr = {}
        for site, F in A.items():
            for t, v in tgt_src.items():
                for kind, idx in (('QQ', QQ), ('LAB', LI)):
                    R, Rs, _ = probe_set(F[:, idx], per_source_to_tokens(v, kind), perm)
                    pr[f'{t}|{kind}|{site}'] = (float(np.mean(R)), float(np.mean(Rs)))
            for t, v in tgt_qt.items():
                R, Rs, _ = probe_set(F[:, QI:QI + 1], v[:, None, None], perm)
                pr[f'{t}|QT|{site}'] = (float(np.mean(R)), float(np.mean(Rs)))
        res[f'probes_{nm}'] = pr
        store[nm] = (head, C, A, src)
        print(tag, nm, 'probes done', flush=True)
    head, C, A, src = store['trained']
    # (2) readout weights per layer / head
    cen = lambda a: a - a.mean(-1, keepdims=True)
    rw = {}
    for l in range(L):
        Aq = C[f'attn{l}'][:, :, QI][:, :, QQ]
        wq = Aq / Aq.sum(2, keepdims=True)
        for h in range(wq.shape[1]):
            lw = cen(np.log(np.maximum(wq[:, h], 1e-30)))
            d = {}
            for bn, b in (('beta', beta), ('beta_lab', beta_lab)):
                lb = cen(np.log(b)); a = float((lw[tr] * lb[tr]).sum() / (lb[tr] ** 2).sum())
                rr = lw[te] - a * lb[te]
                d[bn] = dict(a=a, r2=float(1 - (rr ** 2).sum() / (lw[te] ** 2).sum()))
            d['qq_mass'] = float(Aq[te, h].sum(1).mean())
            rw[f'L{l + 1}H{h}'] = d
    res['readout'] = rw
    # (3) replacements
    Tk = {'comp_means': D['comp_means'][te], 'w_bayes': D['w_bayes'][te], 'comp_vars': D['comp_vars']}
    Xt, mt = X[te], m[te]
    dE = src['E_bayes'] - src['E_lab']

    def klrun(hk=None):
        h = run(trained, Xt, mt, hk=hk)
        kl = P.kl_grid(Tk, h, chunk=200)
        return dict(kl=float(kl.mean()), kl_topcoup=float(kl[topq].mean()), nll=float(P.head_nll_np(h, D['ystar'][te]).mean()))
    rep = {'model': klrun()}
    rng = np.random.default_rng(5)
    for l in range(1, L):                                                    # after block l (resid index l-1)
        Hq = C[f'resid{l - 1}'][:, QQ]                                         # (n, 3, d)
        Atr = np.concatenate([Hq[tr].reshape(-1, Hq.shape[-1]), np.ones((len(tr) * S, 1))], 1)
        coef, *_ = np.linalg.lstsq(Atr, dE[tr].reshape(-1), rcond=None)
        b, c = coef[:-1], float(coef[-1])
        pred = Hq[te] @ b + c
        rep[f'dE_fit_r2_after_block{l}'] = float(1 - ((dE[te] - pred) ** 2).sum() / ((dE[te] - dE[te].mean()) ** 2).sum())
        u = rng.standard_normal(len(b)); u /= np.linalg.norm(u)
        bt = torch.from_numpy(b.astype(np.float32)); ut = torch.from_numpy(u.astype(np.float32)); nb = float(np.linalg.norm(b))

        def mk(rand):
            def f(x, sl):
                x = x.clone(); hq = x[:, QQ]
                dec = hq @ bt + c                                                 # (B, 3)
                if rand:
                    hq = hq - (dec / nb)[..., None] * ut
                else:
                    hq = hq - (dec / nb ** 2)[..., None] * bt
                x[:, QQ] = hq
                return x
            return f
        rep[f'A_dE_{l}'] = klrun({('resid', l - 1): mk(False)})
        rep[f'A_rand_{l}'] = klrun({('resid', l - 1): mk(True)})
    if L >= 3:
        mu_a = torch.from_numpy(C['attn_out2'][tr][:, QI].mean(0)); mu_m = torch.from_numpy(C['mlp_out2'][tr][:, QI].mean(0))
        def setqt(val):
            def f(x, sl):
                x = x.clone(); x[:, QI] = val; return x
            return f
        rep['A_attn3'] = klrun({('attn_out', 2): setqt(mu_a)})
        rep['A_mlp3'] = klrun({('mlp_out', 2): setqt(mu_m)})
        rep['A_L3'] = klrun({('attn_out', 2): setqt(mu_a), ('mlp_out', 2): setqt(mu_m)})
        ma = torch.from_numpy(C['attn_out1'][tr][:, QQ].reshape(-1, C['attn_out1'].shape[-1]).mean(0))
        mm = torch.from_numpy(C['mlp_out1'][tr][:, QQ].reshape(-1, C['mlp_out1'].shape[-1]).mean(0))
        def setqq(val):
            def f(x, sl):
                x = x.clone(); x[:, QQ] = val; return x
            return f
        rep['A_skipQQ2'] = klrun({('attn_out', 1): setqq(ma), ('mlp_out', 1): setqq(mm)})
    res['replace'] = rep
    print(tag, json.dumps(rep)[:600], flush=True)
    json.dump(res, open(outp, 'w'))


if __name__ == '__main__':
    main()
