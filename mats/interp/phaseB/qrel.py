"""E10.3 (PREREG-B amendment 17; triggered by the post-hoc E10.2b model, addendum 1; one-at-a-time shifts, addendum 2):
does layer 1 build the reliability signal that the layer-2 values use?
Per QQ position s: ridge decoder of standardised log own labelled SS from the layer-1 output x1 (fit episodes, one draw);
shift direction u_s = beta / |beta|^2. Patches move x1 at the QQ tokens by Delta u_s only where layer 2 computes the
values (or, control, only keys/queries; or, control, along a random direction orthogonal to beta with the same norm).
Net response d w / d Delta by central differences of autograd effective weights (K = 8, E9 draws); Bayes response by
scaling the labelled residuals so log SS moves by Delta SD. MSD ratios at Delta = +-0.25 with K = 64 (E8 draws).
usage: python qrel.py NET [NET ...] -> runs/NET/qrel.npz ; Bayes side cached in results/qrel_bayes.npz
"""
import math, os, sys, time
import numpy as np
import torch
import nnoise as N
import world as W
import devpred as DP
import qw
import qweights as QWT
import qe9
from qpatch import T, QT, QQ

torch.set_num_threads(1)
DELTA = 0.25; N4 = 4000


def lss_of(d):
    res = d['o'][:, :, :W.QL] - d['y'][:, None, :W.QL]; return np.log((res ** 2).sum(2) + 1e-3)


def decoders(model, sg, nm):
    p = os.path.join('runs', nm, 'qrel_dec.npz')
    if os.path.exists(p):
        z = np.load(p); return {k: z[k] for k in z.files}
    fit, _ = QWT.worlds(); X = torch.from_numpy(fit['X']); al = model.allow_mask(torch.from_numpy(fit['v'])); n = len(X)
    with torch.no_grad():
        x1 = qe9.layer1(model, X, al, sg, qe9.noise_list(n, model.d, torch.Generator().manual_seed(41)))[:, QQ, :].double().numpy()
    lss = lss_of(fit['d']); m_, s_ = lss.mean(), lss.std(); zt = (lss - m_) / s_
    beta, c, U, Rr = [], [], [], []
    rng = np.random.default_rng(7)
    for s in range(3):
        Xs = x1[:, s]; mu = Xs.mean(0); Xc = Xs - mu; lam = 1e-2 * n
        b = np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ (zt[:, s] - zt[:, s].mean()))
        beta.append(b); c.append(zt[:, s].mean() - mu @ b); u = b / (b @ b); U.append(u)
        r = rng.standard_normal(len(b)); r -= (r @ b) / (b @ b) * b; r *= np.linalg.norm(u) / np.linalg.norm(r); Rr.append(r)
    out = dict(beta=np.array(beta), c=np.array(c), u=np.array(U), r=np.array(Rr), lss_mean=np.array(m_), lss_sd=np.array(s_))
    tmp = p + '.tmp.npz'; np.savez(tmp, **out); os.replace(tmp, p); return out


def fwd_shift(model, X, al, sg, z, shift=None, where='values'):
    """shift: (3, d) tensor added to x1 at the QQ tokens, used only for the values ('values') or only for keys/queries ('keys')."""
    n = len(X); b = model.blocks[1]; x1 = qe9.layer1(model, X, al, sg, z)
    xs = x1
    if shift is not None:
        add = torch.zeros(T, x1.shape[-1]); add[QQ] = shift; xs = x1 + add[None]
    sp = lambda t: t.view(n, T, b.h, b.dh).transpose(1, 2)
    q0, k0, v0 = b.qkv(b.ln1(x1)).split(b.h * b.dh, -1)
    if shift is not None:
        q1, k1, v1 = b.qkv(b.ln1(xs)).split(b.h * b.dh, -1)
        if where == 'values': v0 = v1
        else: q0, k0 = q1, k1
    A = ((sp(q0) @ sp(k0).transpose(-1, -2)) / math.sqrt(b.dh)).masked_fill(~al[:, None], float('-inf')).softmax(-1)
    att = b.o((A @ sp(v0)).transpose(1, 2).reshape(n, T, b.h * b.dh))
    xa2 = N.sphere(x1 + att) + sg * z[4]
    u = xa2 + b.mlp[2](model.f_m(2, xa2) + sg * z[5]); r2 = N.sphere(u) + sg * z[6]
    return model.head(r2)


def eff_w(model, X, valid, sg, shift=None, where='values', K=8, bs=500):
    Ws = []
    for bi, i in enumerate(range(0, len(X), bs)):
        g = torch.Generator().manual_seed(2 + 1000 * bi)
        Xi = torch.from_numpy(X[i:i + bs]).clone().requires_grad_(True); al = model.allow_mask(torch.from_numpy(valid[i:i + bs])); m = 0
        for _ in range(K):
            mu, ls, lg = fwd_shift(model, Xi, al, sg, qe9.noise_list(len(Xi), model.d, g), shift, where)
            m = m + (torch.softmax(lg, -1) * mu).sum(-1) / K
        gr, = torch.autograd.grad(m.sum(), Xi); Ws.append(gr[:, QQ, qe9.VAL].numpy().astype(np.float64))
    return np.concatenate(Ws)


def bayes_side():
    p = 'results/qrel_bayes.npz'
    if os.path.exists(p):
        z = np.load(p); return {k: z[k] for k in z.files}
    fit, test = QWT.worlds(); sd = lss_of(fit['d']).std(); d = test['d']
    def scaled(srcs, sign):
        dd = {k: v.copy() for k, v in d.items()}
        for s in srcs:
            dd['o'][:, s, :W.QL] = d['y'][:, :W.QL] + (d['o'][:, s, :W.QL] - d['y'][:, :W.QL]) * math.exp(sign * DELTA * sd / 2)
        return QWT.bayes_weights(dd)
    out = dict(all=(scaled([0, 1, 2], 1) - scaled([0, 1, 2], -1)) / (2 * DELTA))
    for s in range(3): out[f'one{s}'] = (scaled([s], 1) - scaled([s], -1)) / (2 * DELTA)
    np.savez(p, **out); return out


def run(nm):
    t0 = time.time(); model, meta = N.load_run(os.path.join('runs', nm)); model.requires_grad_(False); sg = meta['sigma']
    D = decoders(model, sg, nm); Tz = np.load(os.path.join(DP.TP, 'runs/testset.npz')); X, valid = Tz['X'][:N4], Tz['valid'][:N4]
    # decoding R2 on test (one draw)
    with torch.no_grad():
        al = model.allow_mask(torch.from_numpy(valid)); x1 = qe9.layer1(model, torch.from_numpy(X), al, sg, qe9.noise_list(N4, model.d, torch.Generator().manual_seed(42)))[:, QQ, :].double().numpy()
    fit, test = QWT.worlds(); zt = (lss_of(test['d']) - D['lss_mean']) / D['lss_sd']
    zh = np.stack([x1[:, s] @ D['beta'][s] + D['c'][s] for s in range(3)], 1); dec_r2 = float(1 - ((zt - zh) ** 2).sum() / ((zt - zt.mean()) ** 2).sum())
    U = torch.as_tensor(D['u'], dtype=torch.float32); R = torch.as_tensor(D['r'], dtype=torch.float32)
    out = dict(dec_r2=np.array(dec_r2))
    for name, vec, where in (('val', U, 'values'), ('key', U, 'keys'), ('rnd', R, 'values')):
        out[f'all_{name}'] = (eff_w(model, X, valid, sg, DELTA * vec, where) - eff_w(model, X, valid, sg, -DELTA * vec, where)) / (2 * DELTA)
    for name, vec in (('val', U), ('rnd', R)):
        for s in range(3):
            m = torch.zeros(3, 1); m[s] = 1
            out[f'one{s}_{name}'] = (eff_w(model, X, valid, sg, DELTA * vec * m, 'values') - eff_w(model, X, valid, sg, -DELTA * vec * m, 'values')) / (2 * DELTA)
    # accuracy at +-Delta (value shift, all sources), K = 64
    Tt = {k: Tz[k][:N4] for k in ('comp_means', 'w_bayes', 'mean_bayes', 'var_bayes')}; Tt['comp_vars'] = Tz['comp_vars']
    heads = {v_: ([], [], []) for v_ in ('base', 'plus', 'minus')}; g = torch.Generator().manual_seed(1); Xt = torch.from_numpy(X); K = 64
    with torch.no_grad():
        for _ in range(K):
            z = qe9.noise_list(N4, model.d, g)
            for v_, sh in (('base', None), ('plus', DELTA * U), ('minus', -DELTA * U)):
                mu, ls, lg = fwd_shift(model, Xt, al, sg, z, sh, 'values')
                heads[v_][0].append(mu); heads[v_][1].append(ls); heads[v_][2].append(torch.log_softmax(lg, -1) - math.log(K))
    for v_ in heads:
        dv = qw.deviations_t(tuple(torch.cat(t, 1).numpy() for t in heads[v_]), Tt)
        for f_ in ('mean', 'logsd', 'kl'): out[f'{v_}_{f_}'] = dv[f_]
    np.savez(os.path.join('runs', nm, 'qrel.npz'), **out)
    print(nm, 'done %.0fs' % (time.time() - t0), 'decoding R2 %.3f' % dec_r2, flush=True)


if __name__ == '__main__':
    bayes_side()
    for nm in sys.argv[1:]:
        if not os.path.exists(os.path.join('runs', nm, 'qrel.npz')): run(nm)
