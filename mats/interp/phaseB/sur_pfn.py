"""Surrogate chains on the noisy tiny PFN (2L), PREREG-B amendment 4.

Sites emb, a1, m1, r1, a2, m2, r2; each h_site | parents ~ N(f_site(parents), sigma^2 I). A surrogate replaces f_site by
g_site on the same parents; exact chain rule: KL(real path || surrogate path) = sum_sites E ||f - g||^2 / (2 sigma^2).
(1) per-site KL vs parameters; (2) DV two-sided bounds on the mean NLL and the mean predictive mean vs plain sampling's
95% CI at equal forward passes; (3) skip-merge of the MLP block (m_l, r_l) with the linearised noise covariance.
usage: python sur_pfn.py RUN_DIR -> RUN_DIR/sur.json
"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as Fn
import nnoise as N
import pfn as P
import world as W
from mirun import trace
from sur_small import rrr, dv_bounds

torch.set_num_threads(1)
T = 16


def episodes(n, seed, ds=16):
    rng = np.random.default_rng(seed); d = W.sample_np(n, rng); X, v = W.tokens_np(d, rng, ds)
    return d, X, v


def sph(v): return N.sphere(v)


class Sur:
    """A surrogate for one site: callable on the site's parent tensors, with a parameter count."""
    def __init__(self, fn, cost, name): self.fn, self.cost, self.name = fn, cost, name


def fit_surrogates(model, X, valid, st, sigma):
    """Fit all surrogate families on real noisy paths (st = noisy states from trace)."""
    m = model; d = m.d; S = {}
    Xt = torch.from_numpy(X)
    with torch.no_grad():
        # emb: rank-r input map (per token, pooled)
        Xtok = X.reshape(-1, X.shape[-1]).astype(np.float64); Z = (m.inp(torch.from_numpy(X)).reshape(-1, d)).double().numpy()
        S['emb'] = [Sur(None, d * X.shape[-1] + d, 'exact')]
        for r in (2, 4, 8, 16):
            B, mx, mz = rrr(Xtok, Z, r); Bt = torch.tensor(B, dtype=torch.float32); ct = torch.tensor(mz - B @ mx, dtype=torch.float32)
            S['emb'].append(Sur(lambda Xi, Bt=Bt, ct=ct: sph(Xi @ Bt.T + ct), r * (X.shape[-1] + d) + d, f'LR{r}'))
        for l in range(1, m.depth + 1):
            b = m.blocks[l - 1]; xin = torch.as_tensor(st['emb'] if l == 1 else st[f'r{l - 1}'])
            allow = m.allow_mask(torch.from_numpy(valid))
            A = b.attn_out(xin, allow)                                       # (n, T, d)
            # attention: drop / fixed average pattern / rank-r affine sublayer on the flattened state
            S[f'a{l}'] = [Sur(None, 3 * d * d + 3 * d + d * d + d, 'exact'), Sur(lambda x, al: sph(x), 0, 'drop')]
            q, k, v = b.qkv(b.ln1(xin)).split(b.h * b.dh, -1); nB = len(xin)
            spl = lambda t: t.view(nB, T, b.h, b.dh).transpose(1, 2)
            att = ((spl(q) @ spl(k).transpose(-1, -2)) / math.sqrt(b.dh)).masked_fill(~allow[:, None], float('-inf')).softmax(-1)
            Abar = att.mean(0)                                                # (h, T, T)
            def fixedpat(x, al, b=b, Abar=Abar):
                n_ = len(x); q_, k_, v_ = b.qkv(b.ln1(x)).split(b.h * b.dh, -1)
                vv = v_.view(n_, T, b.h, b.dh).transpose(1, 2)
                return sph(x + b.o((Abar[None] @ vv).transpose(1, 2).reshape(n_, T, b.h * b.dh)))
            S[f'a{l}'].append(Sur(fixedpat, b.h * T * T + 2 * (d * d + d), 'fixedpat'))
            Xf = xin.reshape(nB, -1).double().numpy(); Af = A.reshape(nB, -1).double().numpy()
            for r in (16, 64, 256):
                B, mx, ma = rrr(Xf, Af, r); Bt = torch.tensor(B, dtype=torch.float32); ct = torch.tensor(ma - B @ mx, dtype=torch.float32)
                S[f'a{l}'].append(Sur(lambda x, al, Bt=Bt, ct=ct: sph(x + (x.reshape(len(x), -1) @ Bt.T + ct).view(len(x), T, -1)), r * 2 * T * d + T * d, f'LRattn{r}'))
            # MLP hidden: rank-r pre-activation; TopK transcoder
            xa = torch.as_tensor(st[f'a{l}']); LNa = b.ln2(xa).reshape(-1, d).double().numpy()
            Zm = b.mlp[0](b.ln2(xa)).reshape(-1, 2 * d).double().numpy()
            Gm = Fn.gelu(torch.as_tensor(Zm)).numpy()
            S[f'm{l}'] = [Sur(None, d * 2 * d + 2 * d, 'exact')]
            for r in (2, 4, 8, 16):
                B, mx, mz = rrr(LNa, Zm, r); Bt = torch.tensor(B, dtype=torch.float32); ct = torch.tensor(mz - B @ mx, dtype=torch.float32)
                S[f'm{l}'].append(Sur(lambda xa_, b=b, Bt=Bt, ct=ct: sph(Fn.gelu(b.ln2(xa_) @ Bt.T + ct)), r * 3 * d + 2 * d, f'LR{r}'))
            for Fz, kk in ((64, 4), (256, 16)):
                tc = train_tc(LNa, Gm, Fz, kk, seed=l)
                S[f'm{l}'].append(Sur(lambda xa_, b=b, tc=tc: sph(tc(b.ln2(xa_))), Fz * 3 * d + Fz + 2 * d, f'TC{Fz}k{kk}'))
            # MLP write: rank-r W2
            mm = torch.as_tensor(st[f'm{l}']).reshape(-1, 2 * d).double().numpy(); Zr = b.mlp[2](torch.as_tensor(st[f'm{l}'])).reshape(-1, d).double().numpy()
            S[f'r{l}'] = [Sur(None, 2 * d * d + d, 'exact')]
            for r in (2, 4, 8, 16):
                B, mx, mz = rrr(mm, Zr, r); Bt = torch.tensor(B, dtype=torch.float32); ct = torch.tensor(mz - B @ mx, dtype=torch.float32)
                S[f'r{l}'].append(Sur(lambda xa_, m_, Bt=Bt, ct=ct: sph(xa_ + m_ @ Bt.T + ct), r * 3 * d + d, f'LR{r}'))
    return S


def train_tc(Xin, Y, F, k, seed, steps=3000):
    with torch.enable_grad():
        return _train_tc(Xin, Y, F, k, seed, steps)


def _train_tc(Xin, Y, F, k, seed, steps):
    torch.manual_seed(seed)
    E = torch.nn.Linear(Xin.shape[1], F); Dm = torch.nn.Linear(F, Y.shape[1])
    opt = torch.optim.Adam(list(E.parameters()) + list(Dm.parameters()), lr=3e-3)
    Xt, Yt = torch.tensor(Xin, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32); g = torch.Generator().manual_seed(seed)
    def f(x):
        z = E(x); v, i = z.topk(k, -1); return Dm(torch.zeros_like(z).scatter(-1, i, torch.relu(v)))
    for s in range(steps):
        i = torch.randint(len(Xt), (2048,), generator=g); loss = ((f(Xt[i]) - Yt[i]) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return lambda x: f(x.reshape(-1, x.shape[-1])).view(*x.shape[:-1], Y.shape[1])


def site_values(model, sname, sur, X, valid, st, mu):
    """f (from the trace) and g on the same noisy parents, for one site."""
    m = model; allow = m.allow_mask(torch.from_numpy(valid))
    with torch.no_grad():
        if sname == 'emb':
            g = sur.fn(torch.from_numpy(X))
        elif sname[0] == 'a':
            l = int(sname[1:]); xin = torch.as_tensor(st['emb'] if l == 1 else st[f'r{l - 1}']); g = sur.fn(xin, allow)
        elif sname[0] == 'm':
            g = sur.fn(torch.as_tensor(st[f'a{sname[1:]}']))
        else:
            l = sname[1:]; g = sur.fn(torch.as_tensor(st[f'a{l}']), torch.as_tensor(st[f'm{l}']))
    return mu[sname], g.numpy()


def chain_forward(model, X, valid, sigma, sur, gen):
    """Forward pass with surrogate maps at the sites in `sur` (dict site -> Sur), real elsewhere. Returns head."""
    m = model; allow = m.allow_mask(torch.from_numpy(valid)); z = lambda t: t + sigma * torch.randn(t.shape, generator=gen)
    with torch.no_grad():
        Xt = torch.from_numpy(X)
        x = z(sur['emb'].fn(Xt) if 'emb' in sur else m.f_emb(Xt))
        for l in range(1, m.depth + 1):
            xa = z(sur[f'a{l}'].fn(x, allow) if f'a{l}' in sur else m.f_a(l, x, allow))
            mm = z(sur[f'm{l}'].fn(xa) if f'm{l}' in sur else m.f_m(l, xa))
            x = z(sur[f'r{l}'].fn(xa, mm) if f'r{l}' in sur else m.f_r(l, xa, mm))
        return m.head(x)


def outputs(head, ystar):
    mu, ls, lg = [t.numpy() for t in head]
    nll = P.head_nll_np((mu, ls, lg), ystar)
    w = np.exp(lg - lg.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
    return {'nll': nll, 'mean': (w * mu).sum(1)}


def run(run_dir):
    t_all = time.time()
    model, meta = N.load_run(run_dir); sg = meta['sigma']; s2x2 = 2 * sg ** 2
    dF, XF, vF = episodes(2 ** 14, 81_000); dE, XE, vE = episodes(2 ** 14, 82_000)
    gen = torch.Generator().manual_seed(83_000)
    muF, stF = trace(model, XF, vF, sg, gen); muE, stE = trace(model, XE, vE, sg, gen)
    S = fit_surrogates(model, XF, vF, stF, sg)
    out = dict(sigma=sg, sites={}, configs=[], merges=[])
    for sname, lst in S.items():
        rows = {}
        for sur in lst:
            if sur.fn is None: rows[sur.name] = dict(KL=0.0, cost=sur.cost); continue
            f, g = site_values(model, sname, sur, XE, vE, stE, muE)
            err = ((f - g) ** 2).reshape(len(f), -1).sum(1) / s2x2
            rows[sur.name] = dict(KL=float(err.mean()), KL_se=float(err.std() / math.sqrt(len(err))), cost=int(sur.cost))
        out['sites'][sname] = rows
        print(sname, {k: (round(v['KL'], 1), v['cost']) for k, v in rows.items()}, flush=True)
    # ---- DV bounds: real reference and surrogate chains
    dR, XR, vR = episodes(2 ** 16, 84_000); ys = dR['y'][:, W.QL]
    with torch.no_grad():
        heads = [chain_forward(model, XR[i:i + 4096], vR[i:i + 4096], sg, {}, torch.Generator().manual_seed(85_000 + i)) for i in range(0, len(XR), 4096)]
    Fr = {k: np.concatenate([outputs(h, ys[i * 4096:(i + 1) * 4096])[k] for i, h in enumerate(heads)]) for k in ('nll', 'mean')}
    truth = {k: float(v.mean()) for k, v in Fr.items()}; sd = {k: float(v.std()) for k, v in Fr.items()}
    nKL, nQ = len(XE), 2 ** 13; n_eq = 2 * nKL + nQ
    dQ, XQ, vQ = episodes(nQ, 86_000); yq = dQ['y'][:, W.QL]
    configs = {}
    for sname, lst in S.items():
        for sur in lst:
            if sur.fn is not None: configs[f'{sname}:{sur.name}'] = {sname: sur}
    pick = lambda sname, nm: next(s for s in S[sname] if s.name == nm)
    configs['all-moderate'] = {'emb': pick('emb', 'LR8'), 'a1': pick('a1', 'LRattn64'), 'm1': pick('m1', 'LR8'), 'r1': pick('r1', 'LR8'),
                               'a2': pick('a2', 'LRattn64'), 'm2': pick('m2', 'LR8'), 'r2': pick('r2', 'LR8')}
    for cname, sur in configs.items():
        KL = sum(out['sites'][s][g.name]['KL'] for s, g in sur.items()); cost = sum(out['sites'][s][g.name]['cost'] for s, g in sur.items())
        with torch.no_grad():
            hq = [chain_forward(model, XQ[i:i + 4096], vQ[i:i + 4096], sg, sur, torch.Generator().manual_seed(87_000 + i)) for i in range(0, nQ, 4096)]
        Fq = {k: np.concatenate([outputs(h, yq[i * 4096:(i + 1) * 4096])[k] for i, h in enumerate(hq)]) for k in ('nll', 'mean')}
        row = dict(config=cname, KL=KL, cost=cost, n_eq=n_eq)
        for f in ('nll', 'mean'):
            lo, hi, _, _ = dv_bounds(KL, Fq[f]); ci = 2 * 1.96 * sd[f] / math.sqrt(n_eq)
            mc = 2 * 1.96 * float(Fq[f].std()) / math.sqrt(nQ)
            row[f] = dict(truth=truth[f], q_mean=float(Fq[f].mean()), q_sd=float(Fq[f].std()), lo=lo, hi=hi, width=hi - lo, width_mc=hi - lo + mc, ci_width=ci, width_over_ci=(hi - lo) / ci, width_mc_over_ci=(hi - lo + mc) / ci)
        out['configs'].append(row)
        print(cname, 'KL %.1f' % KL, {f: (round(row[f]['lo'], 3), round(row[f]['hi'], 3), round(row[f]['truth'], 3), round(row[f]['width_over_ci'], 1)) for f in ('nll', 'mean')}, flush=True)
        json.dump(out, open(os.path.join(run_dir, 'sur.json'), 'w'), indent=1)
    # ---- skip-merge of the MLP block (m_l, r_l): per token, a -> r with sigma^2 (I + J J^T), J = d f_r / d m at f_m(a)
    for l in range(1, model.depth + 1):
        b = model.blocks[l - 1]; d = model.d
        a = torch.as_tensor(stE[f'a{l}'][:1024]).reshape(-1, d)            # (n*T, d) token states
        with torch.no_grad():
            m0 = sph(b.hidden_pre(a)); Mr = sph(b.write_mlp(a, m0))
        def fr(mv, av): return sph(b.write_mlp(av, mv))
        Js = []
        for i in range(0, 2048):
            Js.append(torch.autograd.functional.jacobian(lambda mv: fr(mv, a[i]), m0[i]).numpy())
        J = np.stack(Js); Sig = sg ** 2 * (np.eye(d)[None] + J @ np.transpose(J, (0, 2, 1)))
        # Gaussian check (moments): real two-step kernel sampled for 64 token inputs
        kls = []
        for i in range(64):
            with torch.no_grad():
                av = a[i:i + 1].repeat(4096, 1); mv = m0[i:i + 1] + sg * torch.randn(4096, 2 * d); rv = fr(mv, av) + sg * torch.randn(4096, d)
            R = rv.double().numpy(); mR, CR = R.mean(0), np.cov(R.T)
            Si = np.linalg.inv(Sig[i]); dm = mR - Mr[i].double().numpy()
            kls.append(float(0.5 * (np.trace(Si @ CR) - d + np.linalg.slogdet(Sig[i])[1] - np.linalg.slogdet(CR)[1] + dm @ Si @ dm)))
        Sbar = Sig.mean(0); Sbi = np.linalg.inv(Sbar); ldb = np.linalg.slogdet(Sbar)[1]
        Mn = Mr[:2048].double().numpy()
        def klg(Mhat):
            r_ = Mn - Mhat
            return float(0.5 * (np.einsum('de,ned->n', Sbi, Sig) - d + ldb - np.linalg.slogdet(Sig)[1] + np.einsum('nd,de,ne->n', r_, Sbi, r_)).mean() * T)
        merged = {'exact-merge': dict(KL=klg(Mn), cost=out['sites'][f'm{l}']['exact']['cost'] + out['sites'][f'r{l}']['exact']['cost'] + d * (d + 1) // 2)}
        aF = torch.as_tensor(stF[f'a{l}'][:4096]).reshape(-1, d)
        with torch.no_grad():
            MF = sph(b.write_mlp(aF, sph(b.hidden_pre(aF)))).double().numpy(); LF = b.ln2(aF).double().numpy(); LE = b.ln2(a[:2048]).double().numpy()
        aFd = aF.double().numpy(); aEd = a[:2048].double().numpy()
        for r in (2, 4, 8, 16):
            B, mx, my = rrr(LF, MF - aFd, r)                                 # residual-preserving: r ~ sphere(a + B LN2(a) + c)
            Mhat = (torch.as_tensor(aEd + LE @ B.T + (my - B @ mx))); Mhat = sph(Mhat).double().numpy()
            merged[f'mergeLR{r}'] = dict(KL=klg(Mhat), cost=r * 2 * d + d + d * (d + 1) // 2)
        sep = {f'm:{nm1}+r:{nm2}': out['sites'][f'm{l}'][nm1]['KL'] + out['sites'][f'r{l}'][nm2]['KL'] for nm1 in ('LR4', 'LR8', 'LR16', 'exact') for nm2 in ('LR4', 'LR8', 'LR16', 'exact')}
        sepc = {f'm:{nm1}+r:{nm2}': out['sites'][f'm{l}'][nm1]['cost'] + out['sites'][f'r{l}'][nm2]['cost'] for nm1 in ('LR4', 'LR8', 'LR16', 'exact') for nm2 in ('LR4', 'LR8', 'LR16', 'exact')}
        out['merges'].append(dict(block=l, gauss_moment_KL_mean_per_token=float(np.mean(kls)), gauss_moment_KL_max=float(np.max(kls)),
                                  merged=merged, separate={k: dict(KL=sep[k], cost=sepc[k]) for k in sep}))
        print('merge MLP block', l, 'gauss check (per token) %.3f max %.3f' % (np.mean(kls), np.max(kls)), {k: (round(v['KL'], 1), v['cost']) for k, v in merged.items()}, flush=True)
        json.dump(out, open(os.path.join(run_dir, 'sur.json'), 'w'), indent=1)
    out['seconds'] = time.time() - t_all
    json.dump(out, open(os.path.join(run_dir, 'sur.json'), 'w'), indent=1)
    print('done', out['seconds'], flush=True)


if __name__ == '__main__':
    run(sys.argv[1])
