"""Question-weighted surrogate fits for deviation prediction (PREREG-B amendment 7).

Question: the network's deviation from Bayes per episode, F = (mean, log-sd, KL(Bayes||net)) / (Update 9 across-episode sd).
Weighted fit at site s: minimise E||J_s (g - f)||^2 + eps E||g - f||^2, J_s = d F / d h_s (per sample, single noisy
downstream pass with the noise held fixed, autograd on the real net), eps = 1e-3 mean||J||_F^2 / dim.
Unweighted control: output MSE, same parameterisation; closed-form (initial) fits also scored.
End-to-end: whole chains trained jointly on the network's single-draw predictive mean and log-sd, common random numbers.
DAG: restricted attention masks; all attention weights refitted (weighted and unweighted), plus the real weights.
Evaluation: K = 64 noise draws, 2,000 held-out test episodes, R^2 of predicted vs actual deviation; noise floor.
usage: python qw.py RUN_DIR [--smoke] -> RUN_DIR/qw.json (smoke: RUN_DIR/qw_smoke.json)
"""
import copy, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import nnoise as N
import world as W
from sur_pfn import Sur
from sur_small import rrr
import devpred as DP
import dag as DG

torch.set_num_threads(1)
T = 16


def order(depth):
    return ['emb'] + [f'{k}{l}' for l in range(1, depth + 1) for k in 'amr']


# ------------------------------------------------------------------ real trace with stored noise
def trace_nz(model, X, valid, sigma, gen):
    allow = model.allow_mask(torch.from_numpy(valid)); st, nz = {}, {}
    with torch.no_grad():
        def z(name, t):
            e = torch.randn(t.shape, generator=gen); nz[name] = e; return t + sigma * e
        x = z('emb', model.f_emb(torch.from_numpy(X))); st['emb'] = x
        for l in range(1, model.depth + 1):
            xa = z(f'a{l}', model.f_a(l, x, allow)); st[f'a{l}'] = xa
            m = z(f'm{l}', model.f_m(l, xa)); st[f'm{l}'] = m
            x = z(f'r{l}', model.f_r(l, xa, m)); st[f'r{l}'] = x
    return st, nz, allow


def run_from(model, site, h, st, nz, allow, sigma, idx):
    cur = {k: v[idx] for k, v in st.items()}; cur[site] = h
    od = order(model.depth); al = allow[idx]
    for s in od[od.index(site) + 1:]:
        l = int(s[1:])
        if s[0] == 'a': v = model.f_a(l, cur['emb'] if l == 1 else cur[f'r{l - 1}'], al)
        elif s[0] == 'm': v = model.f_m(l, cur[f'a{l}'])
        else: v = model.f_r(l, cur[f'a{l}'], cur[f'm{l}'])
        cur[s] = v + sigma * nz[s][idx]
    return model.head(cur[od[-1]])


# ------------------------------------------------------------------ deviation features (torch, differentiable)
def moments_t(mu, ls, lg):
    w = torch.softmax(lg, -1); m = (w * mu).sum(1); v = (w * (torch.exp(2 * ls) + mu ** 2)).sum(1) - m ** 2
    return m, 0.5 * torch.log(v.clamp_min(1e-12))


def kl_t(mu, ls, lg, Bm, Bv, Bw, ngrid, model_range):
    """KL(Bayes || model mixture) by trapezoid quadrature; grid = +-9 sd of the Bayes comps (and of the model comps if
    model_range, the rule of tiny-pfn pfn.kl_grid)."""
    sd = torch.sqrt(Bv)
    lo = (Bm - 9 * sd).min(1).values; hi = (Bm + 9 * sd).max(1).values
    if model_range:
        lo = torch.minimum(lo, (mu - 9 * torch.exp(ls)).min(1).values); hi = torch.maximum(hi, (mu + 9 * torch.exp(ls)).max(1).values)
    g = (lo[:, None] + (hi - lo)[:, None] * torch.linspace(0, 1, ngrid, dtype=mu.dtype)[None]).detach()
    lpc = torch.log(Bw)[:, None, :] - 0.5 * torch.log(2 * math.pi * Bv) - 0.5 * (g[:, :, None] - Bm[:, None, :]) ** 2 / Bv
    lp = torch.logsumexp(lpc, 2)
    lqc = torch.log_softmax(lg, -1)[:, None, :] - ls[:, None, :] - 0.5 * math.log(2 * math.pi) - 0.5 * ((g[:, :, None] - mu[:, None, :]) / torch.exp(ls)[:, None, :]) ** 2
    lq = torch.logsumexp(lqc, 2)
    return torch.trapezoid(torch.exp(lp) * (lp - lq), g, dim=1)


def features_t(head, B, idx, sds):
    mu, ls, lg = head
    m, logsd = moments_t(mu, ls, lg)
    kl = kl_t(mu, ls, lg, B['m'][idx], B['v'], B['w'][idx], 201, False)
    return torch.stack([(m - B['mean'][idx]) / sds[0], (logsd - B['logsd'][idx]) / sds[1], kl / sds[2]], 1)


def bayes_t(d):
    ref, _, mom, wb = W.references(d)
    f = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)
    return dict(m=f(ref['bayes'][0]), v=f(ref['bayes'][1]), w=f(wb), mean=f(mom['bayes'][0]), logsd=f(0.5 * np.log(mom['bayes'][1])))


def jacobian(model, site, st, nz, allow, sigma, B, sds, bs=256):
    n = st[site].shape[0]; Dd = st[site][0].numel(); J = torch.empty(n, 3, Dd)
    for i in range(0, n, bs):
        idx = torch.arange(i, min(i + bs, n))
        h = st[site][idx].clone().requires_grad_(True)
        F = features_t(run_from(model, site, h, st, nz, allow, sigma, idx), B, idx, sds)
        for k in range(3):
            g, = torch.autograd.grad(F[:, k].sum(), h, retain_graph=k < 2)
            J[idx, k] = g.reshape(len(idx), -1)
    return J


# ------------------------------------------------------------------ surrogate modules (initialised from closed-form fits)
def lowrank_init(X, Y, r):
    B, mx, my = rrr(X.astype(np.float64), Y.astype(np.float64), r)
    U, s, Vt = np.linalg.svd(B, full_matrices=False)
    t = lambda a: nn.Parameter(torch.tensor(a, dtype=torch.float32))
    return t(U[:, :r] * s[:r]), t(Vt[:r].T.copy()), t(my - B @ mx)


class EmbLR(nn.Module):
    def __init__(self, model, X, r):
        super().__init__(); d = model.d
        with torch.no_grad(): Z = model.inp(torch.from_numpy(X)).reshape(-1, d).numpy()
        self.U, self.V, self.c = lowrank_init(X.reshape(-1, X.shape[-1]), Z, r); self.cost = r * (X.shape[-1] + d) + d
    def forward(self, X): return N.sphere(X @ self.V @ self.U.T + self.c)


class HidLR(nn.Module):
    def __init__(self, block, A, r):
        super().__init__(); self.ln2 = block.ln2; d = A.shape[-1]
        with torch.no_grad(): L = block.ln2(A).reshape(-1, d); Z = block.mlp[0](L)
        self.U, self.V, self.c = lowrank_init(L.numpy(), Z.numpy(), r); self.cost = r * 3 * d + 2 * d
    def forward(self, a): return N.sphere(Fn.gelu(self.ln2(a) @ self.V @ self.U.T + self.c))


class WriteLR(nn.Module):
    def __init__(self, block, Mh, r):
        super().__init__(); d = Mh.shape[-1] // 2
        with torch.no_grad(): Z = block.mlp[2](Mh).reshape(-1, d)
        self.U, self.V, self.c = lowrank_init(Mh.reshape(-1, 2 * d).numpy(), Z.numpy(), r); self.cost = r * 3 * d + d
    def forward(self, a, m): return N.sphere(a + m @ self.V @ self.U.T + self.c)


class AttnFP(nn.Module):
    """Fixed learnable pattern (init: data-average attention) + per-head value-output map, rank r (SVD of Wo_k Wv_k) or
    full (r = None: Wv_k, Wo_k). Value biases fold into c because pattern rows sum to 1."""
    def __init__(self, block, Abar, mask, r):
        super().__init__(); self.ln1 = block.ln1; h, dh = block.h, block.dh
        self.mask = mask; self.h = h
        self.P = nn.Parameter(torch.log(Abar.clamp_min(1e-6)).clone())
        Wv = block.qkv.weight[2 * h * dh:].detach(); bv = block.qkv.bias[2 * h * dh:].detach(); Wo = block.o.weight.detach(); bo = block.o.bias.detach()
        Us, Vs = [], []; c = bo.clone()
        for k in range(h):
            Wvk, Wok = Wv[k * dh:(k + 1) * dh], Wo[:, k * dh:(k + 1) * dh]; c = c + Wok @ bv[k * dh:(k + 1) * dh]
            if r:
                U, s, Vt = torch.linalg.svd(Wok @ Wvk); Us.append(U[:, :r] * s[:r]); Vs.append(Vt[:r].T)
            else:
                Us.append(Wok.clone()); Vs.append(Wvk.T.clone())
        self.U = nn.Parameter(torch.stack(Us)); self.V = nn.Parameter(torch.stack(Vs)); self.c = nn.Parameter(c)
        self.cost = int(mask.sum()) * h + self.U.numel() + self.V.numel() + self.c.numel()
    def forward(self, x):
        L = self.ln1(x); A = torch.softmax(self.P.masked_fill(~self.mask[None], float('-inf')), -1)    # (h, T, T)
        VL = torch.einsum('ntd,kdr->nktr', L, self.V)
        y = torch.einsum('kts,nksr->nktr', A, VL)
        return N.sphere(x + torch.einsum('nktr,kdr->ntd', y, self.U) + self.c)


class AttnMasked(nn.Module):
    """Real attention architecture with a fixed restricted mask; attention weights learnable (init = real)."""
    def __init__(self, block, mask):
        super().__init__(); self.ln1 = block.ln1; self.h, self.dh = block.h, block.dh
        self.qkv = copy.deepcopy(block.qkv).requires_grad_(True); self.o = copy.deepcopy(block.o).requires_grad_(True)
        self.mask = mask
        self.cost = sum(p.numel() for p in list(self.qkv.parameters()) + list(self.o.parameters())) + int(mask.sum())
    def forward(self, x):
        n = len(x); q, k, v = self.qkv(self.ln1(x)).split(self.h * self.dh, -1)
        sp = lambda t: t.view(n, T, self.h, self.dh).transpose(1, 2)
        s = (sp(q) @ sp(k).transpose(-1, -2) / math.sqrt(self.dh)).masked_fill(~self.mask[None, None], float('-inf'))
        return N.sphere(x + self.o((s.softmax(-1) @ sp(v)).transpose(1, 2).reshape(n, T, self.h * self.dh)))


def parents(site, st, X, idx):
    if site == 'emb': return (torch.from_numpy(X[idx.numpy()]),)
    l = int(site[1:])
    if site[0] == 'a': return (st['emb'][idx] if l == 1 else st[f'r{l - 1}'][idx],)
    if site[0] == 'm': return (st[f'a{l}'][idx],)
    return (st[f'a{l}'][idx], st[f'm{l}'][idx])


def errors(mod, site, st, nz, X, sigma, idx):
    target = (st[site][idx] - sigma * nz[site][idx]).reshape(len(idx), -1)
    return mod(*parents(site, st, X, idx)).reshape(len(idx), -1) - target


def fit_module(mod, site, st, nz, X, sigma, J, weighted, ntr, steps, bs=256, lr=3e-3, seed=0):
    Dd = st[site][0].numel(); eps = 1e-3 * float((J[:ntr] ** 2).sum((1, 2)).mean()) / Dd
    ps = [p for p in mod.parameters() if p.requires_grad]
    opt = torch.optim.Adam(ps, lr=lr); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps); g = torch.Generator().manual_seed(seed)
    for s in range(steps):
        idx = torch.randint(ntr, (bs,), generator=g); e = errors(mod, site, st, nz, X, sigma, idx)
        if weighted: loss = (torch.einsum('nkd,nd->nk', J[idx], e) ** 2).sum(1).mean() + eps * (e ** 2).sum(1).mean()
        else: loss = (e ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return mod


def val_metrics(mod, site, st, nz, X, sigma, J, ntr):
    idx = torch.arange(ntr, J.shape[0])
    with torch.no_grad():
        e = errors(mod, site, st, nz, X, sigma, idx); je = torch.einsum('nkd,nd->nk', J[idx], e)
    return dict(Jerr=float((je ** 2).sum(1).mean()), Jerr_mean=float((je[:, 0] ** 2).mean()), KL=float((e ** 2).sum(1).mean() / (2 * sigma ** 2)))


def build(site, level, model, st, X, Abar, allow0):
    if site == 'emb': return EmbLR(model, X, level)
    l = int(site[1:]); b = model.blocks[l - 1]
    if site[0] == 'a': return AttnFP(b, Abar[l], allow0, level)
    if site[0] == 'm': return HidLR(b, st[f'a{l}'], level)
    return WriteLR(b, st[f'm{l}'], level)


def chain_nzb(model, mods, Xi, al, sigma, nzb):
    x = (mods['emb'](Xi) if 'emb' in mods else model.f_emb(Xi)) + sigma * nzb['emb']
    for l in range(1, model.depth + 1):
        xa = (mods[f'a{l}'](x) if f'a{l}' in mods else model.f_a(l, x, al)) + sigma * nzb[f'a{l}']
        m = (mods[f'm{l}'](xa) if f'm{l}' in mods else model.f_m(l, xa)) + sigma * nzb[f'm{l}']
        x = (mods[f'r{l}'](xa, m) if f'r{l}' in mods else model.f_r(l, xa, m)) + sigma * nzb[f'r{l}']
    return model.head(x)


def train_e2e(model, mods, X, allow, sigma, st, sds, ntr, steps, bs=256, lr=1e-3, seed=0):
    """Joint training on the single-draw predictive mean and log-sd, fresh common random numbers every step."""
    ps = [p for m in mods.values() for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(ps, lr=lr); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps); g = torch.Generator().manual_seed(seed)
    for s in range(steps):
        idx = torch.randint(ntr, (bs,), generator=g); Xi = torch.from_numpy(X[idx.numpy()]); al = allow[idx]
        nzb = {k: torch.randn((bs,) + v.shape[1:], generator=g) for k, v in st.items()}
        with torch.no_grad():
            tm, tl = moments_t(*chain_nzb(model, {}, Xi, al, sigma, nzb))
        m, logsd = moments_t(*chain_nzb(model, mods, Xi, al, sigma, nzb))
        loss = (((m - tm) / sds[0]) ** 2 + ((logsd - tl) / sds[1]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return mods


# ------------------------------------------------------------------ evaluation with K draws
def deviations_t(head, Tt, ngrid=801, chunk=50):
    """Same quantities and KL quadrature rule as devpred.deviations (tiny-pfn pfn.kl_grid, 801 points), in torch float64."""
    mu, ls, lg = [torch.as_tensor(a, dtype=torch.float64) for a in head]
    m, logsd = moments_t(mu, ls, lg)
    Bm = torch.as_tensor(Tt['comp_means'], dtype=torch.float64); Bv = torch.as_tensor(Tt['comp_vars'], dtype=torch.float64)
    Bw = torch.as_tensor(Tt['w_bayes'], dtype=torch.float64)
    with torch.no_grad():
        kl = torch.cat([kl_t(mu[i:i + chunk], ls[i:i + chunk], lg[i:i + chunk], Bm[i:i + chunk], Bv, Bw[i:i + chunk], ngrid, True) for i in range(0, len(mu), chunk)])
    return dict(mean=m.numpy() - Tt['mean_bayes'], logsd=logsd.numpy() - 0.5 * np.log(Tt['var_bayes']), kl=kl.numpy())


def wrap(mods):
    out = {}
    for s, m in mods.items():
        if s[0] == 'a': out[s] = Sur(lambda x, al, m=m: m(x), 0, 'q')
        elif s[0] == 'r': out[s] = Sur(lambda a, mm, m=m: m(a, mm), 0, 'q')
        else: out[s] = Sur(lambda x, m=m: m(x), 0, 'q')
    return out


def run(run_dir, smoke=False):
    t0 = time.time()
    n_fit, ntr, steps, e2e_steps, n_test, K = (1024, 896, 50, 50, 300, 8) if smoke else (8192, 7168, 2000, 3000, 2000, 64)
    tag = 'qw_smoke' if smoke else 'qw'
    out = dict(K=K, n_test=n_test, n_fit=n_fit, steps=steps, e2e_steps=e2e_steps, rows=[], fits={})
    model, meta = N.load_run(run_dir); sg = meta['sigma']; depth = meta['depth']; model.requires_grad_(False)
    U9 = json.load(open(os.path.join(run_dir, 'devpred.json'))); sds = [U9['dev_sd'][k] for k in ('mean', 'logsd', 'kl')]
    exact_cost = {'emb': model.inp.weight.numel() + model.inp.bias.numel()}
    b0 = model.blocks[0]
    exact_cost['a'] = sum(p.numel() for p in list(b0.qkv.parameters()) + list(b0.o.parameters()))
    exact_cost['m'] = sum(p.numel() for p in b0.mlp[0].parameters()); exact_cost['r'] = sum(p.numel() for p in b0.mlp[2].parameters())
    out['exact_cost'] = exact_cost
    rng = np.random.default_rng(91_000); d = W.sample_np(n_fit, rng); X, valid = W.tokens_np(d, rng, meta['ds'])
    st, nz, allow = trace_nz(model, X, valid, sg, torch.Generator().manual_seed(91_001)); B = bayes_t(d)
    allow0 = torch.as_tensor(DG.base_allow()); assert bool((allow == allow0[None]).all())
    Abar = {}
    with torch.no_grad():
        for l in range(1, depth + 1):
            b = model.blocks[l - 1]; xin = (st['emb'] if l == 1 else st[f'r{l - 1}'])[:2048]; nb = len(xin)
            q, k, v = b.qkv(b.ln1(xin)).split(b.h * b.dh, -1); sp = lambda t: t.view(nb, T, b.h, b.dh).transpose(1, 2)
            Abar[l] = ((sp(q) @ sp(k).transpose(-1, -2)) / math.sqrt(b.dh)).masked_fill(~allow0[None, None], float('-inf')).softmax(-1).mean(0)
    sites = order(depth)
    levels = {'emb': [2, 4, 8], 'a': [2, 4, None], 'm': [2, 4, 8], 'r': [2, 4, 8]}
    Js, fitted = {}, {}
    for s in sites:
        ts = time.time(); J = jacobian(model, s, st, nz, allow, sg, B, sds)
        if s[0] == 'a': Js[s] = J
        out['fits'][s] = dict(J_F2=float(J[:ntr].pow(2).sum((1, 2)).mean()), J_F2_by_feature=[float(J[:ntr, k].pow(2).sum(1).mean()) for k in range(3)])
        for lev in levels['emb' if s == 'emb' else s[0]]:
            base = build(s, lev, model, st, X, Abar, allow0)
            for kind in ('c', 'u', 'w'):
                mod = copy.deepcopy(base)
                if kind != 'c': fit_module(mod, s, st, nz, X, sg, J, kind == 'w', ntr, steps)
                fitted[(s, lev, kind)] = mod
                out['fits'][s][f'{lev}:{kind}'] = val_metrics(mod, s, st, nz, X, sg, J, ntr)
        print(s, 'jacobian + fits %.0fs' % (time.time() - ts), 'J_F2 %.3g' % out['fits'][s]['J_F2'],
              {k: (round(v['Jerr'], 3), round(v['KL'], 1)) for k, v in out['fits'][s].items() if ':' in k}, flush=True)
        del J
    # chains at the <= 25% level and the rank-8 level: closed-form, unweighted, weighted, end-to-end (from weighted)
    chains = {}
    for ctag, lv in (('le25', {'emb': 4, 'a': 2, 'm': 4, 'r': 4}), ('r8', {'emb': 8, 'a': 4, 'm': 8, 'r': 8})):
        get = lambda s, kind: fitted[(s, lv['emb' if s == 'emb' else s[0]], kind)]
        for kind, nm in (('c', 'closed'), ('u', 'unweighted'), ('w', 'weighted')):
            chains[f'chain-{ctag}-{nm}'] = {s: get(s, kind) for s in sites}
        e2e = {s: copy.deepcopy(get(s, 'w')) for s in sites}
        ts = time.time(); chains[f'chain-{ctag}-e2e'] = train_e2e(model, e2e, X, allow, sg, st, sds, ntr, e2e_steps)
        print('e2e', ctag, '%.0fs' % (time.time() - ts), flush=True)
    # DAG: restricted attention, real weights renormalised / refitted unweighted / refitted weighted
    dags = {}
    for v in (['D1', 'D1+lat'] if depth == 2 else ['D1', 'D1-L2lat']):
        M = DG.masks(v, depth); md = {'real': {}, 'unweighted': {}, 'weighted': {}}
        for l, A in M.items():
            if np.array_equal(A, DG.base_allow()): continue
            mask = torch.as_tensor(A); b = model.blocks[l - 1]; s = f'a{l}'
            md['real'][s] = AttnMasked(b, mask).requires_grad_(False)
            for kind in ('unweighted', 'weighted'):
                md[kind][s] = fit_module(AttnMasked(b, mask), s, st, nz, X, sg, Js[s], kind == 'weighted', ntr, steps)
                out['fits'][s][f'DAG-{v}:{kind}'] = val_metrics(md[kind][s], s, st, nz, X, sg, Js[s], ntr)
            out['fits'][s][f'DAG-{v}:real'] = val_metrics(md['real'][s], s, st, nz, X, sg, Js[s], ntr)
        for kind in md: dags[f'DAG-{v}-{kind}'] = md[kind]
    out['fit_seconds'] = time.time() - t0
    print('fits done %.0fs' % out['fit_seconds'], flush=True)
    # ---- evaluation
    Tz = np.load(os.path.join(DP.TP, 'runs/testset.npz'))
    Tt = {k: Tz[k][:n_test] for k in ('comp_means', 'w_bayes', 'mean_bayes', 'var_bayes')}; Tt['comp_vars'] = Tz['comp_vars']
    Xt, vt = Tz['X'][:n_test], Tz['valid'][:n_test]
    def evaluate(sur, seed):
        h, _, _ = DP.marginal(model, Xt, vt, sg, sur, K, seed)
        return deviations_t(h, Tt)
    dev = evaluate({}, 1); dev2 = evaluate({}, 7)
    out['noise_floor'] = {k: DP.r2(dev[k], dev2[k]) for k in dev}; out['zero'] = {k: DP.r2(dev[k], np.zeros_like(dev[k])) for k in dev}
    out['dev_sd'] = {k: float(dev[k].std()) for k in dev}; out['dev_mean'] = {k: float(dev[k].mean()) for k in dev}
    if smoke:                                                                          # quadrature check against devpred.deviations
        h, m_, s_ = DP.marginal(model, Xt, vt, sg, {}, K, 1); ref = DP.deviations(h, m_, s_, Tt)
        out['quad_check_maxabs'] = {k: float(np.abs(ref[k] - dev[k]).max()) for k in dev}
    print('K=%d noise floor' % K, out['noise_floor'], 'zero', out['zero'], 'dev sd', out['dev_sd'], out.get('quad_check_maxabs'), flush=True)
    def score(name, mods):
        dh = evaluate(wrap(mods), 11)
        row = dict(config=name, cost=int(sum(m.cost for m in mods.values())), **{f'R2_{k}': DP.r2(dev[k], dh[k]) for k in dev})
        out['rows'].append(row)
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != 'config'}, flush=True)
        json.dump(out, open(os.path.join(run_dir, f'{tag}.json'), 'w'), indent=1)
    for name, mods in list(chains.items()) + list(dags.items()):
        score(name, mods)
    for (s, lev, kind), mod in fitted.items():
        score(f'{s}:{"fp" if s[0] == "a" else "LR"}{lev if lev else "full"}:{kind}', {s: mod})
    out['seconds'] = time.time() - t0
    json.dump(out, open(os.path.join(run_dir, f'{tag}.json'), 'w'), indent=1)
    print('done', out['seconds'], flush=True)


if __name__ == '__main__':
    run(sys.argv[1], smoke='--smoke' in sys.argv)
