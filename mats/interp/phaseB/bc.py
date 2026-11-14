"""Event-specific backward certificate by the output martingale (PREREG-B amendment 8).

Score Y, event A_t = {Y > t}. phi_l(h) = E[Y | h_l = h] (Doob martingale of the score). The increment at noise site l is a
sigma * Lambda_l Lipschitz function of that site's Gaussian noise (Lambda_l = sup ||grad_{h_l} phi_l|| over reachable states),
hence conditionally sub-Gaussian, and Azuma gives
    P(A_t) <= E_{x ~ exact prior}[min(1, exp(-(t - phi_0(x) - delta)_+^2 / (2 sigma^2 sum_l Lambda_l^2)))] + P(escape).
Reachable: site output on its sphere plus noise with ||Z_l|| <= rho_l (chi^2 tail 2^-60 per site).
Lambda_l: search sup (sampled real-path states from the prior and from tail-tilted inputs, then gradient ascent within the
reachable set, maxima re-evaluated with fresh downstream draws). Expectation over the prior by importance sampling (95% UCB).
Diagnostics: O1 = same formula with rms gradient norms (not a bound); O2 = P(phi_0(x) > t); C2 (PFN) = output ceiling.
usage: python bc.py small SIGMA      -> results/bc_small_s{SIGMA}.json
       python bc.py pfn RUN_DIR      -> RUN_DIR/bc.json
"""
import json, math, os, sys, time
import numpy as np
import torch
from scipy.stats import chi2, beta as betad, norm

torch.set_num_threads(1)
LOG2 = math.log(2)
RHO_LP = 60


def rho_of(D):
    return math.sqrt(chi2.isf(2.0 ** -RHO_LP, D))


def ucb(vals):
    m = float(vals.mean()); se = float(vals.std() / math.sqrt(len(vals))); return m + 1.645 * se, m, se


def samp_bits(p, n_eq):
    k = int(round(p * n_eq)); return -math.log2(float(betad.ppf(0.95, k + 1, max(n_eq - k, 1))))


# =========================================================================== small nets (dec.py, width 8, L = 6)
def ln0_t(a):
    c = a - a.mean(-1, keepdim=True); return c * (math.sqrt(a.shape[-1]) / c.norm(dim=-1, keepdim=True).clamp_min(1e-9))


class SmallNet:
    def __init__(self, net, w, b, sigma):
        self.W = [torch.tensor(W) for W in net.W]; self.b = [torch.tensor(b_) for b_ in net.b]
        self.w = torch.tensor(w); self.bo = float(b); self.sg = sigma; self.L = len(self.W); self.d = net.d

    def g(self, l, h):                                                   # layer l = 1..L map
        return ln0_t(torch.tanh(h @ self.W[l - 1].T + self.b[l - 1]))

    def phi(self, l, h, Zd):
        """E[Y | h_l = h] by K = len(Zd) downstream draws (exact for l >= L-1). h: (n, dim); Zd: (K, L, d)."""
        if l == self.L: return h @ self.w + self.bo
        if l == self.L - 1: return self.g(self.L, h) @ self.w + self.bo
        n, K = h.shape[0], Zd.shape[0]
        hh = h[:, None, :].expand(n, K, h.shape[-1])
        for j in range(l + 1, self.L):
            hh = self.g(j, hh) + self.sg * Zd[None, :, j - 1, :]
        return (self.g(self.L, hh) @ self.w + self.bo).mean(1)

    def phi_se(self, h, Zd):
        """phi_0 at inputs h (n, 4) with its Monte Carlo standard error."""
        n, K = h.shape[0], Zd.shape[0]
        hh = h[:, None, :].expand(n, K, h.shape[-1])
        for j in range(1, self.L):
            hh = self.g(j, hh) + self.sg * Zd[None, :, j - 1, :]
        v = self.g(self.L, hh) @ self.w + self.bo
        return v.mean(1), v.std(1) / math.sqrt(K)

    def paths(self, x, gen):
        hs = [x]; h = x
        for l in range(1, self.L + 1):
            h = self.g(l, h) + self.sg * torch.randn(h.shape[0], self.d, generator=gen, dtype=h.dtype); hs.append(h)
        return hs


def grad_norm(fn, h):
    h = h.detach().clone().requires_grad_(True)
    g, = torch.autograd.grad(fn(h).sum(), h)
    return g.reshape(len(h), -1).norm(dim=1).detach()


def run_small(sigma):
    import dec as Dc, e2 as E2
    t_all = time.time(); torch.manual_seed(0)
    net, w, b = Dc.train_net_ro(8, sigma); S = SmallNet(net, w, b, sigma); L, d = S.L, S.d
    D = json.load(open(os.path.join(E2.OUT, f'dec_d8_s{sigma}.json'))); ts, ptrue = np.array(D['t']), np.array(D['p_true']); rate = D['rate']
    gen = torch.Generator().manual_seed(1); dt = torch.float64
    t_cert = time.time()
    rho = rho_of(d); p_esc = L * 2.0 ** -RHO_LP
    # ---- sampled states: prior and tail-tilted inputs (x shifted along the readout's input gradient)
    Zfix = torch.randn(512, L, d, generator=gen, dtype=dt)
    x0 = torch.randn(4096, 4, generator=gen, dtype=dt)
    gx = torch.autograd.functional.jacobian(lambda x: S.phi(0, x[None], Zfix[:64])[0], torch.zeros(4, dtype=dt))
    xt = torch.randn(4096, 4, generator=gen, dtype=dt) + 3.0 * gx / gx.norm()
    hs_p, hs_t = S.paths(x0, gen), S.paths(xt, gen)
    sites = {}
    for l in range(1, L):                                               # noise sites 1..L-1 (site L: ||w|| exact)
        fn = lambda h, l=l: S.phi(l, h, Zfix[:256])
        gp = torch.cat([grad_norm(fn, hs_p[l][i:i + 512]) for i in range(0, 4096, 512)])
        gt = torch.cat([grad_norm(fn, hs_t[l][i:i + 512]) for i in range(0, 4096, 512)])
        allg = torch.cat([gp, gt]); allh = torch.cat([hs_p[l], hs_t[l]])
        # gradient ascent within the reachable set {u + sigma z: u on the LN0 sphere, ||z|| <= rho}
        top = torch.argsort(-allg)[:128]
        rnd = torch.randn(128, d, generator=gen, dtype=dt) * 2
        v0 = torch.cat([allh[top], rnd]); v = v0.clone().requires_grad_(True); z = torch.zeros_like(v0).requires_grad_(True)
        opt = torch.optim.Adam([v, z], lr=0.05)
        for it in range(150):
            h = ln0_t(v) + sigma * z
            g, = torch.autograd.grad(S.phi(l, h, Zfix[:128]).sum(), h, create_graph=True)
            loss = -(g ** 2).sum(); opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                nz = z.norm(dim=1, keepdim=True); z.mul_(torch.clamp(rho / nz.clamp_min(1e-12), max=1.0))
        with torch.no_grad(): hfin = ln0_t(v) + sigma * z
        Zfresh = torch.randn(2048, L, d, generator=gen, dtype=dt)
        fn2 = lambda h, l=l: S.phi(l, h, Zfresh)
        ga = torch.cat([grad_norm(fn2, hfin[i:i + 32]) for i in range(0, len(hfin), 32)])
        gs = torch.cat([grad_norm(fn2, allh[top][i:i + 32]) for i in range(0, 128, 32)])
        sites[l] = dict(rms_prior=float(gp.pow(2).mean().sqrt()), max_sampled=float(gs.max()), max_ascent=float(ga.max()),
                        Lambda=float(max(gs.max(), ga.max())))
        print('site', l, {k: round(v_, 3) for k, v_ in sites[l].items()}, flush=True)
    wn = float(S.w.norm())
    V = sigma ** 2 * (wn ** 2 + sum(s['Lambda'] ** 2 for s in sites.values()))
    Vrms = sigma ** 2 * (wn ** 2 + sum(s['rms_prior'] ** 2 for s in sites.values()))
    Bphi = wn * math.sqrt(d) + abs(S.bo)
    delta_esc = sum(2 * (Bphi + s['Lambda'] * sigma * (rho + 3)) for s in sites.values()) * 2.0 ** -RHO_LP
    # ---- expectation over the exact input prior: multiple importance sampling (mixture of mean shifts)
    Zphi = torch.randn(1024, L, d, generator=gen, dtype=dt)
    def phi0_se(x):
        out = [S.phi_se(x[i:i + 64], Zphi) for i in range(0, len(x), 64)]
        return torch.cat([o[0] for o in out]), torch.cat([o[1] for o in out])
    centres = [torch.zeros(4, dtype=dt)]
    for t in ts:
        for VV in (V, Vrms, 1e-4):                                      # C1, O1, O2 (narrow: indicator-like)
            best = None
            for r in range(3):
                x = (torch.randn(4, generator=gen, dtype=dt) * 0.5 + 2.0 * gx / gx.norm()).requires_grad_(True); opt = torch.optim.Adam([x], lr=0.05)
                for it in range(200):
                    ph = S.phi(0, x[None], Zfix[:128])[0]
                    obj = -torch.clamp(t - ph, min=0) ** 2 / (2 * VV) - 0.5 * (x ** 2).sum()
                    opt.zero_grad(); (-obj).backward(); opt.step()
                if best is None or float(obj) > best[0]: best = (float(obj), x.detach().clone())
            centres.append(best[1])
    C = torch.stack(centres); nC = len(C); nIS = 2 ** 16
    comp = torch.randint(nC, (nIS,), generator=gen); comp[:nIS // 8] = 0
    xs = C[comp] + torch.randn(nIS, 4, generator=gen, dtype=dt)
    lp0 = -0.5 * (xs ** 2).sum(1)
    lq = torch.logsumexp(-0.5 * ((xs[:, None, :] - C[None]) ** 2).sum(2) + torch.log(torch.tensor([1 / 8 + 7 / 8 / nC] + [7 / 8 / nC] * (nC - 1), dtype=dt))[None], 1)
    wts = torch.exp(lp0 - lq)
    with torch.no_grad(): ph0, se0 = phi0_se(xs)
    t_cert = time.time() - t_cert; n_eq = int(rate * t_cert)
    rows = []
    for t, p in zip(ts, ptrue):
        sh = ph0 + 3 * se0 + delta_esc
        c1v = wts * torch.clamp(torch.exp(-torch.clamp(t - sh, min=0) ** 2 / (2 * V)), max=1.0)
        o1v = wts * torch.clamp(torch.exp(-torch.clamp(t - ph0, min=0) ** 2 / (2 * Vrms)), max=1.0)
        o2v = wts * (ph0 > t).double()
        c1u, c1m, c1s = ucb(c1v.numpy()); o1u, o1m, _ = ucb(o1v.numpy()); _, o2m, o2s = ucb(o2v.numpy())
        bound = c1u + p_esc
        rows.append(dict(t=float(t), log2p=math.log2(p), cert_log2=math.log2(min(1.0, bound)), cert_bits_kept=-math.log2(min(1.0, bound)) / -math.log2(p),
                         cert_rel_se=c1s / max(c1m, 1e-300), O1_log2=math.log2(max(min(1.0, o1m), 1e-300)), O2_log2=math.log2(max(o2m, 1e-300)),
                         O2_rel_se=o2s / max(o2m, 1e-300), samp_log2=-samp_bits(p, n_eq)))
        print({k: (round(v_, 2) if isinstance(v_, float) else v_) for k, v_ in rows[-1].items()}, flush=True)
    out = dict(sigma=sigma, w_norm=wn, sites=sites, V=V, Vrms=Vrms, sd_cert=math.sqrt(V), sd_rms=math.sqrt(Vrms), rho=rho,
               p_escape=p_esc, delta_escape=delta_esc, n_IS=nIS, n_centres=nC, t_cert=t_cert, n_eq=n_eq, rows=rows, seconds=time.time() - t_all)
    json.dump(out, open(os.path.join(E2.OUT, f'bc_small_s{sigma}.json'), 'w'), indent=1)
    print('done', time.time() - t_all, flush=True)


# =========================================================================== PFN (2L, sigma 0.05; score = predictive mean of one noisy pass)
def run_pfn(run_dir, smoke=False):
    import nnoise as N
    import world as W
    import qw
    from sur_pfn import chain_forward
    t_all = time.time()
    model, meta = N.load_run(run_dir); model.requires_grad_(False); sg = meta['sigma']; depth = meta['depth']; Tk = 16
    sites = qw.order(depth)
    D = json.load(open(os.path.join(run_dir, 'diag2.json'))); ts = np.array(D['t'])
    P = np.array([r['p'] for r in D['truth']]); lp_true = np.log2(P).mean(0)
    keep = np.r_[True, np.diff(ts) > 1e-9]; ts, lp_true = ts[keep], lp_true[keep]
    gen = torch.Generator().manual_seed(2)
    allow1 = model.allow_mask(torch.ones(1, Tk, dtype=torch.bool))
    def ymean(head):
        mu, ls, lg = head; return (torch.softmax(lg, -1) * mu).sum(-1)
    def episodes(n, shift, seed):
        rng = np.random.default_rng(seed); d = W.sample_np(n, rng)
        if shift is not None:                                            # shift only the query label; reading errors kept
            e = d['o'][:, :, W.QL] - d['y'][:, W.QL][:, None]
            ys = rng.standard_normal(n) + shift; d['y'][:, W.QL] = ys; d['o'][:, :, W.QL] = ys[:, None] + e
        X, valid = W.tokens_np(d, rng, meta['ds']); return d, X, valid
    def downstream(site, h, ctx, K, g):
        """phi_site: mean over K fresh downstream noise draws of the predictive mean. h: (n, T, D) site output."""
        n = h.shape[0]; od = sites; rep = lambda t: t.repeat_interleave(K, 0)
        cur = {k: rep(v) for k, v in ctx.items()}; cur[site] = rep(h); al = allow1.expand(n * K, Tk, Tk)
        for s in od[od.index(site) + 1:]:
            l = int(s[1:])
            if s[0] == 'a': v = model.f_a(l, cur['emb'] if l == 1 else cur[f'r{l - 1}'], al)
            elif s[0] == 'm': v = model.f_m(l, cur[f'a{l}'])
            else: v = model.f_r(l, cur[f'a{l}'], cur[f'm{l}'])
            cur[s] = v + sg * torch.randn(v.shape, generator=g)
        return ymean(model.head(cur[od[-1]])).view(n, K).mean(1)
    def ctx_of(site, st, idx):
        if site[0] == 'm': return {f'a{site[1:]}': st[f'a{site[1:]}'][idx]}
        return {}
    def gnorm(site, h, ctx, K, seed):
        h = h.detach().clone().requires_grad_(True); g = torch.Generator().manual_seed(seed)
        gr, = torch.autograd.grad(downstream(site, h, ctx, K, g).sum(), h)
        return gr.reshape(len(h), -1).norm(dim=1)
    nS = 512 if smoke else 4096; K = 8 if smoke else 32; bs = 128
    t_cert = time.time()
    # ---- sampled states: prior and tail-tilted episodes (query label shifted to 3.5), real noise
    stS = {}
    for tag, shift, seed in (('prior', None, 11), ('tail', 3.5, 12)):
        d, X, valid = episodes(nS, shift, seed); st, nz, allow = qw.trace_nz(model, X, valid, sg, torch.Generator().manual_seed(seed + 100)); stS[tag] = (st, nz)
    out_sites = {}
    for s in sites:
        Ds = stS['prior'][0][s][0].numel(); rho = rho_of(Ds)
        gp = torch.cat([gnorm(s, stS['prior'][0][s][i:i + bs], ctx_of(s, stS['prior'][0], slice(i, i + bs)), K, 7 + i) for i in range(0, nS, bs)])
        gt = torch.cat([gnorm(s, stS['tail'][0][s][i:i + bs], ctx_of(s, stS['tail'][0], slice(i, i + bs)), K, 9 + i) for i in range(0, nS, bs)])
        # ascent over the site's own noise within ||Z|| <= rho, parents fixed, from the 256 largest sampled norms
        allg = torch.cat([gp, gt]); top = torch.argsort(-allg)[:16 if smoke else 256]
        which = [('prior', int(i)) if i < nS else ('tail', int(i) - nS) for i in top]
        f0 = torch.stack([stS[tg][0][s][i] - sg * stS[tg][1][s][i] for tg, i in which])
        z0 = torch.stack([stS[tg][1][s][i] for tg, i in which])
        ctx = {}
        if s[0] == 'm': ctx = {f'a{s[1:]}': torch.stack([stS[tg][0][f'a{s[1:]}'][i] for tg, i in which])}
        z = z0.clone().requires_grad_(True); opt = torch.optim.Adam([z], lr=0.05)
        for it in range(20 if smoke else 100):
            g = torch.Generator().manual_seed(1000 + it)
            h = f0 + sg * z
            gr, = torch.autograd.grad(downstream(s, h, ctx, 8, g).sum(), h, create_graph=True)
            loss = -(gr ** 2).sum(); opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                nz_ = z.reshape(len(z), -1).norm(dim=1)[:, None, None]; z.mul_(torch.clamp(rho / nz_.clamp_min(1e-12), max=1.0))
        hfin = (f0 + sg * z).detach()
        ga = torch.cat([gnorm(s, hfin[i:i + 8], {k: v[i:i + 8] for k, v in ctx.items()}, 128, 31 + i) for i in range(0, len(hfin), 8)])
        h0 = (f0 + sg * z0)
        g0 = torch.cat([gnorm(s, h0[i:i + 8], {k: v[i:i + 8] for k, v in ctx.items()}, 128, 37 + i) for i in range(0, len(h0), 8)])
        out_sites[s] = dict(dim=Ds, rho=rho, rms_prior=float(gp.pow(2).mean().sqrt()), rms_tail=float(gt.pow(2).mean().sqrt()),
                            max_sampled=float(g0.max()), max_ascent=float(ga.max()), Lambda=float(max(g0.max(), ga.max())))
        print('site', s, {k: (round(v_, 4) if isinstance(v_, float) else v_) for k, v_ in out_sites[s].items()}, flush=True)
    V = sg ** 2 * sum(v['Lambda'] ** 2 for v in out_sites.values())
    Vrms = sg ** 2 * sum(v['rms_prior'] ** 2 for v in out_sites.values())
    p_esc = len(sites) * 2.0 ** -RHO_LP
    delta_esc = sum(2 * (10.0 + v['Lambda'] * sg * (v['rho'] + 3)) for v in out_sites.values()) * 2.0 ** -RHO_LP   # |Y| <= 10 (head ceiling, checked below)
    # ---- C2: output ceiling over reachable final query states (sphere radius sqrt(d) + per-token noise ball)
    dq = model.d; rq = rho_of(dq)
    def head_q(u):
        x = torch.zeros(len(u), Tk, dq); x[:, -1] = u; return ymean(model.head(x))
    v = torch.randn(256 if not smoke else 32, dq, generator=gen).requires_grad_(True); zq = torch.zeros_like(v).requires_grad_(True)
    opt = torch.optim.Adam([v, zq], lr=0.05)
    for it in range(300 if not smoke else 30):
        y = head_q(N.sphere(v) + sg * zq); opt.zero_grad(); (-y.sum()).backward(); opt.step()
        with torch.no_grad(): zq.mul_(torch.clamp(rq / zq.norm(dim=1, keepdim=True).clamp_min(1e-12), max=1.0))
    with torch.no_grad():
        ceil = float(head_q(N.sphere(v) + sg * zq).max()); ceil0 = float(head_q(N.sphere(v)).max())
        yreal = ymean(model.head(stS['tail'][0][sites[-1]])); ymax_real = float(yreal.max())
    print('ceiling (search, with noise ball)', ceil, 'without noise', ceil0, 'max over tail paths', ymax_real, flush=True)
    # ---- expectation over the exact episode prior: defensive mixture of query-label shifts (exact weights)
    shifts = [None, 2.5, 3.5, 4.5]; nIS = 2 ** 11 if smoke else 2 ** 15; KM = 16 if smoke else 64
    ph0, se0, wts = [], [], []
    for j, shf in enumerate(shifts):
        d, X, valid = episodes(nIS // len(shifts), shf, 500 + j); ys = d['y'][:, W.QL]
        lq = np.log(np.mean([norm.pdf(ys, loc=(0 if m is None else m)) for m in shifts], 0)); wts.append(np.exp(norm.logpdf(ys) - lq))
        for i in range(0, len(X), 512):
            vals = []
            with torch.no_grad():
                for k in range(KM):
                    g = torch.Generator().manual_seed(10_000 * j + 100 * k + i)
                    vals.append(ymean(chain_forward(model, X[i:i + 512], valid[i:i + 512], sg, {}, g)))
            vals = torch.stack(vals, 1).double(); ph0.append(vals.mean(1)); se0.append(vals.std(1) / math.sqrt(KM))
    ph0, se0, wts = torch.cat(ph0).numpy(), torch.cat(se0).numpy(), np.concatenate(wts)
    t_cert = time.time() - t_cert
    # pass rate for sampling at equal compute
    rng = np.random.default_rng(3); d, X, valid = episodes(4096, None, 77); t0 = time.time()
    with torch.no_grad():
        chain_forward(model, X, valid, sg, {}, torch.Generator().manual_seed(1))
    rate = 4096 / (time.time() - t0); n_eq = int(rate * t_cert)
    rows = []
    for t, lp in zip(ts, lp_true):
        c1 = wts * np.minimum(1.0, np.exp(-np.maximum(t - ph0 - 3 * se0 - delta_esc, 0) ** 2 / (2 * V)))
        o1 = wts * np.minimum(1.0, np.exp(-np.maximum(t - ph0, 0) ** 2 / (2 * Vrms)))
        o2 = wts * (ph0 > t)
        c1u, c1m, c1s = ucb(c1); _, o1m, _ = ucb(o1); _, o2m, o2s = ucb(o2)
        bound = min(1.0, c1u + p_esc)
        rows.append(dict(t=float(t), log2p=float(lp), cert_log2=math.log2(bound), cert_bits_kept=-math.log2(bound) / -lp,
                         cert_rel_se=c1s / max(c1m, 1e-300), O1_log2=math.log2(max(min(1.0, o1m), 1e-300)), O2_log2=math.log2(max(o2m, 1e-300)),
                         O2_rel_se=o2s / max(o2m, 1e-300), ceiling_minus_t=ceil - float(t), samp_log2=-samp_bits(2.0 ** lp, n_eq)))
        print({k: (round(v_, 3) if isinstance(v_, float) else v_) for k, v_ in rows[-1].items()}, flush=True)
    out = dict(sigma=sg, sites=out_sites, V=V, Vrms=Vrms, sd_cert=math.sqrt(V), sd_rms=math.sqrt(Vrms), p_escape=p_esc, delta_escape=delta_esc,
               ceiling=ceil, ceiling_no_noise=ceil0, max_tail_paths=ymax_real, phi0_max=float(ph0.max()), n_IS=int(len(ph0)), K_phi0=KM,
               t_cert=t_cert, rate=rate, n_eq=n_eq, rows=rows, seconds=time.time() - t_all)
    json.dump(out, open(os.path.join(run_dir, 'bc_smoke.json' if smoke else 'bc.json'), 'w'), indent=1)
    print('done', time.time() - t_all, flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'small':
        run_small(float(sys.argv[2]))
    else:
        run_pfn(sys.argv[2], smoke='--smoke' in sys.argv)
