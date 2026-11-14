"""Task 2 information measurements on a noisy PFN run: I(X; H_site) brackets per site and token group.

usage: python mirun.py RUN_DIR [--M 32768]
writes RUN_DIR/mi.json
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import nnoise as N
import mi
import world as W
import pfn as P

torch.set_num_threads(1)
GROUPS = {'ALL': None, 'QT': 15, 'QQ0': 4}


def trace(model, X, valid, sigma, gen, bs=4096):
    """Noise-free centres mu[site] and noisy states h[site], float32 numpy (n, T, dim)."""
    mu, hs = {}, {}
    with torch.no_grad():
        for i in range(0, len(X), bs):
            Xi = torch.from_numpy(X[i:i + bs]); allow = model.allow_mask(torch.from_numpy(valid[i:i + bs]))
            z = lambda t: t + sigma * torch.randn(t.shape, generator=gen)
            m0 = model.f_emb(Xi); x = z(m0); out = [('emb', m0, x)]
            for l in range(1, model.depth + 1):
                ma = model.f_a(l, x, allow); xa = z(ma); out.append((f'a{l}', ma, xa))
                mm = model.f_m(l, xa); m = z(mm); out.append((f'm{l}', mm, m))
                mr = model.f_r(l, xa, m); x = z(mr); out.append((f'r{l}', mr, x))
            for s, a, b in out:
                mu.setdefault(s, []).append(a.numpy()); hs.setdefault(s, []).append(b.numpy())
    return {k: np.concatenate(v) for k, v in mu.items()}, {k: np.concatenate(v) for k, v in hs.items()}


def gslice(a, g):
    t = GROUPS[g]
    return a.reshape(len(a), -1) if t is None else a[:, t]


def inner_centres(model, X, valid, sigma, K, seed, sites, toks=None, nb=16):
    """(n, K, T', dim) centres per site from K fresh noise chains for each of the n inputs (tokens toks only)."""
    gen = torch.Generator().manual_seed(seed)
    out = {s: [] for s in sites}
    for i in range(0, len(X), nb):
        n = len(X[i:i + nb])
        Xr = np.repeat(X[i:i + nb], K, 0); vr = np.repeat(valid[i:i + nb], K, 0)
        mu, _ = trace(model, Xr, vr, sigma, gen)
        for s in sites:
            m = mu[s] if toks is None else mu[s][:, toks]
            out[s].append(m.reshape(n, K, *m.shape[1:]))
    return {s: np.concatenate(v) for s, v in out.items()}


def run(a):
    model, meta = N.load_run(a.run); sg = meta['sigma']
    assert sg > 0
    rng = np.random.default_rng(50_000)
    d = W.sample_np(a.M, rng); X, valid = W.tokens_np(d, rng, meta['ds'])
    gen = torch.Generator().manual_seed(50_001)
    t0 = time.time()
    mu, hs = trace(model, X, valid, sg, gen)
    sites = model.site_names()
    out = []
    # inner chains: token groups (N_tok targets, K_tok chains); ALL (N_all targets, K_all chains)
    inn_tok = inner_centres(model, X[:a.ntok_in], valid[:a.ntok_in], sg, a.ktok, 50_002, sites, toks=[GROUPS['QT'], GROUPS['QQ0']])
    inn_all = inner_centres(model, X[:a.nall_in], valid[:a.nall_in], sg, a.kall, 50_003, sites)
    for s in sites:
        for g in GROUPS:
            n_t = a.nall if g == 'ALL' else a.ntok
            C = gslice(mu[s], g).astype(np.float64); T = gslice(hs[s][:n_t], g).astype(np.float64)
            D = C.shape[1]
            e = mi.entropy_bracket(T, C, sg, np.arange(n_t))
            hz = mi.hZ_bits(D, sg)
            if s == 'emb':
                hc = dict(hc_in=hz, hc_out=hz, hc_gauss=hz, K=0)
            else:
                if g == 'ALL':
                    inn = inn_all[s]; ni = len(inn); innk = inn.reshape(ni, inn.shape[1], -1)
                else:
                    inn = inn_tok[s]; ni = len(inn); innk = inn[:, :, 0 if g == 'QT' else 1]
                Cin = np.concatenate([gslice(mu[s][:ni], g)[:, None], innk], 1)
                Tin = gslice(hs[s][:ni], g)
                hc = mi.cond_entropy_bracket(Tin.astype(np.float64), Cin.astype(np.float64), sg)
            hc_lo = max(hc['hc_in'], hz); hc_hi = min(hc['hc_out'], hc['hc_gauss'])
            ntok = 16 if g == 'ALL' else 1
            row = dict(site=s, group=g, D=D, sigma=sg, depth=meta['depth'], cap=ntok * mi.gauss_cap_bits(D // ntok, 1.0, sg),
                       I_lo=e['h_in'] - hc_hi, I_hi=e['h_out'] - hc_lo, Ipar_lo=e['h_in'] - hz, Ipar_hi=e['h_out'] - hz,
                       h_in=e['h_in'], h_out=e['h_out'], se=e['se'], hZ=hz, hc_in=hc['hc_in'], hc_out=hc['hc_out'],
                       hc_gauss=hc['hc_gauss'], log2M=math.log2(a.M))
            out.append(row)
            print(f"{s:4s} {g:4s} D={D:4d} I in [{row['I_lo']:7.2f}, {row['I_hi']:7.2f}]  par [{row['Ipar_lo']:6.2f},{row['Ipar_hi']:7.2f}] "
                  f"hc [{hc['hc_in'] - hz:6.2f},{hc['hc_out'] - hz:6.2f}] gauss {hc['hc_gauss'] - hz:6.2f} cap {row['cap']:7.1f} ({time.time() - t0:.0f}s)",
                  flush=True)
            json.dump(out, open(os.path.join(a.run, 'mi.json'), 'w'), indent=1)


if __name__ == '__main__' and not os.environ.get('MIRUN_XG'):
    ap = argparse.ArgumentParser(); ap.add_argument('run')
    ap.add_argument('--M', type=int, default=2 ** 15); ap.add_argument('--ntok', type=int, default=2048)
    ap.add_argument('--nall', type=int, default=1024); ap.add_argument('--ntok_in', type=int, default=256)
    ap.add_argument('--ktok', type=int, default=1024); ap.add_argument('--nall_in', type=int, default=96)
    ap.add_argument('--kall', type=int, default=560)
    run(ap.parse_args())


def run_xg(a):
    """x-level Gaussian-conditional I(X; H_site) for token groups QT, QQ0 (mi.xgauss_bracket)."""
    model, meta = N.load_run(a.run); sg = meta['sigma']
    rng = np.random.default_rng(55_000)
    d = W.sample_np(a.Mx, rng); X, valid = W.tokens_np(d, rng, meta['ds'])
    gen = torch.Generator().manual_seed(55_001)
    _, hs = trace(model, X[:a.nx], valid[:a.nx], sg, gen)
    sites = model.site_names(); toks = [GROUPS['QT'], GROUPS['QQ0']]
    mom = {s: [[], []] for s in sites}
    t0 = time.time()
    for i in range(0, a.Mx, 8):
        inn = inner_centres(model, X[i:i + 8], valid[i:i + 8], sg, a.kx, 56_000 + i, sites, toks=toks, nb=8)
        for s in sites:
            for gi in range(2):
                m, S = mi.moments_from_chains(inn[s][:, :, gi].astype(np.float64), sg)
                if s == 'emb':
                    S = np.broadcast_to(sg ** 2 * np.eye(m.shape[1]), S.shape).copy()
                mom[s][gi].append((m, S))
    print(f'chains done {time.time() - t0:.0f}s', flush=True)
    out = []
    for s in sites:
        for gi, g in enumerate(('QT', 'QQ0')):
            m = np.concatenate([x[0] for x in mom[s][gi]]); S = np.concatenate([x[1] for x in mom[s][gi]])
            r = mi.xgauss_bracket(hs[s][:, toks[gi]].astype(np.float64), m, S, np.arange(a.nx))
            r.update(site=s, group=g, sigma=sg, depth=meta['depth'], D=m.shape[1], hZ=mi.hZ_bits(m.shape[1], sg))
            out.append(r)
            print(f"{s:4s} {g:4s} xg I in [{r['xg_I_lo']:.2f}, {r['xg_I_hi']:.2f}]  hc-hZ {r['xg_hc'] - r['hZ']:.2f}", flush=True)
            json.dump(out, open(os.path.join(a.run, 'mi_xg.json'), 'w'), indent=1)


if __name__ == '__main__' and os.environ.get('MIRUN_XG'):
    ap = argparse.ArgumentParser(); ap.add_argument('run')
    ap.add_argument('--Mx', type=int, default=4096); ap.add_argument('--nx', type=int, default=1024); ap.add_argument('--kx', type=int, default=256)
    run_xg(ap.parse_args())
