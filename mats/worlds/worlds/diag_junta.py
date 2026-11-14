"""W1 diagnostics for one trained run (constructed contexts; every comparison is against exact Bayes on the same
context, so no target sampling is needed; all constructions are seeded and identical across runs of one world).

  C1    contamination: clean balanced context (48 distinct items, 12 per true cell, one fresh reading each) + M in
        {0,16,64,512,4048} readings of ONE dominant item at Hamming distance 8 from the query and in a different
        true cell. Variants: 'fresh' (M fresh readings), 'copy' (1 fresh reading + M-1 exact copies; Bayes unchanged
        after M=1), 'control' (M more balanced distinct items: matched length). Anchors simulated: Bayes and
        R-known Bayes contamination KL(p(D_M) || p(D_0)) and mean shift; model KL(Bayes || model) per M.
        Attention probe: query attention mass on the dominant item's tokens per layer vs its share M/(48+M).
  D1    R-evidence pairs: same items, same multiset of values, values permuted among non-query reports; pairs
        admitted when Bayes moves (|d mean| > 0.5 sd). Model shift vs Bayes shift.
  D2    k copies vs k agreeing fresh readings of the query's single reading (arXiv 2610.16801 control):
        copies -> Bayes unchanged; agreeing fresh -> Bayes sharper. k in {1,4,16,64}.
  D4    whole context x k (Barbero et al. 2024: softmax without a sink sees only proportions): every token
        replicated k times as copies (Bayes unchanged) vs as agreeing fresh readings (Bayes sharper). k in {2,4,8}.
  PROBE ridge probes of context-token residual streams for log count of the token's item (fresh readings),
        log raw count (copies included) and log count of the token's true cell.

usage: python diag_junta.py --run runs/X [--ckpt best.pt] [--nbase 100]
"""
import argparse, json, math, os, time
import numpy as np
import harness as Hn
import junta as J


def _ev(rng, D):
    return rng.standard_normal(D).astype(np.float32)


def _post(world, ep, hidx=None):
    out = []
    for i in range(len(ep['q'])):
        out.append(world.posterior(*world._episode(ep, i), hidx=(ep['h'][i] if hidx else None)))
    return J._stack(out)


def _take(p, idx):
    return tuple(x[idx] for x in p)


# ----------------------------------------------------------------------------------------- C1
def c1_rows(world, seed, nbase, Ms, n_clean=48):
    rng = np.random.default_rng([seed, 101])
    D = world.cfg['ev_dim']; tau = math.sqrt(world.tau2)
    Mmax = max(Ms); rows = []
    for bi in range(nbase):
        h = int(rng.integers(world.H)); s = int(rng.integers(len(world.sig2))); sd = math.sqrt(world.sig2[s])
        g = rng.standard_normal(world.ncell)
        cand = np.unique(rng.integers(0, world.NI, 6 * (n_clean + Mmax) + 4000))
        rng.shuffle(cand)
        cc = world.cell_of(cand[None, :], np.array([h]))[0]
        ncell_used = 1 << len(world.hyps[h])
        q = int(cand[0]); cq = int(cc[0])
        while True:
            flip = rng.choice(world.B, 8, replace=False)
            xd = q ^ int(sum(1 << int(f) for f in flip))
            if world.cell_of(np.array([[xd]]), np.array([h]))[0, 0] != cq: break
        pool = [cand[1:][(cc[1:] == c) & (cand[1:] != xd)] for c in range(ncell_used)]
        per = n_clean // ncell_used
        clean = np.concatenate([p[:per] for p in pool])
        extra = np.stack([p[per:per + Mmax // ncell_used + 1] for p in pool], 1).ravel()[:Mmax]   # interleaved cells
        def v(xs):        # all items here are distinct, so item effects are independent draws
            xs = np.atleast_1d(xs)
            return g[world.cell_of(xs[None, :], np.array([h]))[0]] + rng.standard_normal(len(xs)) * tau
        yc = v(clean) + rng.standard_normal(len(clean)) * sd
        yx = v(extra) + rng.standard_normal(len(extra)) * sd
        yd = v(xd)[0] + rng.standard_normal(Mmax) * sd
        evc = rng.standard_normal((len(clean), D)).astype(np.float32)
        evx = rng.standard_normal((len(extra), D)).astype(np.float32)
        evd = rng.standard_normal((Mmax, D)).astype(np.float32)
        theta = float(v(q)[0])
        base = dict(q=q, target=theta, h=h, s=s)
        for M in Ms:
            n0 = len(clean)
            for var in ('fresh', 'copy', 'control'):
                if var == 'fresh':
                    it = np.r_[clean, np.full(M, xd)]; y = np.r_[yc, yd[:M]]; ev = np.r_[evc, evd[:M]]
                    root = np.arange(n0 + M)
                elif var == 'copy':
                    it = np.r_[clean, np.full(M, xd)]; y = np.r_[yc, np.full(M, yd[0])]
                    ev = np.r_[evc, np.repeat(evd[:1], M, 0)]; root = np.r_[np.arange(n0), np.full(M, n0)]
                else:
                    it = np.r_[clean, extra[:M]]; y = np.r_[yc, yx[:M]]; ev = np.r_[evc, evx[:M]]
                    root = np.arange(n0 + M)
                rows.append(dict(base, items=it, y=y, ev=ev, root=root, var=var, M=M, bi=bi, xd=xd,
                                 ydbar=float(yd[:max(M, 1)].mean())))
    return rows


def run_c1(model, world, seed, nbase, Ms):
    rows = c1_rows(world, seed, nbase, Ms)
    res = {}
    store = {}
    sink = int(model.use_sink)
    for var in ('fresh', 'copy', 'control'):
        for M in Ms:
            rs = [r for r in rows if r['var'] == var and r['M'] == M]
            ep = world.assemble(rs)
            pb = _post(world, ep); po = _post(world, ep, hidx=True)
            pm = Hn.predict(model, ep['X'], ep['mask'])
            att = Hn.query_attention(model, ep['X'], ep['mask'])
            n0 = 48
            dom = np.zeros(ep['X'].shape[1] + sink, bool); dom[sink + n0:sink + n0 + M] = True
            mass = [float(a.mean(1)[:, dom].sum(1).mean()) for a in att] if M > 0 and var != 'control' else None
            sinkm = [float(a.mean(1)[:, 0].mean()) for a in att] if sink else None
            store[(var, M)] = (pb, po, pm, np.array([r['ydbar'] for r in rs]))
            res[f'{var}_M{M}'] = dict(kl_model=Hn._mse(Hn.kl_bayes_model(pb, pm)), attn_mass_dom=mass,
                                     share=M / (n0 + M), attn_sink=sinkm)
    for var in ('fresh', 'copy', 'control'):
        pb0, po0, pm0, _ = store[(var, Ms[0])]
        mb0, mo0, mm0 = Hn.moments(pb0)[0], Hn.moments(po0)[0], Hn.moments(pm0)[0]
        for M in Ms:
            pb, po, pm, yd = store[(var, M)]
            sgn = np.sign(yd - mb0)
            r = res[f'{var}_M{M}']
            r['contam_bayes'] = Hn._mse(Hn.kl_bayes_model(pb, pb0)); r['contam_oracle'] = Hn._mse(Hn.kl_bayes_model(po, po0))
            r['contam_model'] = Hn._mse(Hn.kl_bayes_model(pm, pm0))
            r['shift_bayes'] = float(((Hn.moments(pb)[0] - mb0) * sgn).mean())
            r['shift_oracle'] = float(((Hn.moments(po)[0] - mo0) * sgn).mean())
            r['shift_model'] = float(((Hn.moments(pm)[0] - mm0) * sgn).mean())
    # pre-registered C1 rule
    verdict = []
    for M in Ms[1:]:
        ctl = res[f'control_M{M}']['kl_model'][0]; fr = res[f'fresh_M{M}']['kl_model'][0]
        verdict.append(dict(M=M, control_kl=ctl, fresh_kl=fr, excess=fr - ctl, control_ok=ctl < 0.05))
    ks = [res[f'fresh_M{M}']['kl_model'][0] for M in Ms[1:]]
    slope = float(np.polyfit(np.log2(Ms[1:]), ks, 1)[0])
    refuted = any(v['control_ok'] and v['excess'] > 0.10 for v in verdict) or \
        (slope > 0.03 and any(v['control_ok'] for v in verdict))
    res['rule'] = dict(per_M=verdict, fresh_kl_slope_per_doubling=slope, refuted=bool(refuted),
                       vacuous=not any(v['control_ok'] for v in verdict))
    return res


# ----------------------------------------------------------------------------------------- pairs
def _base_eps(world, seed, E, N, policy='uniform'):
    w0 = J.Junta(dict(world.cfg, p_copy_zero=1.0))
    return w0.sample(np.random.default_rng([seed, N, 7]), E, N, policy)


def _rows_of(ep, i):
    n = ep['n'][i]
    return dict(items=ep['items'][i, :n].copy(), y=ep['y'][i, :n].copy(), root=ep['root'][i, :n].copy(),
                ev=ep['ev'][i, :n].copy(), q=int(ep['q'][i]), h=int(ep['h'][i]), s=int(ep['s'][i]),
                target=float(ep['target'][i]))


def _pair_stats(world, model, A, B):
    ea, eb = world.assemble(A), world.assemble(B)
    pba, pbb = _post(world, ea), _post(world, eb)
    pma, pmb = Hn.predict(model, ea['X'], ea['mask']), Hn.predict(model, eb['X'], eb['mask'])
    (mba, vba), (mbb, vbb) = Hn.moments(pba), Hn.moments(pbb)
    (mma, vma), (mmb, vmb) = Hn.moments(pma), Hn.moments(pmb)
    return dict(d_mean_bayes=mbb - mba, d_mean_model=mmb - mma, logvr_bayes=np.log(vbb / vba),
                logvr_model=np.log(vmb / vma), kl_a=Hn.kl_bayes_model(pba, pma), kl_b=Hn.kl_bayes_model(pbb, pmb),
                sd_a=np.sqrt(vba))


def run_pairs(model, world, seed, nbase):
    D = world.cfg['ev_dim']; out = {}
    rng = np.random.default_rng([seed, 202])
    # D1: permute values among non-query reports
    for N in (64, 256):
        ep = _base_eps(world, seed, 3 * nbase, N)
        A, B = [], []
        for i in range(len(ep['q'])):
            a = _rows_of(ep, i); b = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in a.items()}
            idx = np.nonzero(a['items'] != a['q'])[0]
            b['y'][idx] = a['y'][rng.permutation(idx)]
            A.append(a); B.append(b)
        st = _pair_stats(world, model, A, B)
        keep = np.abs(st['d_mean_bayes']) > 0.5 * st['sd_a']
        db, dm = st['d_mean_bayes'][keep], st['d_mean_model'][keep]
        out[f'D1_N{N}'] = dict(n_pairs=int(keep.sum()), of=len(A),
                              slope=float((db * dm).sum() / (db * db).sum()) if keep.any() else None,
                              corr=float(np.corrcoef(db, dm)[0, 1]) if keep.sum() > 2 else None,
                              sign_agree=float((np.sign(db) == np.sign(dm)).mean()) if keep.any() else None,
                              kl_a=Hn._mse(st['kl_a'][keep]), kl_b=Hn._mse(st['kl_b'][keep]))
    # D2: k copies vs k agreeing fresh readings of the query's single reading
    ep = _base_eps(world, seed, 2 * nbase, 64)
    bases = []
    for i in range(len(ep['q'])):
        a = _rows_of(ep, i)
        j = int(rng.integers(len(a['items'])))
        if (a['items'] == a['items'][j]).sum() != 1: continue
        a['q'] = int(a['items'][j]); a['j'] = j; bases.append(a)
    bases = bases[:nbase]
    for k in (1, 4, 16, 64, 256):
        A, B = [], []
        for a in bases:
            n = len(a['items']); j = a['j']
            cp = dict(a, items=np.r_[a['items'], np.full(k, a['q'])], y=np.r_[a['y'], np.full(k, a['y'][j])],
                      ev=np.r_[a['ev'], np.repeat(a['ev'][j:j + 1], k, 0)], root=np.r_[a['root'], np.full(k, j)])
            fr = dict(a, items=cp['items'], y=cp['y'], ev=np.r_[a['ev'], rng.standard_normal((k, D)).astype(np.float32)],
                      root=np.arange(n + k))
            A.append(cp); B.append(fr)
        s0 = _pair_stats(world, model, bases, A); s1 = _pair_stats(world, model, bases, B)
        out[f'D2_k{k}'] = dict(n=len(bases),
                               copies=dict(logvr_bayes=float(s0['logvr_bayes'].mean()), logvr_model=Hn._mse(s0['logvr_model']),
                                           kl_model=Hn._mse(s0['kl_b'])),
                               fresh=dict(logvr_bayes=float(s1['logvr_bayes'].mean()), logvr_model=Hn._mse(s1['logvr_model']),
                                          kl_model=Hn._mse(s1['kl_b'])),
                               base_kl=Hn._mse(s0['kl_a']))
    # D4: whole context x k as copies vs as agreeing fresh readings
    ep = _base_eps(world, seed, nbase, 64)
    bases = [_rows_of(ep, i) for i in range(len(ep['q']))]
    for k in (2, 4, 8):
        A, B = [], []
        for a in bases:
            n = len(a['items'])
            rep = lambda x: np.concatenate([x] * k)
            cp = dict(a, items=rep(a['items']), y=rep(a['y']), ev=rep(a['ev']), root=rep(a['root']))
            fr = dict(cp, ev=np.r_[a['ev'], rng.standard_normal(((k - 1) * n, D)).astype(np.float32)], root=np.arange(k * n))
            A.append(cp); B.append(fr)
        s0 = _pair_stats(world, model, bases, A); s1 = _pair_stats(world, model, bases, B)
        out[f'D4_k{k}'] = dict(copies=dict(logvr_bayes=float(s0['logvr_bayes'].mean()), logvr_model=Hn._mse(s0['logvr_model']),
                                           d_mean_rms_model=float(np.sqrt((s0['d_mean_model'] ** 2).mean())),
                                           kl_model=Hn._mse(s0['kl_b'])),
                               fresh=dict(logvr_bayes=float(s1['logvr_bayes'].mean()), logvr_model=Hn._mse(s1['logvr_model']),
                                          kl_model=Hn._mse(s1['kl_b'])),
                               base_kl=Hn._mse(s0['kl_a']))
    return out


# ----------------------------------------------------------------------------------------- count probes
def _ridge_r2(H, t, lam=1e-1):
    n = len(H) // 2
    Ha = np.c_[H, np.ones(len(H))]
    w = np.linalg.solve(Ha[:n].T @ Ha[:n] + lam * np.eye(Ha.shape[1]), Ha[:n].T @ t[:n])
    p = Ha[n:] @ w
    return float(1 - ((t[n:] - p) ** 2).mean() / t[n:].var())


def run_probes(model, world, seed, N=128, E=200):
    rng = np.random.default_rng([seed, 303])
    ep = world.sample_train(rng, E, N, N)
    hs = Hn.hidden_states(model, ep['X'], ep['mask'])
    sink = int(model.use_sink)
    Eh, T = ep['items'].shape
    fresh = ep['root'] == np.arange(T)[None]
    cell = world.cell_of(ep['items'], ep['h'])
    t_item, t_raw, t_cell = [], [], []
    for i in range(Eh):
        it = ep['items'][i]
        t_raw.append(np.log((it[:, None] == it[None, :]).sum(1)))
        t_item.append(np.log(((it[:, None] == it[None, :]) & fresh[i][None, :]).sum(1)))
        t_cell.append(np.log(((cell[i][:, None] == cell[i][None, :]) & fresh[i][None, :]).sum(1)))
    tgt = {k: np.concatenate(v) for k, v in (('log_n_item_fresh', t_item), ('log_n_item_raw', t_raw), ('log_n_cell_fresh', t_cell))}
    sub = rng.permutation(Eh * T)[:20000]
    out = {}
    for k, t in tgt.items():
        out[k] = [_ridge_r2(h[:, sink:sink + T].reshape(-1, h.shape[-1])[sub], t[sub]) for h in hs]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True); ap.add_argument('--ckpt', default='best.pt')
    ap.add_argument('--nbase', type=int, default=100); ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--Ms', default='0,16,64,512,4048')
    a = ap.parse_args()
    model, world, meta = Hn.load_run(a.run, a.ckpt)
    t0 = time.time()
    res = dict(run=a.run, ckpt=a.ckpt)
    res['C1'] = run_c1(model, world, a.seed, a.nbase, [int(x) for x in a.Ms.split(',')])
    res['pairs'] = run_pairs(model, world, a.seed, a.nbase)
    res['probes'] = run_probes(model, world, a.seed)
    res['seconds'] = time.time() - t0
    json.dump(res, open(os.path.join(a.run, 'diag.json'), 'w'), indent=1, default=float)
    print(json.dumps(res['C1']['rule']), flush=True)


if __name__ == '__main__':
    main()
