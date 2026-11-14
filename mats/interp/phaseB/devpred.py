"""Deviation prediction (PREREG-B amendment 5): can a surrogate chain predict the noisy PFN's errors relative to Bayes?

Network output per episode = noise-marginalised predictive (K = 8 draws). Deviation d = network - Bayes for the predictive
mean, the predictive log-sd, and the per-episode KL(Bayes || network). A surrogate chain (sur_pfn families at some sites,
real maps elsewhere) gives d_hat the same way. R^2 = 1 - sum (d - d_hat)^2 / sum (d - mean d)^2 on held-out test episodes.
Baselines: zero (network = Bayes), untrained network, noise floor (independent K = 8 estimate of the same network).
Scenarios S1-S9 (amendment 5) are scored on fresh episodes.
usage: python devpred.py RUN_DIR -> RUN_DIR/devpred.json
"""
import json, math, os, sys, time
import numpy as np
import torch
import nnoise as N
import pfn as P
import world as W
from mirun import trace
from sur_pfn import fit_surrogates, chain_forward, episodes

torch.set_num_threads(1)
TP = '/home/agent/workspaces/mats/work/tiny-pfn'


def marginal(model, X, valid, sigma, sur, K, seed, bs=2000):
    """Noise-marginalised predictive as (mu, logs, logits) arrays (n, K*4) plus its mean and log-sd."""
    mus, lss, lgs = [], [], []
    for i in range(0, len(X), bs):
        a, b, c = [], [], []
        for k in range(K):
            g = torch.Generator().manual_seed(seed + 1000 * k + i)
            mu, ls, lg = chain_forward(model, X[i:i + bs], valid[i:i + bs], sigma, sur, g)
            a.append(mu.numpy()); b.append(ls.numpy()); c.append(torch.log_softmax(lg, -1).numpy() - math.log(K))
        mus.append(np.concatenate(a, 1)); lss.append(np.concatenate(b, 1)); lgs.append(np.concatenate(c, 1))
    mu, ls, lg = np.concatenate(mus), np.concatenate(lss), np.concatenate(lgs)
    w = np.exp(lg - lg.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
    m = (w * mu).sum(1); v = (w * (np.exp(2 * ls) + mu ** 2)).sum(1) - m ** 2
    return (mu, ls, lg), m, 0.5 * np.log(v)


def deviations(head, m, logsd, T):
    kl = P.kl_grid(T, head, ngrid=801, chunk=200)
    return dict(mean=m - T['mean_bayes'], logsd=logsd - 0.5 * np.log(T['var_bayes']), kl=kl)


def r2(d, dh):
    return float(1 - ((d - dh) ** 2).sum() / ((d - d.mean()) ** 2).sum())


def scenario_features(d, X, T):
    """Per-episode features for S1-S9 from world variables (d: world sample dict, X tokens, T: Bayes comps / moments)."""
    means, vars_, wb = T['comp_means'], T['comp_vars'], T['w_bayes']
    mb = (wb * means).sum(1); between = (wb * means ** 2).sum(1) - mb ** 2; total = between + (wb * vars_[None]).sum(1)
    _, _, _, _ = None, None, None, None
    o, y, vidx = d['o'], d['y'], d['vidx']
    ostar = o[:, :, W.QL]; spread = ostar.max(1) - ostar.min(1)
    ll = W.labelled_loglik(d); llc = sum(ll[:, s, W.COMB[:, s]] for s in range(W.S)); llc = llc - llc.max(1, keepdims=True)
    wl = np.exp(llc); wl /= wl.sum(1, keepdims=True); ent_lab = -(wl * np.log(np.maximum(wl, 1e-300))).sum(1)
    coupling = (wb * (np.log(np.maximum(wb, 1e-300)) - np.log(np.maximum(wl, 1e-300)))).sum(1)
    semb = np.stack([X[:, 5 * s, :16] for s in range(W.S)], 1).astype(np.float64)
    sn = semb / np.linalg.norm(semb, axis=2, keepdims=True)
    cos = np.stack([np.abs((sn[:, a] * sn[:, b]).sum(1)) for a, b in ((0, 1), (0, 2), (1, 2))], 1).max(1)
    R = o[:, :, :W.QL] - y[:, None, :W.QL]; r2_ = R ** 2; SS = r2_.sum(2); share = (r2_.max(2) / SS)
    return dict(bimod=between / total, spread=spread, ent_lab=ent_lab, coupling=coupling, cos=cos, share=share,
                vidx=vidx, mb=mb, ostar=ostar)


def score_scenarios(F, dev):
    out = {}; dm, ds, kl = dev['mean'], dev['logsd'], dev['kl']
    KLm, RMS = kl.mean(), np.sqrt((dm ** 2).mean())
    def ratio_kl(sel): return float(kl[sel].mean() / KLm)
    def ratio_rms(sel): return float(np.sqrt((dm[sel] ** 2).mean()) / RMS)
    q90 = lambda a: a >= np.quantile(a, 0.9)
    s1 = q90(F['bimod']); out['S1'] = dict(n=int(s1.sum()), KL_ratio=ratio_kl(s1), mean_dlogsd=float(ds[s1].mean()), hit=bool(ratio_kl(s1) >= 2 and ds[s1].mean() < 0))
    s2 = q90(F['spread']) & (F['ent_lab'] > np.median(F['ent_lab'])); out['S2'] = dict(n=int(s2.sum()), KL_ratio=ratio_kl(s2), mean_dlogsd=float(ds[s2].mean()), hit=bool(ratio_kl(s2) >= 3 and ds[s2].mean() < 0))
    s3 = (F['vidx'] == 2).all(1); out['S3'] = dict(n=int(s3.sum()), RMS_ratio=ratio_rms(s3), KL_ratio=ratio_kl(s3), hit=bool(ratio_rms(s3) >= 1.3 and ratio_kl(s3) >= 1.3))
    s4 = (F['vidx'] == 0).all(1); out['S4'] = dict(n=int(s4.sum()), KL_ratio=ratio_kl(s4), hit=bool(ratio_kl(s4) <= 0.6))
    s5 = ((F['vidx'] == 0).sum(1) == 1) & ((F['vidx'] == 2).sum(1) == 2); out['S5'] = dict(n=int(s5.sum()), KL_ratio=ratio_kl(s5), hit=bool(ratio_kl(s5) >= 1.2))
    s6 = q90(F['cos']); out['S6'] = dict(n=int(s6.sum()), RMS_ratio=ratio_rms(s6), hit=bool(ratio_rms(s6) >= 1.1))
    sh = F['share']; has = (sh > 0.7).any(1)
    s_idx = np.argmax(sh, 1); agree = np.sign(dm) == np.sign(F['ostar'][np.arange(len(dm)), s_idx] - F['mb'])
    out['S7'] = dict(n=int(has.sum()), sign_agree=float(agree[has].mean()), baseline_sign_agree=float(agree.mean()), hit=bool(agree[has].mean() >= 0.6))
    s8 = np.abs(F['mb']) > 2.5; v8 = float((dm[s8] * np.sign(F['mb'][s8])).mean()) if s8.any() else float('nan')
    out['S8'] = dict(n=int(s8.sum()), mean_signed=v8, RMS_ratio=ratio_rms(s8) if s8.any() else float('nan'), hit=bool(s8.any() and v8 < 0 and ratio_rms(s8) >= 1.5))
    s9 = q90(F['coupling']); out['S9'] = dict(n=int(s9.sum()), KL_ratio=ratio_kl(s9), hit=bool(ratio_kl(s9) >= 2))
    return out


def run(run_dir, n_test=4000, K=8):
    t0 = time.time()
    model, meta = N.load_run(run_dir); sg = meta['sigma']
    Tz = np.load(os.path.join(TP, 'runs/testset.npz'))
    T = {k: Tz[k][:n_test] for k in ('comp_means', 'w_bayes', 'mean_bayes', 'var_bayes')}; T['comp_vars'] = Tz['comp_vars']
    X, valid = Tz['X'][:n_test], Tz['valid'][:n_test]
    # real network: two independent K-draw marginals (deviation + noise floor)
    h1, m1, s1 = marginal(model, X, valid, sg, {}, K, 1); dev = deviations(h1, m1, s1, T)
    h2, m2, s2 = marginal(model, X, valid, sg, {}, K, 7); dev2 = deviations(h2, m2, s2, T)
    out = dict(sigma=sg, depth=meta['depth'], n_test=n_test, K=K,
               dev_sd={k: float(v.std()) for k, v in dev.items()}, dev_mean={k: float(v.mean()) for k, v in dev.items()},
               noise_floor={k: r2(dev[k], dev2[k]) for k in dev}, zero={k: r2(dev[k], np.zeros_like(dev[k])) for k in dev}, rows=[])
    # untrained baseline
    torch.manual_seed(321); unt = N.NoisyPFN(model.inp.in_features, meta['depth'], meta['width'], meta['heads'], meta['mixk'], sg); unt.eval()
    hu, mu_, su = marginal(unt, X, valid, sg, {}, K, 3); du = deviations(hu, mu_, su, T)
    out['untrained'] = {k: r2(dev[k], du[k]) for k in dev}
    print('noise floor', out['noise_floor'], 'zero', out['zero'], 'untrained', out['untrained'], 'dev sd', out['dev_sd'], flush=True)
    # surrogates fitted on real paths of separate episodes
    dF, XF, vF = episodes(2 ** 14, 81_000); gen = torch.Generator().manual_seed(83_000)
    _, stF = trace(model, XF, vF, sg, gen)
    S = fit_surrogates(model, XF, vF, stF, sg)
    sites = list(S.keys())
    configs = {}
    for s in sites:
        for g in S[s]:
            if g.fn is not None: configs[f'{s}:{g.name}'] = {s: g}
    pick = lambda s, nm: next((g for g in S[s] if g.name == nm), None)
    best_near = {'emb': 'LR16', 'm': 'LR16', 'r': 'LR16', 'a': 'LRattn256'}
    cheap = {'emb': 'LR4', 'm': 'LR4', 'r': 'LR4', 'a': 'fixedpat'}
    for tag, fam in (('near', best_near), ('cheap', cheap)):
        cum = {}
        for s in sites:
            g = pick(s, fam['emb' if s == 'emb' else s[0]]); cum[s] = g
            configs[f'cum-{tag}-to-{s}'] = dict(cum)
        mlp_only = {s: pick(s, fam[s[0]]) for s in sites if s[0] in 'mr'}
        configs[f'mlp-only-{tag}'] = mlp_only
    for cname, sur in configs.items():
        hs_, ms_, ss_ = marginal(model, X, valid, sg, sur, K, 11); dh = deviations(hs_, ms_, ss_, T)
        row = dict(config=cname, **{f'R2_{k}': r2(dev[k], dh[k]) for k in dev}, rmse_mean=float(np.sqrt(((dev['mean'] - dh['mean']) ** 2).mean())))
        out['rows'].append(row)
        print(cname, {k: round(v, 3) for k, v in row.items() if k != 'config'}, flush=True)
        json.dump(out, open(os.path.join(run_dir, 'devpred.json'), 'w'), indent=1)
    # scenarios on fresh episodes
    rng = np.random.default_rng(95_000); d = W.sample_np(2 ** 15, rng); Xs, vs = W.tokens_np(d, rng, meta['ds'])
    ref, _, mom, wb = W.references(d)
    Ts = dict(comp_means=ref['bayes'][0], comp_vars=ref['bayes'][1], w_bayes=wb, mean_bayes=mom['bayes'][0], var_bayes=mom['bayes'][1])
    hh, mm, ss = marginal(model, Xs, vs, sg, {}, K, 21); dv = deviations(hh, mm, ss, Ts)
    F = scenario_features(d, Xs, Ts)
    out['scenarios'] = score_scenarios(F, dv)
    out['scenario_dev_overall'] = dict(KL=float(dv['kl'].mean()), RMS_mean=float(np.sqrt((dv['mean'] ** 2).mean())), mean_dlogsd=float(dv['logsd'].mean()))
    print('scenarios', json.dumps(out['scenarios']), flush=True)
    out['seconds'] = time.time() - t0
    json.dump(out, open(os.path.join(run_dir, 'devpred.json'), 'w'), indent=1)
    print('done', out['seconds'], flush=True)


if __name__ == '__main__':
    run(sys.argv[1])
