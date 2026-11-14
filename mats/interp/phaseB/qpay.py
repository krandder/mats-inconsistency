"""E4 (PREREG-B amendment 11): payoff in compute of a description relative to Bayes.

Questions (distribution level, fixed in advance): Q1-Q10 = average KL(Bayes || network's noise-marginal predictive, K = 64)
over episodes in scenario S1..S9 and overall (scenario thresholds fixed from a reference sample); Q11-Q12 = P(|deviation of
the predictive mean| > tau) with tau at the ground truth's 99% and 99.9% quantiles.
Ground truth: the network on 2^15 fresh episodes, K = 64 each (scenario S8 also from 2^13 extra query-label-tilted episodes
with exact weights).
Estimators at a budget of B forward passes: (a) plain sampling of the network, B / 64 episodes; (b) the description:
its setup cost (network passes used to fit it, plus its own training time converted to passes) plus its evaluation on
fresh episodes (its time converted to passes); answers from as many episodes as the remaining budget allows.
usage: python qpay.py RUN_DIR DESC_KIND [H] -> RUN_DIR/qpay.json   (DESC_KIND: e3a; H = hidden width)
"""
import json, math, os, sys, time
import numpy as np
import torch
import nnoise as N
import world as W
import qw
import devpred as DP
import qbayes as QB
from sur_pfn import chain_forward
from scipy.stats import norm

torch.set_num_threads(1)
K = 64


def scen_masks(F, thr):
    vidx = F['vidx']
    return {'S1': F['bimod'] >= thr['bimod'], 'S2': (F['spread'] >= thr['spread']) & (F['ent_lab'] > thr['ent_lab']),
            'S3': (vidx == 2).all(1), 'S4': (vidx == 0).all(1), 'S5': ((vidx == 0).sum(1) == 1) & ((vidx == 2).sum(1) == 2),
            'S6': F['cos'] >= thr['cos'], 'S7': (F['share'] > 0.7).any(1), 'S8': np.abs(F['mb']) > 2.5, 'S9': F['coupling'] >= thr['coupling'],
            'all': np.ones(len(vidx), bool)}


def world(n, seed, shift=None, ds=16):
    rng = np.random.default_rng(seed); d = W.sample_np(n, rng); wts = np.ones(n)
    if shift is not None:
        e = d['o'][:, :, W.QL] - d['y'][:, W.QL][:, None]
        sgn = np.where(rng.random(n) < 0.5, 1.0, -1.0); ys = sgn * (rng.standard_normal(n) + shift)
        d['y'][:, W.QL] = ys; d['o'][:, :, W.QL] = ys[:, None] + e
        q = 0.5 * norm.pdf(ys, loc=shift) + 0.5 * norm.pdf(ys, loc=-shift); wts = norm.pdf(ys) / q
    X, v = W.tokens_np(d, rng, ds)
    ref, _, mom, wb = W.references(d)
    Tt = dict(comp_means=ref['bayes'][0], comp_vars=ref['bayes'][1], w_bayes=wb, mean_bayes=mom['bayes'][0], var_bayes=mom['bayes'][1])
    return d, X, v, Tt, wts


def net_dev(model, X, v, Tt, sg, seed, Kd=K):
    devs = []
    for i in range(0, len(X), 2000):
        T_ = {k: (a[i:i + 2000] if k != 'comp_vars' else a) for k, a in Tt.items()}
        h, _, _ = DP.marginal(model, X[i:i + 2000], v[i:i + 2000], sg, {}, Kd, seed + i)
        devs.append(qw.deviations_t(h, T_))
    return {k: np.concatenate([d_[k] for d_ in devs]) for k in devs[0]}


def answers(dev, masks, wts, taus):
    a = {}
    for s, m in masks.items():
        w = wts * m; a[f'KL_{s}'] = float((w * dev['kl']).sum() / max(w.sum(), 1e-300))
    for j, t in enumerate(taus):
        a[f'tail_{j}'] = float((wts * (np.abs(dev['mean']) > t)).sum() / wts.sum())
    return a


def run(run_dir, kind='e3a', H=16):
    t0 = time.time(); model, meta = N.load_run(run_dir); model.requires_grad_(False); sg = meta['sigma']
    out = dict(kind=kind, H=H)
    # scenario thresholds from a fixed reference sample
    dR, XR, vR, TR, _ = world(2 ** 14, 96_000); FR = DP.scenario_features(dR, XR, TR)
    thr = {k: float(np.quantile(FR[k], 0.9)) for k in ('bimod', 'spread', 'cos', 'coupling')}; thr['ent_lab'] = float(np.median(FR['ent_lab']))
    out['thresholds'] = thr
    # ---- ground truth
    tg = time.time()
    dG, XG, vG, TG, wG = world(2 ** 15, 97_000); devG = net_dev(model, XG, vG, TG, sg, 1)
    mG = scen_masks(DP.scenario_features(dG, XG, TG), thr)
    taus = [float(np.quantile(np.abs(devG['mean']), q)) for q in (0.99, 0.999)]
    truth = answers(devG, mG, wG, taus)
    dT8, XT8, vT8, TT8, wT8 = world(2 ** 13, 97_500, shift=3.0); devT8 = net_dev(model, XT8, vT8, TT8, sg, 5)
    m8 = np.abs(TT8['mean_bayes']) > 2.5
    truth['KL_S8_tilted'] = float((wT8 * m8 * devT8['kl']).sum() / (wT8 * m8).sum())
    out['truth'] = truth; out['taus'] = taus; out['truth_seconds'] = time.time() - tg
    out['scenario_counts'] = {s: int(m.sum()) for s, m in mG.items()}
    # pass rate of the network (single noisy pass)
    tr = time.time(); chain_forward(model, XG[:4096], vG[:4096], sg, {}, torch.Generator().manual_seed(3)); rate = 4096 / (time.time() - tr)
    out['rate_passes_per_s'] = rate
    print('truth', {k: round(v, 5) for k, v in truth.items()}, 'rate', round(rate), flush=True)
    # ---- the description: E3a Bayes head, fitted here (its setup cost is measured)
    ts = time.time(); n_fit, steps = 16384, 4000
    rng = np.random.default_rng(94_000); d = W.sample_np(n_fit, rng); X, valid = W.tokens_np(d, rng, meta['ds'])
    Fb = QB.bayes_features(d); mu_, sd_ = Fb.mean(0), Fb.std(0) + 1e-6; bF = torch.as_tensor((Fb - mu_) / sd_)
    ref, _, _, _ = W.references(d); Bm = torch.as_tensor(ref['bayes'][0], dtype=torch.float32); Bv = torch.as_tensor(ref['bayes'][1], dtype=torch.float32)
    torch.manual_seed(H); mod = QB.BayesHead(bF.shape[1], H)
    opt = torch.optim.Adam(mod.parameters(), lr=3e-3); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps); g = torch.Generator().manual_seed(H + 1)
    for s in range(steps):
        idx = torch.randint(n_fit, (256,), generator=g)
        with torch.no_grad(): hn = chain_forward(model, X[idx.numpy()], valid[idx.numpy()], sg, {}, g)
        loss = QB.kl_mix(hn, mod(bF[idx]), Bm[idx], Bv).mean(); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    setup_s = time.time() - ts; setup_passes_network = steps * 256
    out['setup_seconds'] = setup_s; out['setup_network_passes'] = setup_passes_network; out['setup_pass_equiv'] = setup_s * rate
    # description evaluation cost per episode (world sample + Bayes features + head + quadrature KL)
    def desc_dev(dd, Tt):
        b = torch.as_tensor((QB.bayes_features(dd) - mu_) / sd_)
        with torch.no_grad(): hd = [t.numpy() for t in mod(b)]
        return qw.deviations_t(hd, Tt)
    te = time.time(); dE, XE, vE, TE, wE = world(2 ** 15, 98_000); devE = desc_dev(dE, TE); mE = scen_masks(DP.scenario_features(dE, XE, TE), thr)
    per_ep_s = (time.time() - te) / 2 ** 15; out['desc_seconds_per_episode'] = per_ep_s; out['desc_pass_equiv_per_episode'] = per_ep_s * rate
    desc_big = answers(devE, mE, wE, taus)
    dE8, XE8, vE8, TE8, wE8 = world(2 ** 15, 98_500, shift=3.0); devE8 = desc_dev(dE8, TE8); m8e = np.abs(TE8['mean_bayes']) > 2.5
    desc_big['KL_S8_tilted'] = float((wE8 * m8e * devE8['kl']).sum() / (wE8 * m8e).sum())
    out['desc_answers_2^15'] = desc_big
    out['desc_rel_error'] = {k: (desc_big[k] - truth[k]) / truth[k] for k in truth if truth[k] > 0}
    print('desc rel errors', {k: round(v, 3) for k, v in out['desc_rel_error'].items()}, flush=True)
    # ---- sampling at budgets B: B / 64 episodes with K = 64 (8 repetitions); description: setup + evaluation
    budgets = [2 ** k for k in range(14, 22)]; reps = 8; rows = []
    # sampling estimates by subsampling the ground-truth pool (independent fresh pool for budgets beyond it)
    dP, XP, vP, TP, _ = world(2 ** 15, 99_000); devP = net_dev(model, XP, vP, TP, sg, 9); mP = scen_masks(DP.scenario_features(dP, XP, TP), thr)
    rngs = np.random.default_rng(5)
    for B in budgets:
        n = max(B // K, 1); est = []
        for r in range(reps):
            if n <= len(XP):
                ii = rngs.choice(len(XP), n, replace=False)
                est.append(answers({k: v[ii] for k, v in devP.items()}, {s: m[ii] for s, m in mP.items()}, np.ones(n), taus))
        se = {k: float(np.sqrt(np.mean([(e[k] - truth[k]) ** 2 for e in est])) / truth[k]) if est and truth[k] > 0 else None for k in truth if k != 'KL_S8_tilted'}
        n_desc = int(max(B - setup_passes_network - setup_s * rate, 0) / max(per_ep_s * rate, 1e-9))
        rows.append(dict(B=B, n_sampling_episodes=n, sampling_rel_rmse=se, desc_available=bool(n_desc > 0), desc_episodes=min(n_desc, 10 ** 9)))
        print(B, n, {k: (round(v, 3) if v is not None else None) for k, v in se.items()}, 'desc episodes', n_desc, flush=True)
        json.dump(dict(out, rows=rows), open(os.path.join(run_dir, 'qpay.json'), 'w'), indent=1)
    out['rows'] = rows; out['seconds'] = time.time() - t0
    json.dump(out, open(os.path.join(run_dir, 'qpay.json'), 'w'), indent=1)
    print('done', out['seconds'], flush=True)


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'e3a', int(sys.argv[3]) if len(sys.argv) > 3 else 16)
