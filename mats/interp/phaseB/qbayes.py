"""E3a (amendment 10): the whole network described relative to Bayes.

Free inputs b(x): exact Bayes quantities of the episode (posterior mean and log-sd, per-source reliability posteriors from
the labelled data and with the query coupling, each source's posterior weight in the mean, bimodality); z-scored.
Description: predictive mixture head (4 components) = MLP_h(b); cost = MLP parameters only. Trained to minimise
E_x E_z KL(network single-draw predictive || description) (= KL to the network's noise-marginal predictive + const), fresh
noise each step. Scored as E1: K = 64 independent-draw deviation R^2 on the 2,000 test episodes; scenario agreement.
usage: python qbayes.py RUN_DIR [--smoke] -> RUN_DIR/qbayes.json
"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import nnoise as N
import world as W
import qw
import devpred as DP
from sur_pfn import chain_forward

torch.set_num_threads(1)


def bayes_features(d):
    ref, _, mom, wb = W.references(d)
    mb, vb = mom['bayes']; n = len(mb)
    lab = W.source_posterior(d).reshape(n, -1)                                    # (n, 9)
    full = np.stack([np.stack([wb[:, W.COMB[:, s] == j].sum(1) for j in range(W.NV)], 1) for s in range(W.S)], 1).reshape(n, -1)
    wsrc = np.stack([(wb * (1.0 / W.VC[:, s] / W.PREC)[None]).sum(1) for s in range(W.S)], 1)   # posterior weight of source s
    means = ref['bayes'][0]; between = (wb * means ** 2).sum(1) - mb ** 2
    F = np.c_[mb, 0.5 * np.log(vb), lab, full, wsrc, between / vb]
    return F.astype(np.float32)                                                    # (n, 24)


class BayesHead(nn.Module):
    def __init__(self, k, h, mixk=4):
        super().__init__()
        self.net = nn.Linear(k, 3 * mixk) if h == 0 else nn.Sequential(nn.Linear(k, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 3 * mixk))
        self.k = mixk; self.cost = sum(p.numel() for p in self.parameters())
    def forward(self, b):
        y = self.net(b); K = self.k
        return y[:, K:2 * K], y[:, 2 * K:].clamp(-6, 3), y[:, :K]


def kl_mix(p, q, Bm, Bv, ngrid=201):
    """KL(p || q) for two mixture heads by quadrature on a grid spanning both and the Bayes comps."""
    (mp, lp_, gp), (mq, lq_, gq) = p, q
    sd = torch.sqrt(Bv)
    lo = torch.minimum((Bm - 9 * sd).min(1).values, (mp - 9 * torch.exp(lp_)).min(1).values)
    hi = torch.maximum((Bm + 9 * sd).max(1).values, (mp + 9 * torch.exp(lp_)).max(1).values)
    g = (lo[:, None] + (hi - lo)[:, None] * torch.linspace(0, 1, ngrid)[None]).detach()
    def lpdf(mu, ls, lg):
        c = torch.log_softmax(lg, -1)[:, None, :] - ls[:, None, :] - 0.5 * math.log(2 * math.pi) - 0.5 * ((g[:, :, None] - mu[:, None, :]) / torch.exp(ls)[:, None, :]) ** 2
        return torch.logsumexp(c, 2)
    a, b_ = lpdf(mp, lp_, gp), lpdf(mq, lq_, gq)
    return torch.trapezoid(torch.exp(a) * (a - b_), g, dim=1)


def run(run_dir, smoke=False):
    t0 = time.time(); model, meta = N.load_run(run_dir); model.requires_grad_(False); sg = meta['sigma']
    n_fit, steps, n_test, K, n_sc = (2048, 100, 300, 8, 4096) if smoke else (16384, 4000, 2000, 64, 2 ** 15)
    rng = np.random.default_rng(94_000); d = W.sample_np(n_fit, rng); X, valid = W.tokens_np(d, rng, meta['ds'])
    Fb = bayes_features(d); mu_, sd_ = Fb.mean(0), Fb.std(0) + 1e-6; bF = torch.as_tensor((Fb - mu_) / sd_)
    ref, _, _, wb = W.references(d); Bm = torch.as_tensor(ref['bayes'][0], dtype=torch.float32); Bv = torch.as_tensor(ref['bayes'][1], dtype=torch.float32)
    allow = model.allow_mask(torch.from_numpy(valid))
    out = dict(rows=[], features=['mu_B', 'log sd_B'] + [f'lab s{s} v{j}' for s in range(3) for j in range(3)] +
               [f'bayes s{s} v{j}' for s in range(3) for j in range(3)] + [f'weight s{s}' for s in range(3)] + ['bimodality'])
    # test set and network deviation (K draws, two independent estimates)
    Tz = np.load(os.path.join(DP.TP, 'runs/testset.npz'))
    Tt = {k: Tz[k][:n_test] for k in ('comp_means', 'w_bayes', 'mean_bayes', 'var_bayes')}; Tt['comp_vars'] = Tz['comp_vars']
    Xt, vt = Tz['X'][:n_test], Tz['valid'][:n_test]
    dT = dict(y=Tz['y'][:n_test], o=Tz['o'][:n_test], v=Tz['v'][:n_test], vidx=Tz['vidx'][:n_test])
    bT = torch.as_tensor((bayes_features(dT) - mu_) / sd_)
    def net_dev(seed):
        h, _, _ = DP.marginal(model, Xt, vt, sg, {}, K, seed); return qw.deviations_t(h, Tt)
    devT, devT2 = net_dev(1), net_dev(7)
    out['floor'] = {k: DP.r2(devT[k], devT2[k]) for k in devT}
    # scenarios: network outcomes (K = 8) on fresh episodes
    rs = np.random.default_rng(95_000); ds_ = W.sample_np(n_sc, rs); Xs, vs = W.tokens_np(ds_, rs, meta['ds'])
    refs, _, moms, wbs = W.references(ds_)
    Ts = dict(comp_means=refs['bayes'][0], comp_vars=refs['bayes'][1], w_bayes=wbs, mean_bayes=moms['bayes'][0], var_bayes=moms['bayes'][1])
    Fsc = DP.scenario_features(ds_, Xs, Ts)
    hs, _, _ = DP.marginal(model, Xs, vs, sg, {}, 8, 21); net_sc = DP.score_scenarios(Fsc, qw.deviations_t(hs, Ts)); out['network_scenarios'] = net_sc
    bS = torch.as_tensor((bayes_features(ds_) - mu_) / sd_)
    print('floor', out['floor'], flush=True)
    for h in ([0, 16] if smoke else [0, 8, 16, 32, 64]):
        torch.manual_seed(h); mod = BayesHead(bF.shape[1], h)
        opt = torch.optim.Adam(mod.parameters(), lr=3e-3); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps); g = torch.Generator().manual_seed(h + 1)
        for s in range(steps):
            idx = torch.randint(n_fit, (256,), generator=g)
            with torch.no_grad(): hn = chain_forward(model, X[idx.numpy()], valid[idx.numpy()], sg, {}, g)
            loss = kl_mix(hn, mod(bF[idx]), Bm[idx], Bv).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        with torch.no_grad():
            hd = [t.numpy() for t in mod(bT)]; dv = qw.deviations_t(hd, Tt)
            hsd = [t.numpy() for t in mod(bS)]; sc = DP.score_scenarios(Fsc, qw.deviations_t(hsd, Ts))
        row = dict(hidden=h, cost=mod.cost, R2={k: DP.r2(devT[k], dv[k]) for k in dv}, scenario_agree=int(sum(sc[k]['hit'] == net_sc[k]['hit'] for k in sc)),
                   scenarios={k: sc[k]['hit'] for k in sc}, final_loss=float(loss.detach()))
        out['rows'].append(row); print(row, flush=True)
        json.dump(out, open(os.path.join(run_dir, 'qbayes_smoke.json' if smoke else 'qbayes.json'), 'w'), indent=1)
    out['seconds'] = time.time() - t0
    json.dump(out, open(os.path.join(run_dir, 'qbayes_smoke.json' if smoke else 'qbayes.json'), 'w'), indent=1)
    print('done', out['seconds'], flush=True)


if __name__ == '__main__':
    run(sys.argv[1], smoke='--smoke' in sys.argv)
