"""N6 (scale-free-theory, 2026-10-29): M1 interaction profile on the 12 noisy phase-B nets (see PREREG-N6.md).

Same design as n5_interaction.py; the predictive is the noise-marginalised mixture over KD noise draws.
KL to Bayes is estimated on the 'source' base episodes as mean over episodes of the KL of the predictive on a y-grid.
Read-only use of work/cm-interp-B/nnoise.py and work/tiny-pfn.
"""
import json, math, sys, os
import numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/cm-interp-B')
sys.path.insert(0, '/home/agent/workspaces/mats/work/scale-free-theory')
import torch
import nnoise as NN
import world as W
from n5_interaction import variants, shares, bayes_logp, DY

RUNS = '/home/agent/workspaces/mats/work/cm-interp-B/runs'
A = ['L2_s0.05', 'L2seed1_s0.05', 'L2mx0_s0.05', 'L2mx1_s0.05', 'L2mx2_s0.05', 'L2nb6_s0.05']
B = ['L2seed2_s0.05', 'L2ws0_s0.05', 'L2ws3_s0.05', 'L2mx4_s0.05', 'L2nb5_s0.05', 'L2nb7_s0.05']
KD = 8


def mix_logp(mu, ls, lg, yq):
    """mu, ls, lg (n, R, K) mixture over R draws x K comps; yq (n, m) -> log p (n, m)."""
    n, R, K = mu.shape
    lw = (lg - math.log(R)).reshape(n, 1, R * K)
    m_ = mu.reshape(n, 1, R * K); s_ = ls.reshape(n, 1, R * K)
    lp = lw - s_ - 0.5 * math.log(2 * math.pi) - 0.5 * ((yq[:, :, None] - m_) / np.exp(s_)) ** 2
    mx = lp.max(-1, keepdims=True)
    return mx[..., 0] + np.log(np.exp(lp - mx).sum(-1))


def main():
    n = 4000
    res = {}
    grid = np.linspace(-5, 5, 801)
    for cut in ['source', 'within']:
        rng = np.random.default_rng(11 if cut == 'source' else 12)
        ds = variants(n, rng, cut)
        ystar = ds[0]['y'][:, W.QL]
        yq = ystar[:, None] + np.concatenate([[0.0], DY])[None]
        LB_ref = []
        for d in ds:
            lp = bayes_logp(d, yq)
            LB_ref.append(lp[:, 1:] - lp[:, :1])
        for name in A + B:
            model, meta = NN.load_run(os.path.join(RUNS, name))
            rngid = np.random.default_rng(99)
            semb = rngid.standard_normal((n, W.S, meta['ds'])) / 4
            qemb = rngid.standard_normal((n, W.Q, meta['ds'])) / 4
            Ln = []
            for vi, d in enumerate(ds):
                X, m = W.assemble(semb, qemb, d['o'], d['y'])
                mu, ls, lg = NN.predict_mix(model, X, m, KD, seed=1234)      # same noise seed for all variants
                lp = mix_logp(mu, ls, lg, yq)
                Ln.append(lp[:, 1:] - lp[:, :1])
                if cut == 'source' and vi == 0:
                    # KL(Bayes || net) on a grid for the base episodes (first 1000)
                    k = 1000
                    lb = bayes_logp({kk: (vv[:k] if hasattr(vv, '__len__') else vv) for kk, vv in d.items()},
                                    np.repeat(grid[None], k, 0))
                    ln = mix_logp(mu[:k], ls[:k], lg[:k], np.repeat(grid[None], k, 0))
                    pb = np.exp(lb - lb.max(1, keepdims=True)); pb /= pb.sum(1, keepdims=True)
                    pn = np.exp(ln - ln.max(1, keepdims=True)); pn /= pn.sum(1, keepdims=True)
                    kl = float(np.mean((pb * (np.log(np.maximum(pb, 1e-300)) - np.log(np.maximum(pn, 1e-300)))).sum(1)))
                    res.setdefault(name, {})['kl_grid'] = kl
            res.setdefault(name, {})[cut] = shares([x.ravel() for x in Ln])
            res[name][cut + '_dev'] = shares([(x - y).ravel() for x, y in zip(Ln, LB_ref)])
            res[name]['group'] = 'A' if name in A else 'B'
            print(cut, name, res[name]['group'], [round(x, 4) for x in res[name][cut]], res[name].get('kl_grid'), flush=True)
    json.dump(res, open('/home/agent/workspaces/mats/work/scale-free-theory/n6_interaction_noisy_v2.json', 'w'), indent=1)
    a_d = [res[k]['source_dev'][0] for k in A]; b_d = [res[k]['source_dev'][0] for k in B]
    print('DEV share (error non-additivity): A', np.round(a_d, 3), 'B', np.round(b_d, 3))
    a_s = [res[k]['source'][0] for k in A]; b_s = [res[k]['source'][0] for k in B]
    print('mean source share A', np.mean(a_s), 'B', np.mean(b_s), 'A below B median:', sum(x < np.median(b_s) for x in a_s))
    allk = [res[k]['kl_grid'] for k in A + B]; alls = a_s + b_s
    print('corr(source share, KL)', np.corrcoef(alls, allk)[0, 1])
    print('source < within:', sum(res[k]['source'][0] < res[k]['within'][0] for k in A + B), 'of 12')


if __name__ == '__main__':
    main()
