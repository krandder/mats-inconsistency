"""Direct check of the 'history-blind' reading (review item 3): per-episode KL(history-blind Bayes || net) and
KL(Bayes || net) on the test set, for given W0 runs. If the net computes the history-blind posterior, the first is
much smaller than the second. usage: python blind_cmp.py RUN [RUN ...]"""
import json, math, os, sys
import numpy as np
import sfworld as W

out = {}
for run in sys.argv[1:]:
    meta = json.load(open(os.path.join(run, 'meta.json')))
    N = meta['N']
    z = np.load(f'data/indep_N{N}_test.npz')
    p = np.load(os.path.join(run, 'pred_test.npz'))
    QL0 = W.QL
    try:
        W.QL = 0
        d0 = dict(o=z['o'][:, :, QL0:QL0 + 1], y=z['y'][:, QL0:QL0 + 1], v=z['v'], blk=z['blk'], grp=z['grp'], N=N,
                  mode='indep', rho=0.0, gsize=1, a1=1.0, a2=1.0)
        B0 = W.bayes(d0)
    finally:
        W.QL = QL0
    kl_blind = W.kl_bayes_model(B0, p['mu'], p['ls'], p['lg'])
    B = dict(grid=z['grid'].astype(np.float64), logp=z['logp'].astype(np.float64))
    kl_bayes = W.kl_bayes_model(B, p['mu'], p['ls'], p['lg'])
    # how far apart are the two references themselves
    lq_b0 = None
    res = dict(N=N, kl_blind_net=float(kl_blind.mean()), kl_blind_net_q90=float(np.quantile(kl_blind, 0.9)),
               kl_bayes_net=float(kl_bayes.mean()), kl_bayes_net_q90=float(np.quantile(kl_bayes, 0.9)))
    out[os.path.basename(run.rstrip('/'))] = res
    print(os.path.basename(run.rstrip('/')), json.dumps(res), flush=True)
json.dump(out, open('results/blind_cmp.json', 'w'), indent=1)
