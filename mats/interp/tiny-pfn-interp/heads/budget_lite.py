"""PREREG Amendment 5: B2 budget (Part 1 + output-moment split) for the head-count / capacity arms.
usage: python budget_lite.py RUN_DIR TAG OUT_JSON   (RUN_DIR absolute or relative to cwd; test set from $TINYPFN_DIR/runs)
"""
import json, math, os, sys
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from locate import TP, W, NTR
from steps import run
from budget import draw, mixture, split_ids, K
sys.path.insert(0, HERE)
import pfn2 as P

torch.set_num_threads(1)


def main():
    rd, tag, outp = sys.argv[1:4]
    meta = json.load(open(os.path.join(rd, 'meta.json')))
    ds = meta['ds']
    D = dict(np.load(os.path.join(TP, 'runs', 'testset.npz')))
    model = P.build(meta); model.load_state_dict(torch.load(os.path.join(rd, 'model.pt'), map_location='cpu'))
    n = len(D['y']); te = np.arange(NTR, n)
    X, m = D['X'], D['valid']
    semb, qemb = split_ids(X, ds)
    Tk = {'comp_means': D['comp_means'][te], 'w_bayes': D['w_bayes'][te], 'comp_vars': D['comp_vars']}
    ys = D['ystar'][te]; mb, vb = D['mean_bayes'][te], D['var_bayes'][te]
    o0, y0, s0, q0 = D['o'][te], D['y'][te], semb[te], qemb[te]
    res = dict(tag=tag, params=P.n_params(model), nll_gap_test=None)
    full = run(model, X, m)
    res['nll_gap_test'] = float((P.head_nll_np(full, D['ystar']) - D['nll_bayes']).mean())
    perep = {}

    def mfix(h, var_too):
        mu, ls, lg = h; mm, vm = P.head_moments(h)
        if not var_too:
            return (mu + (mb - mm)[:, None], ls, lg)
        c = np.sqrt(vb / vm)
        return (mb[:, None] + (mu - mm[:, None]) * c[:, None], ls + np.log(c)[:, None], lg)

    def score(h, name, split=False):
        kl = P.kl_grid(Tk, h, chunk=200)
        perep[name] = kl.astype(np.float32)
        r = dict(kl=float(kl.mean()), nll=float(P.head_nll_np(h, ys).mean()))
        if split:
            r['kl_meanfix'] = float(P.kl_grid(Tk, mfix(h, False), chunk=200).mean())
            r['kl_mvfix'] = float(P.kl_grid(Tk, mfix(h, True), chunk=200).mean())
        res[name] = r
        print(tag, name, r, flush=True)

    def avg(which):
        rng = np.random.default_rng(1000)
        return mixture([run(model, *draw(rng, which, o0, y0, s0, q0, ds)[:2]) for _ in range(K)])

    score(tuple(x[te] for x in full), 'none', split=True)
    score(avg(('id',)), 'id')
    score(avg(('rot',)), 'rot')
    score(avg(('id', 'rot', 'y')), 'all', split=True)
    np.savez_compressed(outp.replace('.json', '_perep.npz'), **perep)
    json.dump(res, open(outp, 'w'))


if __name__ == '__main__':
    main()
