"""E1 diagnostic 2 (post hoc): does the weighted fit exploit the null space of the single-draw Jacobian?
Refits the rank-4 weighted surrogate (as in qw.py) at m2 and r2 (2L seed 4) and compares, on held-out fit episodes,
E||J e||^2 with J at the fit's own noise draw (what the loss saw) against J' at a fresh downstream noise draw, and the
second-order remainder (actual single-draw feature change with the fit's own noise minus J e).
usage: python qw_lin2.py RUN_DIR -> RUN_DIR/qw_lin2.json
"""
import json, os, sys, time
import numpy as np
import torch
import nnoise as N
import world as W
import qw

torch.set_num_threads(1)


def run(run_dir, n_fit=8192, ntr=7168, steps=2000):
    t0 = time.time(); model, meta = N.load_run(run_dir); model.requires_grad_(False); sg = meta['sigma']
    U9 = json.load(open(os.path.join(run_dir, 'devpred.json'))); sds = [U9['dev_sd'][k] for k in ('mean', 'logsd', 'kl')]
    rng = np.random.default_rng(91_000); d = W.sample_np(n_fit, rng); X, valid = W.tokens_np(d, rng, meta['ds'])
    st, nz, allow = qw.trace_nz(model, X, valid, sg, torch.Generator().manual_seed(91_001)); B = qw.bayes_t(d)
    g = torch.Generator().manual_seed(77); nz2 = {k: torch.randn(v.shape, generator=g) for k, v in nz.items()}
    hid = torch.arange(ntr, n_fit); out = dict(rows=[])
    for site in ('m2', 'r2'):
        J = qw.jacobian(model, site, st, nz, allow, sg, B, sds)
        st2 = dict(st)                                                   # same upstream states, fresh downstream noise
        J2 = qw.jacobian(model, site, st2, nz2, allow, sg, B, sds)
        for wtd in (False, True):
            mod = qw.build(site, 4, model, st, X, None, None); qw.fit_module(mod, site, st, nz, X, sg, J, wtd, ntr, steps)
            with torch.no_grad():
                e = qw.errors(mod, site, st, nz, X, sg, hid)
                h = st[site][hid] + (e.view_as(st[site][hid]))            # surrogate output + the trace's own noise
                Fr = qw.features_t(qw.run_from(model, site, st[site][hid], st, nz, allow, sg, hid), B, hid, sds)
                Fs = qw.features_t(qw.run_from(model, site, h, st, nz, allow, sg, hid), B, hid, sds)
                Fr2 = qw.features_t(qw.run_from(model, site, st[site][hid], st, nz2, allow, sg, hid), B, hid, sds)
                Fs2 = qw.features_t(qw.run_from(model, site, h, st, nz2, allow, sg, hid), B, hid, sds)
            je = torch.einsum('nkd,nd->nk', J[hid], e); je2 = torch.einsum('nkd,nd->nk', J2[hid], e)
            row = dict(site=site, weighted=wtd, Je_own=[float((je[:, k] ** 2).mean()) for k in range(3)],
                       Je_fresh=[float((je2[:, k] ** 2).mean()) for k in range(3)],
                       actual_own=[float(((Fs - Fr)[:, k] ** 2).mean()) for k in range(3)],
                       actual_fresh=[float(((Fs2 - Fr2)[:, k] ** 2).mean()) for k in range(3)],
                       remainder_own=[float(((Fs - Fr - je)[:, k] ** 2).mean()) for k in range(3)])
            out['rows'].append(row); print(row, flush=True)
            json.dump(out, open(os.path.join(run_dir, 'qw_lin2.json'), 'w'), indent=1)
    out['seconds'] = time.time() - t0; json.dump(out, open(os.path.join(run_dir, 'qw_lin2.json'), 'w'), indent=1)
    print('done', out['seconds'], flush=True)


if __name__ == '__main__':
    run(sys.argv[1])
