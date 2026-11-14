"""E1 diagnostic: is the question-weighted fit's failure a linearisation failure?
For a few site surrogates (refitted exactly as in qw.py), on 1,024 held-out fit-distribution episodes with common random
numbers: predicted first-order feature error E||J e||^2 vs the actual single-draw feature error E||F(sur) - F(real)||^2,
and the actual error of the K = 16 marginal's normalised deviation features (all three features, normalised as in E1).
usage: python qw_lin.py RUN_DIR -> RUN_DIR/qw_lin.json
"""
import json, math, os, sys, time
import numpy as np
import torch
import nnoise as N
import world as W
import qw

torch.set_num_threads(1)


def feats_single(model, X, valid, sigma, sur, nzs, B, sds, idx):
    """Normalised features of single-draw heads, for each noise dict in nzs; returns (K, n, 3)."""
    out = []
    allow = model.allow_mask(torch.from_numpy(valid))
    with torch.no_grad():
        for nz in nzs:
            st = {}; Xt = torch.from_numpy(X)
            x = (sur['emb'](Xt) if 'emb' in sur else model.f_emb(Xt)) + sigma * nz['emb']
            for l in range(1, model.depth + 1):
                xa = (sur[f'a{l}'](x) if f'a{l}' in sur else model.f_a(l, x, allow)) + sigma * nz[f'a{l}']
                m = (sur[f'm{l}'](xa) if f'm{l}' in sur else model.f_m(l, xa)) + sigma * nz[f'm{l}']
                x = (sur[f'r{l}'](xa, m) if f'r{l}' in sur else model.f_r(l, xa, m)) + sigma * nz[f'r{l}']
            out.append(qw.features_t(model.head(x), B, idx, sds))
    return torch.stack(out)


def run(run_dir, n_fit=8192, ntr=7168, steps=2000):
    t0 = time.time(); model, meta = N.load_run(run_dir); model.requires_grad_(False); sg = meta['sigma']
    U9 = json.load(open(os.path.join(run_dir, 'devpred.json'))); sds = [U9['dev_sd'][k] for k in ('mean', 'logsd', 'kl')]
    rng = np.random.default_rng(91_000); d = W.sample_np(n_fit, rng); X, valid = W.tokens_np(d, rng, meta['ds'])
    st, nz, allow = qw.trace_nz(model, X, valid, sg, torch.Generator().manual_seed(91_001)); B = qw.bayes_t(d)
    # held-out episodes and Bayes refs
    hid = torch.arange(ntr, n_fit); Xh, vh = X[ntr:], valid[ntr:]
    Bh = {k: (v[ntr:] if v.dim() > 0 and v.shape[0] == n_fit else v) for k, v in B.items()}
    g = torch.Generator().manual_seed(5)
    nzs = [{s: torch.randn((len(Xh),) + st[s].shape[1:], generator=g) for s in st} for _ in range(16)]
    ih = torch.arange(len(Xh))
    Freal = feats_single(model, Xh, vh, sg, {}, nzs, Bh, sds, ih)
    out = dict(rows=[])
    for site, lev in (('m2', 4), ('r2', 4), ('r1', 4), ('emb', 8)):
        J = qw.jacobian(model, site, st, nz, allow, sg, B, sds)
        for wtd in (False, True):
            mod = qw.build(site, lev, model, st, X, None, None)
            qw.fit_module(mod, site, st, nz, X, sg, J, wtd, ntr, steps)
            with torch.no_grad():
                e = qw.errors(mod, site, st, nz, X, sg, hid); je = torch.einsum('nkd,nd->nk', J[hid], e)
            Fs = feats_single(model, Xh, vh, sg, {site: mod}, nzs, Bh, sds, ih)
            dF = Fs - Freal                                                # (K, n, 3) actual single-draw change, same noise
            row = dict(site=site, rank=lev, weighted=wtd,
                       pred_lin=[float((je[:, k] ** 2).mean()) for k in range(3)],
                       actual_single=[float((dF[..., k] ** 2).mean()) for k in range(3)],
                       actual_mean_over_draws=[float((dF[..., k].mean(0) ** 2).mean()) for k in range(3)],
                       KL_nats=float((e ** 2).sum(1).mean() / (2 * sg ** 2)))
            out['rows'].append(row); print(row, flush=True)
            json.dump(out, open(os.path.join(run_dir, 'qw_lin.json'), 'w'), indent=1)
    out['seconds'] = time.time() - t0; json.dump(out, open(os.path.join(run_dir, 'qw_lin.json'), 'w'), indent=1)
    print('done', out['seconds'], flush=True)


if __name__ == '__main__':
    run(sys.argv[1])
