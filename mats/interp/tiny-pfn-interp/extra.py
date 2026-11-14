"""POST-HOC additions (written after seeing the pre-registered tables; not part of PREREG decision rules).
(a) Control targets for the query-token mean: the equal-weight mean of the query readings (reference a) explains most
    of the Bayes mean, so the reliability-specific part is probed separately: dmean = mean_bayes - mean_a.
(b) Query coupling at the source level: dE = E_bayes - E_lab (what the query readings add to a source's reliability).
(c) Layer-1 attention vs the SIGNED residual of the key token (own-source labelled keys).
usage: python extra.py RUN_DIR TAG OUT_JSON
"""
import json, os, sys
import numpy as np
import torch
from locate import (TP, W, P, NTR, SITES, QI, Q, QL, S, forward_cache, targets, probe_set, per_source_to_tokens,
                    lab_idx, qq_idx)


def main():
    run, tag, outp = sys.argv[1:4]
    rd = os.path.join(TP, run)
    meta = json.load(open(os.path.join(rd, 'meta.json')))
    ds = meta['ds']
    D = np.load(os.path.join(TP, 'runs', 'testset.npz' if ds == W.DS else f'testset_ds{ds}.npz'))
    X, m = D['X'], D['valid']
    trained = P.build(meta); trained.load_state_dict(torch.load(os.path.join(rd, 'model.pt')))
    torch.manual_seed(meta['seed']); untrained = P.build(meta)
    perm = np.random.default_rng(0).permutation(NTR)
    res = dict(tag=tag)
    te = np.arange(NTR, len(X))
    R = D['o'][:, :, :QL] - D['y'][:, None, :QL]
    for nm, model in (('trained', trained), ('untrained', untrained)):
        acts, attns, head = forward_cache(model, X, m)
        src, _, _, _, beta, El = targets(D, P.head_moments(head)[0])
        dmean = D['mean_bayes'] - D['mean_a']
        dE = src['E_bayes'] - src['E_lab']
        out = {}
        for site in SITES:
            A = acts[site]
            FT, FQ, FL = A[:, QI:QI + 1], A[:, qq_idx()], A[:, lab_idx()]
            for name, v in (('mean_a', D['mean_a']), ('dmean', dmean), ('dmean_lab', D['mean_lab'] - D['mean_a'])):
                out[f'{name}|QT|{site}'] = probe_set(FT, v[:, None, None], perm)[0].tolist()
                out[f'{name}|QQ|{site}'] = probe_set(FQ, np.repeat(v[:, None, None], S, 1), perm)[0].tolist()
            out[f'dE|QQ|{site}'] = probe_set(FQ, per_source_to_tokens(dE, 'QQ'), perm)[0].tolist()
            out[f'dE|LAB|{site}'] = probe_set(FL, per_source_to_tokens(dE, 'LAB'), perm)[0].tolist()
        res[f'probes_{nm}'] = out
        # (c) layer-1 attention weight from QQ / LAB tokens to own-source labelled keys vs signed r and r^2 of the key
        at = {}
        for li, Aat in enumerate(attns):
            for h in range(Aat.shape[1]):
                a = Aat[te, h]
                for frm in ('QQ', 'LAB'):
                    xs, rs_, r2s = [], [], []
                    for s in range(S):
                        for q in ([QL] if frm == 'QQ' else range(QL)):
                            i = s * Q + q
                            for qq in range(QL):
                                j = s * Q + qq
                                if j == i: continue
                                xs.append(a[:, i, j]); rs_.append(R[te, s, qq]); r2s.append(R[te, s, qq] ** 2)
                    x = np.concatenate(xs)
                    at[f'L{li + 1}H{h}|{frm}'] = dict(r_signed=float(np.corrcoef(x, np.concatenate(rs_))[0, 1]),
                                                      r_sq=float(np.corrcoef(x, np.concatenate(r2s))[0, 1]))
        res[f'attn_{nm}'] = at
        print(tag, nm, flush=True)
    json.dump(res, open(outp, 'w'))


if __name__ == '__main__':
    main()
