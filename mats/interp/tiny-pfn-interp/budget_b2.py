"""POST HOC (PREREG Amendment 3c): MLP1-at-QQ replacement with cross-source inputs.
b_cross: z = (log SS_s, o_s*, mean and |difference| of the other two sources' query readings, the other two sources'
log SS (sorted)). Fitted like budget.py Part 3; held-out NLL/KL vs model.
usage: python budget_b2.py RUN_DIR TAG OUT_JSON
"""
import json, os, sys
import numpy as np
import torch
from locate import TP, W, P, NTR, QI, Q, QL, S
from steps import run, QQ
from budget import fit_g

torch.set_num_threads(1)


def zcross(o, y):
    R = o[:, :, :QL] - y[:, None, :QL]
    lss = np.log((R ** 2).sum(2)); os_ = o[:, :, QL]
    f = []
    for s in range(S):
        ot = [t for t in range(S) if t != s]
        a, b = os_[:, ot[0]], os_[:, ot[1]]
        la, lb = lss[:, ot[0]], lss[:, ot[1]]
        f.append(np.stack([lss[:, s], os_[:, s], (a + b) / 2, np.abs(a - b), np.minimum(la, lb), np.maximum(la, lb)], 1))
    return np.stack(f, 1)


def main():
    run_dir, tag, outp = sys.argv[1:4]
    rd = os.path.join(TP, run_dir)
    meta = json.load(open(os.path.join(rd, 'meta.json')))
    D = dict(np.load(os.path.join(TP, 'runs', 'testset.npz')))
    model = P.build(meta); model.load_state_dict(torch.load(os.path.join(rd, 'model.pt')))
    n = len(D['y']); tr = np.arange(NTR); te = np.arange(NTR, n)
    X, m = D['X'], D['valid']
    head, C = run(model, X, m, cache=True)
    mo = C['mlp_out0'][:, QQ]
    Z = zcross(D['o'], D['y'])
    g = fit_g(Z[tr].reshape(-1, Z.shape[-1]), mo[tr].reshape(-1, mo.shape[-1]))
    G = g(Z[te].reshape(-1, Z.shape[-1])).reshape(len(te), S, -1).astype(np.float32)
    r2 = 1 - ((mo[te] - G) ** 2).sum() / ((mo[te] - mo[te].reshape(-1, mo.shape[-1]).mean(0)) ** 2).sum()

    def f(x, sl):
        x = x.clone(); x[:, QQ] = torch.from_numpy(G[sl.start:sl.stop]); return x
    Tk = {'comp_means': D['comp_means'][te], 'w_bayes': D['w_bayes'][te], 'comp_vars': D['comp_vars']}
    out = dict(tag=tag, fit_r2=float(r2))
    for nm, h in (('model', run(model, X[te], m[te])), ('b_cross', run(model, X[te], m[te], hk={('mlp_out', 0): f}))):
        out[nm] = dict(kl=float(P.kl_grid(Tk, h, chunk=200).mean()), nll=float(P.head_nll_np(h, D['ystar'][te]).mean()))
    print(json.dumps(out))
    json.dump(out, open(outp, 'w'))


if __name__ == '__main__':
    main()
