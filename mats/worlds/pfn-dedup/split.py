"""Prior-level vs in-context split (PREREG amendment "prior-level vs in-context split").

R1  additive pool  mu = a_n + sum_r g_n(o_r)  (no token compared with another), fitted on fresh episodes.
R2  duplicate-blind Bayes on the n query readings (no identities, no labelled residuals, all rows distinct sources).
R2eq  R2 after dropping exact duplicate values (leak bracket).
Then each model's MSE to the arm's Bayes mean per split, and the share of its improvement over count-all that R1/R2
reproduce (prior-level) vs not (in-context).

usage: PYTHONPATH=... python split.py RUNS_DIR OUT_JSON [procs]
"""
import glob, json, math, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import world as W

KNOTS = np.arange(-5, 6)
TRAIN_SEED, NTRAIN = 50000, 300_000


def basis(o):
    """(..., 12): o and hat functions of width 1 at the knots."""
    h = np.maximum(0, 1 - np.abs(o[..., None] - KNOTS))
    return np.concatenate([o[..., None], h], -1)


def query_rows(d):
    """Query readings per row (copies included) and validity, (n, 9)."""
    src, valid = W.rows_of(d)
    return np.take_along_axis(d['o'][:, :, W.QL], src, 1), valid


def r1_features(d):
    rows, valid = query_rows(d)
    F = (basis(rows) * valid[..., None]).sum(1)                   # (n, 12)
    return F, valid.sum(1)


def fit_r1(lam=1.0):
    rng = np.random.default_rng(TRAIN_SEED)
    d = W.sample_np(NTRAIN, rng)
    F, nr = r1_features(d); y = d['y'][:, W.QL]
    coef = {}
    for n in np.unique(nr):
        s = nr == n
        A = np.concatenate([F[s], np.ones((s.sum(), 1))], 1)
        coef[int(n)] = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[s])
    return coef


def r1_predict(coef, d):
    F, nr = r1_features(d)
    A = np.concatenate([F, np.ones((len(F), 1))], 1)
    return np.array([A[i] @ coef[int(nr[i])] for i in range(len(F))])


_R2 = {}


def r2_tables(n):
    if n not in _R2:
        P = W.all_partitions(n)
        e = np.array([W.ewens(p) for p in P]); sing = np.array([len(set(p)) == n for p in P])
        prior = 0.7 * e / e.sum() + 0.3 * sing; prior /= prior.sum()
        C = np.stack([W.SE2 * np.eye(n) + W.SG2 * (p[:, None] == p[None, :]) for p in P])
        Ci = np.linalg.inv(C); one = np.ones(n)
        wv = np.einsum('pij,j->pi', Ci, one); prec = 1 + wv.sum(1)
        M = C + 1.0; Mi = np.linalg.inv(M); ld = np.linalg.slogdet(M)[1]
        _R2[n] = (np.log(prior), wv, prec, Mi, ld)
    return _R2[n]


def r2_mean(O):
    """Posterior mean of y* given readings O (m, n) under the duplicate-blind model."""
    lp, wv, prec, Mi, ld = r2_tables(O.shape[1])
    out = np.empty(len(O))
    for i in range(0, len(O), 200):
        o = O[i:i + 200]
        lw = lp[None] - 0.5 * np.einsum('ms,pst,mt->mp', o, Mi, o, optimize=True) - 0.5 * ld[None]
        lw -= lw.max(1, keepdims=True); w = np.exp(lw); w /= w.sum(1, keepdims=True)
        out[i:i + 200] = (w * (o @ wv.T) / prec[None]).sum(1)
    return out


def r2_predict(d, dedupe=False):
    rows, valid = query_rows(d)
    res = np.empty(len(rows))
    keys = []
    for i in range(len(rows)):
        v = rows[i][valid[i]]
        if dedupe:
            v = np.unique(v)
        keys.append(v)
    for n in sorted({len(v) for v in keys}):
        idx = [i for i, v in enumerate(keys) if len(v) == n]
        res[idx] = r2_mean(np.stack([keys[i] for i in idx]))
    return res


def model_mu(path):
    import torch
    from pfn import PFN, DEV, fixed_set, predict, TEST_SEED
    meta = json.load(open(os.path.join(path, 'meta.json')))
    _, X, m = fixed_set(TEST_SEED, 8000, meta['keys'])
    torch.manual_seed(meta['seed'])
    model = PFN(meta['fin'], meta['depth']).to(DEV)
    model.load_state_dict(torch.load(os.path.join(path, 'model.pt'), map_location=DEV))
    mu, _ = predict(model, X, m)
    return {'depth': meta['depth'], 'keys': meta['keys'], 'seed': meta['seed'], 'mu': mu.tolist()}


def main(runs, outp, procs=4):
    from pfn import TEST_SEED
    d = W.sample_np(8000, np.random.default_rng(TEST_SEED))   # same draw order as fixed_set (sample first)
    out, _, (md, _), _ = W.references(d)
    sizes, has_dup = W.redundancy_type(d)
    sp = {'dup_singletons': has_dup & (sizes == 1), 'shared_nodup': (~has_dup) & (sizes >= 2),
          'singletons_nodup': (~has_dup) & (sizes == 1)}
    coef = fit_r1()
    refs = {'a': out['a'][0], 'R1': r1_predict(coef, d), 'R2': r2_predict(d), 'R2eq': r2_predict(d, True),
            'b': out['b'][0], 'c': out['c'][0], 'd': md}
    dirs = sorted(x for x in glob.glob(os.path.join(runs, 'L*_k*_s*')) if os.path.isdir(x))
    with ProcessPoolExecutor(procs) as ex:
        models = list(ex.map(model_mu, dirs))
    res = {'refs': {}, 'models': []}
    for keys in (0, 1):
        bm = out['c'][0] if keys else md
        res['refs'][keys] = {s: {k: float(((v - bm)[sel] ** 2).mean()) for k, v in refs.items()} for s, sel in sp.items()}
    for mm in models:
        bm = out['c'][0] if mm['keys'] else md
        mu = np.array(mm['mu'])
        res['models'].append({**{k: mm[k] for k in ('depth', 'keys', 'seed')},
                              'mse': {s: float(((mu - bm)[sel] ** 2).mean()) for s, sel in sp.items()}})
    res['n'] = {s: int(sel.sum()) for s, sel in sp.items()}
    json.dump(res, open(outp, 'w'), indent=1)
    print(json.dumps(res['refs'], indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4)
