"""E48 input formats for world 2 (PREREG-B amendment 56; t2_world.py's task, unchanged and imported). Exact Bayes depends on
the context only through X^T X, X^T y and y.y, sums over examples of the per-example products. Two new formats give those
products LINEARLY, one token per example:
  prod     context tokens [0 (D), 0, 0, products]; the products are x_a x_b (a <= b), x_a y and y^2 (P = D(D+1)/2 + D + 1);
           no signed raw values, so the input is exactly invariant under per-example flips (x_i, y_i) -> (-x_i, -y_i);
  sgnprod  context tokens [x_i, y_i, 0, products]: the signed raw values as in ctrl, plus the products;
  query token in both: [x*, 0, 1, 0 (P)], as ctrl's query token followed by zeros.
Both new formats have width D + 2 + P; their first D + 2 columns are ctrl's layout. With build() below, a seed's prod and
sgnprod nets start from identical weights, and every parameter they share with the ctrl net of that seed (all but the P new
input columns) is copied from the ctrl initialisation (torch.manual_seed(seed)); the new columns are drawn U(-1/sqrt(D+2),
1/sqrt(D+2)) (the ctrl input columns' bound) from torch.Generator seeded 31337 + seed. ctrl and canon are t2_world.py's.
"""
import math
import numpy as np
import torch
import t2_world as T

FMTS = T.FMTS + ('prod', 'sgnprod')


def nprod(D):
    return D * (D + 1) // 2 + D + 1


def fin(fmt, D):
    return D + 2 + (nprod(D) if fmt in ('prod', 'sgnprod') else 0)


def products(x, y):
    """x (n, N, D), y (n, N) -> (n, N, P): x_a x_b for a <= b, then x_a y, then y^2."""
    D = x.shape[2]
    cols = [x[:, :, a] * x[:, :, b] for a in range(D) for b in range(a, D)]
    cols += [x[:, :, a] * y for a in range(D)]
    cols.append(y * y)
    return np.stack(cols, 2)


def tokens(d, fmt):
    if fmt in T.FMTS:
        return T.tokens(d, fmt)
    x, y, xq = d['x'], d['y'], d['xq']
    n, N, D = x.shape
    X = np.zeros((n, N + 1, fin(fmt, D)), dtype=np.float32)
    if fmt == 'sgnprod':
        X[:, :N, :D] = x; X[:, :N, D] = y
    X[:, :N, D + 2:] = products(x, y)
    X[:, N, :D] = xq; X[:, N, D + 1] = 1.0
    return X, np.ones((n, N + 1), dtype=bool)


def build(meta, P2):
    """The net for meta['fmt'] (pfn2's PFN). ctrl/canon: torch.manual_seed(seed) then PFN(D + 2, ...), as train_t2.py. prod and
    sgnprod: the ctrl initialisation of the same seed in every shared parameter, plus the new input columns (module docstring)."""
    D, fmt, seed = meta['D'], meta['fmt'], meta['seed']
    mk = lambda f: P2.PFN(f, meta['depth'], meta['width'], meta['heads'], mixk=meta['mixk'])
    torch.manual_seed(seed)
    if fmt in T.FMTS:
        return mk(D + 2)
    ref = mk(D + 2)
    model = mk(fin(fmt, D))
    sd, new = ref.state_dict(), model.state_dict()
    g = torch.Generator().manual_seed(31337 + seed)
    with torch.no_grad():
        for k, v in new.items():
            if k == 'inp.weight':
                v[:, :D + 2] = sd[k]
                v[:, D + 2:] = (torch.rand(v.shape[0], v.shape[1] - (D + 2), generator=g) * 2 - 1) / math.sqrt(D + 2)
            else:
                assert v.shape == sd[k].shape, k
                v.copy_(sd[k])
    model.load_state_dict(new)
    return model
