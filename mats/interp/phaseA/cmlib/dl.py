"""Description-length bookkeeping (METHOD §8-9).

DL(q) = DL_struct + DL_param (bits).
- struct_bits: fixed grammar (features from a library, map type, noise type, mixture K, factors, hand-set scalars).
- param_bits: per-tensor uniform quantisation with an fp16 scale; b* = smallest bit depth that moves the
  evaluation (e.g. Delta with common random numbers) by <= tol; DL_param = sum n_eff * b* + 16 * n_tensors.
  BIC cross-check (k/2) log2 N. An ordering of two rungs is claimed only if both codes agree.
- break_even_N: episodes after which a rung pays for its extra bits in a two-part code.
- Ledger: rows per rung, lower convex hull frontier in (DL, Delta_rms).
"""
import copy
import math
import numpy as np

MAP_TYPES = 3          # linear / spline-4 / MLP-8
NOISE_TYPES = 8        # iso / diag / rank-r / full x homo / hetero
FACTOR_TYPES = 4


def struct_bits(n_features=0, library_size=1, map_type=True, noise_type=True, rank_max=None, mixture_K=None,
                K_max=8, gate_features=0, factor=False, hand_scalars=0, new_feature_tokens=0, n_primitives=9):
    b = n_features * math.log2(max(library_size, 1))
    b += new_feature_tokens * math.log2(n_primitives)
    b += math.log2(MAP_TYPES) if map_type else 0
    b += math.log2(NOISE_TYPES) if noise_type else 0
    b += math.log2(rank_max) if rank_max else 0
    if mixture_K is not None:
        b += math.log2(K_max) + gate_features * math.log2(max(library_size, 1))
    b += math.log2(FACTOR_TYPES) if factor else 0
    b += 8 * hand_scalars
    return float(b)


def quantize(x, b):
    """Uniform symmetric quantisation to b bits with an fp16 per-tensor scale."""
    x = np.asarray(x, float)
    m = float(np.max(np.abs(x)))
    if m == 0:
        return x.copy()
    scale = float(np.float16(m / (2 ** (b - 1) - 1)))
    if scale == 0:
        return np.zeros_like(x)
    return np.round(x / scale) * scale


def n_eff(x):
    return int(np.count_nonzero(np.asarray(x)))


def param_bits(model, eval_fn, tol, bits=range(2, 25), N_fit=None):
    """Smallest common bit depth b* such that |eval_fn(quantised model) - eval_fn(model)| <= tol.
    model must expose params() -> {name: array} and set_params(dict). Returns dict with b_star, bits, bic."""
    P = model.params()
    base = eval_fn(model)
    ne = sum(n_eff(v) for v in P.values())
    res = dict(n_eff=ne, n_tensors=len(P), base=base, b_star=None, bits=None, moved=None)
    for b in bits:
        m2 = copy.deepcopy(model)
        m2.set_params({k: quantize(v, b) for k, v in P.items()})
        moved = abs(eval_fn(m2) - base)
        if moved <= tol:
            res.update(b_star=b, bits=ne * b + 16 * len(P), moved=moved)
            break
    if N_fit is not None:
        res['bic_bits'] = 0.5 * ne * math.log2(N_fit)
    return res


def break_even_N(d_bits, d_nats):
    """N* = Delta_DL * ln 2 / (reduction in Delta_rms per episode, nats)."""
    return float('inf') if d_nats <= 0 else d_bits * math.log(2) / d_nats


class Ledger:
    def __init__(self):
        self.rows = []

    def add(self, name, struct, param, delta=None, rms=None, bic=None, cost=None, **extra):
        self.rows.append(dict(name=name, struct=struct, param=param, dl=struct + param, bic=bic, delta=delta,
                              rms=rms, cost=cost, **extra))

    def frontier(self):
        """Lower convex hull of (DL, rms), returned as row names in increasing DL."""
        pts = sorted([(r['dl'], r['rms'], r['name']) for r in self.rows if r['rms'] is not None])
        # keep Pareto points then lower hull
        par, best = [], float('inf')
        for p in pts:
            if p[1] < best:
                par.append(p); best = p[1]
        hull = []
        for p in par:
            while len(hull) >= 2:
                (x1, y1, _), (x2, y2, _) = hull[-2], hull[-1]
                if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) <= 0:
                    hull.pop()
                else:
                    break
            hull.append(p)
        return [h[2] for h in hull]

    def orders_agree(self, a, b):
        """True if the quantised DL and BIC orderings of rungs a, b agree (claim an ordering only then)."""
        ra = next(r for r in self.rows if r['name'] == a); rb = next(r for r in self.rows if r['name'] == b)
        if ra['bic'] is None or rb['bic'] is None:
            return None
        return (ra['dl'] < rb['dl']) == (ra['struct'] + ra['bic'] < rb['struct'] + rb['bic'])
