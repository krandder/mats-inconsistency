"""ARC's rare-event method (section 6.6) on the noisy nets: cumulant propagation through the L - 1 ReLU layers (noise
added to the second cumulant after each ReLU, as in ../kprop_noisy.py), then the last linear layer with bias -t and a
heaviside step, whose first cumulant is the estimate of P(z_L,i > t_i). This is ARC's run_lpe construction
(nonlin = relu x layer + heaviside, b_mean = -b at the heaviside layer) with our weights, per-neuron thresholds and noise.

Several threshold vectors share one propagation; the last linear + heaviside step is repeated per threshold vector
(its FLOPs are counted each time). FLOPs: ARC's NamedFlopCounter, as in part 1.
"""
import os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kprop_noisy import arc_mlp, add_noise_to_K2, KH, NamedFlopCounter, flop_name, WICK_COEF_D   # noqa: E402


@torch.no_grad()
def kprop_tail(tnet, thresholds, k_max=3, factor=True, kind='simple', avg_metric=False, dtype=torch.float64):
    """tnet: tails.tl.TNet (depth L). thresholds: list of (n,) tensors. Returns (list of (n,) estimates, flops, secs,
    flops of the shared propagation)."""
    L, n = tnet.L, tnet.n

    class _N:                                   # the attributes arc_mlp reads
        pass
    nn_ = _N(); nn_.n = n; nn_.W = tnet.W64
    mlp = arc_mlp(nn_, L, dtype)
    kind_e = {'simple': KH.SIMPLE, 'augment': KH.AUGMENT}[kind]
    counter = NamedFlopCounter(); t0 = time.time()
    outs = []
    counter.__enter__()
    try:
        with flop_name('kprop'):
            K_in = {1: torch.zeros(n, dtype=dtype), 2: torch.eye(n, dtype=dtype)}
            K = KH.coerce_input(K_in, k_max=k_max, kind=kind_e)
            skip_metric = not KH.metric_needed(k_max, kind_e)

            def metric_for(l):
                if avg_metric:
                    return mlp.init_scale[l]
                return 1.0 if skip_metric else None
            for l in range(L - 1):
                K = KH.linear_kprop(K, mlp.Ws[l].weight, k_max=k_max, set_metric=metric_for(l), bias=None)
                K = KH.nonlin_kprop(K, nonlin_wick_coef=WICK_COEF_D['relu'], k_max=k_max, kind=kind_e, use_pK=True,
                                    factor=factor)
                if tnet.sigma > 0:
                    K = add_noise_to_K2(K, float(tnet.s[l]) ** 2)
            shared = counter.total()
            for t in thresholds:
                Kt = KH.linear_kprop(K, mlp.Ws[L - 1].weight, k_max=k_max, set_metric=metric_for(L - 1),
                                     bias=-t.to(dtype))
                Kt = KH.nonlin_kprop(Kt, nonlin_wick_coef=WICK_COEF_D['heaviside'], k_max=k_max, kind=kind_e,
                                     use_pK=True, factor=factor)
                outs.append(Kt[1].to_tensor().clone())
    finally:
        counter.__exit__(None, None, None)
    return outs, counter.total(), time.time() - t0, shared


@torch.no_grad()
def kprop_moments(tnet, k_max=3, factor=True, kind='simple', avg_metric=False, dtype=torch.float64, powers=(1, 2, 3, 4)):
    """Raw moments E[z_L,i^p] of the description (the Wick readout of a degree-p polynomial is exact given the cumulants
    the description retains; k_max = 3 is ARC's harmonic approximation budget, not a cut at the third cumulant: SIMPLE
    keeps a radial fourth cumulant, AUGMENT more fourth-order structure). Returns ((len(powers), n) tensor, flops). The
    propagation is shared; each power costs one last linear + polynomial step."""
    from numpy.polynomial import Polynomial
    from functools import partial
    from mlp_kprop.wick import poly_wick_coef
    L, n = tnet.L, tnet.n

    class _N:
        pass
    nn_ = _N(); nn_.n = n; nn_.W = tnet.W64
    mlp = arc_mlp(nn_, L, dtype)
    kind_e = {'simple': KH.SIMPLE, 'augment': KH.AUGMENT}[kind]
    counter = NamedFlopCounter(); outs = []
    counter.__enter__()
    try:
        with flop_name('kprop'):
            K_in = {1: torch.zeros(n, dtype=dtype), 2: torch.eye(n, dtype=dtype)}
            K = KH.coerce_input(K_in, k_max=k_max, kind=kind_e)
            skip_metric = not KH.metric_needed(k_max, kind_e)

            def metric_for(l):
                if avg_metric:
                    return mlp.init_scale[l]
                return 1.0 if skip_metric else None
            for l in range(L - 1):
                K = KH.linear_kprop(K, mlp.Ws[l].weight, k_max=k_max, set_metric=metric_for(l), bias=None)
                K = KH.nonlin_kprop(K, nonlin_wick_coef=WICK_COEF_D['relu'], k_max=k_max, kind=kind_e, use_pK=True,
                                    factor=factor)
                if tnet.sigma > 0:
                    K = add_noise_to_K2(K, float(tnet.s[l]) ** 2)
            for p in powers:
                coef = [0.0] * p + [1.0]
                Kt = KH.linear_kprop(K, mlp.Ws[L - 1].weight, k_max=k_max, set_metric=metric_for(L - 1), bias=None)
                Kt = KH.nonlin_kprop(Kt, nonlin_wick_coef=partial(poly_wick_coef, Polynomial(coef)), k_max=k_max,
                                     kind=kind_e, use_pK=True, factor=factor)
                outs.append(Kt[1].to_tensor().clone())
    finally:
        counter.__exit__(None, None, None)
    return torch.stack(outs), counter.total()


@torch.no_grad()
def kprop_all(tnet, thresholds, k_max=3, factor=True, kind='simple', avg_metric=False, dtype=torch.float64):
    """One propagation; then the heaviside readout for each threshold vector (ARC's tail method) and the four
    polynomial readouts E[z^p], p = 1..4 (U2). Returns dict(heavi=[(n,)...], moments=(4, n), flops_shared,
    flops_heavi_each=[...], flops_moments)."""
    from numpy.polynomial import Polynomial
    from functools import partial
    from mlp_kprop.wick import poly_wick_coef
    L, n = tnet.L, tnet.n

    class _N:
        pass
    nn_ = _N(); nn_.n = n; nn_.W = tnet.W64
    mlp = arc_mlp(nn_, L, dtype)
    kind_e = {'simple': KH.SIMPLE, 'augment': KH.AUGMENT}[kind]
    counter = NamedFlopCounter()
    heavi, mom, fh = [], [], []
    counter.__enter__()
    try:
        with flop_name('kprop'):
            K_in = {1: torch.zeros(n, dtype=dtype), 2: torch.eye(n, dtype=dtype)}
            K = KH.coerce_input(K_in, k_max=k_max, kind=kind_e)
            skip_metric = not KH.metric_needed(k_max, kind_e)

            def metric_for(l):
                if avg_metric:
                    return mlp.init_scale[l]
                return 1.0 if skip_metric else None
            for l in range(L - 1):
                K = KH.linear_kprop(K, mlp.Ws[l].weight, k_max=k_max, set_metric=metric_for(l), bias=None)
                K = KH.nonlin_kprop(K, nonlin_wick_coef=WICK_COEF_D['relu'], k_max=k_max, kind=kind_e, use_pK=True,
                                    factor=factor)
                if tnet.sigma > 0:
                    K = add_noise_to_K2(K, float(tnet.s[l]) ** 2)
            shared = counter.total()
            for t in thresholds:
                f0 = counter.total()
                Kt = KH.linear_kprop(K, mlp.Ws[L - 1].weight, k_max=k_max, set_metric=metric_for(L - 1),
                                     bias=-t.to(dtype))
                Kt = KH.nonlin_kprop(Kt, nonlin_wick_coef=WICK_COEF_D['heaviside'], k_max=k_max, kind=kind_e,
                                     use_pK=True, factor=factor)
                heavi.append(Kt[1].to_tensor().clone()); fh.append(counter.total() - f0)
            f0 = counter.total()
            for p in (1, 2, 3, 4):
                Kt = KH.linear_kprop(K, mlp.Ws[L - 1].weight, k_max=k_max, set_metric=metric_for(L - 1), bias=None)
                Kt = KH.nonlin_kprop(Kt, nonlin_wick_coef=partial(poly_wick_coef, Polynomial([0.0] * p + [1.0])),
                                     k_max=k_max, kind=kind_e, use_pK=True, factor=factor)
                mom.append(Kt[1].to_tensor().clone())
            fm = counter.total() - f0
    finally:
        counter.__exit__(None, None, None)
    return dict(heavi=heavi, moments=torch.stack(mom), flops_shared=shared, flops_heavi_each=fh, flops_moments=fm)
