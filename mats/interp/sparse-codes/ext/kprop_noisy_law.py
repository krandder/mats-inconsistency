"""MODIFIED COPY (sparse-codes lane): kprop_noisy with an input-law argument (K_in = {1: mean, 2: cov});
otherwise identical to work/arc-noisy/kprop_noisy.py.

ARC's cumulant propagation (mlp_cumulant_propagation, main at 93d091a) on the noisy nets.

The noise h_l = a_l + s_l eps_l is independent Gaussian, so it adds s_l^2 to the diagonal of the second cumulant
and changes no other cumulant. This is applied after every nonlinear step except the last (the estimand is the
post-ReLU mean a_L; the noise after it has mean zero). The loop mirrors kprop_harmonic.mlp_kprop exactly
(linear_kprop with the average metric, then nonlin_kprop), with that one insertion.

FLOPs: ARC's own NamedFlopCounter (torch FlopCounterMode + their symmetric-tensor discounts), i.e. ARC's
published cost model; reported next to every estimate.
"""
import math, os, sys, time
import torch

ARC_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ext', 'mlp_cumulant_propagation', 'src')
if ARC_SRC not in sys.path:
    sys.path.insert(0, ARC_SRC)
import logging
logging.getLogger('mlp_kprop').setLevel(logging.ERROR)
from mlp_kprop.mlp import MLP                                   # noqa: E402
from mlp_kprop import kprop_harmonic as KH                      # noqa: E402
from mlp_kprop.harmonic import HTensor                          # noqa: E402
from mlp_kprop.flop_utils import NamedFlopCounter, flop_name    # noqa: E402
from mlp_kprop.wick import WICK_COEF_D                          # noqa: E402


def arc_mlp(net, L, dtype=torch.float64):
    """ARC MLP object carrying our weights: L hidden ReLU layers (+ an unused output layer)."""
    n = net.n
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=L + 1)
    with torch.no_grad():
        for l in range(L):
            mlp.Ws[l].weight.copy_(net.W[l].to(torch.float32))
    mlp = mlp.to(dtype)
    with torch.no_grad():
        for l in range(L):
            mlp.Ws[l].weight.copy_(net.W[l].to(dtype))
    mlp.init_scale = [2.0] * (L + 1)                             # avg metric E[W W^T] = (2/n) n = 2 for every layer
    return mlp


def add_noise_to_K2(K, v):
    """Add v * I to the second cumulant of an HTower (HTensor r=0: full core; r=1: isotropic scalar core)."""
    K2 = K[2]
    if not isinstance(K2, HTensor):
        raise TypeError(type(K2))
    if K2.r == 0:
        core = K2.core.clone(); core.diagonal().add_(v)
        K[2] = HTensor(core=core, r=0, metric=K2.metric)
    elif K2.r == 1 and K2.core.ndim == 0:
        if not K2.has_identity_metric():
            raise ValueError('expected identity metric after the nonlinear step')
        K[2] = HTensor(core=K2.core + v, r=1, n=K2.n, metric=K2.metric)
    else:
        raise ValueError(f'unexpected K[2] layout {K2}')
    return K


@torch.no_grad()
def kprop_noisy(net, L, k_max, factor=False, kind='simple', dtype=torch.float64, count_flops=True, keep=False,
                avg_metric=True, noise=True, mu0=None, C0=None):
    """Estimate of E[a_l] for l = L (or all l <= L if keep). Returns (means, flops, seconds)."""
    kind_e = {'simple': KH.SIMPLE, 'augment': KH.AUGMENT}[kind]
    mlp = arc_mlp(net, L, dtype)
    n = net.n
    counter = NamedFlopCounter() if count_flops else None
    t0 = time.time()
    means = []
    if counter is not None:
        counter.__enter__()
    try:
        with flop_name('kprop'):
            K_in = {1: torch.zeros(n, dtype=dtype) if mu0 is None else mu0.to(dtype),
                    2: torch.eye(n, dtype=dtype) if C0 is None else C0.to(dtype)}
            K = KH.coerce_input(K_in, k_max=k_max, kind=kind_e)
            skip_metric = not KH.metric_needed(k_max, kind_e)
            for l in range(L):
                W = mlp.Ws[l].weight
                if avg_metric:
                    metric = mlp.init_scale[l]
                elif skip_metric:
                    metric = 1.0
                else:
                    metric = None
                K = KH.linear_kprop(K, W, k_max=k_max, set_metric=metric, bias=None)
                K = KH.nonlin_kprop(K, nonlin_wick_coef=WICK_COEF_D['relu'], k_max=k_max, kind=kind_e,
                                    use_pK=True, factor=factor)
                if keep or l == L - 1:
                    means.append(K[1].to_tensor().clone())
                if l < L - 1 and net.sigma > 0 and noise:
                    K = add_noise_to_K2(K, float(net.s[l]) ** 2)
    finally:
        if counter is not None:
            counter.__exit__(None, None, None)
    flops = counter.total() if counter is not None else None
    return (means if keep else means[-1]), flops, time.time() - t0
