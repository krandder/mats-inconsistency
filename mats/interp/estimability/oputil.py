"""Test harness: random inputs with an exactly known representation (Gaussian part + independent non-Gaussian
latents => exact source-form kappa_3/kappa_4), Monte-Carlo comparison helpers."""
import math, torch
import torch.nn.functional as Fn
from prop import Propagator

D64 = torch.float64


def make_input(C, shape, nng, seed, scale=1.0, mean_scale=0.5, K4=True):
    """x = mu + G g + M xi, g ~ N(0,I), xi iid standardized centred Gamma(k=2) (kappa3=sqrt2, kappa4=3).
    Returns representation (m, P, A, c3, c4) and a sampler."""
    g = torch.Generator().manual_seed(seed)
    n = int(torch.tensor(shape).prod())
    mu = mean_scale * torch.randn(C, n, generator=g, dtype=D64)
    G = scale * torch.randn(C, n, n, generator=g, dtype=D64) / math.sqrt(n)
    M = scale * torch.randn(C, n, nng, generator=g, dtype=D64) / math.sqrt(n) if nng else torch.zeros(C, n, 0, dtype=D64)
    kg = 2.0
    c3 = torch.full((C, nng), 2 / math.sqrt(kg), dtype=D64); c4 = torch.full((C, nng), 6 / kg, dtype=D64)
    P = G @ G.transpose(1, 2) + M @ M.transpose(1, 2)

    def sample(B, seed2):
        g2 = torch.Generator().manual_seed(seed2)
        z = torch.randn(C, B, n, generator=g2, dtype=D64)
        x = mu[:, None] + z @ G.transpose(1, 2)
        if nng:
            xi = (torch.distributions.Gamma(kg, 1.0).sample((C, B, nng)).to(D64) - kg) / math.sqrt(kg)
            x = x + xi @ M.transpose(1, 2)
        return x.reshape((C, B) + tuple(shape))
    return dict(m=mu.reshape((C,) + tuple(shape)), P=P.reshape((C,) + tuple(shape) * 2),
                A=M.reshape((C,) + tuple(shape) + (nng,)), c3=c3, c4=c4, sample=sample)


def prop_with(inp, K, n_ref=0):
    pr = Propagator(K=K, dtype=D64, components=[0])
    C = inp['m'].shape[0]
    pr._reset(C)
    if K == 4 and inp['A'].shape[-1]:
        pr.c3 = [inp['c3']]; pr.c4 = [inp['c4']]; pr.J = inp['A'].shape[-1]
        A = inp['A']
    else:
        A = inp['A'][..., :0]
    # reference R = the input itself (flattened) so cross-covariances are tested too
    sh = inp['m'].shape[1:]
    n = int(torch.tensor(sh).prod())
    R = inp['P'].reshape((C,) + tuple(sh) + (n,))
    return pr, (inp['m'], inp['P'], R, A)


def mc_stats(y, x=None):
    """y [C,B,...] -> mean, cov, k3, k4 (flattened), and Cov(y, x) if x given."""
    C, B = y.shape[:2]
    y = y.reshape(C, B, -1)
    mu = y.mean(1); yc = y - mu[:, None]
    cov = yc.transpose(1, 2) @ yc / B
    v = (yc ** 2).mean(1)
    out = dict(mean=mu, cov=cov, k3=(yc ** 3).mean(1), k4=(yc ** 4).mean(1) - 3 * v ** 2)
    if x is not None:
        x = x.reshape(C, B, -1); xc = x - x.mean(1, keepdim=True)
        out['R'] = yc.transpose(1, 2) @ xc / B
    return out


def compare(name, blk, pr, mc, quiet=False):
    m, P, R, A = blk
    C = m.shape[0]
    m = m.reshape(C, -1); n = m.shape[1]
    P = P.reshape(C, n, n); R = R.reshape(C, n, -1)
    sd = mc['cov'].diagonal(0, 1, 2).sqrt()
    k3, k4 = pr.marg(A.reshape(C, n, -1)) if pr.K4 else (torch.zeros_like(m), torch.zeros_like(m))
    off = ~torch.eye(n, dtype=torch.bool)
    r = dict(
        mean=(((m - mc['mean']) / sd) ** 2).mean().sqrt().item(),
        var=((P.diagonal(0, 1, 2) / sd ** 2 - 1) ** 2).mean().sqrt().item(),
        offcov=((P - mc['cov'])[:, off].norm() / mc['cov'][:, off].norm()).item(),
        R=((R - mc['R']).norm() / mc['R'].norm()).item() if 'R' in mc else float('nan'),
        skew=(((k3 - mc['k3']) / sd ** 3) ** 2).mean().sqrt().item(),
        skew_rms=((mc['k3'] / sd ** 3) ** 2).mean().sqrt().item(),
        exk=(((k4 - mc['k4']) / sd ** 4) ** 2).mean().sqrt().item(),
        exk_rms=((mc['k4'] / sd ** 4) ** 2).mean().sqrt().item())
    if not quiet:
        print(f"{name:28s} mean/sd {r['mean']:.4f} var rel {r['var']:.4f} offcov relF {r['offcov']:.4f} "
              f"crosscov relF {r['R']:.4f} | skew err {r['skew']:.4f} (rms {r['skew_rms']:.3f}) "
              f"exkurt err {r['exk']:.4f} (rms {r['exk_rms']:.3f})")
    return r
