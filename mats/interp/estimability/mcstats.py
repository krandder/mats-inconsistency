"""Large-sample Monte-Carlo statistics of a model over the episode distribution (ground truth for the estimators)."""
import math
import torch
import world_small as W
from model import nll


@torch.no_grad()
def mc_truth(net, n, seed, bs=16384, device='cpu', dtype=torch.float64, points=True, upto=None):
    """Streaming moments: output means/vars of mu, logs, nll; per residual point mean, cov, marginal k3/k4.
    Sufficient statistics accumulated in float64."""
    net = net.to(device)
    smp = W.Sampler(seed, device)
    acc = None
    done = 0
    while done < n:
        b = min(bs, n - done)
        d = smp.sample(b)
        X = d['X'].to(dtype)
        mu, ls, res, _ = net(X, upto=upto)
        L = nll(mu, ls, d['ystar'].to(dtype))
        outs = torch.stack([mu, ls, L], 1).double()
        if acc is None:
            acc = dict(o1=torch.zeros(3, dtype=torch.float64, device=device), o2=torch.zeros(3, dtype=torch.float64, device=device),
                       pts=[])
            if points:
                for r in res:
                    N = r[0].numel()
                    acc['pts'].append(dict(s1=torch.zeros(N, dtype=torch.float64, device=device),
                                           s2=torch.zeros(N, N, dtype=torch.float64, device=device),
                                           s3=torch.zeros(N, dtype=torch.float64, device=device),
                                           s4=torch.zeros(N, dtype=torch.float64, device=device)))
        acc['o1'] += outs.sum(0); acc['o2'] += (outs ** 2).sum(0)
        if points:
            for a, r in zip(acc['pts'], res):
                x = r.reshape(b, -1).double()
                a['s1'] += x.sum(0); a['s2'] += x.T @ x; a['s3'] += (x ** 3).sum(0); a['s4'] += (x ** 4).sum(0)
        done += b
    m = acc['o1'] / n
    v = acc['o2'] / n - m ** 2
    out = dict(E_mu=m[0].item(), E_logs=m[1].item(), E_nll=m[2].item(),
               V_mu=v[0].item(), V_logs=v[1].item(), V_nll=v[2].item(), n=n)
    pts = []
    for a in acc['pts']:
        m1 = a['s1'] / n
        cov = a['s2'] / n - m1[:, None] * m1[None]
        r2 = torch.diagonal(a['s2']) / n; r3 = a['s3'] / n; r4 = a['s4'] / n
        c2 = r2 - m1 ** 2
        c3 = r3 - 3 * m1 * r2 + 2 * m1 ** 3
        c4 = r4 - 4 * m1 * r3 + 6 * m1 ** 2 * r2 - 3 * m1 ** 4 - 3 * c2 ** 2
        pts.append(dict(mean=m1.cpu(), cov=cov.cpu(), k3=c3.cpu(), k4=c4.cpu()))
    return out, pts


def stat_errors(p, e):
    """Normalised discrepancies between propagated (p) and reference (e) statistics of one point."""
    sd = e['cov'].diagonal().clamp_min(1e-12).sqrt()
    Dm = 1 / sd
    corr_e = e['cov'] * Dm[:, None] * Dm[None]
    corr_p = p['cov'].to(e['cov'].dtype) * Dm[:, None] * Dm[None]
    return dict(
        mean=(((p['mean'].to(sd.dtype) - e['mean']) / sd) ** 2).mean().sqrt().item(),
        cov=((corr_p - corr_e).norm() / corr_e.norm()).item(),
        skew=(((p['k3'].to(sd.dtype) - e['k3']) / sd ** 3) ** 2).mean().sqrt().item(),
        skew_rms=((e['k3'] / sd ** 3) ** 2).mean().sqrt().item(),
        exk=(((p['k4'].to(sd.dtype) - e['k4']) / sd ** 4) ** 2).mean().sqrt().item(),
        exk_rms=((e['k4'] / sd ** 4) ** 2).mean().sqrt().item())
