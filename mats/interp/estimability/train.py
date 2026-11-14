"""Train the small transformer on the shrunken source world with an optional estimability penalty.

loss = task NLL + lam * mean over residual points p (after every attention and MLP sub-block) of D_p,
D_p = e_mean + e_cov [+ e_skew + e_kurt if K=4], each a scale-free squared discrepancy between the propagated
cumulants (deterministic, from the propagator of order K) and the batch-empirical ones:
  e_mean = mean_i (mu_p - mu_e)^2 / v_i
  e_cov  = || D^-1/2 (S_p - S_e) D^-1/2 ||_F^2 / || D^-1/2 S_e D^-1/2 ||_F^2      (D = diag of empirical cov)
  e_skew = mean_i (k3_p - k3_e)^2 / v_i^3,   e_kurt = mean_i (k4_p - k4_e)^2 / v_i^4
Squares are estimated without the sampling-noise floor: (p - e_half1)(p - e_half2) with the batch split in halves;
normalisers use the full batch and are not detached (the ratios are invariant to rescaling the activations).

usage: python train.py --depth 8 --lam 3 --K 4 --seed 0 --steps 6000 --out DIR
"""
import argparse, json, math, os, time
import numpy as np
import torch
import world_small as W
from model import Net, nll
from prop import Propagator

ap = argparse.ArgumentParser()
ap.add_argument('--depth', type=int, required=True)
ap.add_argument('--lam', type=float, default=0.0)
ap.add_argument('--K', type=int, default=4)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--steps', type=int, default=6000)
ap.add_argument('--batch', type=int, default=512)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--warmup', type=int, default=300)
ap.add_argument('--eval_every', type=int, default=250)
ap.add_argument('--out', required=True)
ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
ap.add_argument('--threads', type=int, default=1)
ap.add_argument('--pen_every', type=int, default=1, help='evaluate the penalty every k-th step with weight k*lam')
ap.add_argument('--freeze_body', type=int, default=0, help='random-features baseline: train only the final LN and head')
a = ap.parse_args()
torch.set_num_threads(a.threads)
DEV = a.device
os.makedirs(a.out, exist_ok=True)
VAL_SEED, N_VAL = 10_000, 16384


def penalty_terms(pp, x, K4):
    """pp: propagated point stats (mixture) ; x: [B, N] batch activations. Returns dict of terms."""
    B = x.shape[0]
    h = B // 2
    xs = (x[:h], x[h:2 * h])
    full_m = x.mean(0); xc = x - full_m
    v = (xc ** 2).mean(0).clamp_min(1e-8)
    S = xc.T @ xc / B
    sd = v.sqrt()
    corrF2 = ((S / (sd[:, None] * sd[None])) ** 2).sum()

    def half(z):
        m = z.mean(0); zc = z - m
        cov = zc.T @ zc / z.shape[0]
        vv = (zc ** 2).mean(0)
        return m, cov, (zc ** 3).mean(0), (zc ** 4).mean(0) - 3 * vv ** 2
    h1, h2 = half(xs[0]), half(xs[1])
    pm, pS, p3, p4 = pp['mean'], pp['cov'], pp['k3'], pp['k4']
    t = {}
    t['mean'] = ((pm - h1[0]) * (pm - h2[0]) / v).mean()
    Dn = 1 / (sd[:, None] * sd[None])
    t['cov'] = ((pS - h1[1]) * (pS - h2[1]) * Dn ** 2).sum() / corrF2
    if K4:
        t['skew'] = ((p3 - h1[2]) * (p3 - h2[2]) / v ** 3).mean()
        t['kurt'] = ((p4 - h1[3]) * (p4 - h2[3]) / v ** 4).mean()
    return t


def main():
    torch.manual_seed(a.seed)
    net = Net(a.depth).to(DEV)
    smp = W.Sampler(1000 + a.seed, DEV)                       # data stream depends on the seed only
    if a.freeze_body:
        for n_, p_ in net.named_parameters():
            p_.requires_grad_(n_.startswith('head') or n_.startswith('lnf'))
    opt = torch.optim.AdamW([p_ for p_ in net.parameters() if p_.requires_grad], lr=a.lr, weight_decay=0.01)
    vs = W.Sampler(VAL_SEED, DEV).sample(N_VAL)
    vb_nll, _, _, _ = W.bayes_nll(vs['o'].cpu().numpy(), vs['y'].cpu().numpy())
    vbayes = float(vb_nll.mean())
    prop = Propagator(K=a.K, dtype=torch.float32, device=DEV) if a.lam > 0 else None
    log, t0, nbad = [], time.time(), 0
    for step in range(1, a.steps + 1):
        lr = a.lr * min(1.0, step / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
        for gq in opt.param_groups: gq['lr'] = lr
        d = smp.sample(a.batch)
        mu, ls, res, _ = net(d['X'])
        task = nll(mu, ls, d['ystar']).mean()
        loss = task
        pen_val, terms_mean = float('nan'), {}
        if prop is not None and step % a.pen_every == 0:
            out = prop.run(net)
            terms = [penalty_terms(pp, r.reshape(a.batch, -1), prop.K4) for pp, r in zip(out['points'][1:], res[1:])]
            pen = torch.stack([sum(t.values()) for t in terms]).mean()
            if torch.isfinite(pen):
                loss = loss + a.lam * a.pen_every * pen
                pen_val = pen.item()
                terms_mean = {k: float(np.mean([t[k].item() for t in terms])) for k in terms[0]}
            else:
                nbad += 1
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_([p_ for p_ in net.parameters() if p_.requires_grad], 1.0)
        if not torch.isfinite(gn):
            opt.zero_grad(set_to_none=True); nbad += 1
            continue
        opt.step()
        if step % a.eval_every == 0 or step == a.steps:
            net.eval()
            with torch.no_grad():
                vl = []
                for i in range(0, N_VAL, 4096):
                    m_, l_, _, _ = net(vs['X'][i:i + 4096])
                    vl.append(nll(m_, l_, vs['ystar'][i:i + 4096]))
                v = torch.cat(vl).mean().item()
            net.train()
            rec = dict(step=step, task=task.item(), pen=pen_val, terms=terms_mean, val=v, val_gap=v - vbayes,
                       sec=time.time() - t0, nbad=nbad)
            log.append(rec)
            print(json.dumps(rec), flush=True)
    torch.save(net.state_dict(), os.path.join(a.out, 'model.pt'))
    meta = dict(vars(a), val_bayes=vbayes, log=log, seconds=time.time() - t0, nbad=nbad)
    json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'))


if __name__ == '__main__':
    main()
