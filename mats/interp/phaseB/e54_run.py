"""E54 (PREREG-B amendment 62; CPU, existing nets): does world 2 show the big-miss fingerprint? Every world-2 net (E40's ctrl and
canon nets, E48's prod and sgnprod nets, E49's pre nets; D = 1 and D = 2; 12 seeds each; best-validation model.pt) on E54's
matched pairs (e54_data.py: both versions of a pair have the same X^T X, X^T y and y.y, so exact Bayes is identical).
Per net and pair, for each version (big: most of the residual sum of squares on one example; spread): the predictive mean and
log-sd (moments of the mixture head), the same moments for the calibration versions (the spread version with its residual
scaled by 0.6 and 0.8), and the KL from Bayes (per-episode grid over Bayes's and both versions' densities, as
e40_eval.family_grid; e26_run.set_stats for each density alone).
usage: python e54_run.py PAIRS_D1.npz PAIRS_D2.npz OUTDIR I0 I1   (rows I0..I1-1 of the sorted net list) -> OUTDIR/<tag>.npz
"""
import glob, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import e26_run as RU  # noqa: E402
import e40_eval as EV  # noqa: E402
import pfn2 as P2  # noqa: E402
import pfn_pre as PP  # noqa: E402
import t2lin as L  # noqa: E402

torch.set_num_threads(1)
HERE = '/home/agent/workspaces/mats/work/cm-interp-B'
PATS = ('runs/e40/e40_d*_s*', 'runs/e48/e48_d*prod_s*', 'runs/e49/e49_d*pre_s*')


def nets():
    out = []
    for p in PATS:
        out += [d for d in glob.glob(os.path.join(HERE, p)) if os.path.isdir(d) and os.path.exists(os.path.join(d, 'meta.json'))]
    return sorted(out)


def load(run):
    meta = json.load(open(os.path.join(run, 'meta.json'))); D, fmt = meta['D'], meta['fmt']
    if meta.get('variant') == 'pre':
        m = PP.build_any(meta, P2, D + 2)
    else:
        m = P2.PFN(L.fin(fmt, D), meta['depth'], meta['width'], meta['heads'], mixk=meta['mixk'])
    m.load_state_dict(torch.load(os.path.join(run, 'model.pt'), map_location='cpu')); m.eval(); m.requires_grad_(False)
    return m, meta


def moments(hd):
    mu, ls, lg = hd
    w = np.exp(lg); m = (w * mu).sum(1)
    return m, 0.5 * np.log((w * (np.exp(2 * ls) + mu ** 2)).sum(1) - m ** 2)


def main():
    pd = {1: np.load(sys.argv[1]), 2: np.load(sys.argv[2])}
    outdir, i0, i1 = sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    os.makedirs(outdir, exist_ok=True)
    for run in nets()[i0:i1]:
        tag = os.path.basename(run); p = os.path.join(outdir, tag + '.npz')
        if os.path.exists(p):
            continue
        t0 = time.time(); model, meta = load(run); D, fmt = meta['D'], meta['fmt']; z = pd[D]
        bay = (z['bayes_mu'], z['bayes_va'], z['bayes_wt'])
        hds = []
        for k in ('big', 'spr'):
            X, v = L.tokens(dict(x=z['x'], y=z['y_' + k], xq=z['xq']), fmt)
            hds.append(EV.heads(model, X, v))
        g, lp = EV.family_grid(bay, hds)
        out = dict(fmt=fmt, D=D, seed=meta['seed'], variant=meta.get('variant', ''), run=run)
        for k, hd in zip(('big', 'spr'), hds):
            kl, _, mass = RU.set_stats([hd], lp, g)
            m, ls = moments(hd)
            out['m_' + k], out['ls_' + k], out['kl_' + k], out['mass_' + k] = m, ls, kl, mass
        for f in ('f60', 'f80'):                              # calibration versions: moments only
            X, v = L.tokens(dict(x=z['x'], y=z['y_spr_' + f], xq=z['xq']), fmt)
            out['m_' + f], out['ls_' + f] = moments(EV.heads(model, X, v))
        np.savez(p + '.tmp.npz', **out); os.replace(p + '.tmp.npz', p)
        print(tag, 'done %.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
