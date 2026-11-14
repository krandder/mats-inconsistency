"""E40 per-net evaluation on CPU (PREREG-B amendment 48): task 2 (t2_world.py) nets, best-validation checkpoint model.pt.
  gap        test NLL minus exact Bayes NLL on 16,000 test episodes (seed 20,002) in the net's format (= KL in expectation);
             also the level-blind reference's gap on the same episodes, and the gap by true noise level;
  sign_pc, sign_kl   on 6,000 fresh episodes (seed 40,040): the family of the base and 8 random patterns of per-example sign flips
             (x_i, y_i) -> (-x_i, -y_i) (seed 40,041; Bayes is unchanged): KL-exact reduction of the normalised geometric mean of
             the 9 densities (uncorrected) and the family's mean KL, as E27; ctrl nets only (canon tokens are identical across the
             family, so the reduction is 0 by construction);
  neg_pc, neg_kl     the same for the base and its joint negation (every example flipped; 2 densities);
  mass       the smallest grid mass of any density (coverage); the per-episode grid spans +-9 sd of every component
             of Bayes and of every density in the family (2,001 points), as pfn2.kl_grid.
usage: python e40_eval.py RUNDIR OUTDIR I0 I1   (rows of the sorted list of RUNDIR/*/meta.json) -> OUTDIR/<tag>.npz
"""
import glob, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import e26_run as RU  # noqa: E402
import pfn2 as P2  # noqa: E402
import t2_world as T  # noqa: E402
import train_t2 as TT  # noqa: E402

torch.set_num_threads(1)
NF = 6000


def load(run):
    meta = json.load(open(os.path.join(run, 'meta.json')))
    m = TT.build(meta); m.load_state_dict(torch.load(os.path.join(run, 'model.pt'), map_location='cpu')); m.eval()
    m.requires_grad_(False)
    return m, meta


def heads(model, X, valid):
    mu, ls, lg = [t.astype(np.float64) for t in P2.predict(model, X, valid)]
    lg = lg - lg.max(1, keepdims=True); lg = lg - np.log(np.exp(lg).sum(1, keepdims=True))
    return mu, ls, lg


def family_grid(bay, hds, ng=2001):
    """Per-episode grid spanning +-9 sd of every Bayes component and of every component of every density in the family (as
    pfn2.kl_grid), with the Bayes log-density on it. bay = (mu, va, wt) from t2_world.bayes; hds = list of net heads."""
    mu, va, wt = bay; sd = np.sqrt(va)
    lo = (mu - 9 * sd).min(1); hi = (mu + 9 * sd).max(1)
    for m, ls, _ in hds:
        lo = np.minimum(lo, (m - 9 * np.exp(ls)).min(1)); hi = np.maximum(hi, (m + 9 * np.exp(ls)).max(1))
    g = lo[:, None] + (hi - lo)[:, None] * np.linspace(0.0, 1.0, ng)[None]
    lp = np.log(wt + 1e-300)[:, None, :] - 0.5 * np.log(2 * math.pi * va)[:, None, :] - 0.5 * (g[:, :, None] - mu[:, None, :]) ** 2 / va[:, None, :]
    mx = lp.max(2, keepdims=True)
    return g, mx[:, :, 0] + np.log(np.exp(lp - mx).sum(2))


def flipped(d, s):
    e = dict(d); e['x'] = d['x'] * s[:, :, None]; e['y'] = d['y'] * s
    return e


def main():
    rundir, outdir, i0, i1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    runs = sorted(glob.glob(os.path.join(rundir, '*/meta.json')))[i0:i1]
    os.makedirs(outdir, exist_ok=True)
    cache = {}
    for mp in runs:
        run = os.path.dirname(mp); tag = os.path.basename(run); p = os.path.join(outdir, tag + '.npz')
        if os.path.exists(p):
            continue
        t0 = time.time(); model, meta = load(run); D, N, fmt = meta['D'], meta['N'], meta['fmt']
        if (D, N) not in cache:
            td = T.sample(16000, np.random.default_rng(20_002), D, N); tref, _ = T.references(td)
            fd = T.sample(NF, np.random.default_rng(40_040), D, N)
            r = np.random.default_rng(40_041)
            pats = [np.ones((NF, N))] + [np.where(r.random((NF, N)) < 0.5, -1.0, 1.0) for _ in range(8)]
            cache[(D, N)] = (td, tref, fd, pats, T.bayes(fd))
        td, tref, fd, pats, bay = cache[(D, N)]
        X, v = T.tokens(td, fmt)
        nll = P2.head_nll_np(P2.predict(model, X, v), td['yq'])
        out = dict(gap=float((nll - tref['bayes']).mean()), blind_gap=float((tref['blind'] - tref['bayes']).mean()), fmt=fmt,
                   fmt0=meta['fmt0'], fmt1=meta['fmt1'], switch=meta['switch'], seed=meta['seed'], D=D, N=N,
                   width=meta['width'], depth=meta['depth'], best_step=meta['best_step'])
        for j in range(len(T.VLEV)):
            s = td['li'] == j; out['gap_lev%d' % j] = float((nll[s] - tref['bayes'][s]).mean())
        if fmt == 'ctrl':
            hds = [heads(model, *T.tokens(flipped(fd, s), fmt)) for s in pats]
            kl, pc, mass = RU.set_stats(hds, *family_grid(bay, hds)[::-1])
            out.update(sign_pc=float(pc.mean()), sign_kl=float(kl.mean()), sign_mass=float(mass.min()))
            hn = [hds[0], heads(model, *T.tokens(flipped(fd, -np.ones((NF, N))), fmt))]
            kl, pc, mass = RU.set_stats(hn, *family_grid(bay, hn)[::-1])
            out.update(neg_pc=float(pc.mean()), neg_kl=float(kl.mean()), neg_mass=float(mass.min()))
        np.savez(p + '.tmp.npz', **out); os.replace(p + '.tmp.npz', p)
        print(tag, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in out.items()}, '%.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
