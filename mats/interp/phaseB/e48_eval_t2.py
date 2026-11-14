"""E48 world-2 per-net evaluation on CPU (PREREG-B amendment 56): e40_eval.py's measures, with t2lin.py's formats. For each net
DIR (model.pt, the best validation NLL):
  gap        test NLL minus exact Bayes NLL on 16,000 test episodes (seed 20,002) in the net's format (= KL in expectation); also
             the level-blind reference's gap and the gap by true noise level;
  sign_pc, sign_kl   on 6,000 fresh episodes (seed 40,040), the base and 8 random per-example flip patterns (seed 40,041), as
             e40_eval.py, for formats in which the signs are visible (ctrl, sgnprod; the products are unchanged by the flips);
  neg_pc, neg_kl     the same for the base and its joint negation.
The per-episode grid and the KL-exact geometric-averaging reduction are e40_eval.py's (family_grid, e26_run.set_stats).
usage: python e48_eval_t2.py RUNDIR OUTDIR I0 I1  -> OUTDIR/<tag>.npz
"""
import glob, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import e26_run as RU  # noqa: E402
import e40_eval as EV  # noqa: E402
import pfn2 as P2  # noqa: E402
import t2_world as T  # noqa: E402
import t2lin as L  # noqa: E402

torch.set_num_threads(1)
NF = EV.NF
SIGNED = ('ctrl', 'sgnprod')


def load(run):
    meta = json.load(open(os.path.join(run, 'meta.json')))
    m = P2.PFN(L.fin(meta['fmt'], meta['D']), meta['depth'], meta['width'], meta['heads'], mixk=meta['mixk'])
    m.load_state_dict(torch.load(os.path.join(run, 'model.pt'), map_location='cpu')); m.eval(); m.requires_grad_(False)
    return m, meta


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
        X, v = L.tokens(td, fmt)
        nll = P2.head_nll_np(P2.predict(model, X, v), td['yq'])
        out = dict(gap=float((nll - tref['bayes']).mean()), blind_gap=float((tref['blind'] - tref['bayes']).mean()), fmt=fmt,
                   seed=meta['seed'], D=D, N=N, width=meta['width'], depth=meta['depth'], best_step=meta['best_step'])
        for j in range(len(T.VLEV)):
            s = td['li'] == j; out['gap_lev%d' % j] = float((nll[s] - tref['bayes'][s]).mean())
        if fmt in SIGNED:
            hds = [EV.heads(model, *L.tokens(EV.flipped(fd, s), fmt)) for s in pats]
            kl, pc, mass = RU.set_stats(hds, *EV.family_grid(bay, hds)[::-1])
            out.update(sign_pc=float(pc.mean()), sign_kl=float(kl.mean()), sign_mass=float(mass.min()))
            hn = [hds[0], EV.heads(model, *L.tokens(EV.flipped(fd, -np.ones((NF, N))), fmt))]
            kl, pc, mass = RU.set_stats(hn, *EV.family_grid(bay, hn)[::-1])
            out.update(neg_pc=float(pc.mean()), neg_kl=float(kl.mean()), neg_mass=float(mass.min()))
        np.savez(p + '.tmp.npz', **out); os.replace(p + '.tmp.npz', p)
        print(tag, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in out.items()}, '%.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
