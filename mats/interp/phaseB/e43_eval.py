"""E43 per-net evaluation on CPU (PREREG-B amendment 51), for the branch nets (runs/e43/*) and their parents (E37's ctrl nets):
  gap        test KL (gap on tiny-pfn's 16,000 test episodes; ctrl format);
  rom_large, rom_middle   E15's contrast on the E12 pairs; type A if rom_middle < 0, B if > 0;
  lab_last, xq            block 2's query-row attention mass on the labelled tokens and block 1's xq (E28's first 3,000 bases);
  pm         the predictive mean on the first 4,000 test episodes (for the functional similarity to the parent).
usage: python e43_eval.py RUNDIR OUTDIR I0 I1  (rows of the sorted list of RUNDIR/*/meta.json) -> OUTDIR/<tag>.npz
"""
import glob, os, sys, time
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import e30_eval as E30  # noqa: E402
import e37_eval as EV  # noqa: E402
import e30_fmt as F  # noqa: E402
import pfn as P  # noqa: E402

torch.set_num_threads(1)
NPM = 4000


def main():
    rundir, outdir, i0, i1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    runs = sorted(glob.glob(os.path.join(rundir, '*/meta.json')))[i0:i1]
    os.makedirs(outdir, exist_ok=True)
    tz = np.load(os.path.join(EV.TP, 'runs/testset.npz'))
    pz = np.load('results/e12_pairs.npz'); z8 = np.load('results/e28_data.npz')
    Xa, va = F.make_X(z8['o'][:3000, 0], z8['y'][:3000], z8['semb'][:3000], z8['qemb'][:3000], 'ctrl')
    for mp in runs:
        run = os.path.dirname(mp); tag = os.path.basename(run); p = os.path.join(outdir, tag + '.npz')
        if os.path.exists(p):
            continue
        t0 = time.time(); model, arm, meta = E30.load(run)
        assert arm == 'ctrl'
        head = P.predict(model, tz['X'], tz['valid'])
        out = dict(gap=float((P.head_nll_np(head, tz['ystar']) - tz['nll_bayes']).mean()), seed=meta['seed'],
                   branch=meta.get('branch', -1), stream2=meta.get('stream2', -1), init_seed=meta.get('init_seed', meta['seed']))
        mu, ls, lg = head
        w = np.exp(lg - lg.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
        out['pm'] = (w * mu).sum(1)[:NPM].astype(np.float64)
        out.update(EV.rom(model, 'ctrl', pz))
        lab, summ, xq = EV.att_masses(model, Xa, va)
        out['lab_last'], out['xq'] = float(lab[-1]), float(xq)
        np.savez(p + '.tmp.npz', **out); os.replace(p + '.tmp.npz', p)
        print(tag, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in out.items() if k != 'pm'}, '%.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
