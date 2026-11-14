"""E48 world-1 per-net evaluation on CPU (PREREG-B amendment 56): e37_eval.py's measures, with e48_fmt.py's formats (the
e30_fmt arms are byte-identical through it). For each net DIR (model.pt, the best validation NLL):
  gap        test KL (test NLL minus exact Bayes NLL on tiny-pfn's 16,000 test episodes) in the net's format;
  rom_large, rom_middle   E15's big-miss contrast on the E12 pairs in the net's format; type A if rom_middle < 0, B if > 0;
  lab_b{l}, summ_b{l}, lab_last, summ_last, xq   as e37_eval.py (E28's first 3,000 bases, the net's format);
  sign_pc, sign_kl   E27's flip_ind family (9 samples, 6,000 bases), uncorrected KL reduction of geometric averaging and the
             family's mean KL, for formats in which the signs are visible (ctrl, sgnmag, sgnsq); the r^2 feature is unchanged by
             the flips.
usage: python e48_eval_w1.py RUNDIR OUTDIR I0 I1  -> OUTDIR/<tag>.npz
"""
import glob, os, sys, time
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import e48_fmt as F  # noqa: E402
import e30_eval as E30  # noqa: E402
import e37_eval as EV  # noqa: E402
import e27_data as E7  # noqa: E402
import pfn as P  # noqa: E402

E30.F = F; EV.F = F                      # route format calls (load, family_cost, rom) through e48_fmt
torch.set_num_threads(1)
SIGNED = ('ctrl', 'sgnmag', 'sgnsq')


def main():
    rundir, outdir, i0, i1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    runs = sorted(glob.glob(os.path.join(rundir, '*/meta.json')))[i0:i1]
    os.makedirs(outdir, exist_ok=True)
    tz = np.load(os.path.join(EV.TP, 'runs/testset.npz'))
    ts, tq = F.ids_from_X(tz['X'])
    X0, _ = F.make_X(tz['o'].astype(np.float64), tz['y'].astype(np.float64), ts, tq, 'ctrl')
    assert np.allclose(X0, tz['X'], atol=1e-6), 'test-set reconstruction'
    pz = np.load('results/e12_pairs.npz')
    z7 = np.load('results/e27_data.npz'); z8 = np.load('results/e28_data.npz')
    for mp in runs:
        run = os.path.dirname(mp); tag = os.path.basename(run); p = os.path.join(outdir, tag + '.npz')
        if os.path.exists(p):
            continue
        t0 = time.time(); model, arm, meta = E30.load(run)
        X, valid = F.make_X(tz['o'].astype(np.float64), tz['y'].astype(np.float64), ts, tq, arm)
        head = P.predict(model, X, valid)
        out = dict(gap=float((P.head_nll_np(head, tz['ystar']) - tz['nll_bayes']).mean()), arm=arm, seed=meta['seed'],
                   best_step=meta.get('best_step', -1), total_steps=meta.get('total_steps', -1),
                   pair_init=bool(meta.get('pair_init', False)))
        out.update(EV.rom(model, arm, pz))
        Xa, va = F.make_X(z8['o'][:EV.NB_ATT, 0], z8['y'][:EV.NB_ATT], z8['semb'][:EV.NB_ATT], z8['qemb'][:EV.NB_ATT], arm)
        lab, summ, xq = EV.att_masses(model, Xa, va)
        for l in range(len(lab)):
            out['lab_b%d' % (l + 1)] = float(lab[l]); out['summ_b%d' % (l + 1)] = float(summ[l])
        out['lab_last'], out['summ_last'], out['xq'] = float(lab[-1]), float(summ[-1]), float(xq)
        if arm in SIGNED:
            out['sign_pc'], out['sign_kl'], out['sign_mass'] = E30.family_cost(model, arm, z7, E7.fam_idx('flip_ind'), 6000)
        np.savez(p + '.tmp.npz', **out); os.replace(p + '.tmp.npz', p)
        print(tag, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in out.items()}, '%.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
