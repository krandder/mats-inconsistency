"""E56 per-net evaluation (PREREG-B amendment 64, planned): e48_eval_w1.py's measures with e56_fmt.py's formats and the pfn_uni
builder (as e53_eval.py), except that rom's effective weights are taken with respect to the feature that carries the summary
readings in the net's format (e56_fmt.READ_COL: 36 for sep, 32 otherwise). For every non-sep net this equals e53_eval.py.
  gap; rom_large, rom_middle (type A if rom_middle < 0); lab_b*, summ_b*, lab_last, summ_last, xq; sign_pc, sign_kl (formats with
  visible signs: ctrl, sgnmag, sgnsq, sep).
usage: python e56_eval.py RUNDIR OUTDIR I0 I1
"""
import glob, os, sys, time
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import pfn2 as P2  # noqa: E402
import pfn_uni as PU  # noqa: E402

P2.build = lambda meta: PU.build_any(meta, P2, P2.fin_of(meta['ds']))
import e56_fmt as F  # noqa: E402
import e30_eval as E30  # noqa: E402
import e37_eval as EV  # noqa: E402
import e27_data as E7  # noqa: E402
import e15_run as R15  # noqa: E402
import pfn as P  # noqa: E402

E30.F = F; EV.F = F
torch.set_num_threads(1)
SIGNED = ('ctrl', 'sgnmag', 'sgnsq', 'sep')


def rom(model, arm, pz):
    src, rg = pz['src'], pz['range']; rows = np.arange(len(src)); Wb = pz['WB0'][rows, src]
    ws = []
    for k in (0, 1):
        ps, pq = F.ids_from_X(pz['X%d' % k])
        Xk, vk = F.make_X(pz['d%d_o' % k].astype(np.float64), pz['d%d_y' % k].astype(np.float64), ps, pq, arm)
        w, _ = R15.eff_w(model, Xk, vk, F.READ_COL[arm]); ws.append(w)
    dw = ws[1][rows, src] - ws[0][rows, src]
    return {'rom_' + rr: float(dw[rg == rr].sum() / Wb[rg == rr].sum()) for rr in ('large', 'middle')}


def main():
    rundir, outdir, i0, i1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    runs = sorted(glob.glob(os.path.join(rundir, '*/meta.json')))[i0:i1]
    os.makedirs(outdir, exist_ok=True)
    tz = np.load(os.path.join(EV.TP, 'runs/testset.npz'))
    ts, tq = F.ids_from_X(tz['X'])
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
                   variant=meta.get('variant', ''), best_step=meta.get('best_step', -1), total_steps=meta.get('total_steps', -1))
        out.update(rom(model, arm, pz))
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
