"""E37-E39 per-net evaluation on CPU (PREREG-B amendments 45-47), after the GPU training. For each net DIR (model.pt = the best
validation NLL in the final format after the switch; final format meta['fmt']):
  gap        test NLL minus exact Bayes NLL on tiny-pfn's 16,000 test episodes, in the net's final format (= KL in expectation);
  rom_large, rom_middle   E15's big-miss contrast on the E12 pairs (as e30_eval); type A if rom_middle < 0, B if > 0;
  lab_b{l}, summ_b{l}     block l's query-row attention mass on the 12 labelled tokens and on the 3 summary tokens (sum over the
             set, mean over heads and episodes), on the base episodes of the first 3,000 bases of E28's data in the net's format;
             lab_last, summ_last = the last block's (E34/E35's marker for 2-layer nets);
  xq         block 1's attention from each labelled token to the other two sources' tokens on the same question (E27's marker);
  sign_pc, sign_kl   on E27's data, flip_ind family (9 samples, 6,000 bases): uncorrected KL reduction of geometric averaging
             and the family's mean KL, for nets whose final format shows the signs (ctrl, sgnmag); as e30_eval.
usage: python e37_eval.py RUNDIR OUTDIR I0 I1   (rows I0..I1-1 of the sorted list of RUNDIR/*/meta.json) -> OUTDIR/<tag>.npz
"""
import glob, os, sys, time
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import e30_eval as E30  # noqa: E402
import e27_data as E7  # noqa: E402
import e15_run as R15  # noqa: E402
import e30_fmt as F  # noqa: E402
import pfn as P  # noqa: E402

torch.set_num_threads(1)
TP = '/home/agent/workspaces/mats/work/tiny-pfn'
LAB = [5 * s + q for s in range(3) for q in range(4)]
SUMM = [4, 9, 14]
NB_ATT = 3000


def att_masses(model, X, valid, bs=1000):
    """Per block l: query-row attention mass on LAB and SUMM; block 1's xq (E27). Means over heads and episodes."""
    nb = len(model.blocks); lab = np.zeros(nb); summ = np.zeros(nb); xq, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out = model(torch.from_numpy(X[i:i + bs]), torch.from_numpy(valid[i:i + bs]), want=True)
            att = out[-1]; b = att[0].shape[0]
            for l in range(nb):
                a = att[l][:, :, -1]                                               # (B, h, T): the query row
                lab[l] += float(a[:, :, LAB].sum(-1).mean()) * b; summ[l] += float(a[:, :, SUMM].sum(-1).mean()) * b
            a1 = att[0].numpy(); t = 0.0
            for s in range(3):
                for q in range(4):
                    t += a1[:, :, 5 * s + q, [5 * u + q for u in range(3) if u != s]].sum(-1).mean()
            xq += t / 12 * b; n += b
    return lab / n, summ / n, xq / n


def rom(model, arm, pz, rows=None):
    src, rg = pz['src'], pz['range']; rows = np.arange(len(src)) if rows is None else rows
    Wb = pz['WB0'][rows, src[rows]]
    ws = []
    for k in (0, 1):
        ps, pq = F.ids_from_X(pz['X%d' % k][rows])
        Xk, vk = F.make_X(pz['d%d_o' % k][rows].astype(np.float64), pz['d%d_y' % k][rows].astype(np.float64), ps, pq, arm)
        w, _ = R15.eff_w(model, Xk, vk, 32); ws.append(w)
    dw = ws[1][np.arange(len(rows)), src[rows]] - ws[0][np.arange(len(rows)), src[rows]]
    out = {}
    for rr in ('large', 'middle'):
        s = rg[rows] == rr
        if s.any():
            out['rom_' + rr] = float(dw[s].sum() / Wb[s].sum())
    return out


def main():
    rundir, outdir, i0, i1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    runs = sorted(glob.glob(os.path.join(rundir, '*/meta.json')))[i0:i1]
    os.makedirs(outdir, exist_ok=True)
    tz = np.load(os.path.join(TP, 'runs/testset.npz'))
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
                   fmt0=meta['fmt0'], fmt1=meta['fmt1'], switch=meta['switch'], width=meta['width'], depth=meta['depth'],
                   best_step=meta.get('best_step', -1), total_steps=meta.get('total_steps', -1))
        out.update(rom(model, arm, pz))
        Xa, va = F.make_X(z8['o'][:NB_ATT, 0], z8['y'][:NB_ATT], z8['semb'][:NB_ATT], z8['qemb'][:NB_ATT], arm)
        lab, summ, xq = att_masses(model, Xa, va)
        for l in range(len(lab)):
            out['lab_b%d' % (l + 1)] = float(lab[l]); out['summ_b%d' % (l + 1)] = float(summ[l])
        out['lab_last'], out['summ_last'], out['xq'] = float(lab[-1]), float(summ[-1]), float(xq)
        if arm in ('ctrl', 'sgnmag'):
            out['sign_pc'], out['sign_kl'], out['sign_mass'] = E30.family_cost(model, arm, z7, E7.fam_idx('flip_ind'), 6000)
        np.savez(p + '.tmp.npz', **out); os.replace(p + '.tmp.npz', p)
        print(tag, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in out.items()}, '%.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
