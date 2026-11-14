"""E30 per-net evaluation on CPU (PREREG-B amendment 38), after the GPU training.
For each trained net runs/e30/<arm>_s<seed>/ (best-validation checkpoint):
  gap        test NLL minus exact Bayes NLL on tiny-pfn's 16,000 test episodes, in the net's own format (= KL in expectation);
  rom_large, rom_middle   E15's big-miss contrast on the E12 pairs (autograd effective weights; the pairs' inputs in the net's
             format); A type = middle-range rom < 0, B type = > 0, as for the bases;
  sign_pc, sign_kl   on E27's data, flip_ind family (9 samples): uncorrected KL reduction of geometric averaging and mean KL
             (arms ctrl, sgnmag, fix; for mag the inputs are identical across the family, so the reduction is 0);
  code_pc, code_kl   on E29's data, source-ID family (9 samples): the same (arms ctrl, mag, sgnmag; fix has fixed codes).
usage: python e30_eval.py I0 I1 -> results/e30/<arm>_s<seed>.npz   (rows I0..I1-1 of the sorted run list)
"""
import glob, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import e26_run as RU  # noqa: E402
import e27_data as E7  # noqa: E402
import e29_data as E9  # noqa: E402
import e15_run as R15  # noqa: E402
import e30_fmt as F  # noqa: E402
import pfn as P  # noqa: E402
import pfn2 as P2  # noqa: E402

torch.set_num_threads(1)
TP = '/home/agent/workspaces/mats/work/tiny-pfn'


def load(run):
    meta = json.load(open(os.path.join(run, 'meta.json'))); arm = meta['fmt']
    P2.fin_of = lambda ds: F.fin(arm, ds)
    m = P2.build(meta); m.load_state_dict(torch.load(os.path.join(run, 'model.pt'), map_location='cpu')); m.eval(); m.requires_grad_(False)
    return m, arm, meta


def family_cost(model, arm, z, idx, nb):
    g = z['bmean'][:nb, None] + RU.GRID[None]
    lpB = RU.ED.mix_logp_points(z['bmeans'][:nb], z['bvars'], z['bwts'][:nb], g)
    hds = []
    for k in idx:
        semb = z['semb'][:nb, k] if z['semb'].ndim == 4 else z['semb'][:nb]
        qemb = z['qemb'][:nb, k] if z['qemb'].ndim == 4 else z['qemb'][:nb]
        X, valid = F.make_X(z['o'][:nb, k], z['y'][:nb], semb, qemb, arm)
        hds.append(RU.heads(model, 'clean', X, valid))
    kl, pc, mass = RU.set_stats(hds, lpB, g)
    return float(pc.mean()), float(kl.mean()), float(mass.min())


def main():
    RUNS = os.environ.get('E30_RUNS', 'runs/e30'); OUT = os.environ.get('E30_OUT', 'results/e30')   # overrides: checks only
    runs = sorted(glob.glob(os.path.join(RUNS, '*_s*/meta.json')))[int(sys.argv[1]):int(sys.argv[2])]
    os.makedirs(OUT, exist_ok=True)
    tz = np.load(os.path.join(TP, 'runs/testset.npz'))
    ts, tq = F.ids_from_X(tz['X'])
    X0, v0 = F.make_X(tz['o'].astype(np.float64), tz['y'].astype(np.float64), ts, tq, 'ctrl')
    assert np.allclose(X0, tz['X'], atol=1e-6), 'test-set reconstruction'
    pz = np.load('results/e12_pairs.npz'); src, rg = pz['src'], pz['range']; rows = np.arange(len(src))
    Wb = pz['WB0'][rows, src]
    z7 = np.load('results/e27_data.npz'); z9 = np.load('results/e29_data.npz')
    for mp in runs:
        run = os.path.dirname(mp); tag = os.path.basename(run); p = os.path.join(OUT, tag + '.npz')
        if os.path.exists(p):
            continue
        t0 = time.time(); model, arm, meta = load(run)
        X, valid = F.make_X(tz['o'].astype(np.float64), tz['y'].astype(np.float64), ts, tq, arm)
        head = P.predict(model, X, valid)
        out = dict(gap=float((P.head_nll_np(head, tz['ystar']) - tz['nll_bayes']).mean()), arm=arm, seed=meta['seed'],
                   best_step=meta.get('best_step', -1), total_steps=meta.get('total_steps', -1))
        ws = []
        for k in (0, 1):
            ps, pq = F.ids_from_X(pz['X%d' % k])
            Xk, vk = F.make_X(pz['d%d_o' % k].astype(np.float64), pz['d%d_y' % k].astype(np.float64), ps, pq, arm)
            w, _ = R15.eff_w(model, Xk, vk, 32); ws.append(w)
        dw = ws[1][rows, src] - ws[0][rows, src]
        for rr in ('large', 'middle'):
            s = rg == rr; out['rom_' + rr] = float(dw[s].sum() / Wb[s].sum())
        if arm in ('ctrl', 'sgnmag', 'fix'):
            out['sign_pc'], out['sign_kl'], out['sign_mass'] = family_cost(model, arm, z7, E7.fam_idx('flip_ind'), 6000)
        if arm in ('ctrl', 'mag', 'sgnmag'):
            out['code_pc'], out['code_kl'], out['code_mass'] = family_cost(model, arm, z9, E9.SID, 6000)
        np.savez(p + '.tmp.npz', **out); os.replace(p + '.tmp.npz', p)
        print(tag, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in out.items()}, '%.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
