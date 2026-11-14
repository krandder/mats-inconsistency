"""E37 checkpoint trajectories on CPU (PREREG-B amendment 45; descriptive). For every E37 net and every saved checkpoint (step 0,
the dense schedule of train_sw.py, and K + 250, 500, 1,000, 2,000, 5,000 after a switch at K):
  kl0, kl1   test KL (gap) on the first 4,000 of tiny-pfn's test episodes in format F0 and in format F1 (equal for a pure arm);
             the format in force at a checkpoint is F0 up to K and F1 after;
  lab, summ, xq   the last block's query-row attention mass on the labelled and on the summary tokens, and block 1's xq (E27), on
             the base episodes of the first 1,000 bases of E28's data, in the format in force;
  rom_large, rom_middle   E15's contrast on the E12 pairs in the format in force, at steps 3,000, 6,000, 10,000, 20,000, 30,000,
             40,000, 60,000 and 90,000 and at K + 5,000 (NaN elsewhere).
usage: python e37_traj.py I0 I1   (rows I0..I1-1 of the sorted list of runs/e37/*/meta.json) -> results/e37_traj/<tag>.npz
"""
import glob, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import e37_eval as EV  # noqa: E402
import e30_fmt as F  # noqa: E402
import pfn as P  # noqa: E402
import pfn2 as P2  # noqa: E402

torch.set_num_threads(1)
ROM_STEPS = {3000, 6000, 10000, 20000, 30000, 40000, 60000, 90000}
NT, NA = 4000, 1000


def main():
    i0, i1 = int(sys.argv[1]), int(sys.argv[2])
    RUNS = os.environ.get('E37_RUNS', 'runs/e37'); OUT = os.environ.get('E37_TRAJ_OUT', 'results/e37_traj')   # overrides: checks only
    runs = sorted(glob.glob(os.path.join(RUNS, '*/meta.json')))[i0:i1]
    os.makedirs(OUT, exist_ok=True)
    tz = np.load(os.path.join(EV.TP, 'runs/testset.npz'))
    ts, tq = F.ids_from_X(tz['X'][:NT])
    pz = np.load('results/e12_pairs.npz'); z8 = np.load('results/e28_data.npz')
    for mp in runs:
        run = os.path.dirname(mp); tag = os.path.basename(run); p = os.path.join(OUT, tag + '.npz')
        if os.path.exists(p):
            continue
        t0 = time.time(); meta = json.load(open(mp)); f0, f1, K = meta['fmt0'], meta['fmt1'], meta['switch']
        P2.fin_of = lambda ds: F.fin(f1, ds)
        model = P2.build(meta); model.eval(); model.requires_grad_(False)
        TX = {f: F.make_X(tz['o'][:NT].astype(np.float64), tz['y'][:NT].astype(np.float64), ts, tq, f) for f in {f0, f1}}
        AX = {f: F.make_X(z8['o'][:NA, 0], z8['y'][:NA], z8['semb'][:NA], z8['qemb'][:NA], f) for f in {f0, f1}}
        cks = sorted(glob.glob(os.path.join(run, 'ckpt/step_*.pt')))
        rows = []
        for ck in cks:
            step = int(os.path.basename(ck)[5:11])
            model.load_state_dict(torch.load(ck, map_location='cpu'))
            fin = f0 if (f0 != f1 and step <= K) else f1
            kl = {f: float((P.head_nll_np(P.predict(model, *TX[f]), tz['ystar'][:NT]) - tz['nll_bayes'][:NT]).mean()) for f in TX}
            lab, summ, xq = EV.att_masses(model, *AX[fin])
            r = dict(step=step, fin=int(fin == f1), kl0=kl[f0], kl1=kl[f1], lab=float(lab[-1]), summ=float(summ[-1]), xq=float(xq),
                     rom_large=np.nan, rom_middle=np.nan)
            if step in ROM_STEPS or (f0 != f1 and step == K + 5000):
                r.update(EV.rom(model, fin, pz))
            rows.append(r)
        out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
        out.update(fmt0=f0, fmt1=f1, switch=K, seed=meta['seed'])
        np.savez(p + '.tmp.npz', **out); os.replace(p + '.tmp.npz', p)
        print(tag, len(rows), 'ckpts %.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
