"""E16 per-checkpoint measures (PREREG-B amendment 23) on the checkpointed replays of base_mix4_s5 and s6 (clean nets).
Same measures as E15 (deterministic) plus the normalised encoding-strength surrogate and the deviation pattern:
  gap      test NLL minus exact Bayes NLL, 16,000 test episodes;
  ws0/ws1  effective weights of the three sources in both versions of every E12 pair; mean0/mean1 predictive means;
  gs0/gs1  normalised surrogate g_sph = |d sphere(h1)[QQ_src] / d o*_src| on the pairs (both versions);
  test_w   effective weights on the first 4,000 test episodes; test_m predictive means there (deviation = test_m - Bayes mean);
  test_gs  g_sph of every source on the first 4,000 test episodes (reliability code).
usage: python e16_run.py SEED [PART NPARTS] -> results/e16/s<SEED>_step<STEP>.npz
"""
import glob, json, os, sys, time
import numpy as np
import torch
import pfn as P
import e15_run as R
import e15_gsph as G

torch.set_num_threads(1)
TP = '/home/agent/workspaces/mats/work/tiny-pfn'


def main():
    seed = int(sys.argv[1]); part, nparts = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
    meta = json.load(open(f'runs/base_mix4_s{seed}/meta.json'))
    cks = sorted(glob.glob(f'runs/dyn_base_s{seed}/ckpt/step_*.pt'))[part::nparts]
    os.makedirs('results/e16', exist_ok=True)
    z = np.load('results/e12_pairs.npz'); src = z['src']; tz = np.load(os.path.join(TP, 'runs/testset.npz')); vi = 32
    T = {k: tz[k] for k in ('X', 'valid', 'ystar', 'nll_bayes')}; Xt, vt = tz['X'][:4000], tz['valid'][:4000]
    for ck in cks:
        step = int(os.path.basename(ck)[5:11]); p = f'results/e16/s{seed}_step{step:06d}.npz'
        if os.path.exists(p):
            continue
        t0 = time.time()
        model = P.build(meta); model.load_state_dict(torch.load(ck, map_location='cpu')); model.eval(); model.requires_grad_(False)
        out = dict(step=step, gap=R.gap(model, T))
        for k in (0, 1):
            out[f'ws{k}'], out[f'mean{k}'] = R.eff_w(model, z[f'X{k}'], z[f'valid{k}'], vi)
            out[f'gs{k}'] = G.g_sph(model, z[f'X{k}'], z[f'valid{k}'], src, vi)
        out['test_w'], out['test_m'] = R.eff_w(model, Xt, vt, vi)
        out['test_gs'] = np.stack([G.g_sph(model, Xt, vt, np.full(4000, s), vi) for s in range(3)], 1)
        np.savez(p + '.tmp.npz', **out); os.replace(p + '.tmp.npz', p)
        print(seed, step, 'gap %.4f' % out['gap'], 'done %.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
