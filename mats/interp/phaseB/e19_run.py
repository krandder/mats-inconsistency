"""E19 measures (PREREG-B amendment 26): the E15 measures on the pre/post-MLP variant nets (best checkpoint) and, descriptively,
on their checkpoints. usage: python e19_run.py RUN_DIR [ckpt] -> results/e19/<run>__best.npz (or __stepXXXXXX.npz for ckpt)
"""
import glob, json, os, sys, time
import numpy as np
import torch
import pfn as P
import pfn_var as V
import e15_run as R

torch.set_num_threads(1)
TP = '/home/agent/workspaces/mats/work/tiny-pfn'


def measure(model, z, T, Xt, vt):
    out = dict(gap=R.gap(model, T))
    for k in (0, 1):
        out[f'ws{k}'], out[f'mean{k}'] = R.eff_w(model, z[f'X{k}'], z[f'valid{k}'], 32)
    out['test_w'], out['test_m'] = R.eff_w(model, Xt, vt, 32)
    return out


def main():
    run = sys.argv[1].rstrip('/'); nm = os.path.basename(run); os.makedirs('results/e19', exist_ok=True)
    meta = json.load(open(os.path.join(run, 'meta.json')))
    z = np.load('results/e12_pairs.npz'); tz = np.load(os.path.join(TP, 'runs/testset.npz'))
    T = {k: tz[k] for k in ('X', 'valid', 'ystar', 'nll_bayes')}; Xt, vt = tz['X'][:4000], tz['valid'][:4000]
    files = [os.path.join(run, 'model.pt')] if len(sys.argv) < 3 else sorted(glob.glob(os.path.join(run, 'ckpt', 'step_*.pt')))
    for f in files:
        tag = 'best' if f.endswith('model.pt') else os.path.basename(f)[:-3]
        p = os.path.join('results/e19', f'{nm}__{tag}.npz')
        if os.path.exists(p):
            continue
        t0 = time.time(); model = V.build(meta) if 'variant' in meta else P.build(meta)
        model.load_state_dict(torch.load(f, map_location='cpu')); model.eval(); model.requires_grad_(False)
        out = measure(model, z, T, Xt, vt)
        np.savez(p + '.tmp.npz', **out); os.replace(p + '.tmp.npz', p)
        print(nm, tag, 'gap %.4f' % out['gap'], 'done %.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
