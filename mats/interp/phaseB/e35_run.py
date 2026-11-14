"""E35 (PREREG-B amendment 43): block 2's query attention to the labelled tokens over E16's checkpoints.
For every checkpoint of both replays (base_mix4_s5 = B lineage, base_mix4_s6 = A lineage), on the base episodes of the first 3,000
bases of E28's data: the query token's block-2 attention mass on the twelve labelled tokens and on the three summary tokens (sum over
the set, mean over heads and episodes).
usage: python e35_run.py -> results/e35.json
"""
import glob, json, math, os
import numpy as np
import torch

import e33_run as E33
import world as W
import pfn as P

torch.set_num_threads(1)
NB = 3000
LAB = [5 * s + q for s in range(3) for q in range(4)]
SUMM = [4, 9, 14]


def masses(model, X, valid):
    with torch.no_grad():
        Xt, vt = torch.from_numpy(X), torch.from_numpy(valid)
        allow = E33.allow_mask(vt)
        x1 = E33.block(model.blocks[0], model.inp(Xt), allow)
        blk = model.blocks[1]; B, T, d = x1.shape
        q, k, v = blk.qkv(blk.ln1(x1)).split(d, -1)
        sp = lambda t: t.view(B, T, blk.h, d // blk.h).transpose(1, 2)
        q, k = sp(q), sp(k)
        a = (q[:, :, -1:] @ k.transpose(-1, -2))[:, :, 0] / math.sqrt(d // blk.h)
        a = a.masked_fill(~allow[:, -1][:, None, :], float('-inf')).softmax(-1)
        return float(a[:, :, LAB].sum(-1).mean()), float(a[:, :, SUMM].sum(-1).mean())


def main():
    z = np.load('results/e28_data.npz')
    X, valid = W.assemble(z['semb'][:NB], z['qemb'][:NB], z['o'][:NB, 0], z['y'][:NB])
    out = {}
    for seed in (5, 6):
        meta = json.load(open(f'runs/base_mix4_s{seed}/meta.json'))
        traj = []
        for ck in sorted(glob.glob(f'runs/dyn_base_s{seed}/ckpt/step_*.pt')):
            step = int(os.path.basename(ck)[5:11])
            model = P.build(meta); model.load_state_dict(torch.load(ck, map_location='cpu')); model.eval()
            ml, ms = masses(model, X, valid)
            traj.append(dict(step=step, lab=ml, summ=ms))
        out[str(seed)] = traj
        print(seed, len(traj), flush=True)
    json.dump(out, open('results/e35.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
