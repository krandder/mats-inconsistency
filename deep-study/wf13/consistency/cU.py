"""cU.py CKPT SIZE K N OUTDIR : M(x_c = 1 | h) for shard K of N of all 2^20 assumption fillings (float32). EVALUATION
SUPPORT for cexact.py only (the exact answer of the score T); no arm reads it."""
import os
import sys

import numpy as np
import torch

PW = os.environ.get('PROBWORLD', '/home/agent/workspaces/mats/work/deep-study/wf7/probworld')
sys.path.insert(0, PW)
sys.dont_write_bytecode = True
import model as MM  # noqa: E402
from model import MASK  # noqa: E402

if __name__ == '__main__':
    ckpt, size, k, n, outdir = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    torch.set_num_threads(int(os.environ.get('NTHREADS', 2)))
    m = MM.build(size, 50)
    m.load_state_dict(torch.load(ckpt, map_location='cpu')['state'])
    m.eval()
    lo, hi = k * 2 ** 20 // n, (k + 1) * 2 ** 20 // n
    out = []
    with torch.no_grad():
        for i in range(lo, hi, 4096):
            z = torch.arange(i, min(i + 4096, hi))
            bits = (z[:, None] >> torch.arange(20)[None, :]) & 1
            tok = torch.cat([bits, torch.full((len(z), 30), MASK, dtype=torch.long)], 1)
            out.append(torch.sigmoid(m(tok).float())[:, 20:].clamp(1e-6, 1 - 1e-6).numpy())
    os.makedirs(outdir, exist_ok=True)
    np.save('%s/U_part_%d.npy' % (outdir, k), np.concatenate(out, 0))
    print('U part', k, 'of', n, 'done', hi - lo, flush=True)
