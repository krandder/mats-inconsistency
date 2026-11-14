"""cdirect.py DATADIR OUTDIR N [N ...] : the memorising / plain-learning baseline. A one-pass any-order transformer
(wf7's size-S body over the 30 check positions) trained ONLY on the same N real check vectors as the amplified predictor
(same training / validation split), no M, no trajectories. Writes OUTDIR/direct_n{N}_d{D}.npz with its t1 forecasts.
No exact inference."""
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

PW = os.environ.get('PROBWORLD', '/home/agent/workspaces/mats/work/deep-study/wf7/probworld')
sys.path.insert(0, PW)
sys.dont_write_bytecode = True
import model as MM  # noqa: E402
from model import MASK  # noqa: E402

if __name__ == '__main__':
    ddir, odir = sys.argv[1], sys.argv[2]
    ns = [int(v) for v in sys.argv[3:]]
    steps = int(os.environ.get('DIRECT_STEPS', 2000))        # 800 in the reported runs (changed after registration: CPU time)
    vrep = int(os.environ.get('DIRECT_VREP', 8))
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dev.type == 'cpu':
        torch.set_num_threads(int(os.environ.get('NTHREADS', 1)))
    import cmodel as CM
    lab = CM.load_set(ddir + '/lab.pt')
    first = lab['copy'] == 0
    nv = int(lab['vec'].max()) + 1
    xs = torch.zeros(nv, 30, dtype=torch.long)
    xs[lab['vec'][first]] = lab['x'][first].long()                 # one row per real vector, in vector order
    xs = xs.to(dev)
    t1 = torch.load(ddir + '/t1.pt')
    ttok = torch.where(t1['mask'], t1['x'].long(), torch.full_like(t1['x'].long(), MASK)).to(dev)
    for n in ns:
        for d in (0, 1):
            if os.environ.get('DIRECT_ONLY_D') is not None and str(d) != os.environ['DIRECT_ONLY_D']:
                continue
            lo, hi = (d * n, (d + 1) * n) if 2 * n <= nv else (0, nv)
            nval = min(n // 4, 1024)
            tr, va = xs[lo:hi - nval], xs[hi - nval:hi]
            torch.manual_seed(5 + d)
            g = torch.Generator(device=dev)
            g.manual_seed(6 + d)
            vrev = MM.random_context(va.shape[0] * vrep, 30, g, dev)
            va8 = va.repeat(vrep, 1)
            m = MM.build('S', 30).to(dev)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / 200) * (0.5 * (1 + math.cos(math.pi * min(1.0, t / steps))) * 0.98 + 0.02))
            best, bp, bstep = 1e9, None, 0
            for step in range(steps + 1):
                if step % 100 == 0:
                    m.eval()
                    with torch.no_grad():
                        vl = float(MM.ao_loss(m, va8, vrev))
                        if vl < best:
                            best, bstep = vl, step
                            bp = torch.sigmoid(m(ttok).float()).cpu().numpy()
                    m.train()
                    if step == steps:
                        break
                x = tr[torch.randint(0, tr.shape[0], (256,), generator=g, device=dev)]
                loss = MM.ao_loss(m, x, MM.random_context(256, 30, g, dev))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
                sched.step()
            tag = 'direct_n%d_d%d' % (n, d)
            np.savez_compressed('%s/%s.npz' % (odir, tag), t1=bp)
            json.dump(dict(arm='direct', n=n, d=d, best_val=best, best_step=bstep), open('%s/%s.json' % (odir, tag), 'w'))
            print(tag, 'val %.4f step %d' % (best, bstep), flush=True)
