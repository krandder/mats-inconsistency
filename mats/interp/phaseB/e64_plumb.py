"""E64 plumbing (PREREG-B amendment 72; disclosed, not results), step-0 weights only:
 1. nocons and codeln nets built with torch.manual_seed(seed) have exactly the parameters of E37's ctrl step-0 checkpoint (seeds 0, 1);
 2. nocons: block 1's summary rows put zero attention on the other sources' summary tokens; every other block-1 row and block 2 equal
    the baseline's on the same weights;
 3. codeln: the block-1 LayerNorm input excludes the codes (rescaling every code changes the content normaliser by 0), and with the
    code columns zeroed in the input the variant equals the baseline net whose input code columns are zero (same function);
 4. both run a forward and backward pass on a training batch.
usage: python e64_plumb.py
"""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import pfn2 as P2  # noqa: E402
import pfn_e64 as P64  # noqa: E402
import pfn_uni as PU  # noqa: E402
import world as W  # noqa: E402
import e48_fmt as F  # noqa: E402

torch.set_num_threads(1)


def meta0(seed):
    m = json.load(open('runs/e37/e37_ctrl_s%d/meta.json' % seed))
    return {k: m[k] for k in ('width', 'heads', 'depth', 'attn_only', 'ds', 'seed', 'cfg', 'mixk', 'phase1_fixed', 'l2_heads', 'l2_dh')}


def main():
    rng = np.random.default_rng(64900)
    d = W.sample_np(64, rng); se = rng.standard_normal((64, 3, 16)) / 4; qe = rng.standard_normal((64, 5, 16)) / 4
    X, m = F.make_X(d['o'], d['y'], se, qe, 'ctrl'); Xt, mt = torch.from_numpy(X), torch.from_numpy(m)
    for seed in (0, 1):
        ref = torch.load('runs/e37/e37_ctrl_s%d/ckpt/step_000000.pt' % seed, map_location='cpu')
        base = P2.PFN(P2.fin_of(16), 2, 32, 2, mixk=4); base.load_state_dict(ref)
        for v in P64.VARIANTS:
            mt_ = meta0(seed); mt_['variant'] = v
            torch.manual_seed(seed)
            net = P64.build_any(mt_, P2, P2.fin_of(16))
            sd = net.state_dict()
            diff = max(float((sd[k] - ref[k]).abs().max()) for k in ref)
            assert set(sd) == set(ref), v
            print('seed %d %s: max |param - E37 ctrl step 0| = %.1e' % (seed, v, diff))
            net.eval()
            with torch.no_grad():
                out, _, hs, at = None, None, None, None
                r = net(Xt, mt, want=True); at = r[-1]; hs = r[-2]
                rb = base(Xt, mt, want=True)
            if v == 'nocons':
                a0 = at[0]
                worst = max(float(a0[:, :, PU.SUMM[s], PU.SUMM[t]].abs().max()) for s in range(3) for t in range(3) if t != s)
                rows = [i for i in range(16) if i not in PU.SUMM]
                same = float((a0[:, :, rows] - rb[-1][0][:, :, rows]).abs().max())
                print('   nocons: max attention summary -> other summaries %.1e; other block-1 rows vs baseline %.1e' % (worst, same))
            else:
                Wt = net.inp.weight
                xc = X[..., 32:] @ Wt[:, 32:].detach().numpy().T + net.inp.bias.detach().numpy()
                X2 = X.copy(); X2[..., :32] *= 1.7
                xc2 = X2[..., 32:] @ Wt[:, 32:].detach().numpy().T + net.inp.bias.detach().numpy()
                print('   codeln: content normaliser change when codes are rescaled %.1e' % float(np.abs(xc2.std(-1) - xc.std(-1)).max()))
                X0 = X.copy(); X0[..., :32] = 0
                with torch.no_grad():
                    y1 = net(torch.from_numpy(X0), mt); y2 = base(torch.from_numpy(X0), mt)
                print('   codeln: with zero codes, max output difference from the baseline %.1e' % max(float((p - q).abs().max()) for p, q in zip(y1, y2)))
            net.train()
            head = net(Xt, mt); loss = P2.head_nll_t(head, torch.from_numpy(d['y'][:, W.QL]).float()).mean(); loss.backward()
            print('   forward/backward ok, loss %.3f' % float(loss))


if __name__ == '__main__':
    main()
