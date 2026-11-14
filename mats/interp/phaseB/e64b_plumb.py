"""E64b plumbing (PREREG-B amendment 72, addendum 1; disclosed, not results), step-0 weights only:
 1. split and splitnc nets built with torch.manual_seed(seed) have exactly E37 ctrl's step-0 parameters (seeds 0, 1);
 2. split: rescaling every code leaves block 1's input h unchanged in its content part's normaliser (the content part of h is
    unchanged), and changing the errors leaves the code part of h unchanged;
 3. the code and content parts' norms in h at initialisation, against the baseline's LayerNorm output split the same way (the scale
    check that pfn_e64's codeln failed: its code part is about 7x weaker);
 4. splitnc puts zero block-1 attention from a summary token on the other summaries; forward and backward passes run.
usage: python e64b_plumb.py
"""
import json, math, sys
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import pfn2 as P2  # noqa: E402
import pfn_e64b as PB  # noqa: E402
import pfn_uni as PU  # noqa: E402
import world as W  # noqa: E402
import e48_fmt as F  # noqa: E402

torch.set_num_threads(1)


def main():
    rng = np.random.default_rng(64901)
    d = W.sample_np(256, rng); se = rng.standard_normal((256, 3, 16)) / 4; qe = rng.standard_normal((256, 5, 16)) / 4
    X, m = F.make_X(d['o'], d['y'], se, qe, 'ctrl'); Xt, mt = torch.from_numpy(X), torch.from_numpy(m)
    for seed in (0, 1):
        ref = torch.load('runs/e37/e37_ctrl_s%d/ckpt/step_000000.pt' % seed, map_location='cpu')
        meta = json.load(open('runs/e37/e37_ctrl_s%d/meta.json' % seed))
        base = P2.PFN(P2.fin_of(16), 2, 32, 2, mixk=4); base.load_state_dict(ref)
        for v in PB.VARIANTS:
            mt_ = {k: meta[k] for k in ('width', 'heads', 'depth', 'attn_only', 'ds', 'seed', 'cfg', 'mixk', 'phase1_fixed', 'l2_heads', 'l2_dh')}
            mt_['variant'] = v
            torch.manual_seed(seed); net = PB.build_any(mt_, P2, P2.fin_of(16)); sd = net.state_dict()
            print('seed %d %s: max |param - E37 ctrl step 0| = %.1e' % (seed, v, max(float((sd[k] - ref[k]).abs().max()) for k in ref)))
            with torch.no_grad():
                blk = net.blocks[0]; Wt = net.inp.weight
                h = PB.split_h(net, blk, Xt)
                X2 = X.copy(); X2[..., :32] *= 1.7
                h2 = PB.split_h(net, blk, torch.from_numpy(X2))
                xk = Xt[..., :32] @ Wt[:, :32].T; xk2 = torch.from_numpy(X2)[..., :32] @ Wt[:, :32].T
                ck = blk.ln1.weight * PB.nrm(xk, blk.ln1.eps) / math.sqrt(2.0); ck2 = blk.ln1.weight * PB.nrm(xk2, blk.ln1.eps) / math.sqrt(2.0)
                print('   content part of h unchanged when codes are rescaled: %.1e' % float(((h - ck) - (h2 - ck2)).abs().max()))
                X3 = X.copy(); X3[..., 32] += 2.0                                                 # change every token's o value
                h3 = PB.split_h(net, blk, torch.from_numpy(X3))
                cc3 = blk.ln1.weight * PB.nrm(torch.from_numpy(X3)[..., 32:] @ Wt[:, 32:].T + net.inp.bias, blk.ln1.eps) / math.sqrt(2.0) + blk.ln1.bias
                print('   code part of h (h minus the separately computed content part) unchanged when the values change: %.1e' % float(((h3 - cc3) - ck).abs().max()))
                cc = blk.ln1.weight * PB.nrm(Xt[..., 32:] @ Wt[:, 32:].T + net.inp.bias, blk.ln1.eps) / math.sqrt(2.0)
                hb = base.blocks[0].ln1(base.inp(Xt)); xb = base.inp(Xt); sc = math.sqrt(32) / xb.sub(xb.mean(-1, keepdim=True)).norm(dim=-1, keepdim=True)
                kb = (xk - xk.mean(-1, keepdim=True)) * sc * base.blocks[0].ln1.weight
                cb = (base.inp(Xt) - xk) ; cb = (cb - cb.mean(-1, keepdim=True)) * sc * base.blocks[0].ln1.weight
                print('   mean norms in h: split code %.2f content %.2f | baseline LayerNorm output code %.2f content %.2f' % (
                    float(ck.norm(dim=-1).mean()), float(cc.norm(dim=-1).mean()), float(kb.norm(dim=-1).mean()), float(cb.norm(dim=-1).mean())))
                if v == 'splitnc':
                    a0 = net(Xt, mt, want=True)[-1][0]
                    print('   splitnc: max attention summary -> other summaries %.1e' % max(float(a0[:, :, PU.SUMM[s], PU.SUMM[t]].abs().max()) for s in range(3) for t in range(3) if t != s))
            head = net(Xt, mt); loss = P2.head_nll_t(head, torch.from_numpy(d['y'][:, W.QL]).float()).mean(); loss.backward()
            print('   forward/backward ok, loss %.3f' % float(loss.detach()))


if __name__ == '__main__':
    main()
