"""E71 plumbing (PREREG-B amendment 79; initial weights only, no trained net of E71): checks of pfn_b1hw's paired initialisation and of
block 1's scales at initialisation.
  1. the 2-head ctrl initialisation built as train_sw builds it equals E37 ctrl's step-0 checkpoint (seeds 0, 5);
  2. for each arm (h2w8, h3w16, h4w16) and seed: shared parameters equal ctrl's; the kept q, k, v rows equal ctrl's; nesting:
     h2w8's q, k, v rows and biases equal E70 b1h4's heads 0 and 2 (step-0 checkpoint of the seed); h3w16's head 2 equals h4w16's
     head 2 in q, k, v; output-map columns are ctrl's times sqrt(32 / (H1 DH1));
  3. scales at initialisation on 512 validation-style episodes (seed 0..3): std of block 1's attention logits of the summary rows
     (over all allowed keys), the mean attention mass of the summary rows on their own labelled tokens, and the ratio of block 1's
     attention-output norm to the residual norm at the summary tokens; for ctrl, b1h4 (= ctrl's values split into 4 heads of 8)
     and the three arms.
usage: python e71_plumb.py
"""
import argparse, math, sys
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import world as W  # noqa: E402
import pfn2 as P2  # noqa: E402
import e30_fmt as F  # noqa: E402
import pfn_b1hw as BH  # noqa: E402

torch.set_num_threads(1)
ARMS = {'h2w8': (2, 8), 'h3w16': (3, 16), 'h4w16': (4, 16)}
SUMM = [4, 9, 14]; LABS = [[s * 5 + q for q in range(4)] for s in range(3)]


def meta_of(seed):
    a = argparse.Namespace(width=32, heads=2, depth=2, attn_only=False, ds=16, mixk=4, phase1_fixed=0, l2_heads=0, l2_dh=0,
                           init_from='', seed=seed)
    return dict(width=32, heads=2, depth=2, attn_only=False, ds=16, seed=seed, cfg=P2.cfg_name(a), mixk=4, phase1_fixed=0,
                l2_heads=0, l2_dh=0, device='cpu')


def ctrl_init(seed):
    torch.manual_seed(seed); P2.fin_of = lambda ds: F.fin('ctrl', ds)
    return P2.build(meta_of(seed))


def arm_init(seed, H1, DH1):
    ref = ctrl_init(seed); meta = meta_of(seed); meta['b1_heads'] = H1; meta['b1_dh'] = DH1
    return BH.paired_init(ref, BH.build(meta), seed), ref


def rows(sd, H, dh, h, part):
    return sd['blocks.0.qkv.weight'][part * H * dh + h * dh: part * H * dh + (h + 1) * dh], \
        sd['blocks.0.qkv.bias'][part * H * dh + h * dh: part * H * dh + (h + 1) * dh]


def scales(model, X, m):
    blk = model.blocks[0]; Xt = torch.from_numpy(X).float(); B, T, _ = Xt.shape
    allow = torch.ones(B, T, T, dtype=torch.bool); allow[:, :, T - 1] = False; allow[:, T - 1, T - 1] = True
    with torch.no_grad():
        x = model.inp(Xt); hh = blk.ln1(x)
        q, k, v = blk.qkv(hh).split(blk.h * blk.dh, -1)
        sp = lambda t: t.view(B, T, blk.h, blk.dh).transpose(1, 2)
        q, k, v = sp(q), sp(k), sp(v)
        s = q @ k.transpose(-1, -2) / math.sqrt(blk.dh)
        sm = s.masked_fill(~allow[:, None], float('-inf')); a = sm.softmax(-1)
        out = blk.o((a @ v).transpose(1, 2).reshape(B, T, blk.h * blk.dh))
    ls = torch.cat([s[:, :, SUMM[i], :T - 1].reshape(-1) for i in range(3)])
    own = np.mean([a[:, :, SUMM[i], LABS[i]].sum(-1).mean().item() for i in range(3)])
    ratio = (out[:, SUMM].norm(dim=-1) / x[:, SUMM].norm(dim=-1)).mean().item()
    return float(ls.std()), float(own), float(ratio)


def main():
    ok = True
    for seed in (0, 5):
        ref = ctrl_init(seed).state_dict()
        ck = torch.load('runs/e37/e37_ctrl_s%d/ckpt/step_000000.pt' % seed, map_location='cpu')
        same = all(torch.equal(ref[k], ck[k]) for k in ck) and set(ref) == set(ck)
        print('1. ctrl init == E37 ctrl s%d step 0:' % seed, same); ok &= same
    for seed in range(12):
        b1 = torch.load('runs/e70/e70_b1h4_s%d/ckpt/step_000000.pt' % seed, map_location='cpu')
        sds = {}
        for arm, (H1, DH1) in ARMS.items():
            mdl, ref = arm_init(seed, H1, DH1); sd = mdl.state_dict(); rs = ref.state_dict(); sds[arm] = sd
            shared = all(torch.equal(sd[k], rs[k]) for k in sd if not k.startswith('blocks.0.qkv') and not k.startswith('blocks.0.o.'))
            kept = all(torch.equal(rows(sd, H1, DH1, h, p)[0], rows(rs, 2, 16, h, p)[0][:DH1]) and
                       torch.equal(rows(sd, H1, DH1, h, p)[1], rows(rs, 2, 16, h, p)[1][:DH1]) for h in range(min(H1, 2)) for p in range(3))
            sc = math.sqrt(32 / (H1 * DH1))
            ocol = all(torch.allclose(sd['blocks.0.o.weight'][:, h * DH1:(h + 1) * DH1], rs['blocks.0.o.weight'][:, h * 16:h * 16 + DH1] * sc)
                       for h in range(min(H1, 2))) and torch.allclose(sd['blocks.0.o.bias'], rs['blocks.0.o.bias'] * sc)
            ok &= shared and kept and ocol
            if seed in (0, 11):
                print('2. s%d %s: shared %s, kept rows %s, output columns %s' % (seed, arm, shared, kept, ocol))
        nest8 = all(torch.equal(rows(sds['h2w8'], 2, 8, h, p)[0], rows(b1, 4, 8, hb, p)[0]) and torch.equal(rows(sds['h2w8'], 2, 8, h, p)[1], rows(b1, 4, 8, hb, p)[1])
                    for h, hb in ((0, 0), (1, 2)) for p in range(3))
        nest16 = all(torch.equal(rows(sds['h3w16'], 3, 16, 2, p)[0], rows(sds['h4w16'], 4, 16, 2, p)[0]) for p in range(3))
        ok &= nest8 and nest16
        if seed in (0, 11):
            print('2. s%d nesting: h2w8 = b1h4 heads 0, 2: %s; h3w16 head 2 = h4w16 head 2: %s' % (seed, nest8, nest16))
    rng = np.random.default_rng(71)
    d = W.sample_np(512, rng)
    se = rng.standard_normal((512, 3, 16)) / 4; qe = rng.standard_normal((512, 5, 16)) / 4
    X, m = F.make_X(d['o'], d['y'], se, qe, 'ctrl', 16)
    print('3. scales at initialisation (logit std over summary rows, own-labelled mass, |attn out| / |x| at summaries):')
    for seed in range(4):
        rowsout = []
        c = ctrl_init(seed); rowsout.append(('ctrl', scales(c, X, m)))
        meta = meta_of(seed); meta.update(heads=4, l2_heads=2, l2_dh=16)
        b = P2.PFN(F.fin('ctrl', 16), 2, 32, 4, False, mixk=4, l2_heads=2, l2_dh=16); b.load_state_dict(c.state_dict())
        rowsout.append(('b1h4', scales(b, X, m)))
        for arm, (H1, DH1) in ARMS.items():
            rowsout.append((arm, scales(arm_init(seed, H1, DH1)[0], X, m)))
        print('   s%d ' % seed + '; '.join('%s %.3f %.3f %.3f' % (n, *v) for n, v in rowsout))
    print('ALL CHECKS PASS' if ok else 'SOME CHECK FAILED')


if __name__ == '__main__':
    main()
