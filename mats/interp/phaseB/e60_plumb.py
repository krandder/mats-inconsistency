"""E60 plumbing (disclosed in amendment 68; step-0 weights only, no training): (1) own and ownf have exactly the baseline's
parameters at step 0 (seed 0); (2) block 1's summary rows put all weight on their own source's four labelled tokens, with
non-uniform (learned) weights; ownf's query row puts 0 on labelled tokens in both blocks; (3) for ownf, zeroing the labelled
tokens' states after block 1 leaves the output unchanged; for own it does not.
usage: python e60_plumb.py
"""
import sys
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import pfn2 as P2  # noqa: E402
import pfn_own as PO  # noqa: E402
import pfn_uni as PU  # noqa: E402
import e48_fmt as F  # noqa: E402
import world as W  # noqa: E402

META = dict(depth=2, width=32, heads=2, mixk=4, ds=16)
rng = np.random.default_rng(3); d = W.sample_np(64, rng)
X, v = F.make_X(d['o'], d['y'], rng.standard_normal((64, 3, 16)) / 4, rng.standard_normal((64, 5, 16)) / 4, 'ctrl')
X, v = torch.from_numpy(X), torch.from_numpy(v)
torch.manual_seed(0); base = P2.PFN(36, 2, 32, 2, mixk=4)
for var in PO.VARIANTS:
    torch.manual_seed(0); m = PO.build_any(dict(META, variant=var), P2, 36).eval()
    same = all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), base.state_dict().values())) and list(m.state_dict()) == list(base.state_dict())
    with torch.no_grad():
        out = m(X, v, want=True)
    att = out[-1]
    own = all(bool(torch.allclose(att[0][:, :, PU.SUMM[s], s * 5:s * 5 + 4].sum(-1), torch.ones(1))) for s in range(3))
    spread = float(att[0][:, :, PU.SUMM[0], 0:4].std(-1).mean())
    q = (float(att[0][:, :, 15, PU.LAB].abs().max()), float(att[1][:, :, 15, PU.LAB].abs().max()))
    print('(1)', var, 'parameters == baseline step 0:', same, '| (2) summary rows sum to 1 on own labelled tokens:', own,
          '| mean within-source sd of weights %.4f (uniform: 0) | query weight on labelled, block 1 / 2: %.3g / %.3g' % ((spread,) + q))
    with torch.no_grad():
        B, T, _ = X.shape; qi = T - 1
        ctx = v.clone(); ctx[:, qi] = False
        allow = ctx[:, None, :].expand(B, T, T).clone(); allow[:, qi, qi] = True
        if var == 'ownf':
            allow[:, qi, PU.LAB] = False
        allow0 = allow.clone()
        for s in range(3):
            allow0[:, PU.SUMM[s], :] = False; allow0[:, PU.SUMM[s], s * 5:s * 5 + 4] = True
        x = m.inp(X); x, _ = PU.block_fwd(m.blocks[0], x, allow0)
        x = x.clone(); x[:, PU.LAB] = 0; x, _ = PU.block_fwd(m.blocks[1], x, allow)
        y = m.out(m.ln(x[:, qi])); y0 = m(X, v)
        y = torch.cat([y[:, 4:8], y[:, 8:].clamp(-6, 3), y[:, :4]], 1); y0 = torch.cat(y0, 1)
    print('(3)', var, 'max |output change| when labelled states are zeroed after block 1: %.3g' % float((y - y0).abs().max()))
