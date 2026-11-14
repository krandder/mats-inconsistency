"""E56 plumbing (disclosed in amendment 64; no registered quantity).
  (1) the smoke runs' step-0 weights: every parameter shared with the ctrl net equals E37 ctrl s0's step 0 (sep) / E53 unif s0's
      step 0 (unif sep); inp.weight's first 36 columns equal ctrl's, its 37th column is new;
  (2) on a sep net, the summary readings enter only through column 36: changing the o column of a summary token changes nothing
      (it is 0 in the format), and the effective weight on column 36 equals the ctrl-format effective weight on column 32 when
      the sep net's column-36 weights are copied from column 32 (a function-identity check at step 0).
usage: python e56_plumb.py
"""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import pfn2 as P2  # noqa: E402
import pfn_uni as PU  # noqa: E402
import e56_fmt as F  # noqa: E402
import world as W  # noqa: E402

SM = '/tmp/e56_smoke'
HERE = '/home/agent/workspaces/mats/work/cm-interp-B'


def init_check():
    for sm, base in (('e56_sep_s0', 'runs/e37/e37_ctrl_s0'), ('e56_unifsep_s0', 'runs/e53/e53_unif_s0')):
        a = torch.load(os.path.join(SM, sm, 'ckpt/step_000000.pt'), map_location='cpu')
        b = torch.load(os.path.join(HERE, base, 'ckpt/step_000000.pt'), map_location='cpu')
        same = all(torch.equal(a[k], b[k]) for k in a if k != 'inp.weight') and set(a) == set(b)
        cols = torch.equal(a['inp.weight'][:, :36], b['inp.weight']) and a['inp.weight'].shape[1] == 37
        print('(1)', sm, 'shared parameters == %s step 0:' % base, same, '| inp.weight first 36 columns equal, 37 columns:', cols)


def routing_check():
    rng = np.random.default_rng(5); d = W.sample_np(64, rng)
    se = rng.standard_normal((64, 3, 16)) / 4; qe = rng.standard_normal((64, 5, 16)) / 4
    Xs, vs = F.make_X(d['o'], d['y'], se, qe, 'sep'); Xc, vc = F.make_X(d['o'], d['y'], se, qe, 'ctrl')
    for variant in (None, 'unif'):
        meta = dict(depth=2, width=32, heads=2, mixk=4, ds=16, variant=variant)
        torch.manual_seed(0); ms = PU.build_any(meta, P2, 37).eval()
        torch.manual_seed(0); mc = PU.build_any(meta, P2, 36).eval()
        sd = ms.state_dict(); sc = mc.state_dict()
        for k in sd:
            if k == 'inp.weight':
                sd[k][:, :36] = sc[k]; sd[k][:, 36] = sc[k][:, 32]
            else:
                sd[k] = sc[k]
        ms.load_state_dict(sd)
        with torch.no_grad():
            os_ = torch.cat(ms(torch.from_numpy(Xs), torch.from_numpy(vs)), 1); oc = torch.cat(mc(torch.from_numpy(Xc), torch.from_numpy(vc)), 1)
        print('(2)', variant or 'learned pool', 'sep net with column 36 = column 32 computes the ctrl net\'s function: max |diff| %.2e' % float((os_ - oc).abs().max()))


if __name__ == '__main__':
    routing_check()
    if os.path.isdir(SM):
        init_check()
