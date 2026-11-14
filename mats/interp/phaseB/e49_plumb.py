"""E49 plumbing (disclosed in amendment 57; no registered quantity).
  (1) step-0 weights of the smoke runs: every shared parameter equals the baseline's step 0 (E37 ctrl s0; E40 d2ctrl s0), and
      the variant has exactly the extra block's parameters in addition;
  (2) the variant with its extra block's last layer zeroed computes exactly the baseline's function on a few episodes;
  (3) synthetic per-net results under /tmp/e49_synth/results/ with planted ratios for e49_summ.py (run it there).
usage: python e49_plumb.py
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import pfn2 as P2  # noqa: E402
import pfn_pre as PP  # noqa: E402
import t2_world as T  # noqa: E402
import e30_fmt as F  # noqa: E402
import world as W  # noqa: E402

SM = '/tmp/e49_smoke'


def check(pre_ck, base_ck, label):
    a = torch.load(pre_ck, map_location='cpu'); b = torch.load(base_ck, map_location='cpu')
    extra = sorted(set(a) - set(b))
    ok = set(b) <= set(a) and all(torch.equal(a[k], b[k]) for k in b) and all(k.startswith('extra.') for k in extra)
    print(label, 'shared params equal the baseline step 0:', ok, '| extra tensors:', extra)


def same_function():
    meta = dict(depth=2, width=32, heads=2, mixk=4, variant='pre', ds=16)
    torch.manual_seed(0); m = PP.build_any(meta, P2, 36)
    torch.manual_seed(0); b = PP.build_any(dict(meta, variant=None), P2, 36)
    with torch.no_grad():
        m.extra[3].weight.zero_(); m.extra[3].bias.zero_()
    rng = np.random.default_rng(3); d = W.sample_np(64, rng)
    X, v = F.make_X(d['o'], d['y'], rng.standard_normal((64, 3, 16)) / 4, rng.standard_normal((64, 5, 16)) / 4, 'ctrl')
    with torch.no_grad():
        o1 = m(torch.from_numpy(X), torch.from_numpy(v)); o2 = b(torch.from_numpy(X), torch.from_numpy(v))
    print('variant with a zeroed extra output layer == baseline function:', all(torch.equal(x, y) for x, y in zip(o1, o2)))


def synth():
    R = '/tmp/e49_synth/results'
    rng = np.random.default_rng(1)
    for s in range(12):
        e = 1 + 0.01 * rng.uniform(-1, 1)
        for sub, nm, k, cs in (('e49w1', 'e49_w1pre_s%d', 0.005, 0.05), ('e37', 'e37_ctrl_s%d', 0.0125, 0.20), ('e37', 'e37_mag_s%d', 0.0055, None)):
            os.makedirs(os.path.join(R, sub), exist_ok=True)
            kw = dict(gap=(0.14 if (nm.startswith('e49') and s == 3) else k * e), rom_large=0.05, rom_middle=-0.01, lab_last=0.2, xq=0.01)
            if cs is not None:
                kw.update(sign_pc=cs * 0.01, sign_kl=0.01)
            np.savez(os.path.join(R, sub, nm % s + '.npz'), **kw)
        for D, (kc, kp) in ((1, (0.003, 0.0027)), (2, (0.018, 0.0099))):
            for sub, nm, k, cs in (('e49t2', 'e49_d%dpre_s%d', kp, 0.30), ('e40', 'e40_d%dctrl_s%d', kc, 0.34)):
                os.makedirs(os.path.join(R, sub), exist_ok=True)
                np.savez(os.path.join(R, sub, nm % (D, s) + '.npz'), gap=k * e, blind_gap=0.17, sign_pc=cs * 0.01, sign_kl=0.01, neg_pc=0.001)
    print('synthetic results written (planted: w1 pre/ctrl 0.40 with seed 3 stalled; D2 pre/ctrl 0.55, D1 0.90; c_sign w1 0.05 vs 0.20,'
          ' D2 0.30 vs 0.34 -> P198 holds, P199 holds, P200 fails)')


if __name__ == '__main__':
    check(f'{SM}/e49_w1pre_s0/ckpt/step_000000.pt', 'runs/e37/e37_ctrl_s0/ckpt/step_000000.pt', 'w1')
    check(f'{SM}/e49_d2pre_s0/ckpt/step_000000.pt', 'runs/e40/e40_d2ctrl_s0/ckpt/step_000000.pt', 't2 D=2')
    same_function()
    synth()
