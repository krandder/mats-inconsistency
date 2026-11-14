"""E50 plumbing (disclosed in amendment 58; no registered quantity).
  (1) the smoke runs' step-0 weights: every parameter equals the baseline's step 0 (E37 ctrl s0 for ctrl and mag; E48 sq s0),
      and exactly blocks.0.ln1.weight and blocks.0.ln1.bias are absent;
  (2) the variant with block 1's LayerNorm restored (weight 1, bias 0) computes exactly the baseline's function at step 0;
  (3) synthetic per-net results under /tmp/e50_synth/results/ with planted values for e50_summ.py (run it there).
usage: python e50_plumb.py
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import pfn2 as P2  # noqa: E402
import pfn_noln as PN  # noqa: E402
import e48_fmt as F  # noqa: E402
import world as W  # noqa: E402

SM = '/tmp/e50_smoke'


def check(ck, base, label):
    a = torch.load(ck, map_location='cpu'); b = torch.load(base, map_location='cpu')
    missing = sorted(set(b) - set(a))
    ok = set(a) <= set(b) and all(torch.equal(a[k], b[k]) for k in a) and missing == ['blocks.0.ln1.bias', 'blocks.0.ln1.weight']
    print(label, 'all remaining params equal the baseline step 0, only ln1 of block 1 absent:', ok, missing)


def same_function():
    meta = dict(depth=2, width=32, heads=2, mixk=4, variant='noln1', ds=16)
    torch.manual_seed(0); m = PN.build_any(meta, P2, 36)
    torch.manual_seed(0); b = PN.build_any(dict(meta, variant=None), P2, 36)
    m.blocks[0].ln1 = nn.LayerNorm(32)
    rng = np.random.default_rng(3); d = W.sample_np(64, rng)
    X, v = F.make_X(d['o'], d['y'], rng.standard_normal((64, 3, 16)) / 4, rng.standard_normal((64, 5, 16)) / 4, 'sq')
    with torch.no_grad():
        o1 = m(torch.from_numpy(X), torch.from_numpy(v)); o2 = b(torch.from_numpy(X), torch.from_numpy(v))
    print('variant with block-1 LayerNorm restored == baseline function:', all(torch.equal(x, y) for x, y in zip(o1, o2)))


def synth():
    R = '/tmp/e50_synth/results'
    rng = np.random.default_rng(2)
    plant = dict(ctrl_noln=(0.010, 0.05, 0.10), mag_noln=(0.006, 0.08, None), sq_noln=(0.004, 0.02, None),
                 ctrl=(0.0127, 0.133, 0.20), mag=(0.0055, 0.073, None), sq=(0.010, 0.130, None))
    where = dict(ctrl_noln=('e50', 'e50_ctrl_s%d'), mag_noln=('e50', 'e50_mag_s%d'), sq_noln=('e50', 'e50_sq_s%d'),
                 ctrl=('e37', 'e37_ctrl_s%d'), mag=('e37', 'e37_mag_s%d'), sq=('e48w1', 'e48_w1sq_s%d'))
    for s in range(12):
        e = 1 + 0.01 * rng.uniform(-1, 1)
        for a, (k, rl, cs) in plant.items():
            sub, nm = where[a]; os.makedirs(os.path.join(R, sub), exist_ok=True)
            kw = dict(gap=(0.14 if (a == 'sq_noln' and s == 4) else k * e), rom_large=rl * e, rom_middle=-0.01, lab_last=0.2, xq=0.01)
            if cs is not None:
                kw.update(sign_pc=cs * 0.01, sign_kl=0.01)
            np.savez(os.path.join(R, sub, nm % s + '.npz'), **kw)
    print('synthetic results written (planted: rom sq_noln/sq 0.02/0.13 -> P201 holds; KL sq_noln/mag_noln 0.67 with seed 4 stalled'
          ' -> P202 holds; rom ctrl_noln/ctrl 0.05/0.133 -> P203 holds)')


if __name__ == '__main__':
    check(f'{SM}/e50_ctrl_s0/ckpt/step_000000.pt', 'runs/e37/e37_ctrl_s0/ckpt/step_000000.pt', 'ctrl')
    check(f'{SM}/e50_mag_s0/ckpt/step_000000.pt', 'runs/e37/e37_mag_s0/ckpt/step_000000.pt', 'mag')
    check(f'{SM}/e50_sq_s0/ckpt/step_000000.pt', 'runs/e48/e48_w1sq_s0/ckpt/step_000000.pt', 'sq')
    same_function()
    synth()
