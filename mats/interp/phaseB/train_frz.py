"""E58 training (PREREG-B amendment 66): train_sw.py's unchanged recipe (world 1, amendment 45's fixed 90,000-step schedule) in
e56_fmt.py's sep format (summary readings in their own input column), optionally a pfn_uni.py variant (--variant unif; default
the baseline learned pool), with the input embedding (inp.weight, inp.bias) replaced by a designed one (--emb FILE from
e58_emb.py) and FROZEN (requires_grad False; AdamW skips parameters without gradients, so no update and no weight decay).
With --pair_init every other parameter starts as in the ctrl / E53 unif net of the same seed; the episode stream is theirs.
usage: python train_frz.py [--variant unif] --emb emb_c2.npz --f0 sep --f1 sep --seed S --pair_init --out DIR [train_sw.py options]
"""
import sys
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import pfn2 as P2  # noqa: E402
import pfn_uni as PU  # noqa: E402
import e56_fmt  # noqa: E402
import train_sw as TS  # noqa: E402

VARIANT = None
if '--variant' in sys.argv:
    i = sys.argv.index('--variant'); VARIANT = sys.argv[i + 1]; del sys.argv[i:i + 2]
    assert VARIANT in PU.VARIANTS, VARIANT
i = sys.argv.index('--emb'); EMB = sys.argv[i + 1]; del sys.argv[i:i + 2]
TS.F = e56_fmt
_base_build = P2.build


def build(meta):
    if VARIANT:
        meta['variant'] = VARIANT
        return PU.build_any(meta, P2, P2.fin_of(meta['ds']))
    return _base_build(meta)


TS.P2.build = build
_build_model = TS.build_model


def build_model(a, fmt):
    model, meta = _build_model(a, fmt)
    z = np.load(EMB)
    with torch.no_grad():
        model.inp.weight.copy_(torch.from_numpy(z['W'])); model.inp.bias.copy_(torch.from_numpy(z['b']))
    model.inp.weight.requires_grad_(False); model.inp.bias.requires_grad_(False)
    meta['frozen_emb'] = EMB; meta['emb_c'] = float(z['c']); meta['emb_kappa'] = float(z['kappa'])
    return model, meta


TS.build_model = build_model

if __name__ == '__main__':
    TS.main()
