"""E70 training (PREREG-B amendment 78, planned): train_sw.py's unchanged recipe for pfn2's baseline net with 4 attention heads (dimension 8)
in block 1 only; block 2 keeps 2 heads of dimension 16 (pfn2's l2_heads / l2_dh). Every qkv and output map keeps its shape, so with the
same seed every parameter starts with exactly the values of the 2-head ctrl net; the config name that seeds the episode stream is made the
2-head one (w32_h2_L2_mix4), so the episodes are the ctrl net's too. meta.json records heads = 4, l2_heads = 2, l2_dh = 16.
usage: python train_b1h4.py --f0 ctrl --f1 ctrl --heads 4 --seed S --out DIR [train_sw.py options]
"""
import sys

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import pfn2 as P2  # noqa: E402
import train_sw as TS  # noqa: E402

_cfg = P2.cfg_name
P2.cfg_name = lambda a: _cfg(a).replace('_h4_', '_h2_')
TS.P2.cfg_name = P2.cfg_name
_build = P2.build


def build(meta):
    meta['l2_heads'] = 2; meta['l2_dh'] = 16
    return _build(meta)


P2.build = build
TS.P2.build = build

if __name__ == '__main__':
    TS.main()
