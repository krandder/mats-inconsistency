"""E69 training (PREREG-B amendment 77, planned): train_sw.py's unchanged recipe (world 1, ctrl format, amendment 45's fixed 90,000-step
schedule) for pfn2's baseline net with 4 attention heads per block (head dimension 8) instead of 2 (dimension 16). The qkv and output
maps keep their shapes, so with the same seed every parameter starts with exactly the values of the 2-head ctrl net; the config name
that seeds the episode stream is made the 2-head one (w32_h2_L2_mix4), so the episodes are the ctrl net's too. Only the split of the
32 attention dimensions into heads differs. meta.json records heads = 4 (its 'cfg' field then reads w32_h2_L2_mix4, by design).
usage: python train_h4.py --f0 ctrl --f1 ctrl --heads 4 --seed S --out DIR [train_sw.py options]
"""
import sys

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import pfn2 as P2  # noqa: E402
import train_sw as TS  # noqa: E402

_cfg = P2.cfg_name
P2.cfg_name = lambda a: _cfg(a).replace('_h4_', '_h2_')
TS.P2.cfg_name = P2.cfg_name

if __name__ == '__main__':
    TS.main()
