"""E71 training (PREREG-B amendment 79): train_sw.py's unchanged recipe (world 1, ctrl format, amendment 45's fixed 90,000-step
schedule, pfn2's loss, optimiser, clipping and data) for pfn2's baseline net with --b1_heads H1 heads of dimension --b1_dh DH1 in
block 1 (pfn_b1hw.py); block 2 keeps 2 heads of dimension 16. --heads stays 2, so the config name that seeds the episode stream is
ctrl's (w32_h2_L2_mix4) and the episodes are the ctrl net's. The initialisation is pfn_b1hw.paired_init from the 2-head ctrl
initialisation of the same seed (train_sw calls torch.manual_seed(seed), then pfn2's build: the same draws as E37's ctrl arm).
meta.json records b1_heads and b1_dh.
usage: python train_b1hw.py --b1_heads H1 --b1_dh DH1 --f0 ctrl --f1 ctrl --seed S --out DIR [train_sw.py options]
"""
import sys

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import pfn2 as P2  # noqa: E402
import train_sw as TS  # noqa: E402
import pfn_b1hw as BH  # noqa: E402


def _take(flag):
    i = sys.argv.index(flag); v = int(sys.argv[i + 1]); del sys.argv[i:i + 2]
    return v


H1, DH1 = _take('--b1_heads'), _take('--b1_dh')
_build = P2.build


def build(meta):
    ref = _build(meta)                       # the 2-head ctrl initialisation of this seed
    meta['b1_heads'] = H1; meta['b1_dh'] = DH1
    return BH.paired_init(ref, BH.build(meta), meta['seed'])


P2.build = build
TS.P2.build = build

if __name__ == '__main__':
    assert '--pair_init' not in sys.argv and '--heads' not in sys.argv
    TS.main()
