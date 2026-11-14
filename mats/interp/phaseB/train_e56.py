"""E56 training (PREREG-B amendment 64, planned): train_sw.py's unchanged recipe (world 1, amendment 45's fixed 90,000-step
schedule) with e56_fmt.py's formats (adds 'sep': summary readings in their own input column), and optionally a pfn_uni.py
variant (--variant uni | unif | unif_noln; default: the baseline pfn2 PFN). With --pair_init and the sep format, every shared
parameter starts as in the ctrl net of the same seed (and as E53's uni / unif nets for those variants), and the extra column is
drawn U(-1/6, 1/6) from torch.Generator(31337 + seed), as E39's sgnmag nets; the episode stream is the ctrl net's.
usage: python train_e56.py [--variant unif] --f0 sep --f1 sep --seed S --pair_init --out DIR [train_sw.py options]
"""
import sys

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
TS.F = e56_fmt
_base_build = P2.build


def build(meta):
    if VARIANT:
        meta['variant'] = VARIANT
        return PU.build_any(meta, P2, P2.fin_of(meta['ds']))
    return _base_build(meta)


TS.P2.build = build

if __name__ == '__main__':
    TS.main()
