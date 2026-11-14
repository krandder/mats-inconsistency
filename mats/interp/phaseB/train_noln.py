"""E50 training (PREREG-B amendment 58): the no-block-1-LayerNorm variant (pfn_noln.py) under train_sw.py's unchanged recipe
(world 1, amendment 45's fixed 90,000-step schedule) with e48_fmt.py's formats (ctrl, mag, sq, ...). The only change is the
model builder (pfn2's build returns the variant; meta['variant'] = 'noln1' is recorded). With the same seed, every remaining
parameter starts as in the baseline net of that seed (E37's ctrl and mag nets, E48's sq nets) and the episode stream is the
baseline's.
usage: python train_noln.py --f0 sq --f1 sq --seed S --out DIR [train_sw.py options]
"""
import sys

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import pfn2 as P2  # noqa: E402
import pfn_noln as PN  # noqa: E402
import e48_fmt  # noqa: E402
import train_sw as TS  # noqa: E402

TS.F = e48_fmt


def build(meta):
    meta['variant'] = 'noln1'
    return PN.build_any(meta, P2, P2.fin_of(meta['ds']))


TS.P2.build = build

if __name__ == '__main__':
    TS.main()
