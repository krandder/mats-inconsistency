"""E49 training (PREREG-B amendment 57): the pre-MLP architecture variant (pfn_pre.py) under the unchanged recipes of
  w1: train_sw.py (world 1, amendment 45's fixed 90,000-step schedule, e30_fmt's formats; here ctrl), and
  t2: train_t2.py (world 2, the same schedule; here ctrl).
The only change is the model builder: pfn2's build (w1) or train_t2's build (t2) returns the variant, and meta['variant'] = 'pre'
is recorded. With the same seed, every shared parameter starts as in the baseline net (E37's or E40's ctrl net of that seed) and
the episode stream is the baseline's (the stream does not depend on the model).
usage: python train_pre.py w1 --f0 ctrl --f1 ctrl --seed S --out DIR [train_sw.py options]
       python train_pre.py t2 --f0 ctrl --f1 ctrl --D 2 --seed S --out DIR [train_t2.py options]
"""
import sys

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import pfn2 as P2  # noqa: E402
import pfn_pre as PP  # noqa: E402


def w1():
    import train_sw as TS

    def build(meta):
        meta['variant'] = 'pre'
        return PP.build_any(meta, P2, P2.fin_of(meta['ds']))
    TS.P2.build = build
    TS.main()


def t2():
    import train_t2 as TT

    def build(meta):
        meta['variant'] = 'pre'
        return PP.build_any(meta, P2, meta['D'] + 2)
    TT.build = build
    TT.main()


if __name__ == '__main__':
    world = sys.argv.pop(1)
    {'w1': w1, 't2': t2}[world]()
