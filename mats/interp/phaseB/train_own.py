"""E60 training (PREREG-B amendment 68): pfn_own.py's variants ('own', 'ownf'; or any pfn_uni variant, or none for the baseline)
under train_sw.py's unchanged recipe (world 1, amendment 45's fixed 90,000-step schedule) with e48_fmt.py's formats. No parameter
is added, so with the same seed every parameter starts as in the baseline net of that seed and the episodes are the baseline's.
usage: python train_own.py [--variant own|ownf|uni|unif] --f0 ctrl --f1 ctrl --seed S --out DIR [train_sw.py options]
"""
import sys

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import pfn2 as P2  # noqa: E402
import pfn_own as PO  # noqa: E402
import e48_fmt  # noqa: E402
import train_sw as TS  # noqa: E402

VARIANT = None
if '--variant' in sys.argv:
    i = sys.argv.index('--variant'); VARIANT = sys.argv[i + 1]; del sys.argv[i:i + 2]
TS.F = e48_fmt
_base_build = P2.build


def build(meta):
    if VARIANT:
        meta['variant'] = VARIANT
        return PO.build_any(meta, P2, P2.fin_of(meta['ds']))
    return _base_build(meta)


TS.P2.build = build

if __name__ == '__main__':
    TS.main()
