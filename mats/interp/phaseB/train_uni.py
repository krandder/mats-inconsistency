"""E53 training (PREREG-B amendment 61): the uniform-pool variants of pfn_uni.py ('uni', 'unif', 'unif_noln') under train_sw.py's
unchanged recipe (world 1, amendment 45's fixed 90,000-step schedule) with e48_fmt.py's formats. The only change is the model
builder: pfn2's build returns the variant named by --variant, recorded as meta['variant']. No parameter is added, so with the same
seed every parameter starts as in the baseline net of that seed (E37's ctrl and mag nets; for unif_noln, E50's no-LayerNorm nets,
i.e. the baseline without blocks.0.ln1), and the episode stream is the baseline's (the config name does not include the variant).
usage: python train_uni.py --variant uni|unif|unif_noln --f0 ctrl --f1 ctrl --seed S --out DIR [train_sw.py options]
"""
import sys

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import pfn2 as P2  # noqa: E402
import pfn_uni as PU  # noqa: E402
import e48_fmt  # noqa: E402
import train_sw as TS  # noqa: E402

i = sys.argv.index('--variant')
VARIANT = sys.argv[i + 1]
del sys.argv[i:i + 2]
assert VARIANT in PU.VARIANTS, VARIANT
TS.F = e48_fmt


def build(meta):
    meta['variant'] = VARIANT
    return PU.build_any(meta, P2, P2.fin_of(meta['ds']))


TS.P2.build = build

if __name__ == '__main__':
    TS.main()
