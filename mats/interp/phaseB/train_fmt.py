"""E30 (PREREG-B amendment 38): train the tiny-pfn recipe (pfn2.py, unchanged training loop) with an input-format arm from
e30_fmt.py (ctrl, mag, sgnmag, fix, both). The patched token builder replaces world.tokens_np for training batches and for the
validation set, so pfn2.train runs unchanged. The per-episode identity draws are always consumed, so with the same seed every arm
has the same initialisation and the same episodes; ctrl's tokens are byte-identical to an unpatched run.
usage: python train_fmt.py ARM train --width 32 --heads 2 --depth 2 --mixk 4 --seed S --out DIR
"""
import argparse, json, os, sys
import numpy as np

ARM = sys.argv.pop(1)
import world as W  # noqa: E402
import pfn2 as P2  # noqa: E402
import e30_fmt as F  # noqa: E402

assert ARM in F.ARMS, ARM


def tokens_fmt(d, rng, ds=W.DS):
    n = len(d['y'])
    semb = rng.standard_normal((n, W.S, ds)) / 4                     # consume the same draws as world.tokens_np
    qemb = rng.standard_normal((n, W.Q, ds)) / 4
    return F.make_X(d['o'], d['y'], semb, qemb, ARM, ds)


W.tokens_np = tokens_fmt
P2.fin_of = lambda ds: F.fin(ARM, ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd'); ap.add_argument('--width', type=int, default=32); ap.add_argument('--heads', type=int, default=2)
    ap.add_argument('--l2_heads', type=int, default=0); ap.add_argument('--l2_dh', type=int, default=0)
    ap.add_argument('--depth', type=int, default=2); ap.add_argument('--attn_only', action='store_true')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out', default='out')
    ap.add_argument('--init_from', default='')
    ap.add_argument('--mixk', type=int, default=0); ap.add_argument('--phase1_fixed', type=int, default=0)
    ap.add_argument('--ds', type=int, default=W.DS)
    a = ap.parse_args()
    assert a.cmd == 'train'
    P2.train(a)
    p = os.path.join(a.out, 'meta.json'); meta = json.load(open(p)); meta['fmt'] = ARM; json.dump(meta, open(p, 'w'))


if __name__ == '__main__':
    main()
