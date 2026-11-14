"""pdata.py -- parity world instances: REAL check vectors from the wf7 generator (the hidden h is discarded here).
Seeds: wf7's test seed for test instances (the same instances as wf7/wf10/wf11, check vectors seen in M's training
removed as there); fresh tags for the loop's training instances. No exact inference."""
import sys

import numpy as np

ROOT = '/home/agent/workspaces/mats/work/deep-study/'
HERE = ROOT + 'wf13/loop/'
sys.path.insert(0, ROOT + 'wf7/probworld')
from world import World, data_seed, pack          # noqa: E402
sys.path.pop(0)


def load_tab(ws):
    d = np.load(HERE + 'out/tab_w%d.npz' % ws)
    return {k: d[k] for k in d.files}


def world(ws, tab):
    return World(ws, n=tab['loga'].shape[0], m=tab['checks'].shape[0])


def test_x(ws, tab, n_keep):
    W = world(ws, tab)
    _, xtr = W.sample(50000, data_seed(ws, 'train'))
    seen = set(pack(xtr).tolist()) if W.m <= 62 else set()
    h, x = W.sample(600, data_seed(ws, 'test'))
    keep = [i for i, k in enumerate(pack(x).tolist() if W.m <= 62 else range(len(x))) if k not in seen][:n_keep]
    return x[keep], h[keep]


def fresh_x(ws, tab, n, tag):
    """Fresh real check vectors for the loop: seed = ws * 1000 + 300 + tag (wf7 used tags 1..11 only)."""
    W = world(ws, tab)
    _, x = W.sample(n, ws * 1000 + 300 + tag)
    return x
