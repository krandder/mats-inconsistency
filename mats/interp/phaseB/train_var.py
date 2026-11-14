"""E19 training (PREREG-B amendment 26): the tiny-pfn recipe (pfn.train, width 32, 2 heads, depth 2, mixture head K = 4) run
unchanged on an architecture variant from pfn_var.py, with train_ckpt's checkpoints.
usage: python train_var.py {pre,post} --seed S --out DIR -> DIR/model.pt, meta.json (+ "variant"), ckpt/
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
import pfn as P            # noqa: E402
import pfn_var as V        # noqa: E402
import train_ckpt as C     # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('variant', choices=['pre', 'post']); ap.add_argument('--seed', type=int, required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, 'ckpt'), exist_ok=False)
    C.ST['dir'] = a.out
    torch.optim.AdamW = C.CkptAdamW

    def build(meta):
        m = V.build(dict(meta, variant=a.variant)); C.ST['model'] = m
        return m
    P.build = build
    P.train(argparse.Namespace(width=32, heads=2, depth=2, attn_only=False, ds=16, seed=a.seed, out=a.out, init_from='', mixk=4,
                               phase1_fixed=0))
    mp = os.path.join(a.out, 'meta.json'); meta = json.load(open(mp)); meta['variant'] = a.variant
    json.dump(meta, open(mp, 'w'))
