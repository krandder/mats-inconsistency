"""E76 per-net evaluation (PREREG-B amendment 84, planned): for a reader-retrained net (train_reader.py) and, for comparison, its source:
the test-NLL gap on tiny-pfn's 16,000 test episodes, rom_large and rom_middle on E12's pairs (e37_eval.rom's autograd weights) and the
detached rom (detq: the summary rows' block-1 queries detached; e72_detq). Block 1 and the embedding are frozen in the reader nets, so
E72's benchmark M3 of the source applies unchanged.
usage: python e76_eval.py OUTDIR RUN [RUN ...] -> OUTDIR/<tag>.json   (tag = the run directory's name)
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import e72_detq as DQ  # noqa: E402   (installs the detq-capable block forwards; inactive unless a block is marked)
import e63_run as E63  # noqa: E402
import e37_eval as EV  # noqa: E402
import pfn as P  # noqa: E402

torch.set_num_threads(1)
TP = '/home/agent/workspaces/mats/work/tiny-pfn'


def main():
    outdir = sys.argv[1]; os.makedirs(outdir, exist_ok=True)
    tz = np.load(os.path.join(TP, 'runs/testset.npz')); z = np.load('results/e12_pairs.npz')
    ts, tq = E63.F.ids_from_X(tz['X'])
    for run in sys.argv[2:]:
        tag = os.path.basename(run.rstrip('/')); p = os.path.join(outdir, tag + '.json')
        if os.path.exists(p):
            continue
        t0 = time.time(); model, arm, meta = E63.load(run, None); model.requires_grad_(False)
        X, valid = E63.F.make_X(tz['o'].astype(np.float64), tz['y'].astype(np.float64), ts, tq, arm)
        gap = float((P.head_nll_np(P.predict(model, X, valid), tz['ystar']) - tz['nll_bayes']).mean())
        o = dict(tag=tag, src=meta.get('reader_src', run), mode=meta.get('reader_mode'), gap=gap, **{'rom_' + k: v for k, v in DQ.rom(model, arm, z).items()})
        model.blocks[0]._detq = True
        o.update({'detq_' + k: v for k, v in DQ.rom(model, arm, z).items()}); model.blocks[0]._detq = False
        o['seconds'] = time.time() - t0
        json.dump(o, open(p + '.tmp', 'w'), indent=1); os.replace(p + '.tmp', p)
        print(tag, json.dumps(o), flush=True)


if __name__ == '__main__':
    main()
