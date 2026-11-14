"""E73 exploration (eighth agent; DEV LINEAGES ONLY, E43 newinit s0, s1, s2, s4, s11, disclosed in a later amendment; not a registered
test): how the departure's three parts form over training. At each checkpoint of a lineage: the net's rom on E12's pairs (base), the
rom with the summary rows' block-1 queries detached (detq; e72_detq), the CB readers M1 and M3 of block 1's own-source pool (e72_run),
the pooling heads (P_h >= 0.5; if none, the head with the largest P_h, flagged), and the checkpoint's test-NLL gap.
Decomposition: base = M3 + (detq - M3) + (base - detq).
usage: python e73x_traj.py OUTJSON STEPS TAG=RUNDIR [...]   (STEPS comma-separated, e.g. 2000,3000,5000,10000,20000,40000,90000)
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import world as W  # noqa: E402
import e72_detq as DQ  # noqa: E402   (patches pfn2.Block.forward and pfn_uni.block_fwd; inactive unless a block is marked)
import e72_run as R  # noqa: E402
import e63_run as E63  # noqa: E402
import e68_rpm as RPM  # noqa: E402
import pfn as P  # noqa: E402

S, Q = 3, 5
TP = '/home/agent/workspaces/mats/work/tiny-pfn'
NSIM = int(os.environ.get('E73_NSIM', 100_000))


def gap(model, tz):
    hd = P.predict(model, tz['X'], tz['valid'] if 'valid' in tz.files else np.ones(tz['X'].shape[:2], dtype=bool))
    return float((P.head_nll_np(hd, tz['ystar']) - tz['nll_bayes']).mean())


def m3_m1(model, arm, meta, z):
    hd = R.heads_of(model, arm, meta); pool = hd['pool']; forced = False
    if not pool:
        pool = [int(np.argmax(hd['P_h']))]; forced = True
    c = hd['c']; o = dict(P_h=hd['P_h'], pool=pool, forced=forced, c=c, L=[hd['L'][h] for h in pool], K=[hd['K'][h] for h in pool])
    o['M1'] = RPM.rom_of(RPM.fit_reader(lambda r: RPM.rpm_stat(r, [(hd['L'][h], hd['K'][h]) for h in pool], c, 'ctrl')), z)
    PX = []
    for k in (0, 1):
        ps, pq = E63.F.ids_from_X(z['X%d' % k])
        PX.append(E63.F.make_X(z['d%d_o' % k].astype(np.float64), z['d%d_y' % k].astype(np.float64), ps, pq, arm)[0])
    rng = np.random.default_rng(7301); d = W.sample_np(NSIM, rng)
    se = rng.standard_normal((NSIM, S, 16)) / 4; qe = rng.standard_normal((NSIM, Q, 16)) / 4
    Xs, _ = E63.F.make_X(d['o'], d['y'], se, qe, arm)
    _, m3s = R.feats(model, Xs, pool); _, m30 = R.feats(model, PX[0], pool); _, m31 = R.feats(model, PX[1], pool)
    m3s, m30, m31 = (M.reshape(M.shape[0], S, -1) for M in (m3s, m30, m31))
    o['M3'] = R.cb_rom_feats(R.fit(m3s.reshape(NSIM * S, -1), d['vidx'].reshape(-1)), m30, m31, z)
    return o


def main():
    outp = sys.argv[1]; steps = [int(x) for x in sys.argv[2].split(',')]
    res = json.load(open(outp)) if os.path.exists(outp) else {}
    z = np.load('results/e12_pairs.npz'); tz = np.load(os.path.join(TP, 'runs/testset.npz'))
    for spec in sys.argv[3:]:
        tag, run = spec.split('=', 1)
        for st in steps:
            key = '%s@%d' % (tag, st)
            if key in res:
                continue
            t0 = time.time(); model, arm, meta = E63.load(run, st); model.requires_grad_(False)
            o = dict(step=st, gap=gap(model, tz), base=DQ.rom(model, arm, z))
            model.blocks[0]._detq = True; o['detq'] = DQ.rom(model, arm, z); model.blocks[0]._detq = False
            o.update(m3_m1(model, arm, meta, z)); o['seconds'] = time.time() - t0
            res[key] = o
            json.dump(res, open(outp, 'w'), indent=1)
            print(key, json.dumps({k: v for k, v in o.items() if k not in ('P_h', 'L', 'K')}), flush=True)


if __name__ == '__main__':
    main()
