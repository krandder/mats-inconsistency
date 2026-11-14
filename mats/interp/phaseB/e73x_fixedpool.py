"""E73 post hoc check (after the review; not registered): is M3's drop at the first checkpoint with two pooling heads a change of the
instrument? M3 recomputed with a FIXED head set (both block-1 heads, each renormalised over the source's own keys, whatever its P_h) at
the checkpoint before and the checkpoint at the pair's first appearance, for E37's four A lineages; as e73_traj.py otherwise
(100,000 simulated episodes, numpy seed 7301).
usage: python e73x_fixedpool.py OUTJSON TAG=RUNDIR:STEP,STEP [...]
"""
import json, os, sys
import numpy as np

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import world as W  # noqa: E402
import e72_run as R  # noqa: E402
import e63_run as E63  # noqa: E402

S, Q, NSIM = 3, 5, 100_000


def main():
    outp = sys.argv[1]; res = json.load(open(outp)) if os.path.exists(outp) else {}
    z = np.load('results/e12_pairs.npz')
    rng = np.random.default_rng(7301); d = W.sample_np(NSIM, rng)
    se = rng.standard_normal((NSIM, S, 16)) / 4; qe = rng.standard_normal((NSIM, Q, 16)) / 4
    for spec in sys.argv[2:]:
        tag, rest = spec.split('=', 1); run, steps = rest.split(':', 1)
        for st in [int(x) for x in steps.split(',')]:
            key = '%s@%d' % (tag, st)
            if key in res:
                continue
            model, arm, meta = E63.load(run, st); model.requires_grad_(False)
            Xs, _ = E63.F.make_X(d['o'], d['y'], se, qe, arm)
            PX = []
            for k in (0, 1):
                ps, pq = E63.F.ids_from_X(z['X%d' % k])
                PX.append(E63.F.make_X(z['d%d_o' % k].astype(np.float64), z['d%d_y' % k].astype(np.float64), ps, pq, arm)[0])
            pool = [0, 1]
            _, m3s = R.feats(model, Xs, pool); _, m30 = R.feats(model, PX[0], pool); _, m31 = R.feats(model, PX[1], pool)
            m3s, m30, m31 = (M.reshape(M.shape[0], S, -1) for M in (m3s, m30, m31))
            res[key] = dict(M3_fixed_both=R.cb_rom_feats(R.fit(m3s.reshape(NSIM * S, -1), d['vidx'].reshape(-1)), m30, m31, z))
            json.dump(res, open(outp, 'w'), indent=1)
            print(key, json.dumps(res[key]), flush=True)


if __name__ == '__main__':
    main()
