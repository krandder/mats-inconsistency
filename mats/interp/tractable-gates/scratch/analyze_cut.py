"""Cut ranks for run_cut_act.py outputs with the cut-rank lane's estimators (work/cut-rank/lib/spec.py, imported
read-only): in-sample pair-Gram tail r_in, held-out tail r_ho, both raw and scale-normalised (the lane's post-hoc
remedy for heavy-tailed network energy; here registered in PREREG.md).
usage: python3 scratch/analyze_cut.py runs/RUN.npz ...  -> runs/an_RUN.json, one summary line each
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'cut-rank', 'lib'))
import spec

EPS = (0.3, 0.1, 0.03, 0.01)


def ranks(M, V):
    F, S = M.shape[:2]
    Ms = [M[:, s].reshape(F, -1).astype(np.float64) for s in range(S)]
    Vbar = float(V.mean())
    ti, lam = spec.tails_in(Ms)
    tho = spec.tails_heldout(Ms)
    return dict(Vbar=Vbar,
                r_in={str(e): spec.eps_rank(ti, e * e * Vbar) for e in EPS},
                r_ho={str(e): spec.eps_rank(tho, e * e * Vbar) for e in EPS},
                tail_in_rel=[float(ti[s] / Vbar) for s in (0, 1, 2, 4, 8, 16, 32, 64, 128) if s < len(ti)],
                tail1_rel=float(ti[1] / Vbar))


def scale_norm(M, V):
    E2 = V + M.mean(1) ** 2
    s_f = np.sqrt(E2.mean(axis=(1, 2)))
    alive = s_f > 0
    M, V, E2, s_f = M[alive], V[alive], E2[alive], s_f[alive]
    t_b = np.sqrt((E2 / s_f[:, None, None] ** 2).mean(axis=(0, 2)))
    t_b[t_b == 0] = np.inf
    return M / s_f[:, None, None, None] / t_b[None, None, :, None], V / s_f[:, None, None] ** 2 / t_b[None, :, None] ** 2


for path in sys.argv[1:]:
    d = np.load(path)
    M, V = d['M'], d['V']
    raw = ranks(M, V)
    Mn, Vn = scale_norm(M, V)
    sn = ranks(Mn, Vn)
    name = os.path.basename(path)[:-4]
    out = dict(run=name, raw=raw, scale_norm=sn)
    json.dump(out, open(os.path.join(os.path.dirname(path), 'an_' + name + '.json'), 'w'), indent=1)
    print(f"{name}: raw r_in {list(raw['r_in'].values())} r_ho {list(raw['r_ho'].values())} | "
          f"scale-norm r_in {list(sn['r_in'].values())} r_ho {list(sn['r_ho'].values())} | tail1 {raw['tail1_rel']:.3f}",
          flush=True)
