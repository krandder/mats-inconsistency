"""E48 POST HOC (descriptive; not registered; as E40's post-review check): on identical episodes and with the same estimator,
each format's KL against the ctrl net's sign-averaged predictor (the family-mean KL minus the KL reduction of geometric
averaging over the base and 8 random sign patterns, an exact identity), per seed.
  world 1: E27's first 6,000 bases, flip_ind family (e30_eval.family_cost, E27's grid). Signed formats (ctrl, sgp, sgnsq): the
           family values already in their evaluation files (results/e37, e39, e48w1: sign_kl, sign_pc; same data and estimator).
           Sign-free formats (mag, sq): the base KL, family_cost on the base only (their family members are identical).
  world 2: E40's 6,000 cost episodes (seed 40,040) with e40_eval's per-episode grid: ctrl and sgnprod from their evaluation files
           (sign_kl, sign_pc); canon and prod: the base KL computed here, as e40_post.py.
usage: python e48_post.py -> results/e48_post.json
"""
import glob, json, os, sys
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import e48_fmt as F  # noqa: E402
import e30_eval as E30  # noqa: E402
import e26_run as RU  # noqa: E402
import e40_eval as EV  # noqa: E402
import e48_eval_t2 as ET  # noqa: E402
import t2_world as T  # noqa: E402
import t2lin as L  # noqa: E402

E30.F = F
torch.set_num_threads(1)


def fam(f):
    z = np.load(f)
    return dict(family_kl=float(z['sign_kl']), sym_kl=float(z['sign_kl'] - z['sign_pc']), gap=float(z['gap']))


def main():
    out = dict(w1={}, w2={})
    z7 = np.load('results/e27_data.npz')
    for a, pat in (('ctrl', 'results/e37/e37_ctrl_s%d.npz'), ('sgp', 'results/e39/e39_sgp_s%d.npz'), ('sgnsq', 'results/e48w1/e48_w1sgnsq_s%d.npz')):
        for s in range(12):
            if os.path.exists(pat % s):
                out['w1']['%s_s%d' % (a, s)] = fam(pat % s)
    for a, pat in (('mag', 'runs/e37/e37_mag_s%d'), ('sq', 'runs/e48/e48_w1sq_s%d')):
        for s in range(12):
            run = pat % s
            if os.path.exists(os.path.join(run, 'model.pt')):
                model, arm, _ = E30.load(run)
                _, kl, mass = E30.family_cost(model, arm, z7, [0], 6000)
                out['w1']['%s_s%d' % (a, s)] = dict(base_kl=kl, mass=mass)
    for D in (1, 2):
        fd = T.sample(EV.NF, np.random.default_rng(40_040), D, 8); bay = T.bayes(fd)
        for a, pat in (('ctrl', 'results/e40/e40_d%dctrl_s%d.npz'), ('sgnprod', 'results/e48t2/e48_d%dsgnprod_s%d.npz')):
            for s in range(12):
                if os.path.exists(pat % (D, s)):
                    out['w2']['d%d%s_s%d' % (D, a, s)] = fam(pat % (D, s))
        for a, pat in (('canon', 'runs/e40/e40_d%dcanon_s%d'), ('prod', 'runs/e48/e48_d%dprod_s%d')):
            for s in range(12):
                run = pat % (D, s)
                if not os.path.exists(os.path.join(run, 'model.pt')):
                    continue
                model, meta = ET.load(run)
                hd = EV.heads(model, *L.tokens(fd, meta['fmt']))
                g, lpB = EV.family_grid(bay, [hd])
                kl, _, mass = RU.set_stats([hd], lpB, g)
                out['w2']['d%d%s_s%d' % (D, a, s)] = dict(base_kl=float(kl.mean()), mass=float(mass.min()))
    json.dump(out, open('results/e48_post.json', 'w'), indent=1, default=float)
    print(len(out['w1']), len(out['w2']))


if __name__ == '__main__':
    main()
