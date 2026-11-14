"""E40 post-review checks (not registered; asked for by the Codex review, results/review_e40.md):
  (a) each canonical net's KL on the same 6,000 cost episodes with the same integrated estimator (per-episode family grid) as the
      ctrl nets' sign-flip costs; compared, on identical episodes, with the ctrl nets' sign-averaged predictor's KL
      (family-mean KL minus the KL reduction of geometric averaging, an exact identity);
  (b) a joint bootstrap over seeds of rho - 1.5 x c_sign at D = 2 (P175's margin) and at D = 1.
usage: python e40_post.py -> results/e40_post.json
"""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import e26_run as RU  # noqa: E402
import e40_eval as EV  # noqa: E402
import t2_world as T  # noqa: E402

torch.set_num_threads(1)


def main():
    res = json.load(open('results/e40.json')); pn = res['per_net']; out = {}
    for D in (1, 2):
        fd = T.sample(EV.NF, np.random.default_rng(40_040), D, 8); bay = T.bayes(fd)
        kc = {}
        for s in range(12):
            model, meta = EV.load('runs/e40/e40_d%dcanon_s%d' % (D, s))
            hd = EV.heads(model, *T.tokens(fd, 'canon'))
            g, lpB = EV.family_grid(bay, [hd])
            kl, _, mass = RU.set_stats([hd], lpB, g)
            kc[s] = float(kl.mean())
        ctrl_avg = {s: pn['d%dctrl_s%d' % (D, s)]['sign_kl'] - pn['d%dctrl_s%d' % (D, s)]['sign_pc'] for s in range(12)}
        ctrl_fam = {s: pn['d%dctrl_s%d' % (D, s)]['sign_kl'] for s in range(12)}
        rng = np.random.default_rng(4040); ss = list(range(12))
        marg = lambda q: (1 - np.mean([pn['d%dcanon_s%d' % (D, s)]['gap'] for s in q]) / np.mean([pn['d%dctrl_s%d' % (D, s)]['gap'] for s in q])
                          - 1.5 * sum(pn['d%dctrl_s%d' % (D, s)]['sign_pc'] for s in q) / sum(pn['d%dctrl_s%d' % (D, s)]['sign_kl'] for s in q))
        bs = [marg(list(rng.choice(ss, 12))) for _ in range(2000)]
        out[D] = dict(canon_kl_cost_episodes=float(np.mean(list(kc.values()))), ctrl_family_kl=float(np.mean(list(ctrl_fam.values()))),
                      ctrl_sign_averaged_kl=float(np.mean(list(ctrl_avg.values()))),
                      canon_below_ctrl_averaged=int(sum(kc[s] < ctrl_avg[s] for s in ss)),
                      margin_rho_minus_1p5_csign=[float(marg(ss)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
        print(D, out[D], flush=True)
    json.dump(out, open('results/e40_post.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
