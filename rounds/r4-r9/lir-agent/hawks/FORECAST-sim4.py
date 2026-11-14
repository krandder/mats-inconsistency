"""Hawk FORECAST sim4: how much counterfactual supervision does ShipHunt (default cfg) leak to the
'no-branching' arm through the committed plan's outcome alone?
For random episodes: committed plan = 3 random distinct cells; candidate = another random 3-cell plan.
Label sources for the candidate's outcome facts (hit_j per cell, success):
  L1 committed hit_facts (all 3 planned cells, fired or not, per harness Outcome.hit_facts)
  L2 truth() occ (revealed hidden state; belief adapter is scored against it)
Report: frac of candidate cell-facts literally shared with the committed plan's facts;
        frac of candidates whose success is determined by placements consistent with L1 (+evidence);
        under L2 it is 1 by construction (noiseless shots, shot_noise=0 default)."""
import sys, json, itertools
sys.path.insert(0, '/home/agent/workspaces/eurisko-study/lir-agent/harness')
import numpy as np
from env.shiphunt import EnvConfig, ShipHunt, placements
cfg = EnvConfig(); P = placements(cfg); n = cfg.n_cells
rng = np.random.default_rng(0)
shared, det_L1, det_fired_only = [], [], []
for ep in range(4000):
    env = ShipHunt(cfg, seed=ep)
    a = rng.choice(n, 3, replace=False); b = rng.choice(n, 3, replace=False)
    out = env.snapshot().execute(list(a))
    shared.append(len(set(a) & set(b)) / 3)
    # placements consistent with committed hit_facts (all planned cells)
    m = np.all(P[:, a] == np.array(out.hit_facts), axis=1)
    succ = P[m][:, b].any(axis=1)
    det_L1.append(bool(succ.all() or (~succ).all()))
    # only cells actually fired (what a realistic env reveals)
    fc = [c for c, _ in out.shots]; fh = [h for _, h in out.shots]
    m2 = np.all(P[:, fc] == np.array(fh), axis=1)
    s2 = P[m2][:, b].any(axis=1)
    det_fired_only.append(bool(s2.all() or (~s2).all()))
res = dict(n_placements=int(len(P)), n_cells=n, episodes=4000,
           mean_frac_candidate_cells_shared_with_committed=float(np.mean(shared)),
           frac_candidate_success_determined_by_committed_hit_facts=float(np.mean(det_L1)),
           frac_candidate_success_determined_by_fired_shots_only=float(np.mean(det_fired_only)),
           frac_determined_by_truth_occ=1.0)
print(json.dumps(res, indent=1))
json.dump(res, open('/home/agent/workspaces/eurisko-study/lir-agent/hawks/FORECAST-sim4.json', 'w'), indent=1)
