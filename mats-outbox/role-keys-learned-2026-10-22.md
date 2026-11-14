# Role keys: does a learned aggregator reach the joint ceiling? (2026-10-22, opus-mats builder; v2 after gate fixes M1-M4)

**Short answer: yes, nearly.** The setting is fresh agents every episode. A small transformer is trained on Y labels only, and its checkpoint is chosen on validation Y log-loss, so no true roles are used anywhere. It lands within 0.004 nats of the keys-and-content ceiling at every sigma (seed means). It clearly beats the keys-only ceiling at p=4 from sigma 0.25 and at p=2 from sigma 0.5; at p=2, sigma 0.25 the two cannot be told apart. A model trained on persistent agents memorises who reads what and does not generalise.

**Setup.** The environment is exactly role_curves.py: parity of 3 bits plus 1 distractor, N=6, accuracy 0.85, key = W·role + sigma·eps with d_key=8. My ceiling code reproduces role_curves.json exactly. Each token is [key, readings], or [key, FSQ b-bit message]. The model is bidirectional with 2 layers, width 64 and no positions. It gets 2k steps of masked-agent pretraining (predict a held-out agent's readings), then 10k Y steps with fresh minibatches. Dev regret is measured on 4,000 episodes, with exact ceilings on the same episodes; 3 seeds.
Command: `role_learn.py --cells all --procs 6 --threads 1 --pretrain 2000 --steps 10000 --only 'seed==S'`.
Selection (M4): each run keeps two checkpoints. One is chosen on val Y log-loss over 20k label-only episodes (the headline). The other is chosen on val KL to true-roles Bayes over 2k episodes, which uses privileged information and was the v1 headline. Rerunning v1's selection reproduces its numbers to within 0.0005. Compute: 240 runs over v1 and v2, about 5.5 L4-hours, under $5 at list price.

**Primary result (fresh roles and offsets every episode)**, regret in nats; learned = Y-selected, raw readings:

| sigma | p=2 keys-only / joint / learned | p=4 keys-only / joint / learned |
|---|---|---|
| 0 | 0 / 0 / 0.0004 | 0 / 0 / 0.0006 |
| 0.25 | 0.0171 / 0.0147 / 0.0167 | 0.0354 / 0.0244 / 0.0284 |
| 0.5 | 0.0952 / 0.0869 / 0.0880 | 0.1803 / 0.1549 / 0.1589 |
| 1 | 0.1217 / 0.1152 / 0.1158 | 0.2276 / 0.2123 / 0.2139 |
| 2 | 0.1251 / 0.1197 / 0.1197 | 0.2335 / 0.2211 / 0.2216 |
| 5 | 0.1257 / 0.1207 / 0.1206 | 0.2344 / 0.2228 / 0.2230 |

- **Changing the selection rule (M4) changes no headline.** On the fresh-agent dev sets, Y-selected and KL-selected differ by at most 0.0013; on the shortcut tests, by at most 0.008 (persistent, sigma 2, column (ii): 0.576 vs 0.568). The largest difference is at p=4, sigma 5 (0.2230 vs 0.2243), where the KL rule had picked an early, noisy checkpoint; the Y rule is closer to the ceiling.
- **Where it falls short:** where the keys are partly informative, at p=4. At sigma 0.25 it gains 64% of what content adds over keys alone (0.0070 of 0.0110); at sigma 0.5, 84%. At p=2, sigma 0.25 it is at the keys-only level (−0.0004, t=−2 with n=3, not significant). For sigma of 1 or more the gap is at most 0.0016. Most best checkpoints come from the last 10% of training, so part of the gap may be undertraining.
- **Blind reference (no keys):** 0.1212 and 0.2231, against an exact content-only optimum of 0.1209 and 0.2230.
- **FSQ:** b=4 matches raw readings everywhere, and b=2 matches at p=2. At p=4, b=2 costs about 0.007 at sigma 0.25 or below (0.0077 at sigma 0, against 0.0030 for the best fixed 2-bit code, computed by fsq2_floor.py). For sigma of 0.5 or more it is within 0.002 of raw.

**Shortcut test** (p=4, raw readings, 24 persistent agents with fixed role and fixed offset, 3 seeds, Y-selected). The bound for the persistent-trained model on (i) is true-roles Bayes, which is 0. Every other bound is the joint sigma ceiling on that test set. For (i) and (ii) that ceiling assumes iid roles, so it is a reference rather than exact Bayes for 6 distinct agents with balanced roles. The prior is the regret of always answering 0.5 on that test set.

| trained on | sigma | (i) same pairing, prior 0.304 | (ii) roles reassigned, prior 0.255 | (iii) new agents, prior 0.266 |
|---|---|---|---|---|
| persistent | 0.5 | 0.001 (bound 0) | 0.260 (bound 0.177) | 0.308 (bound 0.163) |
| persistent | 2 | 0.001 (bound 0) | **0.576** (bound 0.232) | 0.318 (bound 0.236) |
| fresh (primary) | 0.5 | 0.184 (bound 0.181) | 0.180 (bound 0.177) | 0.166 (bound 0.163) |
| fresh (primary) | 2 | 0.268 (bound 0.266) | 0.232 (bound 0.232) | 0.236 (bound 0.236) |

The persistent-trained model memorises agent → role and becomes nearly exact on its roster. On new agents it is worse than the prior (0/3 seeds below the prior at both sigmas). On reassigned roles it sits near the prior at sigma 0.5 (0.260 vs 0.255; 1 of 3 seeds below). At sigma 2 it is confidently wrong: 0.576, more than twice the prior. The fresh-trained model sits on the ceiling in all three columns. So it learned the key → role structure, not identities.

**Takeaway.** An aggregator trained only on outcomes can learn who knows what from noisy agent descriptions plus message content, and reach the provable optimum for unseen agents. This works only when it is trained on varied agents; a fixed roster yields a lookup table.

Files (work/role-curves/): role_learn.py; role_learn.json (v2, 120 runs, both selections); role_learn_summary.py; role_learn_plot.json; role-learned.html (the page, with "what this shows"); fsq2_floor.py.
