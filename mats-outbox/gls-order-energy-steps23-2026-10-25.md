# GLS ordering-energy, steps 2-3: learning C, and search as sampled orderings (2026-10-25)

Follows outbox/gls-order-energy-2026-10-25.md (same worlds: same generator and seeds; uniform/oracle/old-estimator
numbers reproduce results.txt exactly). Code: work/gls-energy/step2_learn_c.py, step3_sampled_orders.py;
raw: step2_results.json/.txt, step3_results.json/.txt. CPU, exact n=3, 200 worlds per setting. Not reviewed.

## Step 2: estimating the error structure C

KL(p || mu), mean over 200 worlds (se in step2_results.txt, typically 5-10% of the mean).

| setting | uniform | GLS true-noise C | old contrast est | 2a per-atom, v-dep | 2b class, this world | 2b class, meta (200 other worlds) | 2c data N=100 | 2c data N=1000 |
|---|---|---|---|---|---|---|---|---|
| 0.3/0.3 fwd | 0.0063 | 0.0058 | 0.0094 | 0.0106 | 0.0067 | 0.0059 | 0.0115 | 0.0045 |
| 0.1/0.5 fwd | 0.0065 | 0.0018 | 0.0051 | 0.0060 | 0.0032 | 0.0018 | 0.0083 | 0.0025 |
| 0.05/1.0 fwd | 0.0251 | 0.0006 | 0.0138 | 0.0179 | 0.0068 | 0.0007 | 0.0096 | 0.0013 |
| 0.05/2.0 fwd | 0.1020 | 0.0008 | 0.0609 | 0.0755 | 0.0273 | 0.0007 | 0.0172 | 0.0057 |
| 0.1/0.5 mixed | 0.0090 | 0.0030 | 0.0077 | 0.0085 | 0.0092 | 0.0091 | 0.0092 | 0.0027 |
| 0.05/1.0 mixed | 0.0344 | 0.0074 | 0.0258 | 0.0277 | 0.0369 | 0.0369 | 0.0099 | 0.0040 |
| 0.05/2.0 mixed | 0.1341 | 0.0327 | 0.1021 | 0.1101 | 0.1410 | 0.1384 | 0.0234 | 0.0166 |

Meta-learned class sigmas (fwd/other) recover the true values: 0.306/0.306, 0.065/0.512, 0.001/1.008, 0.001/1.966
(true 0.3/0.3, 0.1/0.5, 0.05/1.0, 0.05/2.0); in mixed worlds the classes carry no stiffness and both come out equal.

## Step 3: search as sampled orderings

A policy samples B orderings (with replacement) and the energy is GLS over the distinct sampled orderings.
Policies (all full-support, 10% uniform mixture): uniform; stiff-seeking (prefers low estimated path variance);
adversarial (prefers high). Cell = mean KL(p || mu_B) / mean TV(mu_B, full-set GLS mu). 20 draws per world.

0.05/1.0 forward-stiff:

| C | policy | B=1 | 2 | 3 | 4 | 6 | 12 | 200 |
|---|---|---|---|---|---|---|---|---|
| true | uniform | .0924/.125 | .0425/.074 | .0253/.052 | .0173/.040 | .0089/.024 | .0023/.006 | .0006/.000 |
| true | stiff | .0599/.087 | .0219/.041 | .0103/.023 | .0047/.013 | .0021/.006 | .0007/.001 | .0006/.000 |
| true | adversarial | .1164/.151 | .0680/.109 | .0471/.087 | .0351/.072 | .0231/.055 | .0115/.031 | .0006/.000 |
| est (2a) | uniform | .0938/.127 | .0481/.084 | .0336/.063 | .0267/.050 | .0221/.034 | .0187/.012 | .0179/.000 |
| est (2a) | stiff | .0596/.096 | .0330/.061 | .0225/.043 | .0206/.035 | .0177/.022 | .0167/.011 | .0179/.000 |
| est (2a) | adversarial | .1418/.163 | .1066/.131 | .0857/.112 | .0729/.099 | .0608/.083 | .0390/.052 | .0179/.001 |

0.05/1.0 mixed:

| C | policy | B=1 | 2 | 3 | 4 | 6 | 12 | 200 |
|---|---|---|---|---|---|---|---|---|
| true | uniform | .1228/.150 | .0625/.095 | .0374/.065 | .0281/.051 | .0182/.033 | .0105/.011 | .0074/.000 |
| true | stiff | .0926/.119 | .0447/.069 | .0270/.046 | .0201/.035 | .0135/.022 | .0090/.008 | .0074/.000 |
| true | adversarial | .1543/.178 | .0938/.129 | .0653/.101 | .0503/.084 | .0337/.061 | .0180/.030 | .0074/.000 |
| est (2a) | uniform | .1226/.153 | .0674/.099 | .0490/.073 | .0400/.059 | .0333/.040 | .0291/.014 | .0277/.000 |
| est (2a) | stiff | .0801/.115 | .0492/.072 | .0387/.052 | .0340/.042 | .0319/.030 | .0295/.016 | .0277/.000 |
| est (2a) | adversarial | .2001/.203 | .1538/.165 | .1276/.144 | .1034/.124 | .0829/.103 | .0546/.067 | .0278/.000 |

TV between the policies' mean mu at B=12: true C 0.028 / 0.023, est C 0.041 / 0.054 (fwd / mixed); at B=200 the gap to
the full-set target is <= 0.001 for every policy. Pseudo-Gibbs, for comparison: TV 0.111 / 0.084 between two scan
orders, which does not shrink with more sweeps.

## Findings
1. Same target, different speed: every full-support policy converges to the same full-set GLS energy, and its gap
   shrinks with B. Pseudo-Gibbs, by contrast, keeps a fixed scan-order bias.
2. Efficiency is measurable and policy-dependent. With true C (forward-stiff), stiff-seeking reaches KL ~0.005 at B=4;
   uniform needs ~B=10, and adversarial is still at 0.0115 at B=12. The ordering is stiff < uniform < adversarial at
   every B in both settings.
3. C from the model alone (2a per-atom, v-dependent) is not better than the old estimator. One world's 8 configurations
   are too few for 12 per-atom variances.
4. Pooling works when stiffness has structure: class sigmas meta-learned on other worlds reproduce the true-noise GLS
   exactly in forward-stiff worlds (0.0007 vs 0.0006 at 0.05/1.0). They do nothing when stiffness is random per atom.
5. Data helps most: 1000 real samples reach or beat the true-noise reference in every mixed setting (0.0040 vs 0.0074).
   That's because data reveals which atoms happened to be wrong in this world, not only their prior variance (caveat 1).

## Caveats
1. The "true-noise C" reference uses the prior noise variance (delta method), not the realised errors, so it is not an
   upper bound; the data arm can and does beat it.
2. At n=3, B=200 always covers all 6 orderings, so convergence at large B is close to trivial. The informative part is
   the finite-B gap column (TV to the full-set target), which is a real bias for a fixed inconsistent model.
3. With estimated C, a subset can beat the full set (est stiff at B=12: 0.0167 vs 0.0179 for the full set): when C is
   misestimated, leaving out noisy orderings helps. So full-set GLS with a poor C is not the best estimator.
4. One noise model (independent logit noise per atom and conditioning value); n=3 only; not reviewed.
