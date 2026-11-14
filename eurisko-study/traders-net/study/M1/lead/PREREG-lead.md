# M1 lead pre-registration: retarget items (a)-(c) (written 2026-09-28 01:15Z, before any run)

The 00:25Z RETARGET in STUDY-PLAN.md makes M1 the negative-result workstream. Workers 1-4 finished
the original items; the lead does the three retarget items and rewrites CURRENCY.lock.

## (a) Toy where the sign relation is provable (`lead/toy_sign.py`, CPU, no training data)
Model: x in {-1,+1} equiprobable, P(y=1|x) = sigmoid(beta x), beta = 1. A unit contributes w*x to the
logit (price starts at the prior l0 = 0 = the base rate). Quantities in closed form (exact sums over
x,y): gradient credit C(w) = w * E[x (y - sigmoid(w x))]; ablation value V(w) = E[log p_w(y)] - log 1/2
(= exact LMSR settlement of a trader who committed w x against the prior price).
Claim to check: C(beta) = 0, C(w) < 0 < V(w) for beta < w < w0 (w0 = root of V), and for two redundant
units (both carry x, w1 + w2 = w) each unit's credit has the sign of C(w) while each unit's
leave-one-out ablation value is > 0. Simulation arm: online Adam (lr 0.05, 20k steps, 3 seeds) on
the 2-unit model; report accumulated gradient credit vs accumulated committed-trader settlement.
Stop rule: none (a check); any violation of the closed-form claim is reported as a failed lemma.

## (b) Sign census on wealth-net checkpoints (`lead/sign_census.py`, read-only, existing files)
Oracle: exact single-unit zero ablation dBCE on held-out data (wealth-net `decay_oracle_s*.npy`,
final checkpoint, neurons; M1/w4 `results/s*-step*.json` `ablation`, neurons and heads, 4 ckpts).
Important = top 5% by oracle dBCE (primary), top 10% and dBCE > 0.01 (secondary).
Gradient credit: all-time accumulated books wealth (`decay_currencies cum`, snapshot heads);
instantaneous held-out income under rules live / v1 / cp / attnlrp (w4).
Primary metric: fraction of important units with negative credit, mean +- sd over seeds 1-3,
against the base rate (fraction negative over all units). Contrast: exact leave-one-out
settlement (wealth-net `settle_currencies`, loo_train / loo_heldout).
Stop rule: none (measurement).

## (c) Three-rules settlement for attention only (`lead/three_rules.py`, CPU)
Rule: LN detached (exchange rate), linear nodes Euler, S = q.k/sqrt(dh) half/half, softmax as LMSR:
values paid at the prior pi = 1/Q, routing credit to scores by the 16-step IG path from S = 0
(sum = sum_j (a_j - pi_j) u_j). Same checkpoints and held-out data as w4 (3 seeds x 8k/16k/24k/30k).
Checks, in order: (1) conservation residual at every cut / |R| (pass: <= 1e-3, quadrature only;
also 64 steps); (2) split of each head's wealth into value (pi.u) and routing ((a-pi).u) parts,
sign of routing wealth for heads by w4 ablation; (3) Spearman of head wealth (three-rule total,
routing, value) and neuron wealth vs w4's exact ablation, compared to cp and live; (4) lock-in:
Spearman of head wealth ranks 8k vs 30k under three-rule vs cp.
Stop rule: if (1) fails the rule is not conservative on this net and is not locked for attention.
Seeds 1-3; mean +- sd; 3 CPU jobs x 2 threads (load checked first).
