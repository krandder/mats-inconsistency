Model: GPT-6 (Codex)

Checked 2026-10-29 17:45Z using source, JSON, logs, SHA-256 and filesystem timestamps only; no Python or heavy computation.

1. **High: mechanism is not isolated.** Draft lines 16, 76–83 claim pooling-order isolation and that gradient descent initially ignores the new path. `pfn_var.py:27–36` changes every pre-attention token, including queries, hence attention routing, representations and optimisation; post affects only the final query's readout. Behavioural trajectories cannot locate computation. Fix: describe an equal-parameter placement intervention consistent with the account, remove claims about path use, and propose direct ablation as the mechanistic test.

2. **Medium: “forms at binding” overstates timing.** `e19_traj.json["5"].pre` has departure onset 3000 versus sustained slope≥0.6 at 3250; seed 6 has 4500 versus 5000. Gap<0.10 arrives at 4750/6500. `e19_traj.py:24–27` requires three consecutive checkpoints, omitted from the table header. Fix: say departure emerges near, but before these operational acquisition markers; report markers and distinguish exploratory timing from the preregistered trajectory comparison. The 90k column is correct: all four `step_final` values are 90000. Explicitly distinguish seed-5 90k values (.1444/.0474) from best-at-89k (.1575/.0586).

3. **Medium: certainty exceeds replication.** Draft lines 6, 10, 20, 78 assert “changes nothing,” position matters and lineage dependence. `e19.json.predictions.P119` tests only ≥0.75, not equivalence; seed-6 post rises .07788→.08880. Two acquired pre pairs, one per baseline signature, establish neither a lineage interaction nor a general placement effect; better fit remains an alternative explanation. Fix: “post's mean is nearly unchanged; pre reductions differ between these two seeds.” State P118 fails at .5656 without “narrowly.” `runs/var_pre_s7/meta.json.total_steps=50000`, versus 90000 otherwise: replace “never learned” with “did not acquire within its recipe-determined run.”

4. **Medium: toy comparison needs its limits.** Draft lines 84–86 omit E20's different precision-based metric and signed-mean cancellation (Update 28, E20.1: pre −.011 to +.022). Fix: compare relative mean reductions only; call binding competition and unfinished correction possibilities, not “likely reasons.”

**Survive:** reported numbers, P119 passing, P118 failing, registered acquisition exclusion and bootstrap scheme. Source matches registered variants/seeds/recipe/measures; all four hashed code files match. Constructor order and independent NumPy RNG support identical shared initialisation/data, though binary weights were not independently compared. Step-zero timestamps place pre starts at 15:09:36 and post starts after 16:09, after the 15:09:22 hash; no implementation deviation found.

**Need rewording:** partial placement effect, lineage, timing and toy analogy. **Fail:** mechanistic isolation, inferred path use, and literal “no change.”
