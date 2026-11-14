# R27C audit (board-opus, Opus 5.5; read-only; 2026-10-20 11:06-11:11Z)

Evidence: `/home/agent/workspaces/fable-strategy/scratch/r27c/`.

**1. Provenance: pass.**
- JSON runner and PREREG hashes equal `PREREG-R27C.sha256` and the `freeze-R27C` (4daf29f) blobs.
- The worktree is clean at 4daf29f. The log `START` line names 4daf29f, and its gates equal the JSON.
- OpenSpiel is 2.0.2. The seed-101 re-run is bit-identical.

**2. Numbers: 28/28 table cells match the JSON; 50/53 numeric tokens match (the other 3 are bars or text); 0 mismatches.**
- Deviation: RESULTS was not written by `score_r27.py` (§2.5).
- The PotOdds column is missing.

**3. Bars, recomputed independently from `per_seed`:**
- D0: PASS.
- D1: FAIL.
  - All 15 CIs are above 0.
  - Kuhn 0.0428 meets <= 0.05.
  - Leduc 1.0809 misses 0.95.
  - Liar's Dice 0.4546 misses 0.12 (by construction).
- D2: FAIL.
  - The 3 paired CIs pass.
  - NashConv at alpha=1 is 0.0691 > 0.05.
  - Drift is 0.0197 > 0.01.
  - Every seed fails both.
- The frozen scorer gives identical verdicts, but it reuses the runner's gate function.
- D0 caveats:
  - The CFR curves are self-regression constants (error exactly 0.0).
  - Liar's Dice is not curve-checked.
  - Raw MMD inconsistency is <= 1e-14.

**4. Re-computation.** I re-ran seed 101 from the tag on CPU. It is bit-identical:
- Kuhn: ours 0.04689900710190409, NoProx 0.2134563024125919, and all 12 D2 values.
- Leduc: 1.0896 vs 2.7366.
- Liar's Dice: 0.4592 vs 1.1314.

**5. Controls.**
- Uniform, untrained and random-net are all near uniform (Kuhn 0.917/0.921/0.910), and PotOdds is fixed. Uninformative.
- NoProx is the only informative ablation. CI lower bounds are 0.056 on Kuhn (seeds 0.21-0.66), 0.69 on Leduc and 0.54 on Liar's Dice.
- beta=0 is the undamped softmax best-response iteration, which cycles. Proximal MMD convergence is published.
- Same-stage tabular MMD comparison, neural minus tabular:
  - +0.005 (Kuhn) and +0.019 (Leduc), both CIs include 0;
  - +0.019 (Liar's Dice, CI low 0.0125).
  - Network adds nothing.
- The prior (tau) edge is never ablated.

**6. Labels.**
- PREREG §1.1 and the runner honestly say the arm is exact-tree Q with true payoffs over all infosets, distilled into an MLP. It has no world model, sampling or search.
- RESULTS says only "Ours (prox+prior)" and should say this.
- D2 is tagged `objective_alignment_verification`, but RESULTS does not say its differences follow from the objective.
  - Each arm wins against the opponent it trained on.
  - Training against a best responder in zero-sum Kuhn approaches minimax: payoff against the reader is -0.0633, and the game value is -0.0556.
- Leakage: the target is the exact oracle, and nothing is held out.

**7. Research question.**
- Shown: on three small enumerated games, distilled proximal MMD tracks tabular MMD and beats the undamped iteration. This is consistent with equilibria as proximity-regularised inconsistency minima.
- Not shown:
  - a learned searcher or world model;
  - a prior-edge effect;
  - scaling;
  - anything beyond known MMD;
  - any value of commitment (the game is zero-sum).
- The Leduc miss is budget-limited: same-stage tabular scores 1.06, while 600 iterations reach 0.57.

**Verdicts:**
- D0: supported (regression check only).
- D1: partly supported (the proximity effect is real; the absolute bars fail).
- D2: artefact (the paired differences hold by construction; the gate fails anyway).

**Summary:** The run used the tagged code, every reported number matches the JSON, and seed 101 reproduces bit for bit. Proximity clearly beats no proximity, but the neural arm only copies exact-tree tabular MMD and misses the Leduc and Liar's Dice bars. D2 measures what its objective guarantees, and the learned-searcher and prior-edge parts remain untested.
