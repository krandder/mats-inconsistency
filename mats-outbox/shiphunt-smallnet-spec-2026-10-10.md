# ShipHunt small-network experiment: does structured (Bayes-net) training buy data efficiency?

Owner: eurisko. Spec agreed with Kelvin, 2026-10-10. Status: review and price first; no launch until Kelvin OKs.

## Question
For a fixed small network, do structured training targets (k parents per cell) reach better
single-cell forecasts with fewer training boards than plain per-cell training, at the cost of more
compute per board? Secondary: does TD across evidence stages add to that?

## Task (ShipHunt "hard")
- 5x5 board, ships of lengths 3, 2, 2, probe noise eps = 0.2. Probes are random cells, independent of the model.
- Forecast stages: after 0, 2, 4 and 6 probes (same board, cumulative evidence).
- Training uses only the realized true board, never the exact posterior. The exact posterior (enumeration) is for evaluation only.
- Augmentation: the 8 board symmetries are allowed in every arm (not a research variable).

## Model
One small fixed backbone (e.g. a small transformer over 25 cell tokens plus evidence tokens), identical across arms.
Only the output head and training target differ.

## Arms
- k=0: independent per-cell probabilities; loss = sum of per-cell log scores vs the true board.
- k=1, k=2: random cell order per episode. Each cell chooses k parents among earlier cells and outputs a
  conditional table (2^k entries). Loss = log-likelihood of the true board under the resulting Bayes net
  (pick the table entry that matches the parents' true values). Parent choice: enumerate candidates
  (24 singles / 276 pairs), soft-weight them in training, harden to the top choice at eval.
- k=2 fixed-parents control: parents = nearest earlier neighbours (tests whether learned structure matters).
- Masked-cell baseline: k=0 plus an auxiliary "predict hidden true cells from revealed true cells" loss.
- TD variants (applied to the best of the above, plus k=0): stage-t forecast trained toward the stage-t+1
  forecast (stop-gradient), anchored by the true board at the final stage. Optional: a martingale-consistency
  penalty, regularizer only.
- Optional reference: the full true-value chain (each cell given all earlier true cells).

## Evaluation
- Primary: single-cell KL to the exact posterior marginals on held-out boards, at every stage. Single-cell
  forecasts come from each arm's own output; for k>=1, infer marginals from the Bayes net (exact; brute force at this size is fine).
- Secondary: joint log-likelihood of held-out true boards; calibration (ECE); compute per board.
- Data-efficiency sweep: train each arm to convergence (early stopping on a validation set) on
  100, 300, 1k, 3k and 10k boards; at least 3 seeds. Report curves of single-cell KL vs number of boards, and vs compute.
- Reference point (not an arm): a small net trained directly on exact posteriors, i.e. what is achievable. Evaluation only.

## Decision rule (pre-register before running)
"Structured training is more data-efficient" if k=1 or k=2 reaches k=0's 10k-board single-cell KL with
<= 1/3 of the boards (CI over seeds), or beats k=0 at equal boards across the sweep.

## Compute
Kelvin's own GCP (spot L4 or CPU). Cap $50 unless Kelvin raises it. Measure a smoke run first and re-price.

## Not in scope (later)
LLM (Qwen3-8B on Tinker) versions, reasoning chains, model-chosen orderings, larger boards.
