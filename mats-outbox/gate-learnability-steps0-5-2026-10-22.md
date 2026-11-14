# Gate: learnability steps 0-5 (inconsistency-search ebd2298), 2026-10-22

Opus 5.5 gate subagent for opus-mats. The repo was read only, through a detached worktree (/tmp/gate-ebd). A CPU ablation script is at /tmp/gate-keys/run.py; it uses the repo's own `build`/`sample`, the OneCycle schedule over 8000 steps, a float channel and seeds 1 and 2, and stops at 2500 steps. Line numbers below are in aggbench/parity_learn.py unless noted.
**Verdict:** (a) PASS with wording fixes. (b) PASS with caveats. (c) FAIL: an artifact of input scale, not a finding. (d) PASS on numbers but mis-tagged and confounded by (c). (e) PASS, but "leak-free" should read "leak-reduced". (f) confounded, as the builder says. README: hand-copied (numbers correct) and has no provenance.

## 1. Fair closed forms (a)
- No oracle on the decoder side. `FairZ9Code.forecast` decodes only `logT[sym]` (compression.py:1763-1776). The var prior is tallied from fresh structures (1691-1697). Fair Z9 and E2/E3 decoders are scored on `blind(batch)`, whose `NoOracle` raises on any access (1656-1668, 1787/1790, 1874/1877). FairLLR's structure models use seeds 80M+ (1803), which are disjoint from train (0-10M) and dev (10M-20M) (splits.py:10). `agent_statistic(train)` (1864) runs on an unblinded batch but for E2/E3 reads only each agent's own model (207-215).
- fair.json matches the claim: E2 0.1037/0.0565/0.0399; Z9-p2 b=6 0.0147 (45 symbols), with blind_refs at 0.0151 (5.49 bits, entropy 5.38).
- The 6-bit figure is achieved, but it is not a lower bound. Arithmetic: 9 tokens per reading (4 bits x 2 values + distractor) give 45 unordered pairs, log2 45 = 5.49, so 6 bits fixed-length. But b=5 merges symbols by an arbitrary `sym % 2**b` (compression.py:1731). Five of the 45 pairs (dist+dist and (j,0)+(j,1)) carry no per-agent information and could share one symbol. Change "needs ~6 bits" to "a hand code reaches 0.015 at 6 bits; the minimum is unknown".
- README:1005 gives "0.425/0.383/0.306 below 6 bits" from a per-agent decoder that ignores the coupling between agents. The step-0 table uses the joint-prior decoder (b=2: 0.159). Label the first "per-agent decoder". E3-sybil fair is non-monotone in b (b=2 0.068, b=8 0.077), so it is not Bayes under its own prior: mark it "not a bound".

## 2. Anchor (b)
- It is fresh: a new minibatch every step (256, rng [seed,99]). Best-val is picked on val (VAL_SEED, 2000 episodes; 215, 265-271) and scored on a separate dev (DEV_SEED, 4000; 216, 304). JSON dev: 0.0037/0.0033/0.0038.
- Caveat 1: best_step is 7999 in 3 of 3 seeds, so the anchor was still improving and 0.0036 is an upper bound.
- Caveat 2: on readers-random, ISAB (0.002) beats the transformer anchor (0.017; seeds 0.006/0.005/0.040). Score the gap against the best uncompressed reader.

## 3. Step 1 keys (c): a scaling/optimisation artifact, not a plumbing bug
- The code is sound on symmetry and plumbing. Keys are concatenated per agent before a shared `inp` (178-180), so permutation symmetry holds. The problem is scale: 14 key dims standardised to sd 1 (226), against a message of 2, 4 or 16 dims with sd about 0.075 at init. There is no per-part LayerNorm before `inp`; `norm_first` acts only inside the layers.
- JSON: keys1 b=2 symbol entropy 0.69/0.00/0.23 bits (encoder collapsed), best_step 1400/1000/400; keys1 b=4 entropy 3.0-3.6 bits but the decoder ignores the messages (sign fine-tuning moves it to 0.38-0.44).
- CPU ablation, val KL at step 2500, with the encoder gradient norm (`enc` grad):

| arm | escapes | val KL | `enc` grad |
|---|---|---|---|
| keys as shipped | 0/2 | 0.457 | 2e-4 to 5e-4 (collapsing) |
| no keys | 2/2 | 0.151-0.155 | 0.57-0.59 |
| same net, keys zeroed | 2/2 | 0.152-0.158 | not logged |
| keys x0.1 | 2/2 | 0.152-0.172 | not logged |
| raw tokens + keys | 2/2 | 0.047-0.059 | n/a (no encoder) |

- So unit-scale keys starve the encoder of gradient; the decoder and the concatenation work. "Keys hurt" and step 1's "encoder/channel learning is the problem" are not supported. Every keys arm in steps 3 and 5 carried the same handicap.
- Decisive check: rerun step1 keys1 with a separate LayerNorm and projection for the message and for the keys (or message standardised and keys x0.1), plus keys zeroed for the first 1000 steps. Run b in {2,4,inf} x 3 seeds x 8000 steps and log the encoder gradient norm and message sd. Pass: keys1 escapes at least as often as keys0 at every b.

## 4. Matching and leaks
- Step 3 matches step1_keys1 on steps (8000), decoder, lr, keys and val/dev. There is no keys0 curriculum arm, so curriculum and keys effects cannot be separated.
- Tag: the fixed-identity curriculum is not "Y labels only". Its subtasks k=2,3 are built from the true parity bits (114-118), so structure knowledge enters through the task design. The k=2 label (bit0 XOR bit1) is the quantity the aux loss supervises. Tag it "structure-informed curriculum".
- The anchor got no curriculum, so the arms and the anchor did not train on the same data. Run the anchor under each curriculum.
- Aux slot 0 is the raw latent bit 0 (100-103), active for t < 4000, i.e. stages k=2 and k=3 (281). "Latent-label supervision" is the correct tag.
- Step 5: the b=4 arm has keys and the inf arm does not (362-363; admitted). The biased curriculum carries no structure knowledge but is still different data from the anchor's. 0.065/0.065/0.066 suggests a shared plateau; compare it with the blind b=4 reference.
- Step 4 is matched (same spec, steps and decoders, no keys). "Leak-free" is overstated: with readers random the blind count code is still 0.256 against a prior of 0.308 (lower because some bits get zero readers). The failed dvib seed sits at the prior (best_step 400). The step1 keys0 b=2 "escape" (0.358) is partial: above the count level (0.162).

## 5. README table
The table is hand-written: no renderer reads parity_learn.json (grep). All 18 row groups I recomputed match the JSON. However, README:1224-1225 has a doubled separator row. parity_learn.json has `"harness": null`, so the README's "harness ece51b7" cannot be checked, and blind_refs records a dirty "ece51b7 + ... (this commit)".

## Must-fix (in order)
1. Rerun the keys arms with the scale fix from section 3 before any keys result is reported; rerun step 3 with it and add a keys0 curriculum arm.
2. Tag the curriculum "structure-informed" and train the anchor under the same curricula.
3. Generate the table from the JSON with a script and write `harness_info` into the merged JSON.
4. Wording: "needs ~6 bits", the per-agent label, "E3-sybil fair is not a bound", "leak-reduced"; gap = arm minus the best uncompressed reader (anchor unconverged at step 7999).
