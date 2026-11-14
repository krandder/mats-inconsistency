I’m GPT-6, running as Codex. Round 2 follows; no files modified.

1. Strongest objections to the other critics’ worlds

Gemini’s 16-bit junta: its proposed concentration policy explicitly uses hidden R. Observed keys therefore reveal R through the sampling distribution; the stated Gaussian-mixture oracle omits that likelihood. This invalidates the policy-invariance experiment. Concentrate on visible coordinates chosen independently of R, or include the policy likelihood. Larger keys alone do not fix this.

Gemini’s transposition graph: permutations/reflections of state labels may leave the equivalence relation unchanged. With iid state values, such hypotheses are observationally indistinguishable. Specify 32 genuinely different partitions of paths; otherwise “inferring symmetry” is either impossible or unnecessary.

Fable’s modified junta: inverse group multiplicity is the wrong general target. Independent readings of a shared latent value legitimately accumulate precision. Equalizing groups can move predictions away from Bayes. Separate copied events, persistent item effects and independent measurement noise before interpreting count-dependent attention as deduplication.

Fable’s hierarchical cuts: parallel prefix statistics do not make recursive model evidence a two-layer calculation. Each split’s evidence depends on its children’s recursively marginalized evidence. The proposed shallow mechanism assumes away the interesting computation. Also, 2^(2^d−1) counts internal-node bit assignments, not distinct partitions: choices below a stopping node are irrelevant.

2. Portfolio, in build order

World 1: copies, fresh measurements and related items.

Draw θ~N(0,1), item effects b_j~N(0,τ²), and fresh events y_e=θ+b_j+ε_e, with known Gaussian noise. Tokens identify item and event; copying preserves both IDs and value. Predict θ.

Exact Bayes deduplicates events, then pools item means with variance τ²+σ²/n_j. Retain within-item residual statistics for likelihood checks. Randomize IDs and episode values. Paired contexts with identical value histograms but different provenance require different posteriors, defeating fixed discounting.

Predicted mechanism: event equality, item pooling, saturating item precision, final precision-weighted aggregation. References: treating copies as fresh, independent-item inference, exact hierarchical Bayes.

World 2: similarity inferred from context.

Extend World 1 to 16-bit items: θ becomes g(x_R), with R uniform over the 120 coordinate pairs and four independent Gaussian cell values. Queries are unobserved items. Policies use only visible keys and recorded observations.

Exact Bayes enumerates R, using Gaussian item/cell marginal likelihoods and posterior predictions; cost O(120N). Construct matched episodes where evidence favors different R and changes the query posterior. Require an oracle-certified advantage over prior-averaged similarity before admitting an episode to the diagnostic set.

Predicted mechanism: cross-item residual comparisons, relevance-hypothesis weighting, conditional pooling. Compare per-item Bayes, true-R Bayes, full-mixture Bayes and fixed-similarity inference. Increasing N and increasing distinct items are separate axes.

World 3: heterogeneous evidence and constraints.

Draw K=8–16 independent latent bits. Messages report noisy individual bits, noisy pairwise XORs, or noisy subset majorities; each type has a declared error rate. Copies retain event IDs. Predict a queried bit’s probability. This is a toy market of contracts about a shared underlying state.

Exact Bayes enumerates 2^K assignments and multiplies unique-event likelihoods. Aggregate repeated factor types for large-N evaluation; benchmark oracle cost before generating training volume. Include contexts where individual reports are ambiguous but relations resolve the query. Randomize labels and factor graphs.

Predicted mechanism: relation composition and reliability-weighted evidence integration. References: unary-only Bayes, independence-assuming pooling, exact joint Bayes. This tests whether rich messages contribute information unavailable in scalar estimates.

World 4: recursively inferred evidence sharing.

Use a known binary tree. At each reached internal node, stop with probability 1/2 and assign one Gaussian block mean, or split recursively; leaves must stop. Fresh observations measure their block mean with independent noise. Predict an unobserved leaf.

Exact evidence satisfies Z_v=½L_v+½Z_left Z_right; posterior prediction follows the corresponding mixture. Sparse observed/query paths permit O(Nd) processing. Require matched datasets whose local summaries agree but whose descendant evidence changes merge/split odds.

Predicted mechanism: prefix pooling followed by recursive evidence combination. Compare per-leaf, true-partition and full recursive Bayes. Vary depth independently of N; require informative coverage.

3. Cruxes and refutation criteria

Preregister these operational claims. Use ten training seeds, 2,000 held-out episodes per condition, paired interventions and seed-level confidence intervals. Check small oracles by independent enumeration. Report predictive KL in nats, calibration and runtime. These are proposed thresholds, not results.

Attention extrapolates aggregation across policies and lengths. Train at N=16–64; test 256–4096 under uniform, Zipf and recorded-history adaptive policies. Refute the tested architecture’s claim if at least 8/10 seeds achieve KL<0.02 in distribution but KL>0.10 under two held-out policies, with confidence intervals excluding thresholds. Exact streaming sufficient statistics must remain accurate. This does not prove all attention architectures incapable.

A bounded consensus preserves arbitrary future queries. In World 3’s independent-bit subfamily, reveal K bits perfectly, compress before revealing the query, and prohibit rereading. A B-bit state has average bit error at least h₂⁻¹(1−B/K). At B=K/4 this exceeds 21%, while full context has zero error. This analytically refutes universal lossless bounded consensus; query-aware compression or external memory changes the claim.

Minimizing inconsistency suffices for truthful probabilities. Define the inconsistency objective before testing. Refutation: exhibit two globally optimal, equally consistent outputs differing by at least 0.2 in a query probability when exact Bayes is unique. If likelihood and prior terms eliminate the counterexample, those terms are essential to the program’s claim.

More inference steps improve aggregation. For a specified recurrent update, compare 1, 2, 4, 8 and 16 steps on Worlds 3–4. Refute monotonic improvement if 16 steps increase KL by >0.05 over four in 8/10 seeds while the stated inconsistency objective decreases. Include damping and matched-compute controls.

4. Architecture consequences

Build event-addressable memory if causal ablations establish provenance pooling; precision/count channels if saturation causes length failures; recursive shared updates if tree evidence requires sequential composition; query-conditioned retrieval if fixed-state compression loses information. Require targeted ablation to increase KL by ≥0.05 and a predicted intervention to recover ≥75% of that increase before claiming a mechanism. Build a replacement only if it then improves held-out accuracy or compute. Relative to PFNs4BO, DPT, KR-UCT and in-context model-selection work, the defensible contribution is this causal and extrapolation evidence; novelty remains to be verified.
