Model: GPT-6 (Codex)

The portfolio needs sharper hypotheses, valid oracles, and tests that distinguish failure to learn from information loss. Current numerical thresholds are engineering tolerances, not decisive refutations.

1. Cruxes

C1 — No. The contamination slope can reflect incorrect group inference, relay dependence, or generic length failure; it does not isolate count drift. Eight of ten seeds cannot repair that confounding. Corrected criterion: use paired contexts with identical Bayes posteriors while varying redundant representation, alongside matched-length contexts containing genuinely independent evidence. Measure excess predictive log loss and posterior displacement, with uncertainty across training seeds and episodes. A systematic displacement under posterior-preserving transformations refutes invariance; a failure only under added independent evidence refutes evidence scaling. Validate short-context competence first. Relays require a correctly specified dependency oracle.

C2 — No. Global attention need not propagate one graph edge per layer; pointer jumping can beat the assumed linear depth requirement. Six hops barely tests scaling. Conversely, a successful depth-2 model refutes only that particular six-hop necessity claim. Corrected criterion: randomize node labels and serialization, train on short paths, test substantially longer paths, and compare depth, width, recurrent iterations, and training budget. Include an explicit pointer-jumping baseline. A reproducible gain from additional inference iterations on the same trained weights establishes useful test-time computation; it does not prove a general expressivity lower bound. Residual reduction alone is insufficient: measure distance to Bayes.

C3 — No. Thirty-two real-valued slots do not specify B bits; full attention may itself fail; average log loss can hide forgotten items. The analytic bound concerns a particular revealed-bit task, not automatically W2. Corrected criterion: quantize the entire retained state, count all bits and side channels, reveal the query only after compression, and sample queries uniformly over independent items with substantial information. Compare against Bayes and a memory-matched ledger. Test the bound in its native bit task first, then measure W2’s distortion-versus-memory curve. Success at one finite K cannot refute an asymptotic impossibility.

C4 — No. Repeated versus fresh sources changes noise, bias uncertainty, and calibration evidence together. Prediction changes can vanish, making their ratio unstable; closeness to one weak reference does not establish prior-level behavior. Corrected criterion: hold query readings fixed and vary only other-item observations that calibrate the same sources. Construct paired episodes with identical counts and source labels but opposite inferred reliability or bias. Bayes should change its query prediction substantially. Failure to track that change, despite success with supplied calibration, isolates missing in-context source inference.

C5 — No. “Oracle steps” is undefined; auxiliary losses may supply unavailable information or change the target, while warm starts may encode the answer. Corrected criterion: hold architecture, inference inputs, and evaluation distribution fixed. Demonstrate a successful parameter setting, then compare ordinary training with initialization or auxiliary supervision under reported compute budgets. Success proves representability on the evaluated distribution and refutes expressivity as the sole explanation there. Failure of ordinary training is evidence about that training procedure, not a universal optimization barrier.

C6 — Accept, narrowly. Two globally optimal solutions under the exact proposed consistency objective, with different query predictions and unique Bayes, refute that objective’s sufficiency. Use an exact counterexample; 0.2 is unnecessary. The conclusion is that additional identifying information is required—not specifically that both likelihood and prior terms are essential. Specify boundary conditions and exclude mere latent reparameterizations.

2. Worlds

W1: Fatal unless resolved: O(120N) is not generally valid with unknown relay ancestry. Marginalizing hidden copy partitions can dominate inference. Start with observed provenance and exact covariance; separately introduce hidden ancestry only where enumeration remains tractable. Item effects alone establish correlated observations, not learned deduplication.

W2: Fatal for the proposed scaling experiment: exact enumeration costs 2^S, so growing S undermines the reference. Start with known heterogeneous variances and unknown Gaussian biases; grow items and sources with exact Gaussian inference. Add discrete uncertainty only at small S. Also distinguish prior-regularized uniqueness from likelihood identifiability.

W3: Fatal to a reasoning interpretation: longer noisy chains can erase query information, making prior predictions look successful. Match posterior information across lengths, include exact-equality chains, and report excess loss relative to the available information.

W4: Fatal: CTW trees and transposition DAGs are different probabilistic models. Multiplying child evidences double-counts shared descendants in a DAG. Specify one generative model first; retain a tree oracle or use bounded-treewidth exact inference for sharing.

3. Missing

Add a provenance-and-complementarity world: overlapping sources report linear projections of shared primitive measurements, plus genuinely new measurements. Known provenance gives an exact Gaussian oracle. Redundancy, complementary information, source calibration, and graph distance become independently controllable. Later hide provenance in enumerable cases. This directly tests whether aggregation preserves novel information while discounting shared evidence.

4. Twelve-hour priorities

Ranking: W3, simplified W2, W1, W4. W3 offers the cheapest clean reasoning test; W2 best matches aggregation; W1 carries oracle risk; W4 remains underspecified.

First three experiments:
1. Validate W1 oracle and paired posterior-preserving duplication tests.
2. Run W2’s paired source-calibration intervention against Bayes and ablated references.
3. Run W3’s information-matched path sweep, comparing fixed depth with additional tied iterations.

Use initial seeds to screen failures, then replicate informative results. No architecture decision is justified by an unvalidated oracle.
