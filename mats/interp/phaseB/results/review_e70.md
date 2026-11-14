Model: GPT-6 (Codex)

Checked 2026-10-31 09:16Z: source, metadata, NPZ scalars, head JSONs, tables and SHA-256 records; no Python or model execution. References to draft mean update80_draft.md.

1. **High: causal localization exceeds the control.** Draft:4,54,57 attributes the effect to head count and says sign dependence is removed. Count and dimension change together (train_b1h4.py:21; pfn2.py:38); results/e70.json:b1h4.sign_share remains 0.0667. Fix: say block-1 repartitioning suffices without extra block-2 heads; count versus width remains unidentified, and sign dependence decreases.

2. **High: the undecided mechanism becomes a conclusion.** Draft:3,55 says the fixes “do not fully coincide” and remove a shared cost. PREREG-B.md:3936-3951 requires interval thresholds; results/e70.json:predictions.P273 is undecided. The positive log interaction establishes departure from multiplicative independence, not shared mechanistic cost. Fix: retain the descriptive interaction and undecided verdict; describe shared cost only as a hypothesis. State both comparisons in the plain-language rule.

3. **Medium: device reassurance is too broad.** Draft:58 invokes E60's check for GPU ownmag versus CPU own. Update 71:6740,6744 checks four baseline ctrl seeds, not own. Fix: retain this architecture-specific device uncertainty when attributing the 20% reduction to format. b1h4/ctrl and ownmag/mag are GPU/GPU comparisons.

4. **Medium: numerical and role wording.** Draft:19's “a fifth below either” conflicts with P273.ratio_mag=0.6097: reductions are 20% versus own, 39% versus mag. Draft:5's “any learned version” excludes the acknowledged 0.0032 comparator. Fix: “learned-pool arm.” Explicitly call pooling/consensus attention-mass classifications (e70_heads.py:35-39), not demonstrated computations.

5. **Low: loader drops block-2 configuration.** e63_run.py:35 reaches pfn_pre.py:45, which omits l2_heads/l2_dh. Fix: forward these fields. Current block-1-only head measurements remain valid; e48's separate evaluator preserves them.

6. **Low: registration time.** Draft:24 says 07:41Z; prereg_hashes.txt:860 says 07:42:12Z, before GPU-LOG.md:239's launch. Fix timestamp; all listed hashes match.

**Survive:** code-supported identical initialization/episodes, correct training arms, P271/P272/P274, P273 undecided, registered interaction scoring, matching tables/JSON/scalars. **Reword:** causal, device and head-role interpretations. **Fail:** isolated head-count causation, established shared-cost explanation, “20% below either.”
