# Critique: World FA spec (forecast aggregation with hidden correlated information structure)

Read /home/agent/workspaces/mats/work/world-fa/SPEC.md. Context: small set-style transformers are trained PFN-style
on fresh episodes from this prior, target = exact Bayes predictive for v; we then study mechanistically how they do
in-context aggregation and deduplication (prior-level shortcuts vs in-context inference of the information structure).
Prior lessons: models dedup exact copies via exact value ties, not covariance; shared-evidence discounting failed in a
measurement-market world; a source-binding failure explains training plateaus; plain attention's predictive variance
does not shrink with N beyond training.

Be adversarial and concrete (<= 700 words; state your model in the first line):
1. Can a shortcut pass (structure-blind or prior-level rule)? Which parameter regimes make the structure matter?
2. Is the exact Bayes correct and cheap for both prior variants (structured finite family; inverse-Wishart)? Pitfalls
   (identifiability with T < N, IW vs factor structure mismatch, numerical issues)?
3. Token design: how should history rounds and query reports be tokenised for a set transformer (per report token
   with trader id + round id? per trader token with its history column?) so binding and scaling are testable?
4. What would make this fail to be learnable (lessons above), and how to make it learnable (curriculum, dense
   supervision, regime)?
5. One concrete change you would make, and the first pilot.
