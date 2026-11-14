Model: GPT-6 (Codex)

Ranked review; checked 2026-10-30 by reading code, JSON, registration, hashes and drafts. No Python or reruns; raw checkpoint/NPZ contents were not independently recomputed. References to “draft” mean update67_draft.md.

1. **High: unsupported causal subtraction.** Draft:55 says “most of it comes from elsewhere” because measured rom exceeds constrained Bayes. That reader is not a measured component of the network, so subtraction cannot allocate causes. The threefold description also fails across learned cohorts: `e55.json:cohorts.e38_w128mag` has -0.002 measured versus +0.025 predicted. Fix: restrict the numerical example to E43; call the discrepancy model mismatch, with learned weighting a hypothesis. Draft:8’s “so attention can produce it” likewise exceeds E50’s evidence, which leaves later stages possible.

2. **High: retracting a failed test.** Draft:58 says E58 “could not test learned pools”; Update66’s P226 explicitly fails, with both jointly acquired seeds moving oppositely. Fix: retain the failure and label poor acquisition as a post hoc limitation on generalisation.

3. **Medium: reader approximation becomes mechanism.** Draft:15,55 treats the departure as the capped statistic’s departure; `lngeom.py:27-34` correctly computes centred A/U, but the even reduction needs near-constant A and negligible rho. Actual pools retain vectors, including signed information; LayerNorm also includes epsilon and width scaling. Fix: say approximate quantitative agreement at arm means near c=1.4, not an identified scalar bottleneck. Draft:7,56’s tiny KL is the simulated even-reader cost (`e55_cbkl.py:22-32`), not the trained network’s available gain; retain only the weak-incentive hypothesis.

4. **Medium: clustered confirmation needs prominence.** Draft:3-5 omits the shared exploration parents acknowledged at :59. Fix: describe 21/23 as a registered descriptive score across related cohorts, not hundreds of independent replications.

5. **Low: table corrections.** Draft:40 lists six E44 groups; `e55.json:predictions.P216.cohorts` contains five. Draft:19 omits ctrl_s8’s endpoint c=1.62, rom=0.194. P216’s median is -0.754545, rounding to -0.75. Fix these and label E53’s 12-net rows as including exploratory seed 0. Suppress invalid sq CB values in the linked generated table too: its c has r² units.

**Survive:** A/U computation, registered interpolation/acquisition/exclusions, P215-P217 scores and corrected P218; hashes and the finite-rom correction match the log. **Reword:** quantitative agreement, clustering, KL incentive, width/training associations. **Fail:** causal subtraction, attention identification, and “could not test” learned pools.
