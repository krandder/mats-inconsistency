# Member 4b — Round-3 Synthesis & Critique

**Society of Thinkers · `futarchy-fi/inconsistency-search` · 2026-10-18**  
**Author:** Member 4b, GPT-6-Astra via codex-any  
**Evidence boundary:** This adjudication uses only the supplied R11 report, artifact excerpts, GPU ledger, and Member 14 audit. No tools were called, files inspected, or experiments independently rerun. Round-12 specifications below are proposals for registration, not completed experiments.

## 1. Header & Executive Adjudication of `R11-GPU-NEURAL`

**Verdict: R11 establishes a substantial recovery of empirical credibility after Round 2. It demonstrates genuinely trained, blind neural tree search with useful compute scaling. It does not establish strict anytime monotonicity, calibration-only explainability of query error, or generalist superiority in independently measured interactive play.**

Accept **`R11-D0` as PASS within its audited scope**. Retain the reported numerical passes for `R11-D2`, `R11-D3`, and `R11-D5`, while distinguishing their measured endpoints from broader scientific claims. Retain **PARTIAL PASS** for `R11-D1`, `R11-D4`, and `R11-D6`. The self-labeled `m_ql` evaluation is not independent evidence of validity detection, despite meeting its implemented numerical threshold.

### `R11-D0`: what the gate establishes

| Component | Supplied evidence | Adjudication |
| :--- | :--- | :--- |
| Static audit, `S1`–`S6` | Zero violations; all three synthetic canaries caught | PASS for the audited rules and code paths |
| Runtime poisoning, `R1` | `3/3` `PoisonTensor` traps caught for each seed | PASS for the tested oracle fields |
| Counterfactual invariance, `R2` | Label permutation and zeroing give `0.000000000` maximum difference across 12 environments × seeds `101, 102, 103` | Strong evidence that tested inference paths do not depend on those labels |
| Genuine learning, `R3` | Mean parameter displacement `17.9362`; aggregate training loss `15.0813 → 0.6579`, a `95.6%` reduction; mean B=0 KL improvement over uniform `14.85%` | PASS for actual optimization and improvement over the stated reference |
| KV equivalence, `R4` | Cached versus uncached root difference `0.000000000` | PASS at the tested root; deeper cache correctness remains a separate obligation |

The `95.6%` loss reduction is the reduction in the mean endpoint losses. The approximately `94.9%` figure elsewhere in the report corresponds to averaging individual percentage reductions; these are different aggregations and should be labeled accordingly.

The supplied audit reports removal of the Round-2 oracle interpolation, seed-conditioned outcome manipulation, and synthetic redundancy-curve construction. This changes the evidential status of R11 materially. However, a finite AST audit plus runtime canaries establishes **no detected leakage under specified checks**, not a universal proof against every possible information channel. The two supplied accounts also describe the three AST canaries differently; R12 should publish their exact source fixtures and expected rule identifiers.

The GPU ledger records a full production run on **`soup-gpu-r11gpu-1018041347-24711`**, using an **NVIDIA L4 with 22.03 GB VRAM**, CUDA `13.0`, and PyTorch `2.14.1+cu130`. The reported cost ceilings are **`$0.58` production + `$0.08` initial probe = `$0.66` total**. Python execution took `2,128.57 s`; both instances were recorded as deleted. These are supplied historical ledger findings, not a fresh infrastructure check.

### Analytic operator ceiling versus realized neural performance

| Question | Tier-1 analytic operator ceiling: `R10` / `Stage-2` | Tier-1 end-to-end neural run: `R11-GPU-NEURAL` |
| :--- | :--- | :--- |
| What is being tested? | Operator behavior under surrogate or privileged constructions | Randomly initialized, SGD-trained representations used in blind inference |
| Generalist Mini-ECI gain | `+30.73` reported in Stage-2; `+18.0` retained as the R11 target | `+6.95` |
| Witness discovery | `14.2% → 88.5%` | `0.52% → 43.06%` |
| Redundancy-law evidence | Near-perfect fit compromised by circular construction | Empirical neural trajectories; `R²_red = 0.9443` for `GLC-AOM-KV-Tree`, `0.9155` for the generalist |
| Scientific interpretation | A reference for idealized operator behavior, not achieved neural capability | A smaller but substantially more credible demonstration |

“Ceiling” should remain shorthand for an idealized reference: the supplied material does not establish a mathematical upper bound. Likewise, Mini-ECI differences across separately fitted rounds require caution unless the item calibration and scale linking are held fixed.

**Provenance for R11 tables and numerical claims:** supplied excerpts attribute production measurements to `benchmarks/run_r11_gpu_neural.py`, the leakage audit to `src/eval/ast_leak_auditor.py`, and seed records to `results/r11_gpu_neural_results.json`, seeds `101–103`, timestamp `2026-10-18T04:54:16Z`. Hardware and cost evidence comes from the supplied `GPU-LOG.md` lines `2177–2189`; Member 14 supplies the code-level audit account.

## 2. Scientific Analysis of Major Breakthroughs vs. Honest Negative/Partial Results

### Useful tree-search scaling is real; strict monotonicity is not

The two principal `GLC_Tree_Active_PUCT` cohorts show positive aggregate scaling:

| Cohort | ΔMini-ECI, B=0→128 | κ per doubling | `R²_log` | `R²_red` |
| :--- | ---: | ---: | ---: | ---: |
| `Unified-Generalist` | `+6.95` | `+0.7637` | `0.9241` | `0.9155` |
| `GLC-AOM-KV-Tree` | `+5.43` | `+0.7389` | `0.9110` | `0.9443` |

The generalist also improves witness discovery from **`0.52%` to `43.06%`** and reported mean Elo by **`+268.1`**. Together, these support the claim that additional search can extract useful information from imperfect learned representations.

The plausible mechanism is that branch-aware clearing controls redundant or conflicting evidence before it is accumulated, while active allocation directs expansions toward useful hypotheses. Flat pooling can instead count dependent evidence repeatedly or combine mutually exclusive alternatives as if they were jointly valid. The ablation results support this account, but do not independently identify the contribution of every component.

Three qualifications matter:

1. **The trajectories are not strictly monotonic.** Both principal cohorts decline between B=32 and B=64. The generalist has `7/8` non-decreasing transitions. Its mean KL is best at B=8, `0.5225`, and worsens to `0.5319` at B=128. More witnesses and higher composite scores do not imply continuously improving marginal beliefs.
2. **Every flat search baseline has a negative endpoint gain**, spanning `[-36.82, -10.72]` Mini-ECI, but not every fitted slope is negative. `Flat-Ataraxos-MMD` has `κ = +0.8256` with `R²_log = 0.0381`, reflecting recovery after an initial collapse rather than useful overall scaling.
3. **Redundancy-fit success is cohort-specific.** The generalist’s `0.9155` misses the `0.92` bar; `GLC-AOM-KV-Tree` passes at `0.9443`. One cohort’s pass cannot be transferred to another.

Reported Gram crowding is also not yet a fair cross-method comparison: tree search uses `edge_keys_bank`, flat methods use `root_keys_bank`, and the root-only reference receives an assigned value. R12 must use identical measurement definitions.

### `R11-D2`: a strong common factor within this benchmark

The five preregistered numerical conditions pass:

- **PC1:** `64.75%`.
- **Hierarchical omega:** `ω_h = 0.8961`.
- **Explained common variance:** `ECV = 0.7350`.
- **Leave-one-task-out correlation:** `r = 0.7122`.
- **General loadings:** all nine satisfy `λ_g ≥ 0.4891`, above the `0.45` bar.

These results support a useful common performance dimension across the evaluated cohorts. They do not show that the suite is fully unidimensional: Sudoku has a substantial group loading, `0.8622`, compared with its general loading, `0.4891`.

The 108 model–algorithm–budget cohorts are also dependent observations, not 108 independently trained systems. Budget sweeps and severely degraded flat methods can strengthen a common factor. R12 should preserve the original analysis and add family-held-out validation and uncertainty estimates that respect shared checkpoints and seeds.

### `R11-D3`: alignment passes numerically, with limited external validity

The reported **`r = 0.8862`**, **`ρ = 0.9652`**, and **`+751.3` ALL Elo over `Flat-Mix@B=32`** meet the three stated bars. The last comparison uses the generalist’s `1080.3` against `329.0`; the separately named `GLC-AOM-KV-Tree` cohort has `1081.6`.

However, the audit identifies two different outcome sources:

- Actual alternating-seat interactive matches for Hex and Connect-4.
- Supplementary pairwise outcomes generated from the belief-payoff score
  \[
  -1.8\,\mathrm{KL}-0.6\,\mathrm{Brier}
  +0.45\,\mathrm{witness\ rate}.
  \]

These outcomes may be blind without being independent measures of strategic play. Moreover, multiplayer ratings are themselves inputs to Mini-ECI, making its correlation with Elo partly a part–whole relationship.

Thus, **PASS on the reported alignment gate does not establish five-game interactive strategic transfer**. R12 must measure that claim separately.

### `R11-D5`: the clearest causal architectural success

The firewall passes its five headline conditions:

- Gradient transmission, post-RL belief drift, and utility-induced B=0 bias are each reported as **`0.000000`**.
- Utility interventions change actions in **`85.42%`** of cases.
- Skewed belief KL improves from **`0.2612` to `0.2031`**, a **`22.25%`** reduction.
- Reported mean Elo exceeds `Fused` by **`526.2`**.

The key result is selective responsiveness: utility can change action selection while the tested belief channel remains invariant. That is substantially stronger evidence for the firewall than a favorable aggregate score alone.

The performance advantages are aggregate findings, not universal seedwise improvements: seed 101 slightly favors `Fused` on skewed KL and ALL Elo. The Elo caveat above also applies. Finally, these interventions validate the tested firewall pathways, not every claim about episodic memory.

### Three partial results and their causal limits

#### (a) `R11-D1`: neural search gains fall below the analytic target

The **`+6.95`** Mini-ECI gain fails the **`+18.0`** target. Finite training, imperfect query representations, residual calibration errors, and search allocation are plausible contributors.

Neither the endpoint gain nor the final B=128 increase proves non-saturation. A deeper sweep is warranted, but cannot be assumed to close the gap. The correct conclusion is **positive measured neural scaling with unresolved limitations**, not an already-validated path to the analytic result.

#### (b) `R11-D4`: belief protection succeeds; query probability quality fails

At **B=16**, A4 protects variable beliefs relative to A3:

| Environment | A4 Var KL | A3 Var KL | Relative protection |
| :--- | ---: | ---: | ---: |
| Equiv-SAT | `0.0079` | `1.6535` | Approximately `209×` lower |
| Sudoku-6x6 | `0.8518` | `3.5453` | Approximately `4.2×` lower |

A4 also beats the odd-linear A2 query arm, reaching **`58.1%` Equiv-SAT accuracy** and **`78.5%` Sudoku accuracy** at B=16.

The claim that A4 has the highest accuracy of all four arms on both environments is contradicted by the supplied table: **A3 reaches `75.0%` on Equiv-SAT at B=2 and `72.4%` at B=32**. A4’s Equiv-SAT advantage is protection of variable beliefs and improvement over A2, not best raw query accuracy.

At B=32, A4 query KL is **`0.8735` on Equiv-SAT** and **`1.0219` on Sudoku**, failing `≤ 0.15`. The audit identifies a credible train–test mismatch: training exposes the query head to one-step sibling consensus, while inference accumulates auxiliary updates over multiple search steps.

That mismatch can produce correct classifications with excessive confidence on the remaining errors. It does **not**, by itself, prove that calibration explains all excess KL or that two parameters can meet the target. Section 3 gives the proper-score decomposition and the intervention needed to test that hypothesis.

#### (c) `R11-D6`: generalist efficiency wins coexist with specialist strategic advantages

The reported comparison favors the generalist on:

- Mean single-agent KL: **`0.5319` versus `0.7896`**, a **`32.6%`** reduction.
- Scaling slope: **`+0.7637` versus `+0.1881`**, approximately **`4.06×`**.
- Witness rate: **`+3.65` percentage points**.

It favors specialists on reported B=32 mean multiplayer Elo: **`1438.0` versus `1123.6`**.

The generalist receives **15 updates per environment**, compared with **30–60** for specialists. This is a genuine confound, not proof that more generalist updates will eliminate the difference. Negative transfer and task-specific representation demands remain possible.

The reported parameter comparison—**`2.04M` shared world-model parameters versus `24.43M` aggregate specialist parameters**—describes deployment storage, not a 12-fold per-decision model-size difference. The plan head and clearer must also appear in any full-system ledger.

Finally, **`AUROC(m_ql) = 1.0000` is tautological when its target is a median threshold of `m_ql` itself**. The supplied `m_qs = 0.8038` measures agreement with that derived target, not independently verified global solvability. Those semantic claims remain unvalidated.

## 3. Pre-Registration Specification for Round 12 (`R12-GPU-NEURAL-V2` on `NVIDIA L4`)

### Common protocol and mandatory integrity gate

The following is a proposed registration to freeze before execution.

Use one NVIDIA L4 and seeds **`101, 102, 103, 104, 105`**, with newly generated, disjoint train/calibration/test instances. Historical seeds aid comparison; they do not make reused R11 test instances unseen. Save generator versions, instance identifiers, split hashes, checkpoint hashes, configuration, runtime, and cost records.

Use paired comparisons across seeds and instances. Report individual seeds and 95% intervals; never treat budgets sharing a checkpoint as independent replicates. For seed-level mean differences, use

\[
\bar d \pm t_{0.975,4}\frac{s_d}{\sqrt5},
\qquad t_{0.975,4}\approx2.776.
\]

For matches, additionally resample paired openings and seat swaps within seed. Specify all primary contrasts before running; adjust multiple primary superiority tests with Holm’s procedure at `α=0.05`.

**R12-D0 is mandatory for every R12 claim:**

1. Run the static auditor over the full model/search dependency closure, new calibrator, modulation modules, and interactive policy adapters. Require zero violations and detection of all three original canaries; publish their exact fixtures.
2. Add fixtures for aliases, helper-mediated oracle reads, private metadata access, and premature scorer invocation. Require every registered fixture to be caught.
3. Require all three original `PoisonTensor` traps to fire. Poison oracle fields before every forward/search call, including calibration prediction generation.
4. With fixed parameters, randomness, and public inputs, require exact prediction/action/trace invariance under oracle-label permutation and zeroing across every environment, seed, and registered budget.
5. Require genuine parameter updates and the original training checks: parameter displacement `≥0.10` and training-loss reduction `>50%`. Separately report held-out improvement against both uniform and initialization; one reference cannot substitute for the other.
6. Require cached/uncached root agreement `≤10^-5`, plus agreement at selected deeper branches and sibling-order/cache-reuse checks.
7. Keep labels in a separate scoring process or capability-restricted container. Calibration labels may enter the calibration loss **after blind predictions return**; test labels may enter only post-prediction evaluation.

Calibration-label invariance is required for a **frozen calibrator**, not across refits: legitimate calibration must depend on calibration labels. A leakage-gate failure invalidates the affected result, regardless of its score.

### `R12-D1`: calibrated bilinear query readout on `split="cal"`

#### Exact proper-score decomposition

Let \(X\) denote an instance, \(P=p^\star(X)\in[0,1]\) its true Bernoulli query probability, and \(Q=q(X)\) the prediction. Define

\[
\eta(Q)=\mathbb E[P\mid Q],
\qquad
H(p)=-p\log p-(1-p)\log(1-p).
\]

The expected query KL decomposes exactly as

\[
\boxed{
\mathbb E\,D_{\mathrm{KL}}
\bigl(\operatorname{Ber}(P)\Vert\operatorname{Ber}(Q)\bigr)
=
\underbrace{\mathbb E\,D_{\mathrm{KL}}
\bigl(\operatorname{Ber}(\eta(Q))\Vert\operatorname{Ber}(Q)\bigr)}
_{\mathcal C:\ \text{calibration error}}
+
\underbrace{\mathbb E\,D_{\mathrm{KL}}
\bigl(\operatorname{Ber}(P)\Vert\operatorname{Ber}(\eta(Q))\bigr)}
_{\mathcal F:\ \text{refinement deficit}} .
}
\]

The identity follows by conditioning the cross-entropy on \(Q\); its dependence on \(P\) is linear. Equivalently, with \(\pi=\mathbb E[P]\),

\[
\mathcal F=\mathbb E[H(\eta(Q))]-\mathbb E[H(P)],
\]

\[
\mathcal R=H(\pi)-\mathbb E[H(\eta(Q))],
\]

\[
\boxed{
\mathrm{query\_kl}
=
\mathcal C+H(\pi)-\mathbb E[H(P)]-\mathcal R .
}
\]

Here \(\mathcal R\) is resolution. Calibration adjusts probability reliability; resolution measures information retained about the target.

For a fixed score, positive-temperature scaling preserves ranking and its information content. It can reduce calibration error without restoring distinctions the representation has lost. A fitted bias can change classifications; instance-dependent depth normalization can also change ranking and resolution.

**Consequently, the R11 accuracy/KL pairs do not identify \(\mathcal C\) and \(\mathcal F\).** They are compatible with overconfidence, but do not prove a calibration-only failure.

A useful falsifiability check follows for **hard binary targets**, where query KL equals log loss:

\[
\mathrm{query\_kl}\ge (1-\mathrm{accuracy})\log 2.
\]

At `58.1%` accuracy, unchanged classifications imply KL at least approximately **`0.2904`**, already above `0.15`. At the B=32 Sudoku accuracy of `77.8%`, the corresponding bound is approximately **`0.1539`**. Achieving KL `≤0.15` with hard targets requires accuracy of at least approximately **`78.36%`**, a necessary but insufficient condition.

If the benchmark uses soft target probabilities, this hard-label bound does not apply directly. R12 must publish that distinction.

#### Frozen depth normalization and two-parameter calibration

Normalize only the auxiliary world-state input to the query readout. Do not mutate the state used by variable-belief prediction, search selection, or policy execution.

Let \(a_d\) be the accumulated world-auxiliary vector after the \(d\)-th committed update, \(\delta_d=a_d-a_0\), and \(D_a\) its dimension. Define

\[
r_d=
\sqrt{
\epsilon+
\frac{1}{dD_a}\sum_{j=1}^{d}\|\delta_j\|_2^2
},
\qquad
s_d=\max\left(1,\frac{r_d}{r_{\mathrm{ref}}}\right),
\]

\[
\widetilde a_d=a_0+\frac{\delta_d}{s_d}.
\]

Use \(s_0=1\), \(\epsilon=10^{-8}\), and a single frozen \(r_{\mathrm{ref}}\) measured from one-step **training** traces. Distinguish committed-update count from actual branch depth and total search budget; log all three.

Recompute the existing bilinear query logit using \(\widetilde a_d\), then apply

\[
q_{\mathrm{cal}}
=
\sigma\left(\frac{\ell_q(\widetilde a_d)}{\tau_q}+b_q\right),
\qquad \tau_q>0.
\]

Fit **one shared pair** \((\tau_q,b_q)\) across the two primary environments and calibration budgets, with equal environment and budget weights. Do not introduce a separate temperature for each budget.

```python
class QueryCalibrator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.log_tau = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))

    def forward(self, frozen_logits):
        tau = self.log_tau.exp()
        return frozen_logits.to(torch.float64) / tau + self.bias


def normalize_aux_for_query(a0, a_now, sum_delta_sq, n_updates, r_ref):
    if n_updates == 0:
        return a_now
    rms = (sum_delta_sq / (n_updates * a_now.shape[-1]) + 1e-8).sqrt()
    scale = (rms / r_ref.clamp_min(1e-8)).clamp_min(1.0)
    return a0 + (a_now - a0) / scale
```

`sum_delta_sq` is accumulated per instance from squared norms of the **cumulative** deltas. All normalization statistics are detached. Fit with full-batch float64 LBFGS, maximum 200 iterations, `tolerance_grad=1e-9`, `tolerance_change=1e-12`, minimizing weighted `binary_cross_entropy_with_logits` against calibration targets. Freeze all model parameters and constrain temperature to `[0.05, 100]`; report boundary hits.

Calibration predictions must be generated from poisoned batches. Only the detached logits are paired with unpoisoned targets inside the permitted calibration-loss function.

#### Diagnostic arms and pass/fail bars

Run four paired arms: unchanged A4, temperature/bias only, normalization only, and normalization plus temperature/bias. Preserve A1–A3 as references.

Estimate calibration and refinement with cross-fitting. Publish binning/smoothing choices and uncertainty. Finite-bin estimates are approximations: report coarsened-score decomposition and residuals rather than presenting them as the exact continuous population decomposition.

**Primary efficacy PASS requires:**

- At B=32, the upper 95% seed-level confidence bound on mean query KL is **`≤0.15` separately for Equiv-SAT and Sudoku-6x6**.
- D0 passes.
- Variable-belief predictions and search traces remain bitwise unchanged by this readout-only intervention.
- Query accuracy decreases by no more than one percentage point relative to paired uncalibrated A4.

**The stronger “predominantly calibration” diagnosis passes only if** the temperature/bias-only arm removes at least 80% of the estimated calibration component and the estimated refinement deficit is compatible with attaining `0.15`. If the absolute bar fails, report whether calibration improved, refinement remained limiting, or the two-parameter family was insufficient. Do not relabel a lower KL as full success.

### `R12-D2`: iso-update, game-modulated generalist versus specialists

#### Training and modulation

Assign **60 optimizer updates per environment** to both systems, with matched batches, examples, loss reductions, and learning-rate schedules. The generalist therefore receives 720 environment-specific updates over 12 environments. A specialist serving two environments receives 120 updates, balanced equally between them.

Use four arms:

1. Original 15-update generalist.
2. Iso-update generalist.
3. Iso-update generalist with header-conditioned FiLM.
4. Iso-update specialists.

This separates update allocation from modulation. Log actual optimizer steps, examples, tokens, gradient accumulation, wall time, and complete parameter counts. Iso-update equality is not automatically FLOP equality.

Use FiLM as the primary lightweight intervention:

\[
h'_{\ell}
=
\left(1+\gamma_\ell(e)\right)\odot h_\ell+\beta_\ell(e),
\]

where \(e\) is a public environment header. For 12 headers, four layers, and width 128, a direct embedding table has

\[
12\times4\times2\times128=12{,}288
\]

parameters, below 2% of the reported world-model count. Initialize \(\gamma,\beta\) to zero.

Implement an `nn.Embedding(12, 4 * 2 * 128)`. Keep world-belief modulation in the supervised parameter group and exclude it from policy-gradient updates. Utility inputs must not enter belief modulation. Rerun the gradient and post-RL drift firewall checks.

FiLM is the registered primary choice; LoRA may be a separately registered secondary arm, not a post-result substitution. No environment-specific search thresholds are permitted.

#### Pure interactive league

For each strategic environment, execute a complete match:

\[
o_t^i
\longrightarrow
\pi_i(o_t^i,B)
\longrightarrow
a_t^i
\longrightarrow
s_{t+1}
\longrightarrow
\text{terminal payoff}.
\]

A policy receives only legal observations and public history. Hidden state remains inside the simulator. Record actions, seats, terminal outcomes, termination reasons, and replay hashes.

Require **200 paired openings/deals per seed and primary comparison**, each played with seats swapped: 400 games. Freeze openings, opponents, terminal-payoff conversion, and draw handling before evaluation.

Maintain separate datasets and ratings:

- **Interactive Elo:** terminal outcomes from played matches.
- **Belief-payoff ratings:** the R11-style derived score, clearly labeled.
- **Belief quality:** KL, Brier, and witness measures.

Never add belief-derived edges to the interactive league. If a game lacks a complete interactive implementation, mark the five-game claim incomplete rather than substituting belief payoffs.

Fit interactive ratings using a fixed-anchor Bradley–Terry/Elo model:

\[
P(i\text{ beats }j)
=
\frac{1}{1+10^{-(E_i-E_j)/400}},
\]

with the registered draw treatment and paired-seat analysis.

**Primary PASS requires:**

- Exact update/exposure matching and FiLM overhead `<2%`.
- Full D0 and firewall passes.
- At B=32, the lower 95% bound on the equal-game mean interactive Elo difference, modulated generalist minus specialist, exceeds **`−50` Elo**.
- No game has a point deficit worse than **`−100` Elo**.
- The upper 95% bound on generalist-minus-specialist single-agent KL is `≤0`.

Call this **non-inferiority**. Claim superiority only if the interactive Elo difference’s lower bound exceeds zero.

For independent cross-domain validation, correlate a frozen single-agent-only Mini-ECI score with held-out interactive Elo. Register `r≥0.60` as the point-estimate target and require its confidence interval to exclude zero. Do not include the evaluated interactive outcomes in the predictor score.

#### Independent labels for validity and solvability

Replace self-thresholded targets with evaluator-generated labels:

\[
Y_{\mathrm{local}}
=
\mathbf1\{\text{candidate partial assignment violates no constraint}\},
\]

\[
Y_{\mathrm{extendable}}
=
\mathbf1\{\text{candidate has at least one legal completion}\}.
\]

Use independently implemented constraint checks and, where available, an exact solver. Matching one sampled solution is not equivalent to validity when multiple solutions exist.

Evaluate `m_ql` against \(Y_{\mathrm{local}}\) and `m_qs` against \(Y_{\mathrm{extendable}}\). A separate usefulness label may be defined as

\[
Y_{\mathrm{improve}}
=
\mathbf1\{
\mathrm{KL}_{\mathrm{after}}
<
\mathrm{KL}_{\mathrm{before}}-10^{-4}
\},
\]

but must be reported as improvement prediction, not solvability.

Freeze score direction before evaluation. Require lower 95% AUROC bounds of **`0.85` for `m_ql`** and **`0.75` for `m_qs`** on their respective external targets. Single-class samples are undefined evaluations, not passes. Labels are generated only after candidate scores and predictions are frozen.

### `R12-D3`: ultra-deep KV-tree search through B=256

Run

\[
B\in\{0,1,2,4,8,16,32,64,128,256\}
\]

on the NVIDIA L4, recording node expansions, realized depth, KV pages, latency, and peak allocated/reserved VRAM. **B=256 is a search budget, not necessarily tree depth 256.**

Define \(T_b\) explicitly as the actual linear error-propagation operator of the clearing step. For a preconditioned iteration,

\[
x^{(k+1)}=x^{(k)}+P_b(c_b-A_bx^{(k)}),
\qquad
T_b=I-P_bA_b.
\]

Measure the induced infinity norm:

\[
\|T_b\|_\infty
=
\max_i\sum_j |(T_b)_{ij}|.
\]

Then

\[
\rho(T_b)\le\|T_b\|_\infty<1
\]

certifies contraction of that fixed linear update. It does not establish monotonic KL, calibrated probabilities, or global contraction of an unexamined nonlinear search map.

If the clearer is nonlinear, publish its linearization and any state dependence. A local Jacobian bound must be labeled local. Persistent bounded inputs and a uniform contraction margin are needed for a corresponding bounded-trajectory argument.

Implementation requirements:

- Measure every clearing update, not only the last update at B=256.
- Compute diagnostic row sums in float64 on GPU; do not use elementwise maximum as a substitute.
- Save the exact operator or a reproducible compact representation for worst-case traces.
- Check all states and outputs with `torch.isfinite`.
- Preserve sibling cache isolation and compare selected deep branches against uncached execution.
- Report numerical margin \(1-\|T_b\|_\infty\).
- If damping is introduced, apply one environment-independent rule, preregister it, and log every intervention.

For example, the norm calculation is:

```python
row_sums = T.detach().to(torch.float64).abs().sum(dim=-1)
norm_inf = row_sums.amax(dim=-1)
```

**Stability PASS requires:**

- \(\max_{\text{seeds, instances, steps}\le256}\|T_b\|_\infty<1.0\), including the requested \(\|T_{256}\|_\infty<1.0\).
- No NaN, infinity, out-of-memory failure, or silent budget reduction.
- Deep cache equivalence `≤10^-5` and full D0 compliance.

**Useful-extension PASS additionally requires:**

- A positive lower 95% bound for paired `Mini-ECI(256) − Mini-ECI(128)`.
- Positive fitted scaling slope, using a preregistered fit on `B≥1`; report B=0 separately.
- Non-negative point change in witness discovery from B=128 to B=256.
- Upper 95% bound on mean `KL(256) − KL(128)` of at most `0.015`.

Strict anytime monotonicity is a separate claim requiring every registered adjacent-budget contrast to be non-decreasing; a positive regression slope is insufficient.

Repeat externally labeled `m_ql`/`m_qs` evaluation across depth strata. Compute Gram crowding identically for every method:

\[
\bar\rho(K)
=
\frac{1}{n(n-1)}
\sum_{i\ne j}
\left|
\frac{k_i^\top k_j}{\|k_i\|_2\|k_j\|_2}
\right|.
\]

Publish both edge-bank and root-bank results, bank sizes, and zero-vector handling. A missing or undersized bank is `N/A`, never an assigned favorable constant.

**Required artifacts:** the R12 driver, static-audit fixtures, frozen registration, per-seed configuration/checkpoint manifests, calibration parameters, blind prediction records, post-hoc external labels, interactive match replays, and GPU ledger. Every resulting table must identify its generating command and seed-record paths.

## 4. 3-Round Trajectory Rubric Table (`D1`–`D6` on 1.0–10.0 Scale)

These are **Member 4b’s charter-dimension judgments**, not the similarly numbered experimental gates. Earlier columns reproduce the supplied personal scores. R3 projections are conditional assessments if the proposed R12 tests succeed; they are not measured results or guaranteed forecasts.

| Charter dimension | R1 Proj | R2 Math/Unit | R2 Blind-Emp | R2 Proj-R11 | **R3 Realized-R11** | **R3 Proj-R12** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **D1. Universal KV-cache tree machine** | 9.2 | 8.5 | 2.0 | 8.5 | **8.4** | **9.0** |
| **D2. Active fractal prospectus plans** | 8.3 | 7.5 | 1.5 | 8.0 | **8.0** | **8.7** |
| **D3. Bidirectional latent lemma space and queries** | 8.4 | 8.0 | 1.5 | 8.5 | **7.2** | **8.8** |
| **D4. Present-time episodic memory and LIR firewall** | 8.5 | 7.5 | 1.5 | 8.0 | **8.5** | **8.9** |
| **D5. Anytime monotonicity and self-consistency** | 8.0 | 7.5 | 1.0 | 8.0 | **7.4** | **8.7** |
| **D6. Multi-game Elo league and Mini-ECI bifactor IRT** | 8.5 | 8.0 | 1.5 | 8.5 | **6.8** | **8.5** |
| **Unweighted mean** | **8.48** | **7.83** | **1.50** | **8.25** | **7.72** | **8.77** |

The realized scores recognize the genuine neural execution and blind-search recovery while retaining deductions for specific unresolved claims:

- **D1:** credible tree execution and root-cache equivalence; deeper correctness and broader generality remain.
- **D2:** strong steering interventions; limited independent evidence of general strategic planning.
- **D3:** parity-preserving auxiliary structure and variable-belief protection; failed query KL and unresolved refinement limits.
- **D4:** strong firewall interventions; less direct evidence for the broader episodic-memory claim.
- **D5:** positive aggregate scaling and measured contraction; observed regressions and self-labeled validity evaluation.
- **D6:** strong within-suite factor structure; mixed interactive/derived tournament outcomes, part–whole alignment, and unequal training exposure.

**R11 earns confidence in the experimental foundation. R12 should earn confidence in the disputed mechanisms: whether calibration suffices, whether equal training supports interactive generalist competence, and whether deeper search remains both stable and useful under independent evaluation.**