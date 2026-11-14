# Which simple questions reveal a forecaster's dependency structure fastest? (Qwen3-32B, 2026-09-24)

Question (Kelvin, 24 Sep): can we recover the dependency structure of Qwen3-32B's beliefs by asking
only simple questions (P(A), P(A and B), P(B | A), "which of these would change your forecast of A?"),
picking at each step the questions that most reduce our uncertainty about the structure, and how
many questions does that save? Literature: `docs/active-structure-elicitation-lit-20260924.md`.

**Answer.** Yes for edges. Expected-information-gain (EIG) selection over pairwise questions
recovers the strong dependencies with about half the questions that random pairs need:

- EIG on pairs only (`eig_noscr`) reaches edge AUC 0.9 after **57** questions [95% CI 42–64].
  Random pairs need **114** [98–132], similarity-ranked pairs 91, and screening lists first 137.
- The dense fit uses 170 questions per 12-event set.
- On the 12 sets built on the global run's events and date the counts are 50 / 95 / 97 / 139.

It does not help held-out answer prediction. Random or similarity-ranked pairs reach 90% of the
dense fit's held-out gain in about 90–105 questions, and EIG takes 140 or more. Edge-EIG buys
conditionals and never pays to learn the AND bias offset (below).

The screening question ("which of these would change your forecast of A?") is weak evidence.
The model lists 50% of strong partners and 27% of the rest. Asking it first wastes 12 calls, and
EIG with screening available is worse than EIG without it.

## Data (one dense query pool; "asking" = reading a stored answer)

- Model: `qwen3-32b` on DashScope, thinking off, temperature 0, max_tokens 20 (60 for screens).
  Every call was served by qwen3-32b, with no substitution, 0 missing answers and 0 parse retries.
- Prompt: v6_direct, with one query per call and as-of-t text, the same as
  `docs/worldmodel-clusters-20260924.md`. No market data.
- **Main: 24 sets of 12 binary events.** All are test-split parents and are open at t (t = the
  seed's closing week − 7 days).
  - Each set has 8 Manifold-group mates plus 4 unrelated fillers.
  - 6 random sets calibrate the noise model: s00, s04, s17, s20, s21, s22.
  - The other 18 sets are for evaluation.
- **Aligned: 12 sets on the brute-force global run's events and date.** The source is
  `/home/agent/data/worldmodel-global-20260924`, with t = 2025-09-15.
  - Each set is a seed plus its 7 nearest BGE neighbours plus 4 random events.
  - These sets are evaluation only and use the main calibration.
  - The global run's pool is not dense (one AND per sampled pair, no conditionals), so it cannot
    serve as a dense reference itself.
- Per set: 12 singles, all 66 ANDs, all 132 conditionals ("Suppose A resolves YES. What is the
  probability that B resolves YES?") and 12 screening lists.
  - 12 random 3-way ANDs are held out and never offered to the selectors.
  - A fixed 20% of the pair questions are also held out, which leaves 158 askable pair questions.
- 8,436 calls in all (5,616 main and 2,808 aligned), **$3.81** at the capped price (0.7/2.8 $/M).
- Determinism check: 261 aligned prompts are byte-identical to prompts in the global run. Only
  178 of the 261 got the same answer at t=0 (per-call sd 0.29 logit), so DashScope at t=0 is not
  deterministic. The simulation still treats each question as askable only once.

## Method (`scripts/active_structure.py`)

- **Belief model.** An Ising / log-linear model over the 2^12 = 4,096 states (exact enumeration):
  log p(x) = Σ h_i x_i + Σ J_ij x_i x_j.
- **Answer model.** logit(stated) = logit(implied by the model) + b_kind + ε, with ε ~ N(0, σ_kind²).
  - The offset b_kind is learned per query type, with prior sd 0.3 around the calibrated value.
  - Singles fix the level (b = 0).
- **Calibration** on the 6 calibration sets (dense fit, residuals):
  - σ is 0.59 (single), 0.59 (AND) and 0.53 (conditional) logit.
  - The AND offset b_and = **+0.54 logit**, so ANDs are too high. This matches the +0.05 to +0.09
    probability residuals in `docs/averaging-vs-structure-20260924.md`. There, sampling sd is 0.47
    and incoherence sd 0.86 logit.
  - The conditional offset is b_cond = +0.18.
  - Strong edges (|J| ≥ 0.7) make up 8.8% of pairs (π0). Slab sd 1.36, spike sd 0.29.
- **Edge prior.** Spike-and-slab on J_ij. The Laplace fit uses the moment-matched Gaussian, and the
  per-edge P(strong) comes from the Laplace marginal (data message = posterior / prior).
- **Posterior.** MAP by L-BFGS, with the Laplace covariance from the Gauss-Newton Hessian.
- **Screening model.** P(listed | strong) against P(listed | not strong) is 0.34/0.16 for lists of
  3 or fewer, 0.72/0.46 for lists of 4–7, and 0.67/0.70 for longer lists. A screen updates π_ij.
- **Selectors.** Each round asks a batch of 20, refits, and repeats. Every method first asks the 12
  singles, and they are counted.

  | selector | how it picks questions |
  |---|---|
  | `random` | uniform over the unasked pair questions (3 seeds) |
  | `similar` | pairs by BGE cosine, each pair's AND then both conditionals |
  | `screen` | all 12 screening lists first, then flagged pairs (2 flags, then 1), then random |
  | `eig` | greedy batch maximising the expected drop in edge-existence entropy Σ_e H(z_e) − E[H(z_e \| answers)] (linear-Gaussian preposterior, Gauss-Hermite); the batch covariance is updated after each pick, so near-duplicates lose value; screening lists are a query type with their own likelihood |
  | `eig_noscr` | the same, pair questions only |
  | `eig_pred` | the second objective: information about the answers to all 220 3-way ANDs |

- **Reference.** The dense fit on all 170 non-held-out numeric questions, which is the model's own
  beliefs and not the outcomes. Strong edges are |J_ref| ≥ 0.7 (odds ratio ≥ 2). The main sets
  have 52 of 1,188 pairs above this threshold, and the aligned sets 29 of 792.
- **Metrics.**
  - Edge AUC of P(strong) against the strong reference edges, pooled over sets. The dense fit's
    AUC is 1 by construction, so "90% of dense" means AUC 0.9.
  - Spearman correlation of |J| with |J_ref|.
  - Held-out answer KL, with the stated p as a soft label. It runs from 0.080 (singles only) to
    0.045 (dense) on the main sets.
  - Log loss against real outcomes of the fitted model's coherent beliefs, for singles, pair ANDs
    and 3-way ANDs.
  - CIs are from a bootstrap over sets (300 resamples).

## Results: queries needed per method

Queries to reach 90% of the dense fit. The dense fit uses 170 numeric questions, or 182 with the
screens. The main columns are 18 sets and the aligned columns are 12 sets.

| method | edge AUC 0.9 (τ 0.7), main | aligned | edge AUC 0.9 (τ 0.5), main | held-out KL, 90% of dense gain, main | aligned |
|---|---|---|---|---|---|
| random pairs | 114 [98–132] | 95 [80–116] | 119 | 104 [64–146] | 91 [31–130] |
| similarity (BGE top pairs) | 91 [78–121] | 97 [38–138] | 108 | **89** [64–130] | **77** [36–123] |
| screening lists, then flagged pairs | 137 [122–170] | 139 [79–151] | 147 | 119 [91–143] | 118 [33–172] |
| EIG (edges) + screening | 79 [50–89] | 79 [47–91] | 95 | 141 [81–172] | 171 [69–180] |
| **EIG (edges), pairs only** | **57 [42–64]** | **50 [39–73]** | **69** | 138 [114–161] | 147 [66–167] |
| EIG (3-way answers) | 87 [70–98] | 68 [60–79] | 95 | 99 [72–147] | 125 [25–158] |

### Curves on the main sets

Edge AUC (τ 0.7), then held-out KL, then singles log loss against outcomes:

| n queries | 32 | 52 | 72 | 92 | 112 | 152 |
|---|---|---|---|---|---|---|
| random AUC | .69 | .78 | .82 | .85 | .90 | .98 |
| similar AUC | .71 | .76 | .83 | .90 | .94 | .97 |
| screen AUC | .66 | .70 | .73 | .75 | .81 | .93 |
| eig AUC | .70 | .80 | .88 | .94 | .98 | .99 |
| eig_noscr AUC | **.80** | **.88** | **.97** | **.99** | 1.00 | 1.00 |
| eig_pred AUC | .60 | .66 | .85 | .92 | .98 | 1.00 |
| random KL | **.063** | **.054** | **.051** | .049 | .048 | .045 |
| similar KL | .069 | .062 | .052 | **.048** | .047 | .045 |
| eig_noscr KL | .070 | .060 | .058 | .055 | .054 | .046 |
| random singles LL | .590 | .571 | .555 | .549 | .550 | .549 |
| eig_noscr singles LL | .612 | .584 | .555 | .544 | .539 | .542 |

Paired against random at the same n (main): eig_noscr's edge AUC is higher by +0.09 [0.05, 0.15]
at 52 questions and +0.16 [0.13, 0.19] at 72. Its held-out KL is worse by +0.006 [0.001, 0.011].
Full curves, CIs and paired differences are in the chart JSON.

**Real outcomes.** The fitted coherent beliefs beat the raw answers. On the evaluation sets the
raw singles log loss is 0.661. After the dense fit it is 0.548, with pair ANDs at 0.218 and
3-way ANDs at 0.076.

Every method except screening gets within 0.01 of the dense fit's singles log loss by about 90
questions. EIG on pairs slightly beats the dense fit at 112 questions (0.539 against 0.548), which
is within noise.

## Reading

1. **Conditionals are the informative question.** Edge-EIG spends over 90% of its pair budget on
   conditionals: at 92 questions, eig_noscr has asked 73 conditionals and 7 ANDs.
   With the marginals known, P(B | A) against P(B) measures dependence directly. P(A and B)
   confounds dependence with the AND bias (+0.54 logit).
2. **Batch EIG avoids redundancy.** The per-pick covariance update is what makes the gain real,
   as BatchBALD warns: after a pair is asked, its reverse conditional and its AND lose value.
3. **Screening lists are not worth their cost here.** The model often says NONE (70 of 288 lists)
   or lists nearly everything, so the likelihood ratio is about 2. My EIG overvalued them: it
   bought all 12 in round 1, and it scored screen information on the pre-batch edge probabilities.
   This cost it 22 questions against the pairs-only version.
4. **Different objectives buy different questions.** Held-out answer prediction is dominated by the
   levels and the AND offset, which broad coverage (random or similarity) learns early. Edge-EIG
   ignores the offset, because it is not an edge.
   - A deployable selector should put the bias offsets (and, for joints, the eig_pred term) into
     the objective. The next step is a mixed objective: edges + offsets + 3-way predictions.
   - Noise and bias are handled explicitly, as the literature recommends. Without b_and, every
     "AND too high" answer would look like a positive edge.
5. **Similarity is a good cheap prior, not a selector.** BGE similarity helps early on the aligned
   sets (AUC .87 at 52 questions against .78 for random). On the main sets, where group mates are
   loosely related, it is no better than random.

## Caveats

- The reference is the dense fit of the same model family, so "edges" means dependence in the
  model's stated beliefs. It does not mean causal edges or outcome correlations. Conditionals are
  observational (see the literature note).
- About 3 strong edges per set is thin. The per-set AUC is noisy, so the AUCs are pooled over sets.
- The held-out KL range is small (0.080 → 0.045), and its CIs overlap across methods.
- t=0 answers are treated as single draws. Repeats would add information (sd 0.29 logit), but
  only against noise, not against the bias.

## Files

- Code: `scripts/active_structure_collect.py` (pool builder and runner) and
  `scripts/active_structure.py` (calibrate, evaluate, report).
- Data: `/home/agent/data/active-structure-20260924/`. It holds the sets, queries, log,
  calibration.json, eval.json, summary.json, report.txt and bge-cosine.json. The `aligned/`
  folder has the same files.
- Chart JSON: `docs/active-structure-20260924-charts.json` (main) and
  `docs/active-structure-20260924-aligned-charts.json`.
