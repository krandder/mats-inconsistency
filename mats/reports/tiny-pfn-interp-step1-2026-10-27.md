# Tiny-PFN interpretability, step 1 "Locate" (2026-10-27, 08:05Z)

**TL;DR.** In all 5 main seeds, reliability is computed per source at the source's query-question token (QQ) in
layer 1 and read out by the query token in layer 2.
- **Layer-1 attention groups by source.** Each source's tokens attend to that source's labelled tokens about 100× more per token than to other sources' tokens.
- **attn1 → mlp1 builds the per-source quantities.** The layer-1 attention weights are tilted by the key token's residual. log SS_s is decodable at QQ after attn1 and is strong after MLP1 (R² 0.89). MLP1 also gives the variance posterior and the expected precision E_s (R² 0.88).
- **The layer-2 readout is a precision-weighted average.** The query token's layer-2 attention over the three QQ tokens tracks the Bayes precision weights β_s (r = 0.74 and 0.82 for the two heads; untrained ≈ 0).
- **Weighted mean and variance first appear at attn2.** The reliability-weighted part of the Bayes mean and the predictive variance cannot be decoded at the query token before attn2.
- **The plateau-stuck network does none of this.**

No decision needed. Steps 2–3 are not run.

## Setup (PREREG.md, written 07:50Z before any probe was run)
- **Networks:**
  - main: w32_h2_L2 with the warm-started mixture head, seeds 0–4
  - small: w16_h2_L2 at R⁴, seeds 2 and 4
  - plateau: w32_h2_L2_mix4_s3 (cold start, gap 0.144)
  - an untrained control for each, built under the same `torch.manual_seed`
- **Reproduction:** recomputed NLL gaps match the eval.json files (main 0.0056–0.0110).
- **Data:** the fixed test set, split by episode: 12,000 episodes for probe training, 4,000 held out.
- **Probes:** ridge regression, with λ chosen by 5-fold CV grouped by episode.
- **Baselines:** the untrained network, and shuffled targets (shuffled-target R² is at most 0.014 in every cell).
- **POSITIVE (per seed):** R² ≥ 0.5 and at least 0.10 above both baselines. The main model needs this in ≥ 4 of 5 seeds; the small model in both seeds.
- **Token types:**
  - LAB: labelled tokens, 12 per episode
  - QQ: the query question's token for each source, 3 per episode
  - QT: the query token
- **Per-source targets** are probed at the source's own tokens.
- **Dead sites:** context tokens after attn2 and mlp2 are never read by the output.

## Where each intermediate first appears (main model; small model in brackets)

Cells give mean held-out R², trained (untrained). * = POSITIVE; ** = POSITIVE and R² ≥ 0.8.

| Intermediate | Token | emb | attn1 | mlp1 | attn2 | First appears |
|---|---|---|---|---|---|---|
| (1) residual r (own token) | LAB | 1.00 (0.99) | 1.00 | 0.99 | 0.99 | input; linear in raw features, so not learned |
| (1) r² (own token) | LAB | 0.00 | 0.29 (0.01) | 0.89** (0.10) | 0.86** | mlp1 [mlp1] |
| (1) log SS_s | QQ | 0.00 | 0.57* (0.01) | 0.89** (0.02) | 0.89** | attn1 [attn1] |
| (1) log SS_s | LAB | 0.00 | 0.32 | 0.63* | 0.67* | mlp1 [mlp1] |
| (2) p_s, labelled-only variance posterior (mean over its 3 components) | QQ | 0.00 | 0.35 | 0.76* (0.01) | 0.77* | mlp1 [mlp1] |
| (2) E_s, expected precision | QQ | 0.00 | 0.46 | 0.88** (0.02) | 0.88** | mlp1 [attn1] |
| (2) E_s | LAB | 0.00 | 0.25 | 0.57 | 0.61* | attn2 [mlp1] |
| (2) query-reading coupling of the posterior, E_bayes − E_lab † | QQ | 0.00 | 0.01 | 0.25 | 0.31 | not found [not found] |
| (3) β_s, Bayes weight on o_s* | QQ | 0.00 | 0.36 | 0.64* (0.02) | 0.80* | mlp1 [mlp1] |
| (3) entropy of the 27-way weights | QT | 0.00 | 0.11 | 0.14 | 0.46 (0.53 at mlp2) | not found [not found] |
| (4) Bayes mean | QT | 0.00 | 0.89** (0.30) | 0.90** | 0.98** | attn1 (but see the next row) |
| (4) equal-weight mean of the query readings † | QT | 0.00 | 0.99* (0.32) | 0.99* | 0.99* | attn1 |
| (4) Bayes mean minus equal-weight mean † | QT | 0.00 | 0.01 (0.00) | 0.05 | 0.81* (0.00) | **attn2** [attn2] |
| (4) Bayes log variance | QT | 0.00 | 0.19 | 0.23 | 0.88** (0.04) | attn2 [attn2] |

† Added post hoc after the pre-registered tables, as controls (`extra.py`). The pre-registered "Bayes mean at attn1" result is almost entirely the equal-weight average of the query readings, which the query token collects in layer 1. The reliability-specific part first appears at attn2.

Full-Bayes and labelled-only per-source posteriors give the same R² to within 0.02. The part the query readings add on top of the labelled questions is not represented per source (R² ≤ 0.33).

**Plateau network** (query-only behaviour):
- At QQ, no per-source quantity reaches the threshold: log SS 0.36, E_s 0.35, β_s 0.19.
- At QT, the Bayes-minus-equal-weight mean reaches 0.46 at most.
- r² is present at LAB tokens after mlp1 (0.89), but it is never pooled per source.

Full tables are in `work/tiny-pfn-interp/tables.md` and `tables_extra.md`.

## Attention

**Layer 1 groups tokens by source.** The table shows per-token attention to own-source labelled tokens divided by per-token attention to other sources' labelled tokens, from QQ tokens. Untrained ratios are 0.99–1.01 throughout.

| Network | L1H0 | L1H1 | Layer 2 (both heads) |
|---|---|---|---|
| main s0 / s1 / s2 / s3 / s4 | 1.3 / 277 / 94 / 104 / 267 | 107 / 269 / 1.4 / 1.2 / 299 | 0.9–1.1 |
| small s2 / s4 | 124 / 126 | 116 / 113 | 1.0 |
| plateau | 1.1 | 1.2 | 1.0 |

The pre-registered per-head rule passes for L1H0 (4/5 seeds) and fails for L1H1 (3/5). Head indices are arbitrary, though, and every main seed has at least one head that groups by source.

There are two motifs:
- **Two source-pooling heads:** main seeds 1 and 4 and both small seeds. The two heads' weights over own-source labelled keys are tilted in opposite directions by the key's signed residual (r(attention, r_key) ≈ +0.73 in one head and −0.73 in the other; untrained ≈ 0). The difference between two oppositely tilted softmax averages is a second-moment term, which is where SS can come from inside attention.
- **One source-pooling head plus one head that averages the three query readings uniformly** (self 0.33, others 0.66): main seeds 0, 2 and 3. These include the two worst seeds (0.011).

**Layer 2 is the readout.** The query token puts 0.66–0.69 of its attention on the QQ tokens. Its per-source mass correlates with the Bayes weights β_s:

| Head | Main: r(QQ mass, β), per seed | Small | Untrained | Plateau |
|---|---|---|---|---|
| L2H0 | 0.50, 0.94, 0.93, 0.33, 0.97 | 0.97, 0.96 | ≈ 0.00 | 0.39 |
| L2H1 | 0.88, 0.87, 0.65, 0.86, 0.83 | 0.74, 0.76 | ≈ −0.01 | −0.04 |

Both main heads pass the rule (4/5 and 5/5 seeds). Normalising within each episode does not change r. The correlation with E_s alone is lower (0.58–0.63), which fits the softmax normalising across sources, as β does. The query token's layer-1 attention is uncorrelated with β (|r| ≤ 0.13).

## Symmetry
- **Token-permutation equivariance:** max abs difference 1.5e-5 at the sites and 1.5e-6 at the head. Pass.
- **Swapping identity vectors between sources (readings kept in place):**
  - Probes for SS, E_s and β_s at their first site keep R² to within 0.01 in all 8 networks.
  - Scored against the value of the source whose identity vector the token now carries, R² is between −0.15 and −1.24.
  - So the representations follow the readings and permute with the sources. Pass.
- **Finding for step 4 (diagnostic):** the predictive mean depends noticeably on the random identity vectors, which Bayes ignores.
  - Redrawing all identity vectors moves the mean with a per-draw sd of 0.019–0.029 (main), against an sd of 0.040–0.056 for model − Bayes.
  - That noise is 22–33% of the variance of the mean error in the main model and 34–49% in the small model.
  - Part of the deviation is therefore crosstalk between identity vectors, not reliability estimation.

## Algorithm hypothesis (for steps 2–3; not yet tested by intervention)
1. **Layer 1, attention.** Each source's QQ token, and to a lesser degree its LAB tokens, pools over that source's labelled tokens. It finds them by matching the source-identity vector, with softmax weights tilted by the key's residual. That puts a within-source second moment of the residuals into the residual stream.
2. **Layer 1, MLP.** It turns that moment into log SS_s and an expected precision E_s at the QQ token, which also holds the source's own query reading o_s*.
3. **Layer 2, attention.** The query token attends to the three QQ tokens with logits that increase with E_s. The softmax over sources does the normalisation, so the weights approximate β_s, and the values carry o_s*. That yields the reliability-weighted mean correction on top of the equal-weight average the query token collected in layer 1. The total attended precision gives the predictive variance.
4. **Layer 2, MLP and head.** They shape the mixture.
5. **Query-reading coupling.** How much the query readings change each source's reliability is not represented per source. If it exists, it is in the layer-2 query-token computation. That fits where the KL concentrates.

Tests for steps 2–3:
- patch E_s at the QQ token after mlp1
- replace the query token's layer-2 attention with β_s
- ablate one of the two oppositely tilted layer-1 heads

## Files (`/home/agent/workspaces/mats/work/tiny-pfn-interp/`)
- `PREREG.md`
- `locate.py`: probes, attention and symmetry
- `extra.py`: post-hoc controls
- `summarize.py` and `summarize_extra.py`
- `tables.md` and `tables_extra.md`
- `out/*.json`: raw results

tiny-pfn/ is unchanged.

---

# Steps 2–4: intervene, replace, identity vectors (2026-10-27, 08:35Z)

**TL;DR.**
- **(2) Intervene.** Patching only a source's query-question (QQ) token state after MLP1 reproduces the model's own response to changing that source's labelled residuals. It does this exactly in 4 of the 7 near-Bayes networks (shift slope 1.00, R² 0.96 against the Bayes shift) and partly in the three seeds that use the other layer-1 motif (slope 0.68–0.84).
- **(3) Replace.** All the pre-registered replacements fail, and so does the hand-set uniform layer-1 pattern. Replacing the query token's layer-2 weights over sources with the Bayes β_s retains 55–88% of performance. Post hoc, one hand-set layer-1 attention pattern reproduces the model's NLL to within 0.001 nats in 6 of 7 networks: within a source's labelled tokens, weights = softmax(c₁r + c₂r²), where r is each labelled token's residual and c₁, c₂ are fitted per head (they come out at ±1.6 and −0.4).
- **(4) Identity vectors.** Averaging the predictive over 16 resampled ID draws removes 22–28% of KL in the main model. That is just under the pre-registered 25% threshold, so the rule fails in 4 of 5 seeds. It also removes 25–32% of the mean-error variance.
- **Caveat on the replacements.** The best combined replacement keeps 80–90% of performance, but it still costs 0.02–0.09 nats more than the model, which is 2–10× the model's own gap to Bayes. So these replacements describe how the model approximates Bayes, but they are not yet precise enough to explain its errors.

No decision needed.

**Setup.** PREREG.md Amendment 1 was written at 08:02Z, before any result. Amendment 1b was added at 08:20Z, post hoc, after seeing only small_s2's step-3 scores. Code is in `steps.py`; full tables are in `tables_steps.md`.

## Step 2: patches between episodes that differ only in one source's labelled residuals

There are 4,000 held-out pairs. Each cell gives the slope and R² of the model's shift regressed on the Bayes shift, for the mean.

| Network | Full swap | (a) QQ state after MLP1 | (b) layer-1 attention row of QQ only | (c, extra) that source's labelled tokens after MLP1 | (R) random direction, same norm | Untrained, full swap |
|---|---|---|---|---|---|---|
| main s0 | 0.95, 0.92 | 0.69, 0.74 | 0.34, 0.16 | 0.26, 0.36 | 0.13, 0.11 | 0.00 |
| main s1 | 1.01, 0.96 | 1.01, 0.96 | 0.31, 0.18 | 0.00, 0.00 | 0.19, 0.06 | 0.00 |
| main s2 | 0.95, 0.94 | 0.84, 0.89 | 0.60, 0.43 | 0.06, 0.10 | 0.19, 0.17 | 0.00 |
| main s3 | 0.97, 0.93 | 0.68, 0.73 | 0.34, 0.14 | 0.27, 0.36 | 0.13, 0.11 | 0.00 |
| main s4 | 1.00, 0.97 | 1.00, 0.96 | 0.30, 0.16 | 0.00, 0.01 | 0.21, 0.11 | 0.00 |
| small s2 / s4 | 0.98, 0.94 / 0.97, 0.92 | 0.98, 0.95 / 0.98, 0.93 | 0.31, 0.16 / 0.30, 0.17 | 0.00 / 0.00 | 0.21, 0.09 / 0.20, 0.08 | 0.00 |
| plateau | 0.08, 0.04 | 0.00, 0.03 | 0.00, 0.01 | 0.07, 0.04 | 0.00, 0.00 | 0.00 |

The log-variance shift behaves the same way: (a) gives slope 0.93–0.96 and R² 0.93–0.96 in the seeds where (a) carries everything.

**(a) The QQ state after MLP1 is the bottleneck** in main s1 and s4 and in both small seeds: R² 0.99–1.00 against the model's own full-swap shift. In s0 and s3, the seeds with one pooling head plus a head that mixes the query readings, the source's labelled tokens carry part of the effect (c: R² 0.36), and so does (a) (0.74).

**Pre-registered rule:**
- It passes for main s1 and both small seeds.
- It fails for main s0, s2, s3 and s4 only because the random-direction control's R² is 0.11–0.17, above the 0.10 cutoff. The control is confounded: a random kick of the same norm to a reliable source's state tends to make that source look less reliable, so its shift partly lines up with the Bayes shift (slope ≈ 0.2).
- Effect sizes: (a) has slope 1.0 and R² 0.96, against slope 0.2 and R² ≤ 0.17 for the control. The untrained network gives 0.00 everywhere.

**(b) The attention pattern alone is not enough** (R² 0.14–0.43). Patching B's weights over A's values does not transfer the information, because the reliability signal comes from weights and values computed from the same residuals.

**The plateau network hardly responds to the labelled residuals at all** (full-swap R² 0.04).

## Step 3: replacement

Score = (NLL_ablated − NLL_replaced) / (NLL_ablated − NLL_model), on held-out episodes. Ablation costs (the denominators) are 0.05–1.1 nats.

| Replacement | main s0 | s1 | s2 | s3 | s4 | small s2 | small s4 |
|---|---|---|---|---|---|---|---|
| 3a-attn: QQ attention uniform over own labelled tokens (pre-reg) | −0.24 | −0.08 | 0.27 | 0.00 | −0.19 | −0.05 | 0.03 |
| 3a-mlp: MLP1 output at QQ := linear in true [log SS, E] (pre-reg) | −0.39 | −0.57 | −0.22 | −0.12 | −0.76 | −0.97 | −0.99 |
| 3a-both (pre-reg) | −1.20 | −1.26 | −0.55 | −0.45 | −0.92 | −0.10 | −0.90 |
| **3b: query-token layer-2 weights over sources := Bayes β_s (pre-reg)** | **0.83** | **0.55** | **0.88** | **0.88** | **0.55** | **0.71** | **0.64** |
| 3-all (pre-reg) | −1.18 | −1.93 | −0.46 | −0.50 | −1.60 | −0.59 | −1.63 |
| post hoc: within-own-labelled pattern softmax(c₁r + c₂r²) vs flat | **1.00** | **1.00** | 0.23 | **1.00** | **1.00** | **1.00** | **1.00** |
| post hoc: MLP1 output, only the [log SS, E] subspace := true values | 0.81 | 0.42 | 0.83 | 0.81 | 0.52 | 0.57 | 0.11 |
| post hoc: the tilt + the subspace + 3b together | 0.90 | 0.85 | 0.86 | 0.89 | 0.80 | 0.81 | 0.50 |
| NLL of the model / of that combined replacement (Bayes ≈ 0.306) | .318 / .336 | .316 / .376 | .318 / .348 | .318 / .337 | .313 / .401 | .318 / .430 | .321 / .582 |

**Why the pre-registered 3a replacements fail:**
- A uniform pattern deletes the residual tilt, so the QQ token gets only the mean residual and no second moment. It also deletes the self and query-reading attention.
- [log SS, E] explains only 29–53% of the MLP1 output at QQ. Replacing the whole output with that linear reconstruction throws away the rest, including whatever carries o_s*.

**The tilt replacement (post hoc) is close to exact.** In 6 of the 7 networks, the hand-set pattern gives NLL within 0.001 nats of the model's, while flattening costs 0.14–0.27 nats.
- Fitted tilts are c₁ ≈ ±1.5 to 1.6 and c₂ ≈ −0.4.
- In two-head seeds the two heads have opposite signs of c₁, as step 1 found.
- Main s2 is the exception (0.23). Its pooling head has almost no linear tilt (c₁ = 0.23, c₂ = −0.44).

**β at layer 2 is partly captured.** It retains 55–88%, but replacing the model's weights with the exact Bayes β costs 0.01–0.06 nats. The best seeds (s1, s4) lose the most. So the model's layer-2 pattern is not just β, and step 4 should look here for the deviation.

## Step 4: identity-vector noise (all 16,000 episodes)

| Network | Original KL | Constant orthogonal IDs, 3 frames: KL removed | Average over 16 ID draws: gap, KL | Fraction of KL removed | Fraction of mean-error variance removed |
|---|---|---|---|---|---|
| main s0 | 0.0133 | +14%, +9%, +15% | 0.0085, 0.0103 | 23% | 25% |
| main s1 | 0.0077 | +10%, +15%, +8% | 0.0060, 0.0059 | 23% | 32% |
| main s2 | 0.0107 | +27%, +26%, +8% | 0.0075, 0.0077 | 28% | 31% |
| main s3 | 0.0122 | +19%, +17%, +8% | 0.0087, 0.0096 | 22% | 26% |
| main s4 | 0.0067 | +14%, +16%, +16% | 0.0049, 0.0052 | 22% | 29% |
| small s2 / s4 | 0.0107 / 0.0131 | +29% to +36% / +21% to +31% | 0.0075 / 0.0090 KL 0.0072 / 0.0090 | 33% / 31% | 42% / 39% |
| plateau | 0.1439 | 0% | unchanged | 0% | 0% |

The main model's mean NLL gap goes from 0.0089 to 0.0071 when averaged over ID draws.

**The pre-registered rule (≥ 25% of KL removed in ≥ 4/5 main seeds) fails:** 22–28%, only 1 of 5 seeds is at or above 25%. ID noise is a consistent but minority part of the deviation: about a quarter of the KL in the main model and a third in the small one. It is absent in the plateau network, whose error is structural.

Fixed orthogonal IDs help less and depend on the frame (8–36%). A single fixed configuration is not typical of training.

## Revised hypothesis, and what is still not understood
1. **Layer 1 of each QQ token** computes softmax(±c₁r + c₂r²)-weighted averages of its own source's labelled residuals. This is exactly replaceable. The difference between the two oppositely tilted heads gives a second moment.
2. **MLP1** turns that into a reliability code at QQ. Only part of it lies in a linear [log SS, E] subspace (replacement score 0.1–0.8).
3. **The layer-2 softmax of the query token over the QQ tokens** approximates β_s. It is not exactly β, and replacing it with the exact β costs 0.01–0.06 nats.

The remaining error has three candidate sources, to apportion next:
- (i) The ID-vector noise, about 25% of KL; a step-4 estimator could average it out analytically.
- (ii) The layer-2 weights deviating from β, which is consistent with the KL concentrating where the query readings couple the sources.
- (iii) The nonlinear part of the MLP1 reliability code.

## Files (added to `work/tiny-pfn-interp/`)
- `steps.py`
- `summarize_steps.py`
- `tables_steps.md`
- `out/steps_*.json`
- PREREG.md Amendments 1 and 1b

---

# Overnight B1: fresh-seed confirmation of the layer-1 rule (2026-10-28, 04:00Z)

**TL;DR.** Both pre-registered hypotheses hold on 5 freshly trained seeds:
- **H1, the layer-1 rule softmax(c₁r + c₂r²):** passes in 3 of 4 non-plateau seeds.
- **H2, the combined replacement:** passes in 4 of 4.

The fresh seeds also produced two other solutions. Seed 6 got stuck on the query-only plateau before the warm start. Seed 8 found a third solution, with no per-source pooling in layer 1.

**Setup.**
- PREREG Amendment 2 was written at 01:15Z, before training.
- Seeds 5–9 were trained with the same recipe: Gaussian head for 160k steps plus cooldown, then a warm-started K=4 mixture head.
- Training ran on one standard g2-standard-8 VM for 2.5 h. The VM was deleted; GPU-LOG records $2.18 (rate ceiling × age).
- The test set regenerated on the VM matched the local one exactly (sha256 prefixes).

| Seed | Test gap | Layer-1 own/other-source ratio (H0, H1) | Fitted tilts (c₁, c₂) H0; H1 | H1: score, NLL_repl − NLL_model | H1 | H2: score (NLL cost; model gap) | H2 |
|---|---|---|---|---|---|---|---|
| 5 | 0.0062 | 251, 311 | −1.47, −0.32; +1.67, −0.40 | 1.00, +0.0010 | pass | 0.84 (+0.064; 0.008) | pass |
| 6 | 0.142 (plateau, excluded) | 1.2, 1.1 | n/a | n/a | – | n/a | – |
| 7 | 0.0130 | 1.5, 117 | −0.09, +0.15; −1.25, −0.33 | 0.99, +0.0009 | pass | 0.87 (+0.028; 0.013) | pass |
| 8 | 0.0343 | 0.8, 1.1 | +1.54, −0.34; +0.36, +0.29 | −1.15, +0.17 | fail | 0.81 (+0.048; 0.039) | pass |
| 9 | 0.0067 | 328, 283 | −1.64, −0.38; +1.52, −0.34 | 1.00, +0.0012 | pass | 0.83 (+0.089; 0.010) | pass |

**The same two motifs reappear.**
- Seeds 5 and 9 have two oppositely tilted source-pooling heads. They are again the best seeds (gaps 0.006–0.007).
- Seed 7 has one pooling head plus one head that mixes the query readings (gap 0.013).

**Step 1–2 numbers replicate on seeds 5, 7 and 9:**
- At QQ after MLP1, log SS_s gives R² 0.84–0.96 and E_s gives R² 0.82–0.98.
- The query token's layer-2 attention tracks β with r 0.59–0.97.
- The step-2 patch of QQ after MLP1 gives slopes 1.00 / 0.74 / 1.00 and R² 0.97 / 0.78 / 0.97.

**H2 carries the same caveat as before.** The combined replacement retains 81–87% of performance, but it costs 0.03–0.09 nats, which is 2–9× the model's own gap.

**Seed 6.** Its Gaussian body sat at gap ≈ 0.15–0.16 for all 160k steps. The warm start therefore does not prevent the plateau. It only avoids it when the Gaussian body has already escaped. 1 of the 5 fresh Gaussian runs got stuck, against 0 of the original 5.

**Seed 8 is a third solution,** with gap 0.034, between the plateau and the near-Bayes seeds.
- Layer 1 does not group by source (ratios 0.8–1.1).
- No per-source quantity can be decoded at QQ (log SS R² 0.27).
- Patching QQ after MLP1 does nothing (R² 0.04).
- Even so, one query-token layer-2 head still tracks β (r 0.86), and the predictive variance can be decoded at the query token (0.87).
- Where this seed computes reliability has not been found. It is a candidate for a step-1 locate run.

Files: `work/tiny-pfn-interp/b1/` (checkpoints, logs, out/), `summarize_b1.py`, `tables_b1.md`.

# Overnight B2: budget of the model's KL to Bayes (2026-10-28, 04:00Z)

**TL;DR.** On the 5 main seeds, held-out KL is 0.0071–0.0144 (mean 0.0106). It splits as follows:
- **43% is nuisance noise.** This is the part that disappears when the predictive is averaged over inputs Bayes ignores.
  - **Identity-vector crosstalk: 22% (19–25%).** It is a named share under the pre-registered rule: it lowers KL by at least 10% in 5/5 seeds.
  - **Dependence on the direction of a source's residual vector, not just its length (SS_s): 21% (9–32%).** It passes the 10% threshold in only 3/5 seeds, so it is short of the rule.
  - **Labelled y values: 0%.**
- **57% is systematic,** meaning a deterministic function of the Bayes sufficient statistics. It splits into mean mismatch 21%, variance 15% and shape 21%.

The K=4 head's capacity floor is 0.3%. Every attempt to "idealise" the layer-2 weights or the MLP1 code raised KL. The systematic part therefore has a location in output-moment space but no named mechanism yet.

Setup: PREREG Amendment 3 (01:15Z); 3b (02:03Z) and 3c (03:48Z) are post hoc. Code: `budget.py`, `budget_b2.py`, `k4floor.py`; tables in `tables_budget.md`. Bayes is exactly invariant to every resampling used (max change 8e-15).

**Nuisance symmetrisation.** The predictive is averaged over 16 draws; each nuisance's share is a Shapley value.

| | s0 | s1 | s2 | s3 | s4 | Mean |
|---|---|---|---|---|---|---|
| KL of the model (held-out) | 0.0144 | 0.0080 | 0.0107 | 0.0126 | 0.0071 | 0.0106 |
| Identity vectors | 22% | 22% | 25% | 19% | 20% | 22% |
| Residual direction (r_s rotated at fixed SS_s) | 29% | 9% | 25% | 32% | 9% | 21% |
| Labelled y (resampled, r fixed) | 1% | 0% | 0% | 0% | 0% | 0% |
| Left after averaging all three (KL_sys) | 48% | 69% | 51% | 49% | 71% | 57% |

**The residual-direction share follows the layer-1 motif.** It is 9% in the two-pooling-head seeds (s1, s4) and 25–32% in the one-pooling-head seeds (s0, s2, s3). That fits the mechanism. Taking the difference of two oppositely tilted softmax averages cancels the first moment and leaves an SS-like second moment. A single tilted head mixes the mean residual into its reliability signal, and the mean residual is not a sufficient statistic.

**Output-moment split (post hoc).** The KL drop from setting the mean to the Bayes mean, then also matching the variance; whatever remains is shape.

| | Mean | Variance | Shape |
|---|---|---|---|
| Model (KL 0.0106) | 50% | 27% | 24% |
| After nuisance averaging (KL_sys = 57%) | 21% | 15% | 21% |

The best possible K=4 mixture to Bayes has KL 0.00003 (0.3% of the model's KL), so head capacity is not the limit. The nuisance noise is mostly a noisy mean. The systematic part is spread evenly over mean, variance and shape.

**Layer-2 query-token weights vs β** (per head, log weights centred over sources):
- **The weights are flatter than β.** The fitted exponent is a = 0.35–0.81, so each head under-differentiates the sources.
- **One head per seed tracks β closely** (R² 0.85–0.93). In the one-pooling-head seeds s0 and s3, the other head does not (R² 0.14–0.30).
- **Full-Bayes β fits slightly better than labelled-only β** in 9/10 heads. So the layer-2 weights do use the query-reading coupling.
- **The residual on the most-trusted source has a seed-specific sign.**
  - In two-pooling-head seeds it is positive and grows with coupling (s4 H0: +0.007 → +0.054 across coupling quintiles).
  - In the query-mixer heads it goes the other way (s0 and s3 H0: −0.08 to −0.15 in the top quintile).
- **The residual is uncorrelated** with how far a source's reading sits from the other two, and with log SS (|r| ≤ 0.10).

**Every replacement that "idealises" a mechanism raises KL** (means over 5 seeds; the model's KL is 0.0106):

| Replacement | KL | KL with nuisances averaged | Mean MSE / model's |
|---|---|---|---|
| Layer-2 weights := β^a (fitted a, both heads) | 0.0221 | 0.0187 | 0.88 |
| Same, one head only (H0 / H1) | 0.0207 / 0.0208 | 0.016 / 0.016 | 1.86 / 1.44 |
| Weights := labelled-only β^a | 0.0352 | 0.0317 | 2.79 |
| MLP1 at QQ := g(log SS_s, o_s*) | 0.112 (s1, s4: 0.012) | 0.100 | 7.5 |
| MLP1 at QQ := g(log SS_s, mean r, mean r³, o_s*, mean y) | 0.109 (s1, s4: 0.009) | 0.100 | 7.3 |
| MLP1 at QQ := g(own and other sources' readings and log SS), post hoc | 0.0167 (+0.004 to +0.009 over the model) | – | – |

**β weights help the mean but hurt the full predictive.** Replacing both heads' weights with β^a improves the mean slightly (MSE −12%) but doubles KL. Downstream layers are calibrated to the model's own weights, including the 7–15% of their variance that β does not explain. So "the weights are not β" is not an error source by this test.

**In one-pooling-head seeds, the MLP1 code at QQ depends on the other sources' query readings.**
- Replacing it with a function of the source's own statistics alone is catastrophic in s0, s2 and s3 (KL ≈ 0.19).
- Adding the other two query readings brings KL back to within 0.004–0.009 of the model.
- In those seeds, the layer-1 head that mixes the query readings carries the coupling between sources into each source's QQ token.

**Budget so far** (fraction of the model's KL, mean of 5 seeds):

| Component | Share | How it was measured |
|---|---|---|
| Identity-vector crosstalk | 22% | Symmetrisation (named, rule passed) |
| Dependence on residual direction, not only SS (non-sufficient layer-1 pooling) | 21% | Symmetrisation (3/5 seeds ≥ 10%) |
| Systematic: mean | 21% | Output-moment split |
| Systematic: variance | 15% | Output-moment split |
| Systematic: shape | 21% | Output-moment split |
| Head capacity (K=4) | < 1% | Best-fit floor |

No network-internal replacement reduces the three systematic shares; they are located only in output-moment space.

**Where the systematic part lives** (post hoc, descriptive; held-out, averaged over the 5 seeds):
- **Shared across seeds.** After symmetrisation, the per-episode systematic KL correlates 0.81 between seeds (range 0.64–0.92). The raw KL correlates only 0.43. So the systematic deviation belongs to the architecture/recipe rather than to any one seed.
- **Concentrated in multimodal episodes.** It tracks how multimodal the Bayes posterior is: Spearman ρ = 0.65 with the between-component variance fraction, whose top quintile holds 48% of the systematic KL. It also tracks the spread of the query readings (ρ 0.51) and query coupling (ρ 0.27; top quintile 48%).
- **The nuisance part is diffuse** (ρ ≤ 0.25 with every statistic).
- **Candidate mechanism for the shared part (a hypothesis, not tested):** the layer-2 readout forms only two attention-weighted averages of the query readings, one per head, while Bayes mixes 27 reliability hypotheses. The K=4 head floor (0.3%) rules out the output head itself as the limit. The next test would be a 3- or 4-head layer 2 at matched width, checking whether the systematic shape and mean parts shrink.

# Overnight B2(d): mechanistic vs sampling estimate of the expected excess loss (2026-10-28, 04:40Z)

**TL;DR.** The ARC criterion is not met by anything that deserves to be called mechanistic.
- **Sampling beats the flat estimate almost immediately.** "Network = Bayes" is beaten after only 4–16 network calls when per-episode exact KL is used. With the plain NLL gap at y*, it takes about 256–1024 calls.
- **The zero-call estimate does beat sampling for a while,** up to N = 16–1024 calls, and past twice the flat crossover in 4/5 seeds. But its features turn out not to matter: it reduces exactly to "the mean KL of the other four seeds".
- **The mechanism-informed control variate is negligible.** It beats sampling at every N, but its variance reduction is only 6–10%, and the per-episode correlation is only 0.24–0.30. Generic Bayes features give 1–4%.
- **The pre-registered "features add value" test fails in 5/5 seeds:** the variance ratios differ by about 5%, against the required 20%.
- **The reason:** per-episode KL is heavy-tailed and is about 43% noise from IDs and residual direction, which no Bayes-side feature can predict.

**Setup.**
- PREREG Amendment 4 was written at 03:49Z. Code is `arc.py`; results are in `out/arc.json`.
- Pool: 100,000 fresh episodes. The truth is the pool mean of exact per-episode KL.
- Each estimator was resampled 4,000 times at each N.
- Cost: a Bayes reference costs 1.5e-5 s per episode, against 5.5e-5 s for a network forward. The KL quadrature used by the KL-based sampler is not counted, which flatters that sampler. The NLL-gap sampler needs no quadrature.

| Seed | G (truth) | sd of per-episode KL / NLL gap | N where sampling beats flat (KL / NLL gap) | Zero-call estimate: bias; beats KL sampling up to N | Control-variate variance ratio, mechanistic / generic |
|---|---|---|---|---|---|
| 0 | 0.0132 | 0.029 / 0.16 | 4 / ≈ 1024 | −0.0039; 32 | 1.08 / 1.03 |
| 1 | 0.0075 | 0.023 / 0.12 | 16 / ≈ 1024 | +0.0032; 32 | 1.10 / 1.01 |
| 2 | 0.0107 | 0.029 / 0.15 | 8 / ≈ 256 | −0.0008; 1024 | 1.06 / 1.04 |
| 3 | 0.0124 | 0.029 / 0.16 | 4 / ≈ 256 | −0.0029; 64 | 1.07 / 1.03 |
| 4 | 0.0066 | 0.023 / 0.12 | 16 / ≈ 1024 | +0.0043; 16 | 1.10 / 1.01 |

**What would be needed to meet the criterion.** A per-episode prediction of the network's error that explains much more than 10% of its variance. For the systematic part this looks feasible: it is shared across seeds at r = 0.81 and concentrated in multimodal episodes. A surrogate of the systematic error built from the mechanism, for example the two-weighted-average readout, could serve as a strong control variate. The nuisance part, 43% of KL, can only be predicted in expectation, not per episode. That alone limits any control variate to a variance reduction of roughly 2×.

**VM check.** Only one VM was used (tinyb1-1028), and GPU-LOG records its DELETE at 03:46Z. Spend: $2.18 of the $8 cap.

---

# Layer-2 head-count test (2026-10-28, 08:50Z)

**TL;DR.** The shared systematic error (KL_sys) is not a two-head readout bottleneck.
- 4 heads in layer 2 reduce KL_sys by 11% (not significant, p = 0.24), and 3 heads make it worse (−43%).
- A layer-2 width control with the same parameter count as 4 heads gives the same 12%.
- A third layer cuts KL_sys by 46% (0.0056 → 0.0031) and raw KL to 0.0051, the best of any model so far.
- Pre-registered outcome (iv): head count is not the limit. Of the two alternatives, depth helps and layer-2 width barely does.

**Setup.** PREREG Amendment 5 was written at 04:41Z, before training.
- **Arms:** 5 (layer 1 unchanged), seeds 10–14, the same warm-start recipe.
- **Training:** all arms, including the 2-head base, ran on the GPU of one on-demand L4 VM. The VM ran 4.04 h, was deleted at 08:45Z (GPU-LOG DELETE line), and cost $3.51 against the $4 cap. Overnight total: $5.69.
- **Analysis:** budget_lite runs the B2 symmetrisation and moment split on held-out episodes.
- **Exclusions:** 2 of 25 runs stuck on the query-only plateau (base s11 and h3 s10, gap 0.142) and are excluded as pre-registered.
- **Device check:** the GPU base reproduces the CPU-trained main seeds (KL_sys 0.0056 vs 0.0058).

| Arm (layer 2) | Params | Seeds kept | Test gap | KL raw | KL_sys (± se) | Systematic mean / var / shape | r with main seeds' per-episode KL_sys | Reduction vs base |
|---|---|---|---|---|---|---|---|---|
| base: 2 heads × 16 | 18,732 | 4 | 0.0087 | 0.0087 | 0.0056 ± 0.0005 | 0.0020 / 0.0015 / 0.0021 | 0.93 | – |
| h3: 3 heads × 16 | 20,828 | 4 | 0.0173 | 0.0177 | 0.0081 ± 0.0017 | 0.0029 / 0.0029 / 0.0022 | 0.75 | −0.43 |
| h4: 4 heads × 16 | 22,924 | 5 | 0.0095 | 0.0100 | 0.0050 ± 0.0006 | 0.0017 / 0.0015 / 0.0018 | 0.89 | +0.11 |
| w64: 2 heads × 32 (parameter-matched to h4) | 22,924 | 5 | 0.0080 | 0.0086 | 0.0050 ± 0.0002 | 0.0016 / 0.0013 / 0.0020 | 0.93 | +0.12 |
| L3: three layers, 2 heads × 16 each | 27,276 | 5 | 0.0047 | 0.0051 | 0.0031 ± 0.0003 | 0.0009 / 0.0007 / 0.0015 | 0.87 | **+0.46** |

**Pre-registered rules.**
- (i) The bottleneck needs R_h4 ≥ 0.30 and R_h3 ≥ 0.15. It fails: R_h4 = +0.11 (p = 0.24) and R_h3 = −0.43.
- (ii) and (iii) do not apply.
- (iv) holds: head count is not the limit.

**Interpretation.**
- **More heads at the same depth help no more than width.** At 22.9k parameters, more heads and more width give the same small gain.
- **Depth is what shrinks the systematic error.** The third layer reduces all three of its parts: mean −56%, variance −55%, shape −29%.
- **The error pattern is unchanged.** Per-episode KL_sys of the 3-layer model still correlates 0.87 with that of the 2-layer seeds. A third layer shrinks the same error rather than changing which episodes carry it.
- **Depth vs parameters is not cleanly separated.** L3 has 4.4k more parameters than the parameter-matched pair (h4, w64). Since 4.2k extra layer-2 parameters bought only 11–12%, extra parameters alone are unlikely to explain the 46%.
- **Why h3 is worse:** two of its four kept seeds sit at gap 0.025, so its training is less reliable. Its KL_sys spread is large (0.0045–0.0118).
- **Mechanistic reading (hypothesis):** the systematic error comes from too few sequential steps, not too few parallel readouts. A plausible reason is that the coupling of reliabilities through the query readings needs a further round of attention after the per-source reliability code exists. The next test would be the B2 replacement analysis on the 3-layer model: which layer computes the coupling?

Files: `work/tiny-pfn-interp/heads/` (pfn2.py, budget_lite.py, summarize_heads.py, tables_heads.md, hruns/ checkpoints, hout/ results, gpu-run.out).

---

# What the third layer computes (2026-10-28, 09:05Z)

**TL;DR.** The third layer is not a module for the reliability coupling through the query readings.
- **The coupling is never cleanly represented.** The part of each source's posterior that the query readings add (dE = E_bayes − E_lab) peaks at R² 0.46 at the query-question tokens after block 2 of the 3-layer models. That is below the 0.5 threshold, though above the 2-layer models (0.31).
- **What the coupling code there is carries no load.** Removing its linear code leaves KL unchanged, just as removing a random direction of the same size does. The pre-registered rule passes in 0/5 seeds.
- **The 3-layer gain is uniform.** It cuts systematic KL by a uniform 43–51% in every quintile of coupling, of multimodality and of query spread.
- **What layer 3 does compute:** a second, sharper readout. The query token attends to the three query-question tokens again, with weights closer to β (exponent a ≈ 0.79 vs 0.50 in layer 2; R² 0.72 vs 0.57). This refines a coarse estimate formed in layer 2: the Bayes-minus-equal-weight mean goes from R² 0.58 to 0.88, and the log variance from 0.78 to 0.97.
- **The full per-source posterior given the query readings (p_bayes) is found** (R² 0.87 at the query-question tokens after block 2). But it is almost entirely its labelled-only part.

No decision needed.

**Setup.** PREREG Amendment 6 was written at 08:50Z, before the analysis.
- Networks: the 3-layer seeds 10–14 and the 2-layer GPU base seeds 10, 12, 13, 14.
- CPU only, 4 processes, $0.
- Methods: the step-1 probe pipeline with untrained and shuffled baselines, the B2 replacement method, and held-out KL.
- Code: `heads/l3.py`, `summarize_l3.py`, `tables_l3.md`.

**Probes at the query-question (QQ) tokens and the query token (QT).** Cells give held-out R², trained (untrained); * marks POSITIVE in ≥ 4/5 seeds.

| Target | Token | 3-layer: after block 1 | after block 2 | after attn3 | 2-layer: after block 1 | after block 2 |
|---|---|---|---|---|---|---|
| E_lab (labelled-only expected precision) | QQ | 0.73* | 0.92* | 0.93* | 0.93* | 0.91* |
| p_bayes (full per-source posterior given query readings) | QQ | 0.64* | 0.87* | 0.87* | 0.82* | 0.82* |
| dE = E_bayes − E_lab (the coupling part) | QQ | 0.13 | 0.46 (0.02) | 0.46 | 0.17 | 0.31 |
| dp = p_bayes − p_lab | QQ | 0.15 | 0.42 | 0.41 | 0.18 | 0.28 |
| β_s | QQ | 0.53* | 0.84* | 0.88* | 0.67* | 0.83* |
| Bayes mean minus equal-weight mean | QT | 0.03 | 0.60* | 0.88* | 0.07 | 0.87* |
| Bayes mean minus labelled-only Bayes mean (coupling effect on the mean) | QT | 0.06 | 0.09 | 0.39 (0.01) | 0.08 | 0.25 |
| log Bayes variance | QT | 0.28 | 0.81* | 0.97* | 0.25 | 0.96* |

In the 3-layer models, the per-source code at QQ is completed in block 2 (E_lab 0.73 → 0.92), not in block 1 as in the 2-layer models.

**Readout weights.** Query-token attention over the three QQ tokens; R² of the log weights on log β vs on labelled-only log β:
- **3-layer, layer 2:** a 0.46–0.55, R² 0.57 / 0.56–0.57, QQ mass 0.58.
- **3-layer, layer 3:** a 0.76–0.82, R² 0.70–0.74 / 0.67–0.70, QQ mass 0.73.
- **2-layer, layer 2:** a 0.51–0.74, R² 0.69–0.71 / 0.66–0.69.
- **Full β beats labelled-only β by only 0.01–0.04** in every case. So no layer's readout is markedly more "coupled" than another's.

**Replacements, 3-layer** (held-out KL, mean over 5 seeds; model 0.0051, 2-layer base 0.0087):

| Replacement | KL | Interpretation |
|---|---|---|
| Remove the linear coupling code (dE) at QQ after block 2 | 0.0054 | +2% to +24% of the 2-to-3-layer gap, vs +3% to +48% for a random direction of the same size. Rule: 0/5 seeds. |
| Same after block 1 | 0.0062 | Not specific: the random-direction control rises as much |
| QQ tokens' layer-2 update mean-ablated (layer 3 reads QQ as of after block 1) | 0.09–0.95 | Catastrophic: layer 3 needs the QQ code that block 2 completes |
| Layer-3 attention at QT mean-ablated | 0.07–0.36 | Layer-3 attention is the main readout |
| Layer-3 MLP at QT mean-ablated | 0.008–0.015 | Matters about 2× less |

**Where the 3-layer gain lies** (per-episode systematic KL, held-out). The systematic part drops 0.0056 → 0.0031 overall. By quintile, from low to high:

| Statistic | Relative reduction | Share of the total reduction |
|---|---|---|
| Coupling | 0.48, 0.47, 0.48, 0.51, 0.43 | top quintile 46%, the same as its share of the error |
| Between-component variance | 0.53 → 0.42 | – |
| Query spread | 0.52 → 0.37 | – |

**Answer to the question.** No layer carries the coupling as a separable, load-bearing code in either architecture. The extra layer helps by:
- splitting the per-source code over two blocks, and
- adding a second, sharper query-token readout over the same per-source codes.

The result is a proportional shrinking of the same error pattern (per-episode r = 0.87 with the 2-layer seeds), not a fix for the coupling.

**Open question for the next step.** Whether the leftover systematic error is mainly the flattened readout weights (a < 1 even in layer 3). A direct test would sweep a temperature on the layer-3 query-token logits.
