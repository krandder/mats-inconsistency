# The backward bit recursion on small noisy nets (backward-bits lane, 2 November 2026)

Status: FINAL v2 (scored runs complete 20:39Z; gpt-6-astra round 2 applied 21:00Z). Lane bead pool-qxszj1.27 (target M1). Work directory work/backward-bits/:
FORMAL.md (definitions and proofs), PREREG.md + prereg_hashes.txt (registered 14:17:41Z, amendments A1-A7),
notes/DEVIATIONS.md (D1-D9), notes/QUESTIONS.md, results/, figs/, runs_gpu/scored3/ (raw JSON per net and chain),
review/ (gpt-6-astra rounds), GPU-LOG.md.
Labels: PROVEN (proof in FORMAL.md), CITED, MEASURED, HEURISTIC.

## Summary for Kelvin

**Short answer.** Your target (bits of layer l+1 minus the bits the noise erases) is the right ceiling, and it passes its
estimation checks at 262 of 264 steps (2 bootstrap SEs of 0.0100-0.0105 bit against a 0.010 tolerance). But no fixed encoding of a layer carries its own rate's worth of it, and run as a chain the
recursion drifts. With your rule it ends far below what the input knows about the output (median 26% at the input); with
"carry 95%" it never compresses. Your arithmetic is exact only for codes that re-generate the next code's statistics
(stochastic codes); there the counts equal the truth curve, which we measured at every layer. So: certified targets to
the input, yes; certified bit counts of fixed encodings to the input, no: your rule meets the registered criterion on no
config; 3 of 16 config-rule pairs meet it, and on those the 95% level is often not reached, only stably approached (4.3).
Everything is conditional on one fixed 64-cell output quantiser per net and one noise level per config.

1. **What ran.** Model S nets, widths 8 and 16, depths 8 and 16, three seeds each (12 nets); noise chosen on a dev net so
   the first backward step erases about 1 bit of a 64-cell output code (4.6-5.9 bits; output nMSE 0.2-21%). Four rule variants, certificates
   at every step, the truth I*_l = I(h_l; c_L) measured directly, two exactly solvable chains. Pre-registered (14
   predictions: on the 12 run as registered 6 hit, 6 missed; the 2 Gaussian ones re-run in amended form: 1 hit, 1 missed;
   section 8).
2. **The target is certified.** Bootstrap SE 0.001-0.0105 bit (2 steps just over the 0.010 tolerance); Miller-Madow vs Grassberger within 0.006; 4x the noise
   draws and 2x the points move T, I and I* by at most 0.015 bit from the registered sizes (12 checks on 4 nets; with
   only 256 draws I* can be off by 0.03 bit). Estimation is not the limit.
3. **No fixed encoding reaches it at its own rate (proved, FORMAL Props 2-3).** Where the noise erases about a bit
   (0.8 bit or more at the first step: 9 of 12 nets), the best code found with rate <= T carries 51-78% of T at the first
   step and falls >= 10% short at every step (100% of 66 steps on the 16-wide nets). The 95% rule's best codes there have
   6.7-7.9 bits, above the output code's own rate, and still reach 95% on only 2 of these 9 nets (84-94% on the others):
   no candidate tried (up to 256-512 cells) reached 95%. Real-valued Gaussian-channel codes do no better at information
   rate T (44-82% of T): the gap is not about discreteness. Exact 1-D case: the best partition found at rate T carries
   67% of T (four cells).
4. **Your rule drifts down.** Each step keeps 28-89% of its target on the 16-wide nets (median 65%); at the input your chain's code
   knows 2-24% of what the input knows about the output code (16-wide), 27-74% (8-wide; median 26% over all 12 nets).
   Its targets end 0.1-0.6 bit on the 16-wide nets against a truth of 0.13-1.58.
5. **"Carry 95% of the next code" drifts up.** Targets at the input 4.1-7.9 bits against a truth of 0.13-5.2; codes of
   5-9 bits at every layer (above the output code); it keeps 90-99% of the truth only because it never compresses.
6. **Best variant for fixed encodings: the output code as fixed target, 95% per step (PR).** At the input it keeps 74-91%
   of the truth (median 83%) at 1.2-7.5 bits, often by holding a 128-256-cell code for several layers; "no layer exceeds
   its successor" fails at the first step on 9 of 12 nets (the 95% rule's code exceeds the output code by > 0.5 bit).
   17 of its 132 steps did not reach 95% (the most relevant candidate was kept). In the exact chain, 99% per step keeps 98% at about twice the
   surviving bits.
7. **Where your arithmetic is exact (proved, Prop 7).** If each layer's code re-generates the next code's statistics, the
   recursion's targets equal I*_l at every layer, no drift. Those counts are the truth curve: at the input 0.13-1.58
   bits (16-wide), 2.6-5.2 (8-wide). They are bits to regenerate the output code's statistics with shared random bits,
   not bits of a fixed encoding (closed form for analog codes: such codes know 77-88% of I*).
8. **What a fixed encoding of a layer needs** (to carry 95% of what it knows about the output code; empirical, best
   among the candidates tried): 1.1-2.7x I* on the 8-wide nets; on the 16-wide nets 8-9 bits (4.6-7x I*) where 95% was
   reached at all, and most layers did not reach it within 512 cells.
9. **Where it breaks.** (a) The per-step gap and its compounding: the main break. (b) Encoder depth matters only where
   the noise is tiny: on the 8-wide nets learned encoders cannot fit the sharp composed partitions (excess cross-entropy
   up to 1 bit, depth 4 beats depth 2 at up to 12/45 steps); your "use the last encoding" (exact composition, one layer
   deeper per step) is what keeps the bits there. On the 16-wide nets depth 2 is enough (C4 fails at 0-3 steps).
   (c) The fixed absolute noise erases very different amounts across seeds of narrow deep nets (first step 0.02-2.7 bit).
10. **Output distortion, forward codes.** Shared-noise output error follows information: PR codes beat K codes at the
   input on 11 of 12 nets. Codes trained forward on that distortion carry less than the chains' codes at equal rate and
   collapse at the input layers.

No decision needed. Possible next step (not started): run the recursion with stochastic re-simulation codes compressed
by channel simulation (Prop 7) and measure the one-shot code length against I*_l.

## 1. What you asked and what was run

Your procedure (2 Nov, about 12:15Z): train the last layer's code to convergence; compute from the code, near-exactly,
how many bits of layer l-1 the noise erases; train the code at l-1 (or reuse the last encoding) until it converges to the
bits of layer l minus the erased bits; recurse toward the input, with deeper encoders as needed.

Run (PREREG.md):
- Nets: Model S, h_l = relu(W_l h_{l-1}) + sigma xi_l, noiseless output; 8x8 (sigma 0.01), 8x16 (3e-4), 16x8 (0.2),
  16x16 (0.1); seeds 5000-5002. Sigma was set on a dev net (seed 9100) so that the first backward step erases 0.5-2 bits
  of the output code and the input keeps at least 1 bit. The 8-wide nets shrink their state strongly with depth (median
  norm per unit 0.96 at the input, 0.09 at the output on the dev 8x8 net), so their noise has to be tiny; on other
  seeds the same sigma can erase almost nothing (D2).
- The last layer's code: k-means with 64 cells on the output (best of 8 restarts, Lloyd to convergence), 5.4-5.9 bits.
- At every step l = L-2 .. 0: the target T_l = H(c_{l+1}) - E H(c_{l+1} | h_l) from 2^16 fresh points with 1024 noise
  draws each (Miller-Madow; Grassberger as a check); the code at l = a posterior predictor (an MLP that predicts the
  next code's distribution from h_l) followed by the best partition of the predicted distributions (deterministic
  information bottleneck: KL k-means over cell counts, entropy-constrained two-cell codes, annealed DIB paths), or the
  reuse code c_{l+1}(relu(W_{l+1} h)) (your "use the last encoding"; selectable after amendment A2; stacked reuses are
  your "deeper encoders as needed"); selection on a separate set, every number reported on a third; certificates at
  every step of chains K and PR (an independent restart, a depth-4 predictor, a look-ahead control that clusters the
  Monte Carlo posteriors themselves).
- Four versions of the rule (section 2 says why one rule is not enough): K = your rule as written (rate at most the
  target); R = main's second condition (carry at least 95% of the target, rate as needed); P and PR = the same two rules
  with the output code as the fixed target, carried back one noisy layer per step through the codes' cell tables.
- Truth and comparisons: I*_l = I(h_l; c_L) at every layer (Monte Carlo over the whole future noise); each chain code's
  own information about the output code; the chain's prediction of the output code through its kernels; direct codes
  trained at each layer on the output code (the forward comparison) at matched rates; the shared-noise output
  distortion of every code; codes trained forward on that distortion (bit-accounting S4 style); a Gaussian-channel
  version; estimator convergence checks.

## 2. What the target is and what a code can reach (FORMAL.md)

Notation: c_{l+1} is the code one layer later (the target of step l); T_l = H(c_{l+1}) - E H(c_{l+1} | h_l) is your
"bits of layer l+1 minus the bits the noise destroys"; I*_l = I(h_l; c_L) is what layer l knows about the output code.

1. **The target is right as a ceiling (PROVEN).** No function of h_l, deterministic or random, carries more than T_l bits
   about c_{l+1}, and T_l is exactly your R_{l+1} minus the erased bits. Main's formula is correct. T_l <= H(c_{l+1}): no
   target exceeds its successor's rate.
2. **No code carries its own rate (PROVEN).** With Gaussian noise between the layers, every value of c_{l+1} has positive
   probability given any h_l. So a partition code c_l that is not constant can never be a function of c_{l+1}, and
   I(c_l; c_{l+1}) < H(c_l) strictly; the same holds for noisy-channel codes, with I(h_l; z) as the rate. A code whose
   rate equals T_l carries strictly less than T_l. Main's convergence criterion asks for "rate at the target AND
   prediction at the limit" at once; the two are incompatible except in the limit where the gap is small.
3. **Carrying all of T_l takes infinite rate (PROVEN under a checkable condition).** The next code's distribution given
   h_l is a Gaussian-smoothed, hence real-analytic, function of h_l on each linear region of the layer; where it is not
   constant it takes a continuum of values, and a code carrying all of T_l must determine it. HEURISTIC: the shortfall
   falls like 2^(-2R/d), d the dimension the posterior moves in: d/2 extra bits per halving of the shortfall.
4. **The recursion's target counts bits of the next code, not of the output (PROVEN).** If c_{l+1} were sufficient for
   the output code, T_l = I*_l + I(h_l; c_{l+1} | c_L) >= I*_l: the target also pays for the next code's own resolution.
   With a lossy c_{l+1} it can fall below I*_l (a constant next code gives T_l = 0). By point 3 every finite chain is
   lossy, so T_l is a property of the chain's own codes, not of the net.
5. **Where your number is a code length.** If encoder and decoder share random bits, T_l bits per sample (block coding,
   the classical reverse Shannon theorem) or at most T_l + log2(T_l + 1) + 5 bits (one shot, Li and El Gamal 2018) let the
   receiver generate a next code with exactly the right statistics (CITED). That is simulation, not a partition of h_l.
   If instead the receiver must reproduce the sample's own next code given the shared future noise (the S4 setting), it
   needs at least H(c_{l+1} | xi_{l+1}) = T_l + I(h_l; xi_{l+1} | c_{l+1}) >= T_l bits (PROVEN): with shared noise, nothing
   is erased.
6. **What can be certified per step.** T_l, H(c_l) and I(c_l; c_{l+1}) near-exactly (Monte Carlo with bias corrections,
   tested against exact 1-D cases); the consistency facts above; a trained code's optimality only empirically (an
   independent restart, a deeper encoder, clustering the exact posteriors, the 1-D case where dynamic programming gives
   the optimum).

7. **When your recursion is exact (PROVEN, FORMAL Proposition 7, added after registration).** If each layer's code is
   a re-simulation of the next one (draw a fresh copy of the next layer from h_l and apply the next code), the code at
   every layer has exactly the output code's statistics given h_l, and the recursion's targets telescope:
   T_l = I*_l at every layer, with no drift, all the way to the input; I*_{l+1} - I*_l is the information lost across step l (in the
   entropy sense the private noise of such a code decides H(c_L) - I*_l bits).
   These are stochastic codes: their information rate is T_l, they are reached as code lengths only with shared random
   bits (point 5), and as predictors of the output they know less than T_l (point 2). In the linear-Gaussian chain the
   analog code chosen by your rate rule is exactly such a re-simulation: there T_l = I*_l at every layer in closed form,
   and the codes know 81-88% of I*_l about the output (scripts/exact_gauss_chain.py). Deterministic partitions are never
   re-simulations (point 2), which is where the drift below comes from.

So "train until rate = target and prediction = limit" cannot be run as stated for partition codes. It was run with each
half of the criterion as the rule (chains K and R), and with the output code as the fixed target (P, PR). The truth
I*_l measured at every layer (section 3) is exactly what your recursion returns when run with re-simulation codes.

## 3. Per layer: target bits, achieved bits, certificates

How to read the tables (one per config; median over the three seeds, range in brackets; all in bits):
- truth I*_l: what layer l knows about the output code (Monte Carlo over the whole future noise; bootstrap SE about
  0.004 bit). By Proposition 7 this is also what your recursion returns when its codes are re-simulations.
- K (your rule): the target T_l, the rate H of the code chosen (at most T_l + 0.02), what it carries about the next code,
  and what it knows about the output code (I(c_l; c_L), best decoder).
- PR (output code as target, 95% rule): its own target (computed through its cell tables), its rate, what it knows about
  the output code.
- direct: the least rate of a code trained at layer l directly on the output code (or any chain code or reuse code that
  does better) that carries 95% of I*_l; "-" when no candidate reached 95%.
- "direct" entries ">X on n": on n seeds no candidate tried (up to 512 cells; X = the largest rate tried) reached 95%;
  this is not a proof that the needed rate exceeds X (review r2), only that the search did not find a code.
- certified: seeds on which every applicable certificate of the step passed (C1 estimation, C2 predictor, C3 restart,
  C4 depth, C5 look-ahead, C6 consistency).

#### 16x16 (median over seeds [range]; bits)

| l | truth I*_l | K target T | K rate H | K carries I(c;next) | K knows I(c;c_L) | PR target | PR rate H | PR knows I(c;c_L) | direct rate for 0.95 I* | certified K / PR |
|---|---|---|---|---|---|---|---|---|---|---|
| 15 | 5.82 [5.74-5.89] | 5.82 [5.74-5.90] | 5.82 [5.74-5.90] | 5.82 [5.74-5.90] | 5.82 [5.74-5.90] | 5.82 [5.74-5.90] | 5.82 [5.74-5.90] | 5.82 [5.74-5.90] | - | - / - |
| 14 | 4.06 [3.78-4.74] | 4.06 [3.78-4.74] | 3.93 [3.79-4.76] | 2.67 [2.29-3.47] | 2.67 [2.28-3.47] | 4.06 [3.78-4.73] | 7.78 [6.96-7.92] | 3.77 [3.20-4.44] | >7.9 on 3 | 3/3 / 3/3 |
| 13 | 3.33 [2.53-4.11] | 2.90 [2.30-3.75] | 2.88 [2.08-3.70] | 1.96 [1.28-2.62] | 1.88 [1.21-2.51] | 3.01 [2.07-3.75] | 7.77 [6.88-7.88] | 3.11 [2.21-3.80] | 8.89; >8.0 on 2 | 3/3 / 3/3 |
| 12 | 2.83 [1.75-3.62] | 2.23 [1.30-2.90] | 2.24 [1.17-2.89] | 1.50 [0.65-2.08] | 1.38 [0.65-1.97] | 2.41 [1.29-3.06] | 7.77 [6.70-7.77] | 2.66 [1.55-3.33] | 8.94; >7.9 on 2 | 2/3 / 3/3 |
| 11 | 2.28 [1.19-3.16] | 1.59 [0.75-2.33] | 1.54 [0.72-2.21] | 0.97 [0.34-1.62] | 0.91 [0.36-1.51] | 1.86 [0.84-2.51] | 7.58 [6.51-7.74] | 2.15 [1.07-2.91] | 8.91; >7.8 on 2 | 2/3 / 3/3 |
| 10 | 1.93 [0.85-2.86] | 1.15 [0.48-1.88] | 0.87 [0.47-1.79] | 0.60 [0.25-1.37] | 0.55 [0.23-1.22] | 1.52 [0.58-2.13] | 7.39 [6.18-7.68] | 1.81 [0.77-2.63] | 8.39 [7.94-8.84]; >6.5 on 1 | 3/3 / 3/3 |
| 9 | 1.73 [0.62-2.57] | 0.73 [0.33-1.57] | 0.72 [0.33-1.51] | 0.46 [0.19-1.17] | 0.46 [0.15-1.02] | 1.33 [0.41-1.86] | 7.17 [5.87-7.56] | 1.63 [0.56-2.38] | 8.29 [7.87-8.71]; >6.3 on 1 | 3/3 / 3/3 |
| 8 | 1.63 [0.45-2.35] | 0.65 [0.22-1.32] | 0.65 [0.23-1.00] | 0.51 [0.13-0.78] | 0.42 [0.10-0.68] | 1.20 [0.28-1.65] | 5.51 [4.64-7.42] | 1.46 [0.40-2.19] | 8.16 [7.75-8.56]; >5.9 on 1 | 3/3 / 2/3 |
| 7 | 1.55 [0.33-2.20] | 0.60 [0.17-0.88] | 0.61 [0.19-0.88] | 0.40 [0.11-0.51] | 0.39 [0.07-0.54] | 1.07 [0.21-1.50] | 4.10 [2.79-7.33] | 1.37 [0.27-2.05] | 8.06 [7.66-8.47]; >5.9 on 1 | 3/3 / 2/3 |
| 6 | 1.50 [0.29-2.02] | 0.57 [0.16-0.75] | 0.59 [0.18-0.72] | 0.45 [0.12-0.50] | 0.38 [0.07-0.39] | 0.98 [0.18-1.36] | 4.10 [2.17-4.51] | 1.33 [0.23-1.79] | 7.95 [7.57-8.32]; >6.9 on 1 | 3/3 / 3/3 |
| 5 | 1.45 [0.24-1.92] | 0.55 [0.14-0.64] | 0.56 [0.16-0.61] | 0.43 [0.11-0.48] | 0.29 [0.06-0.36] | 0.92 [0.14-1.23] | 3.78 [1.88-4.08] | 1.29 [0.20-1.64] | 7.88 [7.54-8.23]; >6.7 on 1 | 3/3 / 3/3 |
| 4 | 1.42 [0.21-1.80] | 0.50 [0.13-0.53] | 0.48 [0.15-0.55] | 0.32 [0.10-0.47] | 0.20 [0.06-0.35] | 0.86 [0.12-1.10] | 3.44 [1.64-4.07] | 1.27 [0.17-1.50] | 7.83 [7.52-8.15]; >6.4 on 1 | 3/3 / 2/3 |
| 3 | 1.38 [0.18-1.69] | 0.37 [0.12-0.50] | 0.35 [0.13-0.52] | 0.22 [0.08-0.43] | 0.13 [0.05-0.33] | 0.81 [0.09-0.98] | 3.09 [1.38-3.49] | 1.17 [0.14-1.38] | 7.79 [7.49-8.08]; >6.2 on 1 | 3/3 / 2/3 |
| 2 | 1.31 [0.16-1.59] | 0.27 [0.11-0.47] | 0.29 [0.12-0.48] | 0.17 [0.08-0.37] | 0.09 [0.04-0.29] | 0.72 [0.08-0.87] | 2.66 [1.25-3.18] | 1.08 [0.12-1.24] | 8.85 [8.82-8.88]; >7.9 on 1 | 3/3 / 3/3 |
| 1 | 1.26 [0.14-1.50] | 0.22 [0.10-0.42] | 0.20 [0.11-0.43] | 0.12 [0.07-0.33] | 0.05 [0.04-0.26] | 0.63 [0.07-0.78] | 2.58 [1.22-2.94] | 1.00 [0.11-1.16] | 8.83 [8.80-8.87]; >7.9 on 1 | 3/3 / 3/3 |
| 0 | 1.20 [0.13-1.44] | 0.16 [0.09-0.38] | 0.16 [0.10-0.39] | 0.08 [0.07-0.28] | 0.04 [0.03-0.22] | 0.56 [0.06-0.69] | 2.48 [1.18-2.93] | 0.95 [0.10-1.08] | 8.82 [8.79-8.86]; >7.8 on 1 | 1/3 / 3/3 |

#### 16x8 (median over seeds [range]; bits)

| l | truth I*_l | K target T | K rate H | K carries I(c;next) | K knows I(c;c_L) | PR target | PR rate H | PR knows I(c;c_L) | direct rate for 0.95 I* | certified K / PR |
|---|---|---|---|---|---|---|---|---|---|---|
| 7 | 5.82 [5.82-5.86] | 5.82 [5.81-5.85] | 5.82 [5.81-5.85] | 5.82 [5.81-5.85] | 5.82 [5.82-5.86] | 5.82 [5.81-5.85] | 5.82 [5.81-5.85] | 5.82 [5.82-5.86] | - | - / - |
| 6 | 4.22 [3.13-4.27] | 4.23 [3.13-4.28] | 4.13 [2.97-4.26] | 2.86 [1.59-3.01] | 2.85 [1.59-2.99] | 4.22 [3.13-4.27] | 7.88 [6.95-7.90] | 3.89 [2.63-3.90] | >7.9 on 3 | 3/3 / 3/3 |
| 5 | 3.52 [1.84-3.52] | 3.17 [1.48-3.26] | 3.13 [1.32-3.21] | 2.16 [0.67-2.22] | 2.04 [0.64-2.13] | 3.10 [1.45-3.18] | 7.82 [6.62-7.86] | 3.19 [1.54-3.26] | >7.9 on 3 | 3/3 / 3/3 |
| 4 | 2.80 [1.20-3.08] | 2.26 [0.80-2.54] | 2.27 [0.81-2.46] | 1.48 [0.41-1.64] | 1.38 [0.37-1.49] | 2.28 [0.82-2.59] | 7.76 [6.10-7.79] | 2.55 [1.01-2.86] | 8.92; >7.9 on 2 | 1/3 / 3/3 |
| 3 | 2.26 [0.81-2.63] | 1.63 [0.54-1.82] | 1.50 [0.53-1.57] | 1.00 [0.29-1.01] | 0.92 [0.22-0.93] | 1.73 [0.49-2.11] | 7.67 [5.42-7.71] | 2.07 [0.68-2.43] | 8.87; >7.9 on 2 | 2/3 / 3/3 |
| 2 | 1.89 [0.61-2.16] | 1.09 [0.38-1.14] | 0.97 [0.35-0.99] | 0.58 [0.21-0.66] | 0.51 [0.13-0.60] | 1.38 [0.34-1.66] | 7.59 [4.94-7.63] | 1.73 [0.51-1.98] | 8.87; >7.8 on 2 | 3/3 / 3/3 |
| 1 | 1.61 [0.51-1.84] | 0.70 [0.26-0.76] | 0.72 [0.25-0.72] | 0.20 [0.15-0.32] | 0.37 [0.09-0.45] | 1.13 [0.25-1.34] | 7.50 [4.69-7.55] | 1.47 [0.43-1.67] | >8.8 on 3 | 2/3 / 3/3 |
| 0 | 1.39 [0.44-1.58] | 0.53 [0.18-0.57] | 0.46 [0.21-0.47] | 0.25 [0.12-0.26] | 0.23 [0.07-0.29] | 0.94 [0.19-1.09] | 7.45 [4.58-7.51] | 1.26 [0.36-1.41] | >8.8 on 3 | 2/3 / 3/3 |

#### 8x16 (median over seeds [range]; bits)

| l | truth I*_l | K target T | K rate H | K carries I(c;next) | K knows I(c;c_L) | PR target | PR rate H | PR knows I(c;c_L) | direct rate for 0.95 I* | certified K / PR |
|---|---|---|---|---|---|---|---|---|---|---|
| 15 | 5.34 [5.00-5.51] | 5.34 [4.98-5.52] | 5.34 [4.98-5.52] | 5.34 [4.98-5.52] | 5.34 [5.00-5.51] | 5.34 [4.98-5.52] | 5.34 [4.98-5.52] | 5.34 [5.00-5.51] | - | - / - |
| 14 | 4.79 [3.79-5.34] | 4.79 [3.80-5.33] | 4.73 [3.78-5.35] | 4.38 [2.69-5.30] | 4.38 [2.69-5.30] | 4.78 [3.79-5.33] | 5.34 [4.97-6.81] | 4.62 [3.52-5.30] | 5.17 [4.99-5.35]; >6.8 on 1 | 2/3 / 3/3 |
| 13 | 4.66 [3.25-5.32] | 4.57 [2.94-5.32] | 4.59 [2.94-5.34] | 4.24 [2.18-5.12] | 4.17 [2.09-5.12] | 4.44 [2.94-5.28] | 5.33 [4.85-6.74] | 4.35 [3.01-5.09] | 5.32 [4.97-7.85] | 1/3 / 1/3 |
| 12 | 4.54 [2.92-5.32] | 4.46 [2.48-5.30] | 4.47 [2.50-5.32] | 4.16 [1.84-5.09] | 3.98 [1.68-5.06] | 4.11 [2.48-5.03] | 5.25 [4.70-6.59] | 4.18 [2.72-4.95] | 5.32 [4.93-7.73] | 1/3 / 1/3 |
| 11 | 4.45 [2.71-5.30] | 4.30 [2.15-5.29] | 4.26 [2.12-5.30] | 3.88 [1.65-5.10] | 3.76 [1.44-4.98] | 3.88 [2.22-4.80] | 5.08 [4.51-5.23] | 3.99 [2.47-4.82] | 5.31 [4.90-7.40] | 1/3 / 1/3 |
| 10 | 4.42 [2.61-5.30] | 4.18 [1.96-5.26] | 4.20 [1.85-5.23] | 3.96 [1.59-5.01] | 3.70 [1.32-4.88] | 3.67 [2.02-4.62] | 4.51 [4.37-5.20] | 3.90 [2.34-4.68] | 5.32 [4.88-7.14] | 1/3 / 1/3 |
| 9 | 4.41 [2.57-5.29] | 4.13 [1.79-5.19] | 4.08 [1.47-5.18] | 3.69 [1.35-4.95] | 3.55 [1.09-4.80] | 3.50 [1.89-4.43] | 4.49 [4.08-5.00] | 3.80 [2.25-4.49] | 5.32 [4.88-6.74] | 1/3 / 1/3 |
| 8 | 4.40 [2.55-5.28] | 4.03 [1.43-5.13] | 4.00 [0.99-5.07] | 3.72 [0.89-4.72] | 3.47 [0.70-4.59] | 3.32 [1.77-4.20] | 4.38 [3.91-4.94] | 3.67 [2.20-4.37] | 5.32 [4.88-6.63] | 1/3 / 0/3 |
| 7 | 4.39 [2.56-5.25] | 3.97 [0.98-5.01] | 3.98 [0.99-4.94] | 3.67 [0.95-4.63] | 3.41 [0.69-4.49] | 3.15 [1.67-4.02] | 4.31 [3.84-4.83] | 3.56 [2.16-4.23] | 5.31 [4.87-6.64] | 1/3 / 0/3 |
| 6 | 4.39 [2.54-5.23] | 3.94 [0.98-4.88] | 3.93 [0.99-4.89] | 3.61 [0.97-4.51] | 3.31 [0.70-4.34] | 3.02 [1.59-3.81] | 4.32 [3.84-4.70] | 3.58 [2.15-4.07] | 5.31 [4.89-6.62] | 1/3 / 0/3 |
| 5 | 4.38 [2.54-5.20] | 3.90 [0.98-4.81] | 3.92 [0.99-4.80] | 3.68 [0.97-4.38] | 3.29 [0.70-4.15] | 2.99 [1.58-3.61] | 4.20 [3.84-4.65] | 3.44 [2.16-3.94] | 5.31 [4.88-6.62] | 1/3 / 0/3 |
| 4 | 4.39 [2.54-5.20] | 3.90 [0.98-4.74] | 3.92 [0.99-4.65] | 3.66 [0.97-4.16] | 3.25 [0.70-3.95] | 2.86 [1.58-3.43] | 4.02 [3.84-4.64] | 3.30 [2.15-3.85] | 5.76 [4.89-6.63]; >5.3 on 1 | 1/3 / 0/3 |
| 3 | 4.38 [2.54-5.17] | 3.90 [0.98-4.59] | 3.91 [0.99-4.59] | 3.86 [0.98-4.04] | 3.25 [0.69-3.79] | 2.71 [1.58-3.26] | 4.01 [3.85-4.63] | 3.30 [2.16-3.85] | 5.76 [4.89-6.64]; >5.3 on 1 | 2/3 / 1/3 |
| 2 | 4.38 [2.55-5.17] | 3.92 [0.99-4.56] | 3.93 [0.99-4.58] | 3.90 [0.98-3.93] | 3.25 [0.69-3.63] | 2.70 [1.57-3.24] | 4.01 [3.84-4.63] | 3.30 [2.16-3.84] | 5.76 [4.89-6.63]; >5.3 on 1 | 2/3 / 2/3 |
| 1 | 4.38 [2.54-5.18] | 3.91 [0.99-4.56] | 3.92 [0.99-4.56] | 3.53 [0.98-3.89] | 3.25 [0.69-3.30] | 2.70 [1.56-3.23] | 4.01 [3.83-4.63] | 3.31 [2.15-3.85] | 5.76 [4.89-6.63]; >5.2 on 1 | 2/3 / 3/3 |
| 0 | 4.37 [2.55-5.18] | 3.90 [0.99-4.55] | 3.91 [0.99-4.56] | 3.89 [0.99-4.53] | 3.25 [0.69-3.30] | 2.70 [1.56-3.23] | 4.02 [3.83-4.64] | 3.30 [2.16-3.85] | 5.76 [4.88-6.64]; >5.2 on 1 | 3/3 / 3/3 |

#### 8x8 (median over seeds [range]; bits)

| l | truth I*_l | K target T | K rate H | K carries I(c;next) | K knows I(c;c_L) | PR target | PR rate H | PR knows I(c;c_L) | direct rate for 0.95 I* | certified K / PR |
|---|---|---|---|---|---|---|---|---|---|---|
| 7 | 5.57 [4.65-5.58] | 5.58 [4.65-5.59] | 5.58 [4.65-5.59] | 5.58 [4.65-5.59] | 5.58 [4.65-5.58] | 5.58 [4.65-5.59] | 5.58 [4.65-5.59] | 5.58 [4.65-5.58] | - | - / - |
| 6 | 4.77 [4.45-4.81] | 4.78 [4.45-4.82] | 4.60 [4.32-4.72] | 3.73 [3.71-4.00] | 3.72 [3.70-4.01] | 4.78 [4.45-4.80] | 6.94 [4.63-7.26] | 4.55 [4.25-4.59] | 6.94 [4.63-7.01] | 3/3 / 3/3 |
| 5 | 4.37 [4.35-4.54] | 4.17 [4.04-4.22] | 4.11 [3.81-4.11] | 3.46 [3.06-3.83] | 3.21 [2.93-3.79] | 4.08 [4.08-4.27] | 6.37 [4.53-6.93] | 4.06 [4.06-4.25] | 7.01 [5.18-7.45] | 2/3 / 1/3 |
| 4 | 4.29 [4.17-4.40] | 3.88 [3.47-4.00] | 3.86 [3.47-3.90] | 3.31 [2.87-3.63] | 2.94 [2.61-3.51] | 3.81 [3.67-3.91] | 5.78 [4.47-6.33] | 3.88 [3.78-4.02] | 6.93 [5.18-7.25] | 2/3 / 2/3 |
| 3 | 4.19 [4.06-4.26] | 3.59 [3.19-3.74] | 3.55 [2.94-3.64] | 3.03 [2.46-3.33] | 2.68 [2.18-3.26] | 3.54 [3.31-3.56] | 5.47 [4.33-5.71] | 3.69 [3.61-3.79] | 6.91 [5.14-7.19] | 1/3 / 1/3 |
| 2 | 4.15 [3.98-4.23] | 3.45 [2.75-3.55] | 3.42 [2.71-3.46] | 3.09 [2.34-3.15] | 2.59 [2.06-3.06] | 3.28 [3.03-3.33] | 5.29 [4.21-5.35] | 3.55 [3.47-3.64] | 6.89 [5.15-7.38] | 1/3 / 0/3 |
| 1 | 4.07 [3.97-4.23] | 3.34 [2.61-3.35] | 3.22 [2.46-3.34] | 2.85 [2.16-2.93] | 2.52 [1.86-2.80] | 3.09 [2.86-3.12] | 5.31 [4.23-5.34] | 3.51 [3.48-3.65] | 7.14 [6.91-7.37]; >7.2 on 1 | 0/3 / 1/3 |
| 0 | 3.97 [3.93-4.22] | 3.12 [2.41-3.27] | 3.13 [2.14-3.17] | 2.58 [1.79-2.62] | 2.32 [1.57-2.59] | 2.96 [2.78-3.05] | 5.31 [4.22-5.35] | 3.45 [3.43-3.64] | 7.19 [6.89-8.47] | 0/3 / 3/3 |

#### Empirical bracket per layer: bits a code needs to carry 95% of what the layer knows about the output code

Lower end 0.95 I*_l (no code carrying that much can have less rate); upper end the least rate of a certified chain-PR code or a direct code that carries at least 0.95 I*_l on the test set. Median over seeds [range]; "-" = no such code found.

| config | l | 0.95 I*_l | upper end | upper / lower | truth loses from l+1 to l |
|---|---|---|---|---|---|
| 8x8 | 6 | 4.54 [4.23-4.57] | 6.94 [4.63-7.01] | 1.53 [1.09-1.53] | 0.77 [0.19-0.80] |
| 8x8 | 5 | 4.15 [4.13-4.31] | 7.01 [5.18-7.45] | 1.69 [1.25-1.73] | 0.27 [0.08-0.42] |
| 8x8 | 4 | 4.08 [3.97-4.18] | 6.93 [5.18-7.25] | 1.73 [1.27-1.75] | 0.14 [0.08-0.18] |
| 8x8 | 3 | 3.98 [3.86-4.05] | 6.91 [5.14-7.19] | 1.78 [1.29-1.79] | 0.11 [0.10-0.14] |
| 8x8 | 2 | 3.94 [3.78-4.02] | 6.89 [5.15-7.38] | 1.82 [1.31-1.83] | 0.04 [0.03-0.09] |
| 8x8 | 1 | 3.86 [3.77-4.02] | 7.14 [6.91-7.37] | 1.83 [1.83-1.84] | 0.01 [0.01-0.08] |
| 8x8 | 0 | 3.77 [3.74-4.01] | 7.19 [6.89-8.47] | 1.91 [1.85-2.11] | 0.03 [0.01-0.09] |
| 8x16 | 14 | 4.55 [3.60-5.07] | 5.16 [4.97-5.34] | 1.07 [1.05-1.09] | 0.21 [0.00-1.71] |
| 8x16 | 13 | 4.43 [3.08-5.05] | 5.32 [4.97-7.85] | 1.12 [1.05-2.54] | 0.13 [0.02-0.55] |
| 8x16 | 12 | 4.31 [2.77-5.05] | 5.32 [4.93-7.73] | 1.14 [1.05-2.79] | 0.12 [0.00-0.33] |
| 8x16 | 11 | 4.23 [2.57-5.04] | 5.31 [4.90-7.40] | 1.16 [1.05-2.88] | 0.09 [0.02-0.21] |
| 8x16 | 10 | 4.20 [2.48-5.04] | 5.32 [4.88-7.14] | 1.16 [1.06-2.87] | 0.03 [-0.00-0.09] |
| 8x16 | 9 | 4.19 [2.44-5.03] | 5.32 [4.88-6.74] | 1.16 [1.06-2.76] | 0.01 [0.01-0.05] |
| 8x16 | 8 | 4.18 [2.43-5.01] | 5.32 [4.88-6.63] | 1.17 [1.06-2.73] | 0.01 [0.01-0.01] |
| 8x16 | 7 | 4.17 [2.43-4.98] | 5.31 [4.87-6.64] | 1.17 [1.07-2.74] | 0.01 [-0.00-0.03] |
| 8x16 | 6 | 4.17 [2.41-4.96] | 5.31 [4.89-6.62] | 1.17 [1.07-2.75] | 0.02 [-0.00-0.02] |
| 8x16 | 5 | 4.16 [2.42-4.94] | 5.31 [4.88-6.62] | 1.17 [1.08-2.74] | 0.02 [-0.01-0.03] |
| 8x16 | 4 | 4.17 [2.41-4.94] | 5.76 [4.89-6.63] | 1.96 [1.17-2.75] | 0.00 [-0.01-0.01] |
| 8x16 | 3 | 4.16 [2.41-4.92] | 5.76 [4.89-6.64] | 1.96 [1.18-2.75] | 0.01 [-0.00-0.02] |
| 8x16 | 2 | 4.16 [2.42-4.91] | 5.76 [4.89-6.63] | 1.96 [1.18-2.74] | -0.00 [-0.01-0.00] |
| 8x16 | 1 | 4.16 [2.41-4.92] | 5.76 [4.89-6.63] | 1.96 [1.18-2.75] | 0.00 [-0.01-0.01] |
| 8x16 | 0 | 4.15 [2.42-4.92] | 5.76 [4.88-6.64] | 1.96 [1.18-2.74] | 0.00 [-0.01-0.01] |
| 16x8 | 6 | 4.01 [2.98-4.06] | - | - | 1.60 [1.54-2.73] |
| 16x8 | 5 | 3.35 [1.74-3.35] | - | - | 0.75 [0.70-1.30] |
| 16x8 | 4 | 2.66 [1.14-2.92] | 8.92 | 3.05 | 0.64 [0.44-0.72] |
| 16x8 | 3 | 2.14 [0.77-2.49] | 8.87 | 3.56 | 0.45 [0.39-0.54] |
| 16x8 | 2 | 1.79 [0.58-2.05] | 8.87 | 4.33 | 0.37 [0.20-0.47] |
| 16x8 | 1 | 1.53 [0.49-1.75] | - | - | 0.28 [0.10-0.32] |
| 16x8 | 0 | 1.32 [0.42-1.50] | - | - | 0.22 [0.07-0.26] |
| 16x16 | 14 | 3.86 [3.59-4.51] | - | - | 1.75 [1.00-2.12] |
| 16x16 | 13 | 3.16 [2.40-3.91] | 8.89 | 2.27 | 0.73 [0.63-1.24] |
| 16x16 | 12 | 2.69 [1.66-3.44] | 8.94 | 2.60 | 0.50 [0.50-0.78] |
| 16x16 | 11 | 2.17 [1.13-3.00] | 8.91 | 2.97 | 0.54 [0.46-0.56] |
| 16x16 | 10 | 1.83 [0.81-2.71] | 8.39 [7.94-8.84] | 3.80 [3.26-4.34] | 0.34 [0.30-0.36] |
| 16x16 | 9 | 1.65 [0.59-2.44] | 8.29 [7.87-8.71] | 4.17 [3.57-4.78] | 0.23 [0.19-0.29] |
| 16x16 | 8 | 1.55 [0.43-2.24] | 8.16 [7.75-8.56] | 4.42 [3.83-5.02] | 0.18 [0.11-0.22] |
| 16x16 | 7 | 1.47 [0.32-2.09] | 8.06 [7.66-8.47] | 4.62 [4.05-5.19] | 0.12 [0.07-0.15] |
| 16x16 | 6 | 1.42 [0.28-1.92] | 7.95 [7.57-8.32] | 4.83 [4.33-5.32] | 0.06 [0.04-0.18] |
| 16x16 | 5 | 1.38 [0.23-1.82] | 7.88 [7.54-8.23] | 4.98 [4.51-5.46] | 0.05 [0.04-0.10] |
| 16x16 | 4 | 1.35 [0.20-1.71] | 7.83 [7.52-8.15] | 5.16 [4.76-5.56] | 0.04 [0.03-0.12] |
| 16x16 | 3 | 1.31 [0.17-1.61] | 7.79 [7.49-8.08] | 5.38 [5.03-5.72] | 0.05 [0.03-0.11] |
| 16x16 | 2 | 1.25 [0.15-1.51] | 8.85 [8.82-8.88] | 6.48 [5.84-7.12] | 0.07 [0.02-0.10] |
| 16x16 | 1 | 1.20 [0.14-1.43] | 8.83 [8.80-8.87] | 6.78 [6.16-7.41] | 0.05 [0.01-0.09] |
| 16x16 | 0 | 1.14 [0.13-1.36] | 8.82 [8.79-8.86] | 7.11 [6.44-7.78] | 0.06 [0.01-0.07] |

![bits by layer](figs/bits_by_layer.png)

## 4. Where it breaks

**4.1 The target (C1, C6).** Passes at 262 of 264 steps of chains K and PR (the 2 exceptions: bootstrap SEs of 0.0100 and
0.0105 bit at 8x8 seed 5002, l = 3, 4; Miller-Madow and Grassberger agree there within 0.0003). Consistency
(T_l <= H(c_{l+1}), relevance <= min(rate, target)) holds at every step. The truth decreases toward the input on every
net within 0.01 + 3 SE; with the registered tolerance of 0.01 alone, 2 layers exceed it by 0.0008 and 0.0017 bit (8x16
seed 5000, l = 1; seed 5001, l = 0), within their sampling error. Convergence checks (chain K, seed 5000, l = L-2, middle,
0): 4x the noise draws and 2x the points move T, I(c_l; c_{l+1}), I*_l and I(c_l; c_L) by at most 0.015 bit; With M = 256 the
targets land within 0.014 bit but the truth I* can move by up to 0.032 bit (16x8 seed 5000, l = 0). MEASURED.

**4.2 The gap at a step (Props 2-3; MEASURED).** Two regimes, set by how much the noise erases (first-step table below):
- Real erasure (first step 0.8-2.7 bits erased; all 16-wide nets, 8x8 seeds 5000-5001, 8x16 seed 5001): the best code
  with rate <= T carries 51-78% of T at the first step, and every one of the 66 noisy steps on the 16-wide nets falls at
  least 10% short (median keeps 65%). Carrying 95% of T at the first step needs 6.7-7.9 bits (often more than 256 cells),
  above the output code's own 5.7-5.9 bits (on 7 of the 9 nets
  no candidate tried reached 95%; the best reached 84-94%); the noise-free reuse code carries 74-86% at about the output code's rate.
- Near-noiseless (8x16 seeds 5000, 5002; 8x8 seed 5002; first step 0.02-0.2 bit erased): the reuse code carries 90-99%
  of T at essentially the next code's rate. Here B1 fails (33 of 45 8x16 steps are within 10% of their target).

#### First backward step (l = L-2): the gap, per net

| net | H(c_L) | erased | T | rate rule: H, I (I/T) | 95% rule (R): H, I | reuse: H, I |
|---|---|---|---|---|---|---|
| 16x16 s5000 | 5.74 | 1.00 | 4.74 | 4.76, 3.47 (73%) | 7.78, 4.43 | 5.74, 4.09 |
| 16x16 s5001 | 5.90 | 2.11 | 3.78 | 3.79, 2.29 (60%) | 6.95, 3.22 | 5.85, 3.02 |
| 16x16 s5002 | 5.82 | 1.75 | 4.06 | 3.93, 2.67 (66%) | 7.91, 3.78 | 5.76, 3.43 |
| 16x8 s5000 | 5.82 | 1.54 | 4.28 | 4.13, 2.86 (67%) | 7.89, 3.89 | 5.79, 3.56 |
| 16x8 s5001 | 5.85 | 2.72 | 3.13 | 2.97, 1.59 (51%) | 6.94, 2.63 | 5.73, 2.43 |
| 16x8 s5002 | 5.81 | 1.59 | 4.23 | 4.26, 3.01 (71%) | 7.90, 3.91 | 5.78, 3.58 |
| 8x16 s5000 | 5.34 | 0.02 | 5.33 | 5.35, 5.30 (99%) | 5.36, 5.30 | 5.35, 5.30 |
| 8x16 s5001 | 5.52 | 1.72 | 3.80 | 3.78, 2.69 (71%) | 6.71, 3.51 | 5.38, 3.30 |
| 8x16 s5002 | 4.98 | 0.20 | 4.79 | 4.73, 4.38 (91%) | 4.99, 4.62 | 4.98, 4.61 |
| 8x8 s5000 | 5.58 | 0.80 | 4.78 | 4.72, 3.73 (78%) | 6.98, 4.56 | 5.56, 4.29 |
| 8x8 s5001 | 5.59 | 0.78 | 4.82 | 4.60, 3.71 (77%) | 7.23, 4.59 | 5.58, 4.31 |
| 8x8 s5002 | 4.65 | 0.19 | 4.45 | 4.32, 4.00 (90%) | 4.64, 4.26 | 4.63, 4.24 |

**4.3 Accumulated error (MEASURED).** What each chain's code at the input knows about the output code, against the truth:

#### At the input (l = 0): share of the truth each chain keeps, per net

| net | I*_0 | K: I_out / I* | R: I_out / I* | P: I_out / I* | PR: I_out / I* | PR: rate | PR: I_chain / I_out | K target T_0 | R target T_0 |
|---|---|---|---|---|---|---|---|---|---|
| 16x16 s5000 | 1.44 | 2% | 97% | 0% | 75% | 2.48 | 88% | 0.16 | 5.80 |
| 16x16 s5001 | 0.13 | 24% | 93% | 0% | 77% | 1.18 | 90% | 0.09 | 6.47 |
| 16x16 s5002 | 1.20 | 19% | 97% | 0% | 80% | 2.93 | 87% | 0.38 | 6.02 |
| 16x8 s5000 | 1.39 | 21% | 93% | 0% | 91% | 7.45 | 93% | 0.57 | 4.51 |
| 16x8 s5001 | 0.44 | 15% | 91% | 0% | 82% | 4.58 | 86% | 0.18 | 4.14 |
| 16x8 s5002 | 1.58 | 14% | 94% | 1% | 89% | 7.51 | 94% | 0.53 | 4.67 |
| 8x16 s5000 | 5.18 | 64% | 90% | 0% | 74% | 4.64 | 95% | 4.55 | 5.14 |
| 8x16 s5001 | 2.55 | 27% | 99% | 0% | 85% | 3.83 | 92% | 0.99 | 7.79 |
| 8x16 s5002 | 4.37 | 74% | 95% | 0% | 76% | 4.02 | 93% | 3.90 | 4.87 |
| 8x8 s5000 | 3.93 | 40% | 97% | 0% | 88% | 5.31 | 96% | 2.41 | 7.73 |
| 8x8 s5001 | 4.22 | 55% | 97% | 0% | 86% | 5.35 | 97% | 3.27 | 7.90 |
| 8x8 s5002 | 3.97 | 65% | 94% | 30% | 86% | 4.22 | 97% | 3.12 | 4.63 |

- K (your rule): 2-24% of the truth on the 16-wide nets, 27-74% on the 8-wide ones; its targets end far below the
  truth (B7: 11 of 12). Its own chain of cell kernels predicts the output code worse still.
- P (output target, rate rule): 0% on 11 of 12 nets. With the target routed through the cell tables, the reuse code
  never fits under the rate rule, so every step compresses and the losses compound to nothing.
- R (next-code target, 95% rule): 90-99% of the truth, because it keeps 5-9-bit codes at every layer; its targets at the
  input (4.1-7.9 bits) exceed the truth (0.13-5.2) on 11 of 12 nets (B6): the over-count of FORMAL Prop 6, measured.
- PR (output target, 95% rule): 74-91% at 1.2-7.5 bits; its own kernel table predicts the output code at 86-97% of what
  the code knows (B8 needed 90% on 9 nets: 8). PR's own targets undercount the truth (FORMAL section 7: exact cell
  averages are not sufficiency).
- PREREG section 6 ("certified all the way": every step certified on >= 2 seeds and >= 80% of I* at every layer):
  met by 16x8 PR, 8x16 R and 8x8 R only; not by your rule on any config. Caveat (review r2): for the 95%-rule chains
  the restart and depth certificates compare rates even at steps where 95% was not reached (17 PR steps, most R steps
  on the 16-wide nets); there they certify a stable fallback, not attainment of 95%.

**4.4 Encoder depth (C2, C4, C5; MEASURED).**

#### Deeper encoders: does depth help, and where is the reuse composition chosen?

| config | C4 fails (depth 4 beats depth 2), K | C4 fails, PR | excess CE (bits), K, upper half of layers | excess CE, K, lower half | reuse selected: K / R / P / PR (steps) | direct predictor excess CE, upper / lower half |
|---|---|---|---|---|---|---|
| 8x8 | 4/21 | 4/21 | 0.030 [0.010-0.049] | 0.065 [0.019-0.293] | 0 / 15 / 0 / 7 | 0.030 [0.010-0.047] / 0.070 [0.017-0.409] |
| 8x16 | 6/45 | 12/45 | 0.086 [0.006-0.241] | 0.188 [0.008-1.090] | 14 / 35 / 8 / 20 | 0.091 [0.011-0.251] / 0.321 [0.038-1.867] |
| 16x8 | 3/21 | 0/21 | 0.014 [0.004-0.018] | 0.007 [0.004-0.014] | 1 / 6 / 4 / 18 | 0.017 [0.014-0.018] / 0.019 [0.015-0.035] |
| 16x16 | 0/45 | 2/45 | 0.006 [0.001-0.022] | 0.003 [0.001-0.023] | 5 / 15 / 17 / 24 | 0.014 [0.011-0.017] / 0.014 [0.012-0.056] |

Where the noise smooths the posteriors (16-wide), a depth-2 predictor is enough (excess cross-entropy 0.003-0.02 bit;
depth 4 beats it beyond tolerance at 0-3 of 45 steps). On the 8-wide nets, where the noise is tiny, the posterior is
almost a sharp partition (the next code composed with the layer): the predictor's excess cross-entropy grows toward the
input (up to 1.1 bits), depth 4 helps at up to 12 of 45 steps, and clustering the exact Monte Carlo posteriors beats the
learned code beyond 0.03 bit at 32 of 66 8-wide K steps (C5; 3 of 66 on the 16-wide nets). There the code that keeps the bits is your "use the last encoding", the
exact composition, one net layer deeper per reuse; the chains chose it at up to 35 steps per config.

#### Certificate pass rates (chains K and PR, all steps)

| certificate | K | PR |
|---|---|---|
| C1 | 130/132 (98%) | 132/132 (100%) |
| C2 | 108/132 (82%) | 106/132 (80%) |
| C3 | 124/132 (94%) | 129/132 (98%) |
| C4 | 119/132 (90%) | 114/132 (86%) |
| C5 | 97/132 (73%) | - |
| C6 | 132/132 (100%) | 132/132 (100%) |
| certified | 87/132 (66%) | 90/132 (68%) |

**4.5 Verdict.** The target passes its checks at 262 of 264 steps; the codes are certified at 66-68% of steps (failures concentrated on
the low-noise 8-wide nets: C2, C4, C5); the recursion gives counts of output-relevant bits all the way to the input only
in the combinations of 4.3, and with your rule in none.

## 5. Shared-noise output distortion and forward-trained codes

What is measured (run_truth.py, run_fwd.py): for a code of layer l, each cell is decoded to a state (the cell's mean
state, or states optimised for this score), and the decoded state runs through the real rest of the net with the same
future noise as the sample: crn = E|F(h_hat, Xi) - F(h, Xi)|^2 / V (the bit-accounting S4 score). Section 2, point 5
says why this is a different count: with the noise shared, nothing is erased. The forward-trained codes are K-cell codes
trained directly on crn (an MLP encoder with a straight-through hard assignment and learned cell states), at 5 layers per
net, K = 2 ... 64; they sit on the same (rate, crn) and (rate, information about c_L) planes as the chains' codes.

Results (results/scores.json; crn with the cell-mean decoder, normalised by the output variance; 0-bit code about
0.8-1.6): at the input the PR code's crn is below the K code's on 11 of 12 nets (B12), e.g. 16x16 seed 5000: K 1.71,
PR 0.67, direct code at rate I* 0.69; at l = 14 there: K 0.13, PR 0.05. Shared-noise error follows the information a
code carries. Forward-trained codes (seed 5000, 3 layers; table) carry less about the output code than the chains' codes
at similar rate and have higher crn at l = L-2 (16x16: 64 cells, rate 4.83, 2.67 bits known, crn 0.21; chain K: rate
4.76, 3.47 bits, crn 0.13); at the input layers their training collapses to about 0 bits. So the backward codes are the
better codes here; forward training of this kind is not a usable baseline at the input.

#### Forward-trained codes (trained on the shared-noise distortion) against the chains' codes

| net | l | I*_l | forward codes: K cells -> (H, I(c;c_L), crn) | chain K (H, I_out, crn) | chain PR (H, I_out, crn) |
|---|---|---|---|---|---|
| n16L16_s5000_sig0.1_K64 | 14 | 4.73 | 2: (0.93, 0.55, 0.584); 4: (1.82, 1.14, 0.435); 8: (2.49, 1.52, 0.358); 16: (3.32, 1.99, 0.280); 32: (3.92, 2.32, 0.264); 64: (4.83, 2.67, 0.205) | (4.76, 3.47, 0.130) | (7.78, 4.44, 0.051) |
| n16L16_s5000_sig0.1_K64 | 7 | 2.19 | 2: (0.96, 0.52, 0.495); 4: (1.85, 0.91, 0.325); 8: (2.48, 1.05, 0.286); 16: (3.03, 1.20, 0.317); 32: (3.86, 1.45, 0.209); 64: (3.68, 1.32, 0.233) | (0.88, 0.54, 0.623) | (7.33, 2.05, 0.068) |
| n16L16_s5000_sig0.1_K64 | 0 | 1.44 | 2: (0.24, 0.16, 0.660); 4: (0.05, 0.03, 0.802); 8: (0.07, 0.05, 0.995); 16: (0.03, 0.02, 0.831); 32: (0.00, 0.00, 0.868); 64: (0.01, 0.01, 0.940) | (0.16, 0.04, 1.708) | (2.48, 1.08, 0.667) |
| n16L8_s5000_sig0.2_K64 | 6 | 4.27 | 2: (0.95, 0.55, 0.545); 4: (1.88, 1.06, 0.398); 8: (2.53, 1.42, 0.344); 16: (3.32, 1.75, 0.312); 32: (4.12, 2.07, 0.253); 64: (4.68, 2.38, 0.255) | (4.13, 2.85, 0.140) | (7.90, 3.89, 0.070) |
| n16L8_s5000_sig0.2_K64 | 3 | 2.25 | 2: (0.84, 0.44, 0.469); 4: (1.45, 0.74, 0.336); 8: (2.14, 0.99, 0.275); 16: (3.29, 1.19, 0.258); 32: (3.98, 1.39, 0.205); 64: (4.48, 1.52, 0.164) | (1.57, 0.92, 0.388) | (7.67, 2.07, 0.056) |
| n16L8_s5000_sig0.2_K64 | 0 | 1.40 | 2: (0.35, 0.19, 0.584); 4: (0.24, 0.14, 0.628); 8: (0.00, 0.00, 0.804); 16: (0.00, 0.00, 0.798); 32: (0.01, 0.01, 1.061); 64: (0.07, 0.04, 0.893) | (0.47, 0.29, 1.015) | (7.45, 1.26, 0.347) |
| n8L16_s5000_sig0.0003_K64 | 14 | 5.33 | 2: (0.93, 0.65, 0.434); 4: (1.55, 1.12, 0.229); 8: (2.16, 1.69, 0.206); 16: (2.64, 2.38, 0.037); 32: (2.47, 1.77, 0.171); 64: (3.01, 2.40, 0.287) | (5.35, 5.30, 0.002) | (5.34, 5.30, 0.002) |
| n8L16_s5000_sig0.0003_K64 | 7 | 5.24 | 2: (0.93, 0.62, 0.428); 4: (1.30, 0.94, 0.239); 8: (1.89, 1.50, 0.138); 16: (2.07, 1.62, 0.121); 32: (2.52, 2.08, 0.063); 64: (1.46, 1.11, 0.171) | (4.94, 4.49, 0.019) | (4.83, 4.23, 0.020) |
| n8L16_s5000_sig0.0003_K64 | 0 | 5.17 | 2: (0.00, 0.00, 1.003); 4: (1.23, 0.70, 1.263); 8: (0.00, 0.00, 1.000); 16: (0.97, 0.73, 1.152); 32: (0.00, 0.00, 1.001); 64: (0.07, 0.06, 1.247) | (4.56, 3.30, 0.738) | (4.64, 3.85, 0.707) |
| n8L8_s5000_sig0.01_K64 | 6 | 4.78 | 2: (0.90, 0.56, 0.584); 4: (1.89, 1.19, 0.411); 8: (2.68, 1.77, 0.343); 16: (2.94, 1.94, 0.277); 32: (3.33, 2.20, 0.308); 64: (3.72, 2.48, 0.232) | (4.72, 3.72, 0.084) | (6.94, 4.55, 0.027) |
| n8L8_s5000_sig0.01_K64 | 3 | 4.07 | 2: (0.95, 0.59, 0.599); 4: (1.92, 1.08, 0.519); 8: (2.22, 1.22, 0.437); 16: (2.79, 1.79, 0.363); 32: (3.14, 1.88, 0.313); 64: (2.20, 1.42, 0.445) | (2.94, 2.18, 0.335) | (5.47, 3.61, 0.055) |
| n8L8_s5000_sig0.01_K64 | 0 | 3.94 | 2: (0.08, 0.07, 1.021); 4: (0.92, 0.69, 1.053); 8: (1.07, 0.40, 1.056); 16: (1.99, 0.86, 0.860); 32: (0.02, 0.02, 1.061); 64: (0.59, 0.46, 0.866) | (2.14, 1.57, 0.687) | (5.31, 3.45, 0.167) |

## 6. Gaussian-channel version

What is measured (run_gauss.py; chain K's steps at 5 layers per net): a real-valued code z = u(h_l) + N(0, I_8) trained
for the same target (the next code), for 8 values of the rate price. Rate I(h_l; z): Monte Carlo information of the
empirical mixture, with and without the source component and at N and N/2 (the Kolchinsky-Tracey pairwise bounds are
reported too but are up to 1.8 bits apart, too loose to use); relevance: a decoder lower bound and the mixture-posterior
plug-in (at N and N/2). Exact reference (section 7.1): in 1-D a linear analog code at rate T carries 72% of T against
61% for the best partition.

Amended (D8): the registered rate price saturated (rates of about 14 bits on the 16-wide nets), so the comparison uses
power-constrained codes (per-dimension SNR 0.03-4), seed 5000 of each config, 3 layers. The information estimates are
reliable: the four Monte Carlo rate estimates agree within 0.15 bit at 98% of points with rate <= 6 (B13); relevance by
decoder bound and mixture plug-in agree within 0.065 bit (median 0.005).

#### Gaussian channel against partitions at rate T (chain K steps)

| net | l | T | partition (H <= T): I | Gaussian at info rate T: I |
|---|---|---|---|---|
| n16L16_s5000_sig0.1_K64 | 14 | 4.75 | 3.47 (73%) | 3.33 (70%) |
| n16L16_s5000_sig0.1_K64 | 7 | 0.88 | 0.40 (46%) | 0.65 (74%) |
| n16L16_s5000_sig0.1_K64 | 0 | 0.15 | 0.08 (51%) | 0.07 (44%) |
| n16L8_s5000_sig0.2_K64 | 6 | 4.28 | 2.86 (67%) | 2.81 (66%) |
| n16L8_s5000_sig0.2_K64 | 3 | 1.62 | 1.01 (62%) | 1.02 (63%) |
| n16L8_s5000_sig0.2_K64 | 0 | 0.58 | 0.26 (46%) | 0.37 (65%) |
| n8L16_s5000_sig0.0003_K64 | 14 | 5.34 | 5.30 (99%) | 4.37 (82%) |
| n8L16_s5000_sig0.0003_K64 | 7 | 5.00 | 4.63 (92%) | 3.94 (79%) |
| n8L16_s5000_sig0.0003_K64 | 0 | 4.54 | 4.53 (100%) | 2.54 (56%) |
| n8L8_s5000_sig0.01_K64 | 6 | 4.78 | 3.73 (78%) | 3.57 (75%) |
| n8L8_s5000_sig0.01_K64 | 3 | 3.19 | 2.46 (77%) | 2.40 (75%) |
| n8L8_s5000_sig0.01_K64 | 0 | 2.41 | 1.79 (74%) | 1.61 (67%) |

At information rate T (amended comparison, D8) the analog code carries 44-82% of T, about the same as the partition (B14 missed: better by >= 0.1
bit at 2 of 12 points; worse on the near-noiseless 8x16 net). The gap of section 4.2 is not an artefact of discrete
codes, as Proposition 2 says for any code; in the exact 1-D case the linear analog code does better (72% vs at least 67%).

## 7. Exactly solvable checks (no training, no Monte Carlo)

**7.1 One step (scripts/exact_1d.py, figs/exact_1d.png; MEASURED exactly).** h ~ N(0, 1); the next code is a 16-level
quantiser of h + 0.3 xi. T = 1.708 bits. Among optimal fixed-cell-count partitions, the best with rate at most T carries
1.04 bits (61% of T); an entropy-constrained four-interval partition (cuts -1.30, 0, 1.30; rate 1.7075) carries 1.14
(67%; found by the round-2 reviewer). A linear Gaussian-channel code z = a h + N(0, 1) at information rate T carries
1.24 (72%). Carrying 95% of T takes 3.5
bits as a partition and 3.2 as a linear analog code. The shortfall halves for every 0.5 extra bit of partition rate
(d = 1 in section 2's heuristic). The trained pipeline (predictor + clustering) matches the dynamic-programming
optimum within 0.004 bit at 2, 4 and 8 cells (tests/test_output.txt).

**7.2 The whole recursion on a 1-D Gaussian chain (scripts/exact_chain_1d.py; results/exact_chain_1d_*.json).**
h_l = rho h_{l-1} + sqrt(1 - rho^2) xi_l, 16 layers, output code a 32-cell Lloyd-Max quantiser (4.92 bits). Every code is
the optimal interval partition (dynamic programming over cell counts and over a rate penalty), every posterior is a
normal-CDF difference: the truth and every posterior are exact (quadrature), the codes are the best within a large
candidate set (not the full rate-constrained optimum). Candidates: the optimal
partition for every cell count up to 96, the rate-penalised optima, every 2-cell code and rate-penalised 3-6-cell codes.
rho = 0.999 (the first step erases 0.94 bit, as on the nets):

| l | truth I*_l | K: target / rate / knows | P: target / knows | R: target / rate / knows | PR: target / rate / knows | direct: rate for 95% of I* |
|---|---|---|---|---|---|---|
| 14 | 3.98 | 3.98 / 3.97 / 3.39 | 3.98 / 3.39 | 3.98 / 4.92 / 3.78 | 3.98 / 4.92 / 3.78 | 4.93 |
| 12 | 3.44 | 3.08 / 2.96 / 2.51 | 2.33 / 2.04 | 3.97 / 4.92 / 3.31 | 3.02 / 4.23 / 3.17 | 4.65 |
| 9 | 3.05 | 2.33 / 2.29 / 1.90 | 0.65 / 0.59 | 3.97 / 4.92 / 2.95 | 2.22 / 3.60 / 2.67 | 4.40 |
| 5 | 2.73 | 0.91 / 0.92 / 0.79 | 0.00 / 0.02 | 3.97 / 4.92 / 2.66 | 1.55 / 2.87 / 2.17 | 4.12 |
| 0 | 2.47 | 0.79 / 0.81 / 0.66 | 0.00 / 0.01 | 3.97 / 4.92 / 2.42 | 1.09 / 2.57 / 1.83 | 4.03 |

("knows" = I(c_l; c_L), the code's information about the output code with the best decoder.)
- Your rule as written (K) keeps 77-86% of its target per step while the codes are fine, then collapses to a 2-cell code
  (about 0.8 bit, l <= 5) that the noise barely erodes, and stays there: at the input it knows 0.66 of the 2.47 bits the
  input has about the output code (27%). With the output code as the target (P) the rate rule does worse: the cell
  tables blur the target, which falls to 0.001 bit.
- The 95% rule with the next code as target (R) picks the reuse code at every step: the output code's partition pulled
  back through the noise-free maps. It never compresses (4.92 bits at every layer) and its target stays at 3.98 bits
  while the truth falls to 2.47: the over-count of section 2, point 4, exactly.
- The 95% rule with the output code as target (PR) compresses (2.57 bits at the input) and keeps 1.83 of the 2.47 bits
  (74%). Its codes are as good as the best direct codes at their own rates (within 0.05 bit at every layer; at the input
  the best direct code with rate 2.48 carries 1.85); what it loses comes from its target, which is computed through the
  chain's cell tables and undercounts (1.09 bits at the input against 2.47).
- The count (best within the candidate set): carrying 95% of what layer l knows about the output code takes 4.0-4.9 bits at every layer, while
  what it knows falls from 3.98 to 2.47 bits. At the input that is 4.03 bits of rate for 2.47 bits of information.
- How strict the corrected rule must be (chain PR at other thresholds, rho = 0.999; figs/exact_chain_1d.png): keeping
  90% of the target at every step keeps 68% of the truth at the input (rate 2.20 bits); 95%: 74% (2.57); 99%: 98% (4.68
  bits at the input, 6.14 at the first step, above the output code's 4.92); 99.9%: 99.9% (6.8-7.5 bits at every layer).
  So the recursion can carry the surviving bits all the way down, but only by paying about twice the surviving
  information in rate, and by letting early codes exceed their successors.
- rho = 0.998 (more noise per step): K settles at about 0.7 bit and knows 0.52 of 2.01 bits at the input (26%); P fades
  to 0.01; R stays at 4.75 bits per layer with its target stuck at 3.65 (99% of the truth kept, no compression); PR
  keeps 1.49 of 2.01 bits (74%) at 2.22 bits; 95% of the truth at the input takes 3.70 bits.
- A coarser output code (16 cells, 3.87 bits, output nMSE 1.0%; rho = 0.999): K knows 26% of the truth at the input,
  P about 0.5%, PR 76% at 2.52 bits; R's rate climbs to 4.88 bits, above the output code's own 3.87; 95% of the truth at
  the input takes 3.87 bits for 2.36 bits of information. The picture is the same for the two output-code sizes tested.
- Grid check (review r1 #20): on a grid twice as fine (1601 points) the input-layer numbers move by at most about 0.1
  bit (truth 2.48; K knows 0.67, 27%; PR 1.83 at 2.48 bits, 74%). The rate-rule answers are "best within the candidate
  set"; the rate-constrained frontier is not fully enumerated (FORMAL section 10).

![exact 1-D chain](figs/exact_chain_1d.png)

**7.3 The recursion with analog codes, in closed form (scripts/exact_gauss_chain.py).** Same chain, but every code is an
analog channel z_l = b_l h_l + N(0, 1) (for jointly Gaussian variables the information-bottleneck optimum has this form),
the output code an analog code carrying 4.92 bits. Everything is a closed-form Gaussian information.
- Your rule (rate = target): the target equals the truth at every layer (rho = 0.999: 4.17, 3.57, 2.94, 2.67, 2.52 bits
  at l = 14, 12, 7, 3, 0, each equal to I(h_l; z_out)). No drift: the analog code chosen by your rule reproduces the
  next code's statistics exactly (FORMAL Proposition 7). As predictors of the output these codes know 88% (l = 14) to 81%
  (l = 0) of the truth. rho = 0.998: target = truth at every layer again; the codes know 87% to 77%.
- 95% rule: the rate stays at about 4.97 bits at every layer (no compression), the target stays at 4.19 bits while the
  truth falls to 2.52, and the codes know 95-99% of the truth: the same over-count as chain R in 7.2.
So the procedure's arithmetic ("bits of layer l minus the erased bits") is exact for codes that re-generate the next
code's statistics (sufficient; not shown to be necessary). For such codes the information rate is I*_l; the bits that
the code's own private noise decides (the entropy-sense erasure) are H(c_L) - I*_l, and I*_{l+1} - I*_l is the information
lost across the transition.

## 8. Pre-registration scorecard

Registered 14:17:41Z, amendments A1-A6 (notes/DEVIATIONS.md D1-D9; no threshold changed). B1-B12 were run as registered:
6 hit, 6 missed. B13-B14 were re-run after the outcome of the registered version was seen (D8: amended codes, seed 5000
only, 12 layers), so they are not registered outcomes: 1 hit, 1 missed.

| | prediction | result |
|---|---|---|
| B1 | rate-rule deficit >= 10% of T at >= 90% of noisy steps | MISS: 73% (100% on the 16-wide nets; fails on the near-noiseless 8x16 nets) |
| B2 | 95% rule needs >= T + 1 bit at the first step on >= 10 of 12 | MISS: 9 of 12 (the 3 near-noiseless nets need no extra rate) |
| B3 | ... and exceeds the output code by > 0.5 bit on >= 9 of 12 | HIT: 9 of 12 |
| B4 | your rule keeps <= 60% of the truth at the input on >= 9 of 12 | HIT: 9 of 12 |
| B5 | PR keeps >= 80% at the input on >= 9 of 12 | MISS: 7 of 12 (74-91%) |
| B6 | R's target exceeds the truth by >= 0.25 bit at the input on >= 9 | HIT: 11 of 12 |
| B7 | K's target falls >= 0.25 bit below the truth at the input on >= 9 | HIT: 11 of 12 |
| B8 | PR's kernel chain predicts >= 90% of what its code knows, >= 9 nets | MISS: 8 of 12 (86-97%) |
| B9 | direct code at K's rate within 0.15 bit of K's code at >= 80% | HIT: 95% |
| B10 | depth 4 never beats depth 2 beyond tolerance at >= 90% of steps | MISS: 88% (fails on the 8-wide nets) |
| B11 | every certificate passes at >= 90% of K and PR steps | MISS: 67% |
| B12 | PR's code has lower shared-noise error than K's at the input, >= 9 | HIT: 11 of 12 |
| B13 | Gaussian rate estimates agree within 0.15 bit at >= 90% | HIT: 98% (amended power-constrained codes, D8) |
| B14 | Gaussian code beats the partition at rate T by >= 0.1 bit at >= 75% | MISS: 17% (amended codes, D8) |


## 9. Process

- Timeline (2 Nov, UTC): brief about 12:20; QUESTIONS and FORMAL 12:40-13:00; library and known-answer tests (19/19)
  13:00-13:20; dev calibration on seed 9100 (CPU) 13:10-13:40; GPU dev runs 13:45-14:17; PREREG registered 14:17:41;
  scored runs from 14:18, restarted twice before results (scored3 from 15:21; chains done 18:50, evaluations 20:39);
  exact 1-D checks on CPU while the GPU ran.
- Deviations (notes/DEVIATIONS.md, each with new hashes in prereg_hashes.txt): D1 CUDA out-of-memory at the first launch
  (dev and scored queues together), relaunch with 6 workers; D2 the registered candidate set lacked your "use the last
  encoding" option, found on the first scored steps (8x16 seed 5000, where the noise erases 0.02 bit and a trained
  code loses to the reuse code), added as a selectable candidate, scored chains restarted from scratch (amendment A2);
  D3 the Gaussian and forward comparisons run at 5 layers per net (as in the hashed job file), not at every step as the
  PREREG text said; D4 a dependency scheduler replaced the two-phase pool (same command lines); D5 the round-1 review
  fixes (test leakage in P/PR selection, empty table rows, a crash path, the scorer) and a full restart; D6-D7 spot
  preemptions and an on-demand VM; D8 the last evaluations reduced to seed 5000 per config and the Gaussian comparison
  re-run with power-constrained codes (the registered rate price saturated). Rules, tolerances and the 14 predictions
  were not changed. The PREREG text lists the first dev run's budgets; the hashed code (what ran) used smaller ones (D5).
- Compute: CPU at nice 19 / ionice 3, at most 2 heavy processes (calibration, tests, exact 1-D checks); GPU one L4 at a
  time (GPU-LOG.md, budget group backbits, cap $10): five VMs (two spot preemptions, one stockout, then on demand),
  $5.68 at the $0.87/h ceiling rate; all deleted (last at 20:40Z).
- Review: gpt-6-astra via codex-any, round 1 on FORMAL, PREREG and the code while the scored runs ran (review/r1),
  round 2 on this report (review/review_r2.raw, model line verified): 14 findings, no number found wrong in the
  2,880 checked against the raw runs; the interpretation fixes (unreached 95% codes, the exact 1-D figure, C1 failures,
  amended predictions, candidate-set labels) are applied here; two residual leaks in the direct comparator are
  disclosed (D9) and not rerun.
