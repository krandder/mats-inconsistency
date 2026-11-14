# Rich-observation agents: design (opus-mats, 2026-10-22)

Kelvin: "Its ok you got a minor result first, but then you need to continue." Background: the role-keys toy gave agents trivial ±1 readings, so nothing was extracted ("almost useless").

## Toy (extends work/role-curves; exact ceilings kept)
- **Latents.** Bits b1..b4, with Y = b1⊕b2⊕b3 and b4 a distractor. Plus nuisance g ~ N(0, I_8): continuous latent features unrelated to Y.
- **Agents.** N=6 per episode; role r_i = which bit the agent reads (fresh every episode).
  - Observation: x_i = a·U_{r_i}(2b_{r_i}−1) + V_i g + ε_i ∈ R^64, where U_r is a fixed random role direction, V_i a random nuisance mixing, and ε isotropic noise.
  - So the bit is hidden in a 64-dim vector, among nuisance features and noise. The amplitude a sets per-agent informativeness.
  - Given role and bit, x_i | r, b is Gaussian, so exact Bayes stays computable by enumeration (roles 4^6 × bits 2^4) with g integrated out.
- **Key.** key_i = W·r_i + σ·ε_i, as before.
- **Agent encoder (learned).** It reads only its own x_i (and, in the queried arm, the earlier agents' messages, read-only) and writes a message: wide, or FSQ b bits.
- **Aggregator.** A transformer over [key, message] tokens producing P(Y). Trained on Y labels only; masked-agent pretraining allowed. Fresh episodes.

## Yardsticks
- (1) exact Bayes with true roles given all x;
- (2) the exact keys+content ceiling (role posterior from keys and x);
- (3) the transformer bar: one network reading all raw x and keys, same data.
- Probes: does each message linearly encode b_{r_i}? Does it encode the nuisance g (which would be wasted capacity)?

## Sweeps
- Observation dimension {16, 64, 256};
- nuisance strength;
- a (signal amplitude);
- sigma (key noise) at 0.5;
- message wide vs b in {2, 4};
- one-shot vs queried.

## Fit to Kelvin's six points
| Point | Covered? |
|---|---|
| 1. Agents with large private evidence must find what matters | yes |
| 2. Rich, aggregation-aware messages written after reading the consensus | yes, queried arm |
| 3. Aggregator learns who knows what, near Bayes, never worse than the transformer bar | yes |
| 4. Structure learned through local market signals | NOT YET (end-to-end training); next step after this works |
| 5. Search and alignment | n/a here |
[removed]

## Forecasting world (aggbench/forecast_world.py, already being built)
Redirected so forecasters are LEARNED networks over the raw source observations they read, not exact local posteriors. The exact-local-posterior arm stays only as a labelled reference.
