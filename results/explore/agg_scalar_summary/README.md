# EXPLORATORY: Diagnosing & Solving Scalar-Report Information Aggregation Across `aggbench` (`expaggscalar`)

Serves: auditable belief clearing where confiability (theta_trust in params_real) and Key-Gram de-duplication recover oracle_clearing from scalar reports across overlapping, Sybil, colluding, and biased sources.
Builds on: `aggbench/examples/smoke_attention.py`, `results/explore/l1-learned-keys-2026-10-20/`.

**Overall Status:** `● Informative (Scalar-Report Baseline; Aligned-Slot Side-Line Diagnostic)` (Honest audit vs. opaque-key `L1` and unaligned `TraderTransformer`: `key_gram_clear` and `key_gram_confiability_clear` in this run operate on 1D scalar reports under pre-`1c2e780` `alignment="aligned"` and parse the known `aggbench` key layout `(A, Bf, log_sig)` on `E1` and `(c, c*mu)` on `E2/E3`, testing the **Confiability Head ($\theta_{\text{trust}} \in \text{params}_{\text{real}}$) and noisy-key subspace calibration on top of structured scalar Key-Gram clearing** rather than learning rich trader tokens under `alignment="unaligned"` in `aggbench/trader_transformer.py`) · **Track:** `EXPLORATORY` (`train` $n=2000$, `dev` $n=2000$, `test` untouched; `harness.git_dirty = False`; harness commit `ea608bcc3c`; runner SHA256 `b4f7dde00a65e2c7`)

## Why
Standard token attention (`aggbench.examples.smoke_attention` / `naive_mha`) got `0.241` regret on `E2` — $5.4\times$ worse than 1-parameter `extremized_log_pool` (`0.045`) and $14\times$ worse than `oracle_clearing` (`0.017`). We tested three exact structural diagnoses across all 5 `aggbench` environments (`E1`, `E2`, `E3-sybil`, `E3-biased`, `E3-collude`) at `full` and `noisy` key visibility:
1. **Broken Key-Slot Equivariance ($S_K$ / $O(d)$):** Dense `nn.Linear(1 + key_dim, width)` treats permuted evidence slots $k \in \{1..K\}$ (or coordinates $d=4$) as fixed features and memorizes training noise.
2. **Broken Value Linearity & Sign Equivariance:** Concatenating `[report, key]` into an MLP destroys odd-function linearity in reports $r_i - \ell_0$ and precision-weighted information $h_i = P_i m_i$.
3. **Missing Key-Gram De-Duplication & Cross-Agent Confiability ($\theta_{\text{trust}} \in \text{params}_{\text{real}}$):** Keys describe coverage, not honesty (`E3-biased` has an honest-looking key while reporting $\ell_0 + 3(\ell - \ell_0) + 1.5$), and `noisy` keys corrupt self-precision and Sybil key matching unless calibrated against report-precision subspaces (`E1`), duplicate-report consensus (`E3-sybil`), and cross-agent Gram likelihood (`E3-biased`).

## What we did
Across all 10 `(env, visibility)` pairs ($5 \text{ envs} \times 2 \text{ visibilities}$), we trained three learned aggregators strictly on `tr.truth` ($n=2000$ `train` episodes) and evaluated each with `untrained_init` and `random_network` controls (`run_with_controls`) plus all 6 baselines on $n=2000$ `dev` episodes:
- `naive_mha`: standard `nn.MultiheadAttention` over concatenated `[report, key]` tokens.
- `key_gram_clear`: $S_N \times S_K$ (and $O(d)$) equivariant Key-Gram de-duplication clearing $(G + \lambda I)^{-1} \text{diag}(G) r$ without cross-agent outlier confiability.
- `key_gram_confiability_clear`: equivariant Key-Gram clearing augmented with a reality-calibrated Confiability Head ($\theta_{\text{trust}} \in \text{params}_{\text{real}}$) performing vectorized $(N+1)$-hypothesis Gram-covariance outlier de-biasing, duplicate-report key consensus (`E2`/`E3`), and noiseless report-precision subspace calibration (`E1`).

## Master Summary Table (`dev` $n=2000$, Mean Regret to `bayes` [95% CI])

| Env | Vis | Status | `oracle_clearing` | `extremized_log_pool` | `naive_mha` | `key_gram_clear` | `key_gram_confiability_clear` | $\Delta$ vs `extremized` [95% CI] | $\Delta$ vs `oracle_clearing` [95% CI] |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `E1` | `full` | **● Gain** | `-0.0000` | `0.8293` | `2.6119` | `0.0027` | **`0.0057`** | `-0.8236 [-0.8801, -0.7671]` | `+0.0057 [+0.0004, +0.0108]` |
| `E1` | `noisy` | **● Gain** | `-0.0000` | `0.8293` | `2.6188` | `0.8209` | **`0.5928`** | `-0.2364 [-0.2715, -0.2004]` | `+0.5928 [+0.5458, +0.6413]` |
| `E2` | `full` | **● Gain** | `0.0166` | `0.0448` | `0.0563` | `0.0178` | **`0.0178`** | `-0.0270 [-0.0356, -0.0180]` | `+0.0012 [+0.0001, +0.0022]` |
| `E2` | `noisy` | **● Informative** | `0.0166` | `0.0448` | `0.0561` | `0.0406` | **`0.0409`** | `-0.0039 [-0.0110, +0.0037]` | `+0.0243 [+0.0154, +0.0337]` |
| `E3-sybil` | `full` | **● Gain** | `0.0166` | `0.0778` | `0.0674` | `0.0179` | **`0.0178`** | `-0.0600 [-0.0723, -0.0475]` | `+0.0012 [+0.0001, +0.0022]` |
| `E3-sybil` | `noisy` | **● Gain** | `0.0166` | `0.0778` | `0.0727` | `0.0495` | **`0.0421`** | `-0.0357 [-0.0470, -0.0239]` | `+0.0255 [+0.0157, +0.0353]` |
| `E3-biased` | `full` | **● Gain** | `0.0166` | `0.0763` | `0.0725` | `0.0626` | **`0.0309`** | `-0.0453 [-0.0552, -0.0355]` | `+0.0143 [+0.0076, +0.0213]` |
| `E3-biased` | `noisy` | **● Gain** | `0.0166` | `0.0763` | `0.0702` | `0.0753` | **`0.0514`** | `-0.0249 [-0.0344, -0.0160]` | `+0.0348 [+0.0243, +0.0452]` |
| `E3-collude` | `full` | **● Gain** | `0.0175` | `0.0302` | `0.0478` | `0.0185` | **`0.0186`** | `-0.0117 [-0.0175, -0.0058]` | `+0.0010 [-0.0008, +0.0029]` |
| `E3-collude` | `noisy` | **● Informative** | `0.0175` | `0.0302` | `0.0437` | `0.0279` | **`0.0283`** | `-0.0020 [-0.0065, +0.0024]` | `+0.0108 [+0.0042, +0.0167]` |

## Control Verification (`run_with_controls`, Mean Regret to `bayes`)

| Env | Vis | `naive_mha` (trained / init / rand) | `key_gram_clear` (trained / init / rand) | `key_gram_confiability_clear` (trained / init / rand) |
|---|---|---:|---:|---:|
| `E1` | `full` | `2.6119` / `4.5316` / `4.6997` | `0.0027` / `64.0374` / `473.4415` | **`0.0057`** / `52.0467` / `3.3411` |
| `E1` | `noisy` | `2.6188` / `4.5350` / `4.7194` | `0.8209` / `80.0304` / `965.2805` | **`0.5928`** / `53.4389` / `2.7052` |
| `E2` | `full` | `0.0563` / `0.4520` / `0.4540` | `0.0178` / `0.2054` / `0.3888` | **`0.0178`** / `0.2261` / `0.1068` |
| `E2` | `noisy` | `0.0561` / `0.4510` / `0.4549` | `0.0406` / `0.2319` / `0.4316` | **`0.0409`** / `0.2566` / `0.0834` |
| `E3-sybil` | `full` | `0.0674` / `0.4525` / `0.4544` | `0.0179` / `0.2143` / `0.3762` | **`0.0178`** / `0.2404` / `0.1035` |
| `E3-sybil` | `noisy` | `0.0727` / `0.4525` / `0.4525` | `0.0495` / `0.2465` / `0.4236` | **`0.0421`** / `0.3040` / `0.0656` |
| `E3-biased` | `full` | `0.0725` / `0.4593` / `0.4545` | `0.0626` / `0.1708` / `0.4257` | **`0.0309`** / `0.1943` / `0.2334` |
| `E3-biased` | `noisy` | `0.0702` / `0.4583` / `0.4550` | `0.0753` / `0.2037` / `0.4324` | **`0.0514`** / `0.2234` / `0.0919` |
| `E3-collude` | `full` | `0.0478` / `0.4639` / `0.4634` | `0.0185` / `0.2291` / `0.4101` | **`0.0186`** / `0.2376` / `0.0882` |
| `E3-collude` | `noisy` | `0.0437` / `0.4625` / `0.4624` | `0.0279` / `0.2299` / `0.4291` | **`0.0283`** / `0.2595` / `0.0885` |

## Key Findings
1. **`naive_mha` fails across every environment (`0.044–2.619` regret):** Concatenating `[report, key]` into a slot-indexed linear projection breaks $S_K$ piece permutation equivariance and value linearity, performing worse than `extremized_log_pool` across all `E1`, `E2`, and `E3-collude` conditions.
2. **`key_gram_clear` solves honest/Sybil/colluding de-duplication at `full` visibility, but fails on `E3-biased` (`0.0626` vs `0.0166` ceiling) and `E1 noisy` (`0.8209`):** Because a biased agent's key looks honest, pure key-overlap Gram clearing gives the biased agent full weight.
3. **`key_gram_confiability_clear` achieves `Gain` across all 5 `full`-visibility environments and `E1`/`E3-sybil`/`E3-biased` at `noisy` visibility (8 `Gain`, 2 `Informative`):**
   - On `E1` (`full`), both `key_gram_clear` (`0.0027`) and `key_gram_confiability_clear` (`0.0057`) approach the exact `0.0000` Bayes ceiling (vs `0.8293` for `extremized_log_pool`). On `E1` (`noisy`), projecting noisy keys onto the noiseless report-precision subspace $\Delta P_i = P_i - I_d$ with per-agent confiability $\beta_i$ drops regret to **`0.5928`** (vs `0.8293` for `extremized_log_pool` and `0.8209` for `key_gram_clear`).
   - On `E3-biased` (`full`), the vectorized $(N+1)$-hypothesis Gram likelihood Confiability Head cuts regret from `0.0763` (`extremized_log_pool`) and `0.0626` (`key_gram_clear`) down to **`0.0309`** (`-0.0453` nats vs `extremized_log_pool`), and on `E3-biased` (`noisy`) down to **`0.0514`** (`-0.0249` nats vs `extremized_log_pool`).
   - On `E3-sybil` (`noisy`), duplicate-report key consensus inside `key_gram_confiability_clear` cuts regret from `0.0778` (`extremized_log_pool`) and `0.0495` (`key_gram_clear`) down to **`0.0421`**.
4. **Remaining gap to `bayes` (`0.017` nats on `E2/E3`):** Even `oracle_clearing` has `0.017` regret on `E2/E3` because scalar reports lose the decomposition of an agent's log-odds across individual pieces whenever the union is not linearly identified (`union_identified` is $3.1\%$ on `E2`). Closing the remaining `0.017` nat gap requires high-dimensional vector messages (`Track-E Runner 2`).
