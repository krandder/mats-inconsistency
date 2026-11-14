# Compression Decoupling Diagnostic (`#3887` & `#3894`: `Z9-p2` & `E2` at `b = 2`, Label-Only, 3 Seeds)

**Track: `EXPLORATORY` (`gate must-fixes applied per #3894`)**

Decouples encoder learning, decoder learning, and per-episode structure keys (`key.full` / `oracle`) at `b = 2` bits/agent across `3` seeds (`0, 1, 2`), label-only (`Y` labels only), with **fresh episodes at every step on `Z9-p2`**, **best-val checkpoint selection (`best_val_kl`) on both `Z9-p2` and `E2`**, and **`key_scale = 0.1` (`in_norm=False`) + `key_warmup = 1000` steps on `(iii) iii_both_learned_with_keys`**:
- **`(i)` Fixed closed-form encoder + learned decoder (`i_enc_cf_dec_learned_*`):**
  - `i_enc_cf_dec_learned_msg_only`: standard set decoder reading **only** the `b = 2`-bit closed-form messages `m` (`raw_code` on `Z9-p2`, `llr_quantizer` on `E2`).
  - `i_enc_cf_dec_learned_with_keys`: set decoder reading `[m, key_structure]`, receiving the per-agent structure keys (`key.full` on `Z9-p2`; `own_model` on `E2`), reported alongside the **same-steps uncompressed Transformer anchor** (`8000` steps on `Z9-p2`, `4000` steps on `E2`) to separate the decoder's finite-step ceiling from the `2`-bit quantization gap.
- **`(ii)` Learned encoder + `oracle-structure + oracle-codebook decoder [PROVENANCE FULL]` (`ii_enc_learned_dec_fixed_bayes*`):**
  - **Scope & Provenance Note (`#3894` Must-Fix 1):** Arm `(ii)` is an **`oracle-structure + oracle-codebook decoder [PROVENANCE FULL]`** diagnostic, **not** a blind fixed decoder. On `Z9-p2`, `decode_fixed_bayes_z9` uses per-episode `batch.oracle['var']` and `batch.oracle['acc']` (`make_z9_lik_tensor`); on `E2`, `decode_fixed_bayes_e2_eval` uses per-episode `batch.oracle['cov']`, `batch.oracle['mu']` (`gaussian_report_model`), and the oracle quantizer's cell thresholds `quant.thresholds(scale, l0)`. Therefore, `(ii)` shows **only** that the encoder MLP can learn to match a known codebook when the decoder already knows the per-episode structure and codebook cells; it says **nothing** about blind compression.
- **`(iii)` Both learned end-to-end (`iii_both_learned_*`):**
  - `iii_both_learned_msg_only`: message-only set decoder `decode(m)` without structure keys at the decoder (`[BLIND DECODER: messages only]`).
  - `iii_both_learned_with_keys`: both encoder (`b = 2` bits verified by `Channel(2).send`) and structure-key-aware set decoder `decode([m, key_structure])` (`STRUCTURE-KNOWN (decoder reads keys['full'] outside b)`) trained end-to-end on `Y` labels with `key_scale = 0.1` (`in_norm=False`) and `key_warmup = 1000` steps (`best_val_kl` restricted to `t >= key_warmup`).
  - **Spec & Channel Scope Note (`#3896` Must-Fix 5):** On `Z9-p2`, arm `(iii)` (`0/3` escaped) ran on the **old `Z9-p2` spec (`prior = 0.4567`, deterministic `2` readers per parity bit + `6` distractor readers = `14` readers / `7` agents)** with **Gaussian / Gumbel channels**, whereas `tt-compression-joint-remedies-20261022` ran on **leak-reduced `Z9-p2-lf` (`prior = 0.2702`, `Dirichlet` random assignment of `24` parity + `2` distractor readers = `26` readers / `13` agents)** with **`FSQ`**. Comparing `(iii)` here against `joint-remedies` does **not** isolate "FSQ fixes joint learning" because the environment spec and channel are confounded.

## Environment: `Z9-p2` (`b = 2` bits/agent, `steps = 8000`)

### 1. Anchors & Plug-in Tabulated Message-Only Floor (`n_dev = 2000`)

| Anchor | Regret vs Bayes (nats) | 95% Bootstrap CI / Seed SD | Notes |
| --- | --- | --- | --- |
| `prior (b = 0)` | `0.4567` | `[`0.4485`, `0.4649`]` | No messages |
| `same_steps_uncompressed_transformer` (`8000` steps, `b = inf`) | `0.0047` | `[`0.0035`, `0.0064`]` | Same-steps uncompressed Transformer anchor (`best_val_kl` checkpoint across 3 seeds) |
| `closed_form_oracle_decoder` (`raw_code` / `llr_quantizer` `[PROVENANCE FULL]`) | `0.0000` | `[`0.0000`, `0.0000`]` | Oracle-structure Bayes decoder given 2-bit messages **plus** per-episode structure (`oracle['var']` / `oracle['cov','mu']`) |
| `plug-in tabulated message-only floor (±0.001 plug-in bias)` | `0.3234` | `[`0.3095`, `0.3376`]` | Plug-in tabulated message-only floor (`120` multiset cells on `Z9-p2`, `56` on `E2`, estimated from `100,000` train episodes with Laplace smoothing; carries `±0.001` finite-sample plug-in bias) |

### 2. Decoupling Matrix (`3` seeds `0, 1, 2`, label-only, `best_val_kl` checkpoint)

| Arm | Provenance / Decoder Structure | Channel | Mean Regret (`regret_quant`) | Gap to Same-Steps Anchor | Seed SD | Mean 95% CI | Per-Seed Regrets (`s0, s1, s2`) | Escaped (`val_kl < 0.9 * prior`) | Escape Steps (`s0, s1, s2`) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `i_enc_cf_dec_learned_msg_only` | `[BLIND DECODER: messages only]` | `gauss` | `0.3230` | `+0.3184` | `0.0004` | `[`0.3100`, `0.3366`]` | `0.3236`, `0.3229`, `0.3225` | `3/3` | `200`, `200`, `200` |
| `i_enc_cf_dec_learned_msg_only` | `[BLIND DECODER: messages only]` | `gumbel` | `0.3230` | `+0.3183` | `0.0005` | `[`0.3094`, `0.3373`]` | `0.3223`, `0.3233`, `0.3233` | `3/3` | `200`, `200`, `400` |
| `i_enc_cf_dec_learned_with_keys` | `STRUCTURE-KNOWN (decoder reads keys['full'] outside b)` | `gauss` | `0.0573` | `+0.0527` | `0.0338` | `[`0.0523`, `0.0627`]` | `0.0736`, `0.0882`, `0.0103` | `3/3` | `200`, `200`, `200` |
| `i_enc_cf_dec_learned_with_keys` | `STRUCTURE-KNOWN (decoder reads keys['full'] outside b)` | `gumbel` | `0.1085` | `+0.1038` | `0.0673` | `[`0.0995`, `0.1173`]` | `0.1560`, `0.0132`, `0.1562` | `3/3` | `200`, `400`, `200` |
| `ii_enc_learned_dec_fixed_bayes` | `oracle-structure + oracle-codebook decoder [PROVENANCE FULL]` | `gauss` | `0.6371` | `+0.6324` | `0.0148` | `[`0.6089`, `0.6648`]` | `0.6339`, `0.6207`, `0.6565` | `0/3` | `None`, `None`, `None` |
| `ii_enc_learned_dec_fixed_bayes` | `oracle-structure + oracle-codebook decoder [PROVENANCE FULL]` | `gumbel` | `0.7786` | `+0.7740` | `0.0809` | `[`0.7422`, `0.8170`]` | `0.8921`, `0.7346`, `0.7093` | `0/3` | `None`, `None`, `None` |
| `ii_enc_learned_obs_only_dec_fixed_bayes` | `oracle-structure + oracle-codebook decoder [PROVENANCE FULL]` | `gauss` | `0.6834` | `+0.6787` | `0.0680` | `[`0.6507`, `0.7161`]` | `0.6437`, `0.6273`, `0.7791` | `0/3` | `None`, `None`, `None` |
| `ii_enc_learned_obs_only_dec_fixed_bayes` | `oracle-structure + oracle-codebook decoder [PROVENANCE FULL]` | `gumbel` | `0.0000` | `-0.0047` | `0.0000` | `[`0.0000`, `0.0000`]` | `0.0000`, `0.0000`, `0.0000` | `3/3` | `1400`, `1400`, `1400` |
| `iii_both_learned_msg_only` | `[BLIND DECODER: messages only]` | `gauss` | `0.4536` | `+0.4490` | `0.0057` | `[`0.4452`, `0.4619`]` | `0.4456`, `0.4574`, `0.4580` | `0/3` | `None`, `None`, `None` |
| `iii_both_learned_msg_only` | `[BLIND DECODER: messages only]` | `gumbel` | `0.4580` | `+0.4533` | `0.0004` | `[`0.4497`, `0.4663`]` | `0.4585`, `0.4576`, `0.4579` | `0/3` | `None`, `None`, `None` |
| `iii_both_learned_with_keys` | `STRUCTURE-KNOWN (decoder reads keys['full'] outside b)` | `gauss` | `0.4582` | `+0.4535` | `0.0005` | `[`0.4500`, `0.4665`]` | `0.4589`, `0.4578`, `0.4579` | `0/3` | `None`, `None`, `None` |
| `iii_both_learned_with_keys` | `STRUCTURE-KNOWN (decoder reads keys['full'] outside b)` | `gumbel` | `0.4578` | `+0.4532` | `0.0003` | `[`0.4497`, `0.4662`]` | `0.4580`, `0.4574`, `0.4581` | `0/3` | `None`, `None`, `None` |

## Environment: `E2` (`b = 2` bits/agent, `steps = 4000`)

### 1. Anchors & Plug-in Tabulated Message-Only Floor (`n_dev = 2000`)

| Anchor | Regret vs Bayes (nats) | 95% Bootstrap CI / Seed SD | Notes |
| --- | --- | --- | --- |
| `prior (b = 0)` | `0.4184` | `[`0.4077`, `0.4286`]` | No messages |
| `same_steps_uncompressed_transformer` (`4000` steps, `b = inf`) | `0.0432` | `[`0.0394`, `0.0474`]` | Same-steps uncompressed Transformer anchor (`best_val_kl` checkpoint across 3 seeds) |
| `closed_form_oracle_decoder` (`raw_code` / `llr_quantizer` `[PROVENANCE FULL]`) | `0.0331` | `[`0.0298`, `0.0368`]` | Oracle-structure Bayes decoder given 2-bit messages **plus** per-episode structure (`oracle['var']` / `oracle['cov','mu']`) |
| `plug-in tabulated message-only floor (±0.001 plug-in bias)` | `0.0555` | `[`0.0511`, `0.0604`]` | Plug-in tabulated message-only floor (`120` multiset cells on `Z9-p2`, `56` on `E2`, estimated from `100,000` train episodes with Laplace smoothing; carries `±0.001` finite-sample plug-in bias) |

### 2. Decoupling Matrix (`3` seeds `0, 1, 2`, label-only, `best_val_kl` checkpoint)

| Arm | Provenance / Decoder Structure | Channel | Mean Regret (`regret_quant`) | Gap to Same-Steps Anchor | Seed SD | Mean 95% CI | Per-Seed Regrets (`s0, s1, s2`) | Escaped (`val_kl < 0.9 * prior`) | Escape Steps (`s0, s1, s2`) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `i_enc_cf_dec_learned_msg_only` | `[BLIND DECODER: messages only]` | `gauss` | `0.0557` | `+0.0125` | `0.0004` | `[`0.0511`, `0.0608`]` | `0.0562`, `0.0554`, `0.0555` | `3/3` | `200`, `200`, `200` |
| `i_enc_cf_dec_learned_msg_only` | `[BLIND DECODER: messages only]` | `gumbel` | `0.0556` | `+0.0124` | `0.0001` | `[`0.0510`, `0.0605`]` | `0.0556`, `0.0556`, `0.0555` | `3/3` | `200`, `200`, `200` |
| `i_enc_cf_dec_learned_with_keys` | `STRUCTURE-KNOWN (decoder reads keys['full'] outside b)` | `gauss` | `0.0476` | `+0.0044` | `0.0003` | `[`0.0438`, `0.0519`]` | `0.0480`, `0.0474`, `0.0474` | `3/3` | `200`, `200`, `200` |
| `i_enc_cf_dec_learned_with_keys` | `STRUCTURE-KNOWN (decoder reads keys['full'] outside b)` | `gumbel` | `0.0494` | `+0.0062` | `0.0004` | `[`0.0456`, `0.0536`]` | `0.0499`, `0.0489`, `0.0494` | `3/3` | `200`, `200`, `0` |
| `ii_enc_learned_dec_fixed_bayes` | `oracle-structure + oracle-codebook decoder [PROVENANCE FULL]` | `gauss` | `0.0646` | `+0.0214` | `0.0010` | `[`0.0591`, `0.0706`]` | `0.0646`, `0.0659`, `0.0635` | `3/3` | `200`, `200`, `200` |
| `ii_enc_learned_dec_fixed_bayes` | `oracle-structure + oracle-codebook decoder [PROVENANCE FULL]` | `gumbel` | `0.0362` | `-0.0070` | `0.0004` | `[`0.0330`, `0.0396`]` | `0.0356`, `0.0365`, `0.0366` | `3/3` | `200`, `200`, `200` |
| `ii_enc_learned_obs_only_dec_fixed_bayes` | `oracle-structure + oracle-codebook decoder [PROVENANCE FULL]` | `gauss` | `0.0592` | `+0.0160` | `0.0017` | `[`0.0542`, `0.0649`]` | `0.0578`, `0.0615`, `0.0583` | `3/3` | `200`, `200`, `200` |
| `ii_enc_learned_obs_only_dec_fixed_bayes` | `oracle-structure + oracle-codebook decoder [PROVENANCE FULL]` | `gumbel` | `0.0334` | `-0.0098` | `0.0006` | `[`0.0301`, `0.0370`]` | `0.0339`, `0.0337`, `0.0326` | `3/3` | `200`, `200`, `200` |
| `iii_both_learned_msg_only` | `[BLIND DECODER: messages only]` | `gauss` | `0.0710` | `+0.0278` | `0.0012` | `[`0.0655`, `0.0767`]` | `0.0721`, `0.0694`, `0.0716` | `3/3` | `200`, `200`, `200` |
| `iii_both_learned_msg_only` | `[BLIND DECODER: messages only]` | `gumbel` | `0.0696` | `+0.0264` | `0.0103` | `[`0.0644`, `0.0750`]` | `0.0841`, `0.0636`, `0.0612` | `3/3` | `200`, `200`, `200` |
| `iii_both_learned_with_keys` | `STRUCTURE-KNOWN (decoder reads keys['full'] outside b)` | `gauss` | `0.0611` | `+0.0179` | `0.0024` | `[`0.0564`, `0.0662`]` | `0.0644`, `0.0599`, `0.0590` | `3/3` | `200`, `200`, `200` |
| `iii_both_learned_with_keys` | `STRUCTURE-KNOWN (decoder reads keys['full'] outside b)` | `gumbel` | `0.0610` | `+0.0178` | `0.0062` | `[`0.0561`, `0.0664`]` | `0.0526`, `0.0673`, `0.0632` | `3/3` | `200`, `200`, `200` |

