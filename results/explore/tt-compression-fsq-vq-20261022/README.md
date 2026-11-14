# Step 6 Quantizer Symbol-Usage Study (`#3889` & `#3894`: `FSQ`, `VQ-EMA-Reset`, and Matched Controls at `b = 2` on `E2` & `Z8`)

**Track: `EXPLORATORY` (`gate must-fixes applied per #3894`)**

Tests quantizers and matched controls at `b = 2` bits/agent (`2^b = 4` symbols) on `E2` and `Z8` (`3` seeds `0, 1, 2`, label-only, `n_train = 100,000`, `n_dev = 2,000`, `steps = 4,000`, `best_val_kl` checkpoint selection):
- **Baselines:** `gauss_noisy_ste` (`tanh + N(0, 1/3)` in train, `sign` at eval) and `gumbel_st` (Gumbel-Softmax STE over 4 symbols).
- **Matched Controls (`#3894` Must-Fix 3 — isolating `B8` train/score mismatch vs rounding geometry vs usage loss):**
  - `noiseless_sign_ste`: `tanh(u)` + noiseless `sign` STE (`m = s + (sign(s) - s).detach()`) with **zero usage loss** (isolates removing the `N(0, 1/3)` train/score mismatch `B8` in `gauss_noisy_ste`).
  - `fsq_1d_4level_no_usage`: 1D 4-level `FSQ` (`L=[4]`) with **zero usage loss**.
  - `fsq_2d_2x2_no_usage`: 2D binary `FSQ` (`L=[2, 2]`) with **zero usage loss**.
- **Usage-Regularized Quantizers:** `fsq_1d_4level` (`L=[4]` + soft usage loss), `fsq_2d_2x2` (`L=[2, 2]` + soft usage loss), and `vq_ema_reset` (`K=4` EMA codebook + dead-code reset + usage entropy loss).
- **Scope & Statistical-Tie Note (`#3894` Must-Fix 3):** On `Z8`, `FSQ` vs blind `extremized_q` is a **statistical tie** within the 95% bootstrap CI shown below, and on `E2`, `FSQ` narrows the gap to `extremized_q` but does **not** beat `extremized_q`. All `gap(b=2)` values below are scored against the **same-data (`n_train = 100,000`, `4000`-step) uncompressed Transformer anchor** (`same_data_full_context_transformer`).

## Environment: `E2` (`b = 2` bits/agent, `2^b = 4` symbols)

- **Anchors (`n_dev = 2000`):** `prior (b=0)` = `0.4184`, `same_data_full_context_transformer (n_train=100k, 4000 steps, b=inf)` = `0.0419`, `llr_quantizer [PROVENANCE FULL] (b=2)` = `0.0331`, `extremized_q (blind, b=2)` = `0.0565` (CI `[`0.0519`, `0.0618`]`), `run_closed_form 2k-train full_context_transformer` = `0.0350`

| Quantizer Arm | Mean Regret (`regret_quant`) | Gap to Same-Data Anchor | Seed SD | Mean 95% CI | Per-Seed Regrets (`s0, s1, s2`) | Mean Agent Entropy `H(M_i)` (bits / `2.0000`) | Pooled Entropy `H(M)` (bits) | Distinct Symbols (`s0, s1, s2`) | Paired Gain vs `gauss_noisy_ste` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gauss_noisy_ste` | `0.0705` | `+0.0286` | `0.0015` | `[`0.0650`, `0.0766`]` | `0.0706`, `0.0687`, `0.0722` | `1.5623 ± 0.0013` | `1.5627` | `3`, `3`, `3` | `0.0000` (ref) |
| `noiseless_sign_ste` | `0.0689` | `+0.0271` | `0.0088` | `[`0.0635`, `0.0744`]` | `0.0740`, `0.0762`, `0.0565` | `1.6586 ± 0.2034` | `1.6593` | `3`, `3`, `4` | `+0.0016 ± 0.0101` |
| `gumbel_st` | `0.0689` | `+0.0270` | `0.0082` | `[`0.0635`, `0.0746`]` | `0.0698`, `0.0784`, `0.0585` | `1.6564 ± 0.2233` | `1.6572` | `3`, `3`, `4` | `+0.0016 ± 0.0096` |
| `fsq_1d_4level_no_usage` | `0.0591` | `+0.0172` | `0.0017` | `[`0.0543`, `0.0642`]` | `0.0589`, `0.0612`, `0.0571` | `1.9819 ± 0.0178` | `1.9831` | `4`, `4`, `4` | `+0.0114 ± 0.0031` |
| `fsq_1d_4level` | `0.0629` | `+0.0210` | `0.0030` | `[`0.0582`, `0.0681`]` | `0.0596`, `0.0669`, `0.0622` | `1.9763 ± 0.0204` | `1.9773` | `4`, `4`, `4` | `+0.0076 ± 0.0042` |
| `fsq_2d_2x2_no_usage` | `0.0689` | `+0.0271` | `0.0088` | `[`0.0635`, `0.0744`]` | `0.0740`, `0.0762`, `0.0565` | `1.6586 ± 0.2034` | `1.6593` | `3`, `3`, `4` | `+0.0016 ± 0.0101` |
| `fsq_2d_2x2` | `0.0598` | `+0.0179` | `0.0022` | `[`0.0548`, `0.0649`]` | `0.0581`, `0.0629`, `0.0583` | `1.9852 ± 0.0060` | `1.9861` | `4`, `4`, `4` | `+0.0107 ± 0.0036` |
| `vq_ema_reset` | `0.0585` | `+0.0166` | `0.0007` | `[`0.0537`, `0.0634`]` | `0.0592`, `0.0575`, `0.0587` | `1.9941 ± 0.0021` | `1.9949` | `4`, `4`, `4` | `+0.0121 ± 0.0010` |

## Environment: `Z8` (`b = 2` bits/agent, `2^b = 4` symbols)

- **Anchors (`n_dev = 2000`):** `prior (b=0)` = `0.4500`, `same_data_full_context_transformer (n_train=100k, 4000 steps, b=inf)` = `0.0402`, `llr_quantizer [PROVENANCE FULL] (b=2)` = `0.0337`, `extremized_q (blind, b=2)` = `0.0504` (CI `[`0.0460`, `0.0549`]`), `run_closed_form 2k-train full_context_transformer` = `0.0364`

| Quantizer Arm | Mean Regret (`regret_quant`) | Gap to Same-Data Anchor | Seed SD | Mean 95% CI | Per-Seed Regrets (`s0, s1, s2`) | Mean Agent Entropy `H(M_i)` (bits / `2.0000`) | Pooled Entropy `H(M)` (bits) | Distinct Symbols (`s0, s1, s2`) | Paired Gain vs `gauss_noisy_ste` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gauss_noisy_ste` | `0.0645` | `+0.0243` | `0.0005` | `[`0.0590`, `0.0705`]` | `0.0640`, `0.0643`, `0.0652` | `1.5810 ± 0.0015` | `1.5818` | `3`, `3`, `3` | `0.0000` (ref) |
| `noiseless_sign_ste` | `0.0620` | `+0.0218` | `0.0025` | `[`0.0565`, `0.0678`]` | `0.0650`, `0.0624`, `0.0588` | `1.6983 ± 0.1792` | `1.6989` | `3`, `3`, `4` | `+0.0025 ± 0.0030` |
| `gumbel_st` | `0.0618` | `+0.0216` | `0.0042` | `[`0.0563`, `0.0678`]` | `0.0643`, `0.0653`, `0.0559` | `1.7042 ± 0.1867` | `1.7050` | `3`, `3`, `4` | `+0.0027 ± 0.0047` |
| `fsq_1d_4level_no_usage` | `0.0496` | `+0.0094` | `0.0009` | `[`0.0453`, `0.0542`]` | `0.0491`, `0.0509`, `0.0489` | `1.9960 ± 0.0014` | `1.9965` | `4`, `4`, `4` | `+0.0149 ± 0.0012` |
| `fsq_1d_4level` | `0.0508` | `+0.0106` | `0.0008` | `[`0.0465`, `0.0553`]` | `0.0516`, `0.0511`, `0.0497` | `1.9966 ± 0.0014` | `1.9973` | `4`, `4`, `4` | `+0.0137 ± 0.0013` |
| `fsq_2d_2x2_no_usage` | `0.0620` | `+0.0218` | `0.0025` | `[`0.0565`, `0.0678`]` | `0.0650`, `0.0624`, `0.0588` | `1.6983 ± 0.1792` | `1.6989` | `3`, `3`, `4` | `+0.0025 ± 0.0030` |
| `fsq_2d_2x2` | `0.0514` | `+0.0112` | `0.0006` | `[`0.0466`, `0.0564`]` | `0.0514`, `0.0521`, `0.0507` | `1.9897 ± 0.0042` | `1.9904` | `4`, `4`, `4` | `+0.0131 ± 0.0010` |
| `vq_ema_reset` | `0.0507` | `+0.0105` | `0.0006` | `[`0.0464`, `0.0554`]` | `0.0502`, `0.0515`, `0.0505` | `1.9971 ± 0.0003` | `1.9977` | `4`, `4`, `4` | `+0.0138 ± 0.0008` |

