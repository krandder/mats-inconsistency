# Exploratory: Compressed Aggregation Benchmark — `TraderTransformer` & `Queried-View` (`b`-Bit Uplink)

- **Commit:** `b1bf581adf4799c90fed38131917d55ec56bca55`
- **Script:** `benchmarks/explore_trader_transformer_compression.py` + `aggbench/compression.py`
- **Protocol:** 3 seeds (`[0, 1, 2]`), `n_train=100000`, `n_dev=2000`, `steps=3000` (`z9_steps=4500`), `batch_size=512`, `width=128`, `layers=4`
- **Channel & Locality Enforcement:** Every finite-$b$ learned message is quantized to hard bits at evaluation time, packed into integer symbols in $[0, 2^b)$, verified by `aggbench.compression.Channel(b).send(syms)`, and checked for strict agent-local / causal prefix invariance by `verify_net_locality()`.
- **Supervision & Controls (`gate-e1gls-compv2-2026-10-21.md`):** All primary learned arms (`self_summary_b`, `random_encoder_b`, `dvib_oneshot_q0`, `coa_chain_b`, `fixed_state_nb`, `queried_learned_qinf`, `trader_transformer_b_qinf`) are trained **strictly on the final outcome label $Y$ (`label-only`)**. On `Z9-p2` ($k=4$ parity, $p=2$ readings/agent), each individual agent's 2 readings have zero marginal mutual information with $Y$, so `self_summary_b` is null by construction (sitting at the prior); we therefore include `random_encoder_b` (frozen random $b$-bit encoder + trained consensus set decoder) as a non-aware control, and tag the diagnostic curriculum+auxiliary parity arms `*_curr_aux` explicitly as `[LATENT-LABEL SUPERVISION]`.

- **Learnability Review Corrections (`B1`, `B5`, `B11`, `#3888`–`#3889`):**
  1. **`B1` (Provenance-Full Closed Forms vs. Blind Decoders):** Closed-form `raw_code` (`Z9-p2`) and `llr_quantizer` (`E2`–`Z8`) pass per-episode structure (`oracle['var','acc']` / `oracle['cov','mu']`) to the decoder out-of-band and are tagged `[PROVENANCE FULL]`. The blind message-only Bayes floor at $b=2$ is `0.3225` (`raw_code` symbols without keys on `Z9-p2`; count-leak optimum `0.1600`, coordinate-search bound `0.1510`) and `0.0555` (`llr_quantizer` symbols without keys on `E2`).
  2. **`B5` (`FCT` Column Relabel & Recomputed $b_{90}/\rho$):** On `Z9-p2`, the earlier `0.2583` figure was `dvib_oneshot_q0` at $b=\infty$ (which passes through a 16-dim encoder bottleneck), NOT `full_context_transformer`. It is now explicitly separated as `dvib_b_inf` (`0.2583`), while `FCT` ($b=\infty$) on `Z9-p2` reports the genuine uncompressed full-context anchor (`0.0617` on the 100k pool; `0.0020` on fresh stream), with all $\rho_{\text{FCT}}(b)$, $b_{90}(\text{FCT})$, and `gap_to_fct(b) = regret(arm, b) - regret(FCT)` recomputed against it.
  3. **`B11` (Unmatched Comparisons Disclosed):** `fixed_state_nb` maintains an $S = \min(N \cdot b, 32)$-bit recurrent state across agents (`consensus_bits = N*b`, not $b$ bits/agent) and is explicitly tagged `[S = min(N*b, 32)-bit state, NOT b bits/agent]`. In the v2 runs below, the 4-output sequential arms (`queried_learned_qinf`, `trader_transformer_b_qinf`) included a `0.15` auxiliary prefix-BCE weight on non-`Z9` finite-$b$ runs and omitted `tanh` at $b=\infty$ (both fixed in `explore_trader_transformer_compression.py`).

## 1. Headline Summary: $b_{90}$ and Peak Label-Only Gains

| Env | $N$ | Prior | Floor | `FCT` ($b=\infty$) | `dvib_b_inf` ($b=\infty$) | `llr_quantizer` [PROVENANCE FULL] $b_{90}$ | `self_summary_b` $b_{90}$ (FCT) | `random_encoder_b` $b_{90}$ (FCT) | `dvib_oneshot_q0` $b_{90}$ (FCT) | `queried_learned_qinf` $b_{90}$ (FCT) | `trader_transformer_b_qinf` $b_{90}$ (FCT) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `E2` | `5` | `0.4184` | `0.0000` | `0.0350` | `0.0338` | `2` | `4` | `8` | `2` | `2` | `4` |
| `E3-sybil` | `8` | `0.4184` | `0.0000` | `0.0420` | `0.0403` | `2` | `4` | `8` | `4` | `2` | `4` |
| `Z9-p2` | `7` | `0.4567` | `0.0000` | `0.0617` | `0.2583` | `None` | `None` | `None` | `None` | `None` | `None` |
| `E4` | `6` | `0.3532` | `0.0472` | `0.0795` | `0.0795` | `None` | `4` | `None` | `2` | `2` | `4` |
| `Z8` | `5` | `0.4500` | `0.0000` | `0.0364` | `0.0338` | `2` | `2` | `8` | `2` | `2` | `2` |

## 2. `E2` ($N=5$, family=`pieces`, Prior `0.4184`, Floor `0.0000`, FCT `0.0350`, `dvib_b_inf` `0.0338`)

### 2.1 Closed-Form Reference Baselines (`aggbench.compression`, dev regret)

| Closed-Form Arm | Provenance / Query | $b=1$ | $b=2$ | $b=4$ | $b=8$ | $b=inf$ | $b_{90}$ (floor) |
|---|---|---:|---:|---:|---:|---:|---:|
| `llr_quantizer` | none (one-shot) `[PROVENANCE FULL: oracle structure at decoder]` | `0.0762` | `0.0331` | `0.0182` | `0.0158` | `0.0158` | `2` |
| `extremized_q` | none (one-shot) `[BLIND DECODER]` | `0.1039` | `0.0565` | `0.0418` | `0.0409` | `0.0407` | `4` |
| `trimmed_q` | none (one-shot) `[BLIND DECODER]` | `0.1163` | `0.0759` | `0.0602` | `0.0586` | `0.0586` | `None` |
| `coa_chain_scalar` | none (one-shot) `[BLIND DECODER]` | `0.1773` | `0.0973` | `0.0488` | `0.0416` | `0.0417` | `8` |
| `fixed_state` | none (one-shot) `[BLIND DECODER]` | `0.0438` | `0.0417` | `0.0417` | `0.0417` | `0.0417` | `2` |
| `top_w` | none (one-shot) `[PROVENANCE FULL: oracle structure at decoder]` | `0.4184` | `0.3507` | `0.1198` | `0.0371` | `-0.0000` | `8` |
| `k_of_n` | none (one-shot) `[PROVENANCE FULL: oracle structure at decoder]` | `0.2279` | `0.1475` | `0.0829` | `0.0352` | `-0.0000` | `8` |
| `queried_llr_qinf` | unbounded `[PROVENANCE FULL: oracle structure at decoder]` | `0.0762` | `0.0331` | `0.0182` | `0.0158` | `—` | `2` |
| `queried_llr_qb` | 1 bits `[PROVENANCE FULL: oracle structure at decoder]` | `0.0825` | `0.0331` | `0.0182` | `0.0158` | `—` | `2` |

### 2.2 Learned Compression Arms (3-Seed Mean ± SD Regret, `Untrained` & `RandomNet` Controls)

| Learned Arm | Protocol / State Budget | $b=1$ | $b=2$ | $b=4$ | $b=8$ | $b=inf$ | $b_{90}$ (floor) | $b_{90}$ (FCT) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `self_summary_b` | none (one-shot) | `0.1213 ± 0.0040` | `0.0734 ± 0.0038` | `0.0480 ± 0.0012` | `0.0420 ± 0.0004` | `0.0356 ± 0.0005` | `None` | `4` |
| `random_encoder_b` | none (one-shot) | `0.2941 ± 0.1236` | `0.2624 ± 0.1143` | `0.0943 ± 0.0135` | `0.0653 ± 0.0094` | `0.0364 ± 0.0005` | `None` | `8` |
| `dvib_oneshot_q0` | none (one-shot) | `0.1110 ± 0.0026` | `0.0693 ± 0.0016` | `0.0518 ± 0.0013` | `0.0408 ± 0.0013` | `0.0338 ± 0.0006` | `8` | `2` |
| `coa_chain_b` | state_chain [S = b-bit state] | `0.1958 ± 0.0013` | `0.1082 ± 0.0062` | `0.0689 ± 0.0014` | `0.0561 ± 0.0013` | `0.0359 ± 0.0007` | `None` | `4` |
| `fixed_state_nb` | state_chain [S = min(N*b, 32)-bit state, NOT b bits/agent] | `0.0634 ± 0.0022` | `0.0553 ± 0.0005` | `0.0484 ± 0.0006` | `0.0455 ± 0.0010` | `0.0359 ± 0.0007` | `None` | `1` |
| `queried_learned_qinf` | unbounded (read-only) | `0.1079 ± 0.0005` | `0.0661 ± 0.0012` | `0.0481 ± 0.0018` | `0.0397 ± 0.0005` | `0.0335 ± 0.0005` | `8` | `2` |
| `trader_transformer_b_qinf` | unbounded (read-only) | `0.1217 ± 0.0049` | `0.0796 ± 0.0059` | `0.0547 ± 0.0009` | `0.0469 ± 0.0014` | `0.0555 ± 0.0013` | `None` | `4` |

### 2.2b Gap to Uncompressed Full-Context Transformer (`gap(b) = regret(arm, b) - regret(FCT)`, `#3888`)

| Learned Arm | $b=1$ | $b=2$ | $b=4$ | $b=8$ | $b=inf$ |
|---|---:|---:|---:|---:|---:|
| `gap_to_fct(self_summary_b)` | `+0.0863` | `+0.0384` | `+0.0130` | `+0.0070` | `+0.0007` |
| `gap_to_fct(random_encoder_b)` | `+0.2591` | `+0.2274` | `+0.0593` | `+0.0303` | `+0.0014` |
| `gap_to_fct(dvib_oneshot_q0)` | `+0.0760` | `+0.0343` | `+0.0168` | `+0.0058` | `-0.0011` |
| `gap_to_fct(coa_chain_b)` | `+0.1608` | `+0.0732` | `+0.0339` | `+0.0212` | `+0.0009` |
| `gap_to_fct(fixed_state_nb)` | `+0.0284` | `+0.0203` | `+0.0134` | `+0.0105` | `+0.0009` |
| `gap_to_fct(queried_learned_qinf)` | `+0.0729` | `+0.0311` | `+0.0131` | `+0.0047` | `-0.0015` |
| `gap_to_fct(trader_transformer_b_qinf)` | `+0.0867` | `+0.0446` | `+0.0197` | `+0.0119` | `+0.0205` |

### 2.3 Paired Gains vs `self_summary_b`, `random_encoder_b`, and One-Shot `dvib_oneshot_q0`

| Comparison | $b=1$ | $b=2$ | $b=4$ | $b=8$ | $b=inf$ |
|---|---:|---:|---:|---:|---:|
| `gain(random_encoder_b vs self_summary_b)` | `-0.1728 ± 0.1210` | `-0.1890 ± 0.1145` | `-0.0463 ± 0.0134` | `-0.0233 ± 0.0094` | `-0.0008 ± 0.0001` |
| `gain(dvib_oneshot_q0 vs self_summary_b)` | `+0.0103 ± 0.0017` | `+0.0041 ± 0.0044` | `-0.0039 ± 0.0018` | `+0.0012 ± 0.0013` | `+0.0018 ± 0.0010` |
| `gain(coa_chain_b vs self_summary_b)` | `-0.0745 ± 0.0050` | `-0.0348 ± 0.0082` | `-0.0210 ± 0.0016` | `-0.0141 ± 0.0010` | `-0.0002 ± 0.0007` |
| `gain(fixed_state_nb vs self_summary_b)` | `+0.0579 ± 0.0034` | `+0.0181 ± 0.0036` | `-0.0004 ± 0.0008` | `-0.0035 ± 0.0014` | `-0.0002 ± 0.0007` |
| `gain(queried_learned_qinf vs self_summary_b)` | `+0.0135 ± 0.0037` | `+0.0073 ± 0.0040` | `-0.0001 ± 0.0015` | `+0.0023 ± 0.0009` | `+0.0021 ± 0.0010` |
| `gain(trader_transformer_b_qinf vs self_summary_b)` | `-0.0004 ± 0.0021` | `-0.0062 ± 0.0041` | `-0.0067 ± 0.0021` | `-0.0049 ± 0.0014` | `-0.0198 ± 0.0008` |
| `gain(self_summary_b vs random_encoder_b)` | `+0.1728 ± 0.1210` | `+0.1890 ± 0.1145` | `+0.0463 ± 0.0134` | `+0.0233 ± 0.0094` | `+0.0008 ± 0.0001` |
| `gain(dvib_oneshot_q0 vs random_encoder_b)` | `+0.1831 ± 0.1226` | `+0.1931 ± 0.1158` | `+0.0425 ± 0.0148` | `+0.0245 ± 0.0106` | `+0.0026 ± 0.0009` |
| `gain(coa_chain_b vs random_encoder_b)` | `+0.0983 ± 0.1249` | `+0.1542 ± 0.1083` | `+0.0254 ± 0.0121` | `+0.0091 ± 0.0087` | `+0.0005 ± 0.0008` |
| `gain(fixed_state_nb vs random_encoder_b)` | `+0.2308 ± 0.1214` | `+0.2071 ± 0.1138` | `+0.0459 ± 0.0131` | `+0.0198 ± 0.0095` | `+0.0005 ± 0.0008` |
| `gain(queried_learned_qinf vs random_encoder_b)` | `+0.1863 ± 0.1232` | `+0.1963 ± 0.1155` | `+0.0462 ± 0.0150` | `+0.0256 ± 0.0094` | `+0.0029 ± 0.0010` |
| `gain(trader_transformer_b_qinf vs random_encoder_b)` | `+0.1724 ± 0.1224` | `+0.1828 ± 0.1105` | `+0.0396 ± 0.0138` | `+0.0184 ± 0.0108` | `-0.0191 ± 0.0008` |
| `query_gain(queried_learned_qinf vs dvib_oneshot_q0)` | `+0.0031 ± 0.0024` | `+0.0032 ± 0.0005` | `+0.0038 ± 0.0010` | `+0.0011 ± 0.0017` | `+0.0003 ± 0.0005` |
| `query_gain(trader_transformer_b_qinf vs dvib_oneshot_q0)` | `-0.0107 ± 0.0023` | `-0.0103 ± 0.0073` | `-0.0029 ± 0.0015` | `-0.0061 ± 0.0006` | `-0.0216 ± 0.0018` |

### 2.4 Rule 9.1 Controls & Channel Symbol Count Audit ($b=1, 2$)

| Learned Arm | $b$ | Trained Regret | Untrained Init | Scale-Matched RandomNet | Max Distinct Symbols ($\le 2^b$) |
|---|---:|---:|---:|---:|---:|
| `self_summary_b` | `1` | `0.1213` | `0.4584` | `0.4337` | `2` |
| `self_summary_b` | `2` | `0.0734` | `0.4774` | `0.4034` | `4` |
| `random_encoder_b` | `1` | `0.2941` | `0.4584` | `0.4384` | `2` |
| `random_encoder_b` | `2` | `0.2624` | `0.4774` | `0.4406` | `4` |
| `dvib_oneshot_q0` | `1` | `0.1110` | `0.4584` | `0.4447` | `2` |
| `dvib_oneshot_q0` | `2` | `0.0693` | `0.4774` | `0.4224` | `3` |
| `coa_chain_b` | `1` | `0.1958` | `0.4662` | `0.4601` | `2` |
| `coa_chain_b` | `2` | `0.1082` | `0.4346` | `0.4346` | `4` |
| `fixed_state_nb` | `1` | `0.0634` | `0.4201` | `0.4524` | `9` |
| `fixed_state_nb` | `2` | `0.0553` | `0.4280` | `0.4482` | `59` |
| `queried_learned_qinf` | `1` | `0.1079` | `0.3873` | `0.4852` | `2` |
| `queried_learned_qinf` | `2` | `0.0661` | `0.4232` | `0.4193` | `3` |
| `trader_transformer_b_qinf` | `1` | `0.1217` | `0.5222` | `0.4533` | `2` |
| `trader_transformer_b_qinf` | `2` | `0.0796` | `0.4328` | `0.4228` | `3` |

## 2. `E3-sybil` ($N=8$, family=`pieces`, Prior `0.4184`, Floor `0.0000`, FCT `0.0420`, `dvib_b_inf` `0.0403`)

### 2.1 Closed-Form Reference Baselines (`aggbench.compression`, dev regret)

| Closed-Form Arm | Provenance / Query | $b=1$ | $b=2$ | $b=4$ | $b=8$ | $b=inf$ | $b_{90}$ (floor) |
|---|---|---:|---:|---:|---:|---:|---:|
| `llr_quantizer` | none (one-shot) `[PROVENANCE FULL: oracle structure at decoder]` | `0.0762` | `0.0334` | `0.0180` | `0.0158` | `0.0158` | `2` |
| `extremized_q` | none (one-shot) `[BLIND DECODER]` | `0.1579` | `0.0987` | `0.0760` | `0.0745` | `0.0747` | `None` |
| `trimmed_q` | none (one-shot) `[BLIND DECODER]` | `0.1737` | `0.1248` | `0.1057` | `0.1052` | `0.1043` | `None` |
| `coa_chain_scalar` | none (one-shot) `[BLIND DECODER]` | `0.1672` | `0.1114` | `0.0957` | `0.0807` | `0.0784` | `None` |
| `fixed_state` | none (one-shot) `[BLIND DECODER]` | `0.0807` | `0.0784` | `0.0784` | `0.0784` | `0.0784` | `None` |
| `top_w` | none (one-shot) `[PROVENANCE FULL: oracle structure at decoder]` | `0.4184` | `0.3507` | `0.1198` | `0.0371` | `-0.0000` | `8` |
| `k_of_n` | none (one-shot) `[PROVENANCE FULL: oracle structure at decoder]` | `0.1697` | `0.1118` | `0.0677` | `0.0354` | `-0.0000` | `8` |
| `queried_llr_qinf` | unbounded `[PROVENANCE FULL: oracle structure at decoder]` | `0.0762` | `0.0334` | `0.0180` | `0.0158` | `—` | `2` |
| `queried_llr_qb` | 1 bits `[PROVENANCE FULL: oracle structure at decoder]` | `0.0770` | `0.0334` | `0.0180` | `0.0158` | `—` | `2` |

### 2.2 Learned Compression Arms (3-Seed Mean ± SD Regret, `Untrained` & `RandomNet` Controls)

| Learned Arm | Protocol / State Budget | $b=1$ | $b=2$ | $b=4$ | $b=8$ | $b=inf$ | $b_{90}$ (floor) | $b_{90}$ (FCT) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `self_summary_b` | none (one-shot) | `0.1442 ± 0.0008` | `0.0826 ± 0.0089` | `0.0524 ± 0.0024` | `0.0457 ± 0.0008` | `0.0360 ± 0.0005` | `None` | `4` |
| `random_encoder_b` | none (one-shot) | `0.2968 ± 0.1172` | `0.2660 ± 0.1019` | `0.1110 ± 0.0163` | `0.0731 ± 0.0096` | `0.0392 ± 0.0008` | `None` | `8` |
| `dvib_oneshot_q0` | none (one-shot) | `0.1292 ± 0.0035` | `0.0842 ± 0.0045` | `0.0551 ± 0.0022` | `0.0429 ± 0.0006` | `0.0403 ± 0.0008` | `None` | `4` |
| `coa_chain_b` | state_chain [S = b-bit state] | `0.2358 ± 0.0055` | `0.1282 ± 0.0063` | `0.0872 ± 0.0009` | `0.0704 ± 0.0024` | `0.0456 ± 0.0007` | `None` | `8` |
| `fixed_state_nb` | state_chain [S = min(N*b, 32)-bit state, NOT b bits/agent] | `0.0704 ± 0.0024` | `0.0574 ± 0.0032` | `0.0570 ± 0.0014` | `0.0570 ± 0.0014` | `0.0456 ± 0.0007` | `None` | `1` |
| `queried_learned_qinf` | unbounded (read-only) | `0.1338 ± 0.0025` | `0.0683 ± 0.0018` | `0.0549 ± 0.0021` | `0.0479 ± 0.0009` | `0.0402 ± 0.0014` | `None` | `2` |
| `trader_transformer_b_qinf` | unbounded (read-only) | `0.1360 ± 0.0053` | `0.0851 ± 0.0009` | `0.0580 ± 0.0022` | `0.0561 ± 0.0016` | `0.0629 ± 0.0028` | `None` | `4` |

### 2.2b Gap to Uncompressed Full-Context Transformer (`gap(b) = regret(arm, b) - regret(FCT)`, `#3888`)

| Learned Arm | $b=1$ | $b=2$ | $b=4$ | $b=8$ | $b=inf$ |
|---|---:|---:|---:|---:|---:|
| `gap_to_fct(self_summary_b)` | `+0.1022` | `+0.0406` | `+0.0103` | `+0.0036` | `-0.0061` |
| `gap_to_fct(random_encoder_b)` | `+0.2548` | `+0.2240` | `+0.0689` | `+0.0311` | `-0.0029` |
| `gap_to_fct(dvib_oneshot_q0)` | `+0.0872` | `+0.0422` | `+0.0131` | `+0.0009` | `-0.0018` |
| `gap_to_fct(coa_chain_b)` | `+0.1938` | `+0.0861` | `+0.0452` | `+0.0284` | `+0.0036` |
| `gap_to_fct(fixed_state_nb)` | `+0.0284` | `+0.0154` | `+0.0149` | `+0.0149` | `+0.0036` |
| `gap_to_fct(queried_learned_qinf)` | `+0.0917` | `+0.0262` | `+0.0129` | `+0.0058` | `-0.0018` |
| `gap_to_fct(trader_transformer_b_qinf)` | `+0.0940` | `+0.0431` | `+0.0159` | `+0.0141` | `+0.0209` |

### 2.3 Paired Gains vs `self_summary_b`, `random_encoder_b`, and One-Shot `dvib_oneshot_q0`

| Comparison | $b=1$ | $b=2$ | $b=4$ | $b=8$ | $b=inf$ |
|---|---:|---:|---:|---:|---:|
| `gain(random_encoder_b vs self_summary_b)` | `-0.1526 ± 0.1177` | `-0.1834 ± 0.1049` | `-0.0586 ± 0.0172` | `-0.0274 ± 0.0103` | `-0.0032 ± 0.0006` |
| `gain(dvib_oneshot_q0 vs self_summary_b)` | `+0.0150 ± 0.0041` | `-0.0016 ± 0.0105` | `-0.0028 ± 0.0034` | `+0.0027 ± 0.0011` | `-0.0043 ± 0.0012` |
| `gain(coa_chain_b vs self_summary_b)` | `-0.0916 ± 0.0055` | `-0.0456 ± 0.0119` | `-0.0348 ± 0.0021` | `-0.0247 ± 0.0017` | `-0.0097 ± 0.0003` |
| `gain(fixed_state_nb vs self_summary_b)` | `+0.0738 ± 0.0017` | `+0.0252 ± 0.0122` | `-0.0046 ± 0.0021` | `-0.0113 ± 0.0010` | `-0.0097 ± 0.0003` |
| `gain(queried_learned_qinf vs self_summary_b)` | `+0.0105 ± 0.0026` | `+0.0143 ± 0.0107` | `-0.0025 ± 0.0022` | `-0.0022 ± 0.0016` | `-0.0042 ± 0.0010` |
| `gain(trader_transformer_b_qinf vs self_summary_b)` | `+0.0082 ± 0.0051` | `-0.0025 ± 0.0088` | `-0.0056 ± 0.0046` | `-0.0105 ± 0.0023` | `-0.0270 ± 0.0023` |
| `gain(self_summary_b vs random_encoder_b)` | `+0.1526 ± 0.1177` | `+0.1834 ± 0.1049` | `+0.0586 ± 0.0172` | `+0.0274 ± 0.0103` | `+0.0032 ± 0.0006` |
| `gain(dvib_oneshot_q0 vs random_encoder_b)` | `+0.1676 ± 0.1177` | `+0.1818 ± 0.0975` | `+0.0558 ± 0.0184` | `+0.0302 ± 0.0092` | `-0.0011 ± 0.0010` |
| `gain(coa_chain_b vs random_encoder_b)` | `+0.0609 ± 0.1131` | `+0.1378 ± 0.0957` | `+0.0238 ± 0.0157` | `+0.0027 ± 0.0113` | `-0.0065 ± 0.0003` |
| `gain(fixed_state_nb vs random_encoder_b)` | `+0.2264 ± 0.1192` | `+0.2086 ± 0.1011` | `+0.0540 ± 0.0176` | `+0.0161 ± 0.0100` | `-0.0065 ± 0.0003` |
| `gain(queried_learned_qinf vs random_encoder_b)` | `+0.1630 ± 0.1194` | `+0.1977 ± 0.1009` | `+0.0561 ± 0.0150` | `+0.0253 ± 0.0094` | `-0.0010 ± 0.0015` |
| `gain(trader_transformer_b_qinf vs random_encoder_b)` | `+0.1608 ± 0.1222` | `+0.1809 ± 0.1012` | `+0.0530 ± 0.0161` | `+0.0170 ± 0.0090` | `-0.0237 ± 0.0022` |
| `query_gain(queried_learned_qinf vs dvib_oneshot_q0)` | `-0.0046 ± 0.0029` | `+0.0160 ± 0.0040` | `+0.0002 ± 0.0042` | `-0.0049 ± 0.0014` | `+0.0001 ± 0.0022` |
| `query_gain(trader_transformer_b_qinf vs dvib_oneshot_q0)` | `-0.0068 ± 0.0049` | `-0.0008 ± 0.0037` | `-0.0028 ± 0.0027` | `-0.0132 ± 0.0019` | `-0.0226 ± 0.0032` |

### 2.4 Rule 9.1 Controls & Channel Symbol Count Audit ($b=1, 2$)

| Learned Arm | $b$ | Trained Regret | Untrained Init | Scale-Matched RandomNet | Max Distinct Symbols ($\le 2^b$) |
|---|---:|---:|---:|---:|---:|
| `self_summary_b` | `1` | `0.1442` | `0.4624` | `0.4370` | `2` |
| `self_summary_b` | `2` | `0.0826` | `0.4794` | `0.4023` | `4` |
| `random_encoder_b` | `1` | `0.2968` | `0.4624` | `0.4414` | `2` |
| `random_encoder_b` | `2` | `0.2660` | `0.4794` | `0.4346` | `4` |
| `dvib_oneshot_q0` | `1` | `0.1292` | `0.4624` | `0.4478` | `2` |
| `dvib_oneshot_q0` | `2` | `0.0842` | `0.4794` | `0.4189` | `3` |
| `coa_chain_b` | `1` | `0.2358` | `0.4654` | `0.4537` | `2` |
| `coa_chain_b` | `2` | `0.1282` | `0.4347` | `0.4321` | `4` |
| `fixed_state_nb` | `1` | `0.0704` | `0.4249` | `0.4396` | `26` |
| `fixed_state_nb` | `2` | `0.0574` | `0.4199` | `0.4395` | `259` |
| `queried_learned_qinf` | `1` | `0.1338` | `0.3932` | `0.4766` | `2` |
| `queried_learned_qinf` | `2` | `0.0683` | `0.4241` | `0.4191` | `3` |
| `trader_transformer_b_qinf` | `1` | `0.1360` | `0.5112` | `0.4510` | `2` |
| `trader_transformer_b_qinf` | `2` | `0.0851` | `0.4057` | `0.4295` | `3` |

## 2. `Z9-p2` ($N=7$, family=`grouped`, Prior `0.4567`, Floor `0.0000`, FCT `0.0617`, `dvib_b_inf` `0.2583`)

### 2.1 Closed-Form Reference Baselines (`aggbench.compression`, dev regret)

| Closed-Form Arm | Provenance / Query | $b=1$ | $b=2$ | $b=4$ | $b=8$ | $b=inf$ | $b_{90}$ (floor) |
|---|---|---:|---:|---:|---:|---:|---:|
| `llr_quantizer` | none (one-shot) `[PROVENANCE FULL: oracle structure at decoder]` | `0.4567` | `0.4567` | `0.4567` | `0.4567` | `0.4567` | `None` |
| `raw_code` | none (one-shot) `[PROVENANCE FULL: oracle structure at decoder]` | `0.3595` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `2` |

### 2.2 Learned Compression Arms (3-Seed Mean ± SD Regret, `Untrained` & `RandomNet` Controls)

| Learned Arm | Protocol / State Budget | $b=1$ | $b=2$ | $b=4$ | $b=8$ | $b=inf$ | $b_{90}$ (floor) | $b_{90}$ (FCT) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `self_summary_b` | none (one-shot) | `0.4568 ± 0.0002` | `0.4569 ± 0.0003` | `0.4589 ± 0.0011` | `0.4927 ± 0.0298` | `0.3774 ± 0.1197` | `None` | `None` |
| `random_encoder_b` | none (one-shot) | `0.4567 ± 0.0001` | `0.4567 ± 0.0001` | `0.4567 ± 0.0001` | `0.4568 ± 0.0001` | `0.4567 ± 0.0001` | `None` | `None` |
| `dvib_oneshot_q0` | none (one-shot) | `0.4567 ± 0.0002` | `0.3918 ± 0.0918` | `0.2666 ± 0.1345` | `0.2087 ± 0.0582` | `0.2583 ± 0.1405` | `None` | `None` |
| `coa_chain_b` | state_chain [S = b-bit state] | `0.4567 ± 0.0004` | `0.4508 ± 0.0086` | `0.4169 ± 0.0302` | `0.2781 ± 0.0387` | `0.1589 ± 0.0008` | `None` | `None` |
| `fixed_state_nb` | state_chain [S = min(N*b, 32)-bit state, NOT b bits/agent] | `0.2943 ± 0.0203` | `0.2590 ± 0.0545` | `0.3756 ± 0.1172` | `0.1915 ± 0.0025` | `0.1589 ± 0.0008` | `None` | `None` |
| `queried_learned_qinf` | unbounded (read-only) | `0.4571 ± 0.0003` | `0.4575 ± 0.0004` | `0.3641 ± 0.1308` | `0.2868 ± 0.1233` | `0.3827 ± 0.1051` | `None` | `None` |
| `trader_transformer_b_qinf` | unbounded (read-only) | `0.4565 ± 0.0004` | `0.4568 ± 0.0003` | `0.4565 ± 0.0003` | `0.3608 ± 0.1359` | `0.3612 ± 0.1355` | `None` | `None` |
| `dvib_oneshot_q0_curr_aux` | none (one-shot) [LATENT-LABEL SUPERVISION] | `0.4052 ± 0.0360` | `0.3437 ± 0.0407` | `0.1174 ± 0.0119` | `0.0589 ± 0.0079` | `0.0111 ± 0.0002` | `None` | `8` |
| `trader_transformer_b_qinf_curr_aux` | unbounded (read-only) [LATENT-LABEL SUPERVISION] | `0.4123 ± 0.0121` | `0.3158 ± 0.0326` | `0.1359 ± 0.0268` | `0.0720 ± 0.0149` | `0.0646 ± 0.0035` | `None` | `8` |

### 2.2b Gap to Uncompressed Full-Context Transformer (`gap(b) = regret(arm, b) - regret(FCT)`, `#3888`)

| Learned Arm | $b=1$ | $b=2$ | $b=4$ | $b=8$ | $b=inf$ |
|---|---:|---:|---:|---:|---:|
| `gap_to_fct(self_summary_b)` | `+0.3951` | `+0.3952` | `+0.3972` | `+0.4310` | `+0.3157` |
| `gap_to_fct(random_encoder_b)` | `+0.3950` | `+0.3950` | `+0.3950` | `+0.3951` | `+0.3950` |
| `gap_to_fct(dvib_oneshot_q0)` | `+0.3950` | `+0.3301` | `+0.2049` | `+0.1470` | `+0.1966` |
| `gap_to_fct(coa_chain_b)` | `+0.3950` | `+0.3891` | `+0.3552` | `+0.2164` | `+0.0972` |
| `gap_to_fct(fixed_state_nb)` | `+0.2326` | `+0.1973` | `+0.3139` | `+0.1298` | `+0.0972` |
| `gap_to_fct(queried_learned_qinf)` | `+0.3954` | `+0.3958` | `+0.3024` | `+0.2251` | `+0.3210` |
| `gap_to_fct(trader_transformer_b_qinf)` | `+0.3948` | `+0.3951` | `+0.3948` | `+0.2991` | `+0.2995` |
| `gap_to_fct(dvib_oneshot_q0_curr_aux)` | `+0.3435` | `+0.2820` | `+0.0557` | `-0.0028` | `-0.0506` |
| `gap_to_fct(trader_transformer_b_qinf_curr_aux)` | `+0.3506` | `+0.2541` | `+0.0742` | `+0.0103` | `+0.0029` |

### 2.3 Paired Gains vs `self_summary_b`, `random_encoder_b`, and One-Shot `dvib_oneshot_q0`

| Comparison | $b=1$ | $b=2$ | $b=4$ | $b=8$ | $b=inf$ |
|---|---:|---:|---:|---:|---:|
| `gain(random_encoder_b vs self_summary_b)` | `+0.0001 ± 0.0003` | `+0.0002 ± 0.0004` | `+0.0022 ± 0.0011` | `+0.0359 ± 0.0298` | `-0.0793 ± 0.1198` |
| `gain(dvib_oneshot_q0 vs self_summary_b)` | `+0.0001 ± 0.0002` | `+0.0651 ± 0.0916` | `+0.1922 ± 0.1344` | `+0.2840 ± 0.0827` | `+0.1191 ± 0.1328` |
| `gain(coa_chain_b vs self_summary_b)` | `+0.0001 ± 0.0003` | `+0.0061 ± 0.0090` | `+0.0419 ± 0.0293` | `+0.2146 ± 0.0488` | `+0.2184 ± 0.1202` |
| `gain(fixed_state_nb vs self_summary_b)` | `+0.1625 ± 0.0201` | `+0.1979 ± 0.0548` | `+0.0832 ± 0.1162` | `+0.3012 ± 0.0292` | `+0.2184 ± 0.1202` |
| `gain(queried_learned_qinf vs self_summary_b)` | `-0.0003 ± 0.0005` | `-0.0006 ± 0.0008` | `+0.0947 ± 0.1298` | `+0.2059 ± 0.1493` | `-0.0053 ± 0.1952` |
| `gain(trader_transformer_b_qinf vs self_summary_b)` | `+0.0003 ± 0.0006` | `+0.0001 ± 0.0005` | `+0.0023 ± 0.0009` | `+0.1319 ± 0.1649` | `+0.0162 ± 0.0158` |
| `gain(dvib_oneshot_q0_curr_aux vs self_summary_b)` | `+0.0516 ± 0.0358` | `+0.1132 ± 0.0410` | `+0.3415 ± 0.0130` | `+0.4339 ± 0.0369` | `+0.3663 ± 0.1197` |
| `gain(trader_transformer_b_qinf_curr_aux vs self_summary_b)` | `+0.0446 ± 0.0123` | `+0.1411 ± 0.0323` | `+0.3229 ± 0.0278` | `+0.4207 ± 0.0205` | `+0.3127 ± 0.1229` |
| `gain(self_summary_b vs random_encoder_b)` | `-0.0001 ± 0.0003` | `-0.0002 ± 0.0004` | `-0.0022 ± 0.0011` | `-0.0359 ± 0.0298` | `+0.0793 ± 0.1198` |
| `gain(dvib_oneshot_q0 vs random_encoder_b)` | `+0.0000 ± 0.0002` | `+0.0650 ± 0.0919` | `+0.1900 ± 0.1346` | `+0.2481 ± 0.0582` | `+0.1985 ± 0.1405` |
| `gain(coa_chain_b vs random_encoder_b)` | `+0.0000 ± 0.0004` | `+0.0059 ± 0.0086` | `+0.0398 ± 0.0301` | `+0.1787 ± 0.0388` | `+0.2978 ± 0.0008` |
| `gain(fixed_state_nb vs random_encoder_b)` | `+0.1624 ± 0.0203` | `+0.1977 ± 0.0544` | `+0.0810 ± 0.1172` | `+0.2653 ± 0.0025` | `+0.2978 ± 0.0008` |
| `gain(queried_learned_qinf vs random_encoder_b)` | `-0.0004 ± 0.0002` | `-0.0008 ± 0.0003` | `+0.0926 ± 0.1308` | `+0.1700 ± 0.1233` | `+0.0740 ± 0.1051` |
| `gain(trader_transformer_b_qinf vs random_encoder_b)` | `+0.0002 ± 0.0003` | `-0.0000 ± 0.0002` | `+0.0001 ± 0.0003` | `+0.0960 ± 0.1360` | `+0.0955 ± 0.1356` |
| `gain(dvib_oneshot_q0_curr_aux vs random_encoder_b)` | `+0.0515 ± 0.0359` | `+0.1130 ± 0.0406` | `+0.3393 ± 0.0120` | `+0.3979 ± 0.0080` | `+0.4457 ± 0.0002` |
| `gain(trader_transformer_b_qinf_curr_aux vs random_encoder_b)` | `+0.0444 ± 0.0121` | `+0.1409 ± 0.0326` | `+0.3208 ± 0.0269` | `+0.3848 ± 0.0149` | `+0.3921 ± 0.0034` |
| `query_gain(queried_learned_qinf vs dvib_oneshot_q0)` | `-0.0004 ± 0.0003` | `-0.0657 ± 0.0922` | `-0.0975 ± 0.1311` | `-0.0781 ± 0.0668` | `-0.1244 ± 0.1258` |
| `query_gain(trader_transformer_b_qinf vs dvib_oneshot_q0)` | `+0.0002 ± 0.0005` | `-0.0650 ± 0.0921` | `-0.1899 ± 0.1344` | `-0.1521 ± 0.1156` | `-0.1029 ± 0.1390` |

### 2.4 Rule 9.1 Controls & Channel Symbol Count Audit ($b=1, 2$)

| Learned Arm | $b$ | Trained Regret | Untrained Init | Scale-Matched RandomNet | Max Distinct Symbols ($\le 2^b$) |
|---|---:|---:|---:|---:|---:|
| `self_summary_b` | `1` | `0.4568` | `0.4672` | `0.4624` | `2` |
| `self_summary_b` | `2` | `0.4569` | `0.5101` | `0.4694` | `4` |
| `random_encoder_b` | `1` | `0.4567` | `0.4672` | `0.4621` | `1` |
| `random_encoder_b` | `2` | `0.4567` | `0.5101` | `0.4698` | `4` |
| `dvib_oneshot_q0` | `1` | `0.4567` | `0.4672` | `0.4629` | `2` |
| `dvib_oneshot_q0` | `2` | `0.3918` | `0.5101` | `0.4739` | `4` |
| `coa_chain_b` | `1` | `0.4567` | `0.4621` | `0.4761` | `2` |
| `coa_chain_b` | `2` | `0.4508` | `0.4720` | `0.4597` | `4` |
| `fixed_state_nb` | `1` | `0.2943` | `0.4598` | `0.4828` | `39` |
| `fixed_state_nb` | `2` | `0.2590` | `0.4591` | `0.4802` | `224` |
| `queried_learned_qinf` | `1` | `0.4571` | `0.4743` | `0.4790` | `2` |
| `queried_learned_qinf` | `2` | `0.4575` | `0.4776` | `0.4636` | `4` |
| `trader_transformer_b_qinf` | `1` | `0.4565` | `0.4904` | `0.4620` | `2` |
| `trader_transformer_b_qinf` | `2` | `0.4568` | `0.5018` | `0.5019` | `4` |
| `dvib_oneshot_q0_curr_aux` | `1` | `0.4052` | `0.4672` | `0.4725` | `2` |
| `dvib_oneshot_q0_curr_aux` | `2` | `0.3437` | `0.5101` | `0.5155` | `4` |
| `trader_transformer_b_qinf_curr_aux` | `1` | `0.4123` | `0.4904` | `0.4773` | `2` |
| `trader_transformer_b_qinf_curr_aux` | `2` | `0.3158` | `0.5018` | `0.5970` | `4` |

## 2. `E4` ($N=6$, family=`rollout`, Prior `0.3532`, Floor `0.0472`, FCT `0.0795`, `dvib_b_inf` `0.0795`)

### 2.1 Closed-Form Reference Baselines (`aggbench.compression`, dev regret)

| Closed-Form Arm | Provenance / Query | $b=1$ | $b=2$ | $b=4$ | $b=8$ | $b=inf$ | $b_{90}$ (floor) |
|---|---|---:|---:|---:|---:|---:|---:|
| `llr_quantizer` | none (one-shot) `[PROVENANCE FULL: oracle structure at decoder]` | `0.1287` | `0.0939` | `0.0820` | `0.0802` | `0.0798` | `None` |
| `extremized_q` | none (one-shot) `[BLIND DECODER]` | `0.1474` | `0.1123` | `0.1022` | `0.1014` | `0.1014` | `None` |
| `trimmed_q` | none (one-shot) `[BLIND DECODER]` | `0.1557` | `0.1247` | `0.1118` | `0.1107` | `0.1107` | `None` |
| `coa_chain_scalar` | none (one-shot) `[BLIND DECODER]` | `0.2452` | `0.1512` | `0.1183` | `0.1034` | `0.1029` | `None` |
| `fixed_state` | none (one-shot) `[BLIND DECODER]` | `0.1047` | `0.1030` | `0.1029` | `0.1029` | `0.1029` | `None` |
| `queried_llr_qinf` | unbounded `[PROVENANCE FULL: oracle structure at decoder]` | `0.1287` | `0.0939` | `0.0820` | `0.0802` | `—` | `None` |
| `queried_llr_qb` | 1 bits `[PROVENANCE FULL: oracle structure at decoder]` | `0.1287` | `0.0939` | `0.0813` | `0.0802` | `—` | `None` |

### 2.2 Learned Compression Arms (3-Seed Mean ± SD Regret, `Untrained` & `RandomNet` Controls)

| Learned Arm | Protocol / State Budget | $b=1$ | $b=2$ | $b=4$ | $b=8$ | $b=inf$ | $b_{90}$ (floor) | $b_{90}$ (FCT) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `self_summary_b` | none (one-shot) | `0.1390 ± 0.0016` | `0.1073 ± 0.0013` | `0.0918 ± 0.0007` | `0.0844 ± 0.0005` | `0.0789 ± 0.0005` | `None` | `4` |
| `random_encoder_b` | none (one-shot) | `0.3473 ± 0.0081` | `0.2166 ± 0.0962` | `0.1572 ± 0.0296` | `0.1186 ± 0.0090` | `0.0901 ± 0.0005` | `None` | `None` |
| `dvib_oneshot_q0` | none (one-shot) | `0.1394 ± 0.0046` | `0.1061 ± 0.0018` | `0.0901 ± 0.0011` | `0.0849 ± 0.0012` | `0.0795 ± 0.0007` | `None` | `2` |
| `coa_chain_b` | state_chain [S = b-bit state] | `0.2200 ± 0.0016` | `0.1534 ± 0.0024` | `0.1121 ± 0.0036` | `0.1022 ± 0.0017` | `0.0802 ± 0.0002` | `None` | `8` |
| `fixed_state_nb` | state_chain [S = min(N*b, 32)-bit state, NOT b bits/agent] | `0.1049 ± 0.0020` | `0.0976 ± 0.0019` | `0.0936 ± 0.0002` | `0.0906 ± 0.0009` | `0.0802 ± 0.0002` | `None` | `1` |
| `queried_learned_qinf` | unbounded (read-only) | `0.1327 ± 0.0000` | `0.1034 ± 0.0012` | `0.0897 ± 0.0003` | `0.0848 ± 0.0015` | `0.0794 ± 0.0001` | `None` | `2` |
| `trader_transformer_b_qinf` | unbounded (read-only) | `0.1355 ± 0.0045` | `0.1103 ± 0.0004` | `0.0959 ± 0.0006` | `0.0908 ± 0.0014` | `0.0993 ± 0.0061` | `None` | `4` |

### 2.2b Gap to Uncompressed Full-Context Transformer (`gap(b) = regret(arm, b) - regret(FCT)`, `#3888`)

| Learned Arm | $b=1$ | $b=2$ | $b=4$ | $b=8$ | $b=inf$ |
|---|---:|---:|---:|---:|---:|
| `gap_to_fct(self_summary_b)` | `+0.0595` | `+0.0279` | `+0.0123` | `+0.0050` | `-0.0006` |
| `gap_to_fct(random_encoder_b)` | `+0.2678` | `+0.1371` | `+0.0777` | `+0.0391` | `+0.0106` |
| `gap_to_fct(dvib_oneshot_q0)` | `+0.0599` | `+0.0266` | `+0.0106` | `+0.0054` | `+0.0000` |
| `gap_to_fct(coa_chain_b)` | `+0.1405` | `+0.0739` | `+0.0326` | `+0.0227` | `+0.0008` |
| `gap_to_fct(fixed_state_nb)` | `+0.0254` | `+0.0181` | `+0.0141` | `+0.0111` | `+0.0008` |
| `gap_to_fct(queried_learned_qinf)` | `+0.0533` | `+0.0239` | `+0.0102` | `+0.0053` | `-0.0001` |
| `gap_to_fct(trader_transformer_b_qinf)` | `+0.0560` | `+0.0308` | `+0.0165` | `+0.0113` | `+0.0199` |

### 2.3 Paired Gains vs `self_summary_b`, `random_encoder_b`, and One-Shot `dvib_oneshot_q0`

| Comparison | $b=1$ | $b=2$ | $b=4$ | $b=8$ | $b=inf$ |
|---|---:|---:|---:|---:|---:|
| `gain(random_encoder_b vs self_summary_b)` | `-0.2083 ± 0.0090` | `-0.1092 ± 0.0956` | `-0.0654 ± 0.0290` | `-0.0341 ± 0.0093` | `-0.0112 ± 0.0009` |
| `gain(dvib_oneshot_q0 vs self_summary_b)` | `-0.0004 ± 0.0037` | `+0.0013 ± 0.0012` | `+0.0017 ± 0.0006` | `-0.0005 ± 0.0017` | `-0.0006 ± 0.0003` |
| `gain(coa_chain_b vs self_summary_b)` | `-0.0810 ± 0.0032` | `-0.0461 ± 0.0027` | `-0.0203 ± 0.0032` | `-0.0178 ± 0.0021` | `-0.0014 ± 0.0003` |
| `gain(fixed_state_nb vs self_summary_b)` | `+0.0341 ± 0.0036` | `+0.0098 ± 0.0012` | `-0.0018 ± 0.0009` | `-0.0061 ± 0.0008` | `-0.0014 ± 0.0003` |
| `gain(queried_learned_qinf vs self_summary_b)` | `+0.0063 ± 0.0016` | `+0.0039 ± 0.0015` | `+0.0021 ± 0.0008` | `-0.0004 ± 0.0013` | `-0.0005 ± 0.0006` |
| `gain(trader_transformer_b_qinf vs self_summary_b)` | `+0.0035 ± 0.0048` | `-0.0029 ± 0.0017` | `-0.0041 ± 0.0014` | `-0.0063 ± 0.0019` | `-0.0205 ± 0.0065` |
| `gain(self_summary_b vs random_encoder_b)` | `+0.2083 ± 0.0090` | `+0.1092 ± 0.0956` | `+0.0654 ± 0.0290` | `+0.0341 ± 0.0093` | `+0.0112 ± 0.0009` |
| `gain(dvib_oneshot_q0 vs random_encoder_b)` | `+0.2079 ± 0.0126` | `+0.1105 ± 0.0967` | `+0.0671 ± 0.0293` | `+0.0336 ± 0.0085` | `+0.0106 ± 0.0009` |
| `gain(coa_chain_b vs random_encoder_b)` | `+0.1273 ± 0.0074` | `+0.0631 ± 0.0939` | `+0.0451 ± 0.0262` | `+0.0164 ± 0.0074` | `+0.0098 ± 0.0007` |
| `gain(fixed_state_nb vs random_encoder_b)` | `+0.2424 ± 0.0067` | `+0.1190 ± 0.0968` | `+0.0635 ± 0.0298` | `+0.0280 ± 0.0099` | `+0.0098 ± 0.0007` |
| `gain(queried_learned_qinf vs random_encoder_b)` | `+0.2146 ± 0.0081` | `+0.1132 ± 0.0971` | `+0.0674 ± 0.0299` | `+0.0338 ± 0.0105` | `+0.0107 ± 0.0004` |
| `gain(trader_transformer_b_qinf vs random_encoder_b)` | `+0.2118 ± 0.0049` | `+0.1063 ± 0.0962` | `+0.0612 ± 0.0302` | `+0.0278 ± 0.0082` | `-0.0093 ± 0.0061` |
| `query_gain(queried_learned_qinf vs dvib_oneshot_q0)` | `+0.0066 ± 0.0047` | `+0.0026 ± 0.0009` | `+0.0004 ± 0.0010` | `+0.0001 ± 0.0022` | `+0.0001 ± 0.0007` |
| `query_gain(trader_transformer_b_qinf vs dvib_oneshot_q0)` | `+0.0039 ± 0.0084` | `-0.0042 ± 0.0021` | `-0.0059 ± 0.0016` | `-0.0058 ± 0.0004` | `-0.0199 ± 0.0068` |

### 2.4 Rule 9.1 Controls & Channel Symbol Count Audit ($b=1, 2$)

| Learned Arm | $b$ | Trained Regret | Untrained Init | Scale-Matched RandomNet | Max Distinct Symbols ($\le 2^b$) |
|---|---:|---:|---:|---:|---:|
| `self_summary_b` | `1` | `0.1390` | `0.3733` | `0.3875` | `2` |
| `self_summary_b` | `2` | `0.1073` | `0.4283` | `0.3863` | `3` |
| `random_encoder_b` | `1` | `0.3473` | `0.3733` | `0.3837` | `2` |
| `random_encoder_b` | `2` | `0.2166` | `0.4283` | `0.3842` | `3` |
| `dvib_oneshot_q0` | `1` | `0.1394` | `0.3733` | `0.4032` | `2` |
| `dvib_oneshot_q0` | `2` | `0.1061` | `0.4283` | `0.4023` | `3` |
| `coa_chain_b` | `1` | `0.2200` | `0.3534` | `0.3976` | `2` |
| `coa_chain_b` | `2` | `0.1534` | `0.3608` | `0.3882` | `3` |
| `fixed_state_nb` | `1` | `0.1049` | `0.3578` | `0.3751` | `16` |
| `fixed_state_nb` | `2` | `0.0976` | `0.3616` | `0.3624` | `117` |
| `queried_learned_qinf` | `1` | `0.1327` | `0.3606` | `0.4485` | `2` |
| `queried_learned_qinf` | `2` | `0.1034` | `0.3574` | `0.3938` | `3` |
| `trader_transformer_b_qinf` | `1` | `0.1355` | `0.4156` | `0.3568` | `2` |
| `trader_transformer_b_qinf` | `2` | `0.1103` | `0.4285` | `0.3614` | `3` |

## 2. `Z8` ($N=5$, family=`privateread`, Prior `0.4500`, Floor `0.0000`, FCT `0.0364`, `dvib_b_inf` `0.0338`)

### 2.1 Closed-Form Reference Baselines (`aggbench.compression`, dev regret)

| Closed-Form Arm | Provenance / Query | $b=1$ | $b=2$ | $b=4$ | $b=8$ | $b=inf$ | $b_{90}$ (floor) |
|---|---|---:|---:|---:|---:|---:|---:|
| `llr_quantizer` | none (one-shot) `[PROVENANCE FULL: oracle structure at decoder]` | `0.0750` | `0.0337` | `0.0184` | `0.0168` | `0.0168` | `2` |
| `extremized_q` | none (one-shot) `[BLIND DECODER]` | `0.0943` | `0.0504` | `0.0318` | `0.0305` | `0.0304` | `4` |
| `trimmed_q` | none (one-shot) `[BLIND DECODER]` | `0.1060` | `0.0626` | `0.0486` | `0.0468` | `0.0468` | `None` |
| `coa_chain_scalar` | none (one-shot) `[BLIND DECODER]` | `0.1886` | `0.0994` | `0.0370` | `0.0300` | `0.0302` | `4` |
| `fixed_state` | none (one-shot) `[BLIND DECODER]` | `0.0338` | `0.0301` | `0.0302` | `0.0302` | `0.0302` | `1` |
| `queried_llr_qinf` | unbounded `[PROVENANCE FULL: oracle structure at decoder]` | `0.0750` | `0.0337` | `0.0184` | `0.0168` | `—` | `2` |
| `queried_llr_qb` | 1 bits `[PROVENANCE FULL: oracle structure at decoder]` | `0.0787` | `0.0337` | `0.0186` | `0.0170` | `—` | `2` |

### 2.2 Learned Compression Arms (3-Seed Mean ± SD Regret, `Untrained` & `RandomNet` Controls)

| Learned Arm | Protocol / State Budget | $b=1$ | $b=2$ | $b=4$ | $b=8$ | $b=inf$ | $b_{90}$ (floor) | $b_{90}$ (FCT) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `self_summary_b` | none (one-shot) | `0.1209 ± 0.0012` | `0.0651 ± 0.0038` | `0.0400 ± 0.0020` | `0.0345 ± 0.0007` | `0.0301 ± 0.0003` | `4` | `2` |
| `random_encoder_b` | none (one-shot) | `0.3134 ± 0.1378` | `0.2686 ± 0.1285` | `0.0828 ± 0.0107` | `0.0610 ± 0.0118` | `0.0301 ± 0.0003` | `None` | `8` |
| `dvib_oneshot_q0` | none (one-shot) | `0.0948 ± 0.0021` | `0.0611 ± 0.0012` | `0.0447 ± 0.0025` | `0.0384 ± 0.0008` | `0.0338 ± 0.0003` | `4` | `2` |
| `coa_chain_b` | state_chain [S = b-bit state] | `0.2116 ± 0.0016` | `0.1096 ± 0.0089` | `0.0670 ± 0.0010` | `0.0512 ± 0.0014` | `0.0319 ± 0.0005` | `None` | `4` |
| `fixed_state_nb` | state_chain [S = min(N*b, 32)-bit state, NOT b bits/agent] | `0.0588 ± 0.0020` | `0.0511 ± 0.0016` | `0.0449 ± 0.0007` | `0.0429 ± 0.0003` | `0.0319 ± 0.0005` | `4` | `1` |
| `queried_learned_qinf` | unbounded (read-only) | `0.0968 ± 0.0020` | `0.0608 ± 0.0016` | `0.0447 ± 0.0011` | `0.0405 ± 0.0001` | `0.0321 ± 0.0003` | `4` | `2` |
| `trader_transformer_b_qinf` | unbounded (read-only) | `0.1094 ± 0.0029` | `0.0702 ± 0.0012` | `0.0513 ± 0.0017` | `0.0470 ± 0.0007` | `0.0549 ± 0.0032` | `None` | `2` |

### 2.2b Gap to Uncompressed Full-Context Transformer (`gap(b) = regret(arm, b) - regret(FCT)`, `#3888`)

| Learned Arm | $b=1$ | $b=2$ | $b=4$ | $b=8$ | $b=inf$ |
|---|---:|---:|---:|---:|---:|
| `gap_to_fct(self_summary_b)` | `+0.0844` | `+0.0286` | `+0.0036` | `-0.0020` | `-0.0064` |
| `gap_to_fct(random_encoder_b)` | `+0.2770` | `+0.2322` | `+0.0464` | `+0.0245` | `-0.0063` |
| `gap_to_fct(dvib_oneshot_q0)` | `+0.0583` | `+0.0247` | `+0.0082` | `+0.0020` | `-0.0026` |
| `gap_to_fct(coa_chain_b)` | `+0.1752` | `+0.0731` | `+0.0305` | `+0.0148` | `-0.0045` |
| `gap_to_fct(fixed_state_nb)` | `+0.0223` | `+0.0147` | `+0.0085` | `+0.0064` | `-0.0045` |
| `gap_to_fct(queried_learned_qinf)` | `+0.0604` | `+0.0244` | `+0.0083` | `+0.0040` | `-0.0043` |
| `gap_to_fct(trader_transformer_b_qinf)` | `+0.0730` | `+0.0338` | `+0.0149` | `+0.0106` | `+0.0184` |

### 2.3 Paired Gains vs `self_summary_b`, `random_encoder_b`, and One-Shot `dvib_oneshot_q0`

| Comparison | $b=1$ | $b=2$ | $b=4$ | $b=8$ | $b=inf$ |
|---|---:|---:|---:|---:|---:|
| `gain(random_encoder_b vs self_summary_b)` | `-0.1925 ± 0.1378` | `-0.2035 ± 0.1305` | `-0.0428 ± 0.0110` | `-0.0265 ± 0.0111` | `-0.0000 ± 0.0004` |
| `gain(dvib_oneshot_q0 vs self_summary_b)` | `+0.0261 ± 0.0010` | `+0.0040 ± 0.0027` | `-0.0047 ± 0.0014` | `-0.0040 ± 0.0012` | `-0.0037 ± 0.0000` |
| `gain(coa_chain_b vs self_summary_b)` | `-0.0908 ± 0.0024` | `-0.0445 ± 0.0116` | `-0.0270 ± 0.0019` | `-0.0168 ± 0.0018` | `-0.0018 ± 0.0004` |
| `gain(fixed_state_nb vs self_summary_b)` | `+0.0621 ± 0.0032` | `+0.0140 ± 0.0040` | `-0.0049 ± 0.0022` | `-0.0084 ± 0.0005` | `-0.0018 ± 0.0004` |
| `gain(queried_learned_qinf vs self_summary_b)` | `+0.0240 ± 0.0008` | `+0.0043 ± 0.0047` | `-0.0047 ± 0.0031` | `-0.0060 ± 0.0007` | `-0.0020 ± 0.0001` |
| `gain(trader_transformer_b_qinf vs self_summary_b)` | `+0.0115 ± 0.0018` | `-0.0052 ± 0.0027` | `-0.0113 ± 0.0023` | `-0.0126 ± 0.0001` | `-0.0248 ± 0.0029` |
| `gain(self_summary_b vs random_encoder_b)` | `+0.1925 ± 0.1378` | `+0.2035 ± 0.1305` | `+0.0428 ± 0.0110` | `+0.0265 ± 0.0111` | `+0.0000 ± 0.0004` |
| `gain(dvib_oneshot_q0 vs random_encoder_b)` | `+0.2186 ± 0.1386` | `+0.2075 ± 0.1287` | `+0.0382 ± 0.0097` | `+0.0226 ± 0.0122` | `-0.0037 ± 0.0004` |
| `gain(coa_chain_b vs random_encoder_b)` | `+0.1018 ± 0.1393` | `+0.1590 ± 0.1198` | `+0.0158 ± 0.0098` | `+0.0097 ± 0.0126` | `-0.0018 ± 0.0008` |
| `gain(fixed_state_nb vs random_encoder_b)` | `+0.2546 ± 0.1368` | `+0.2175 ± 0.1272` | `+0.0379 ± 0.0100` | `+0.0181 ± 0.0115` | `-0.0018 ± 0.0008` |
| `gain(queried_learned_qinf vs random_encoder_b)` | `+0.2166 ± 0.1381` | `+0.2078 ± 0.1295` | `+0.0381 ± 0.0106` | `+0.0205 ± 0.0117` | `-0.0020 ± 0.0003` |
| `gain(trader_transformer_b_qinf vs random_encoder_b)` | `+0.2040 ± 0.1369` | `+0.1984 ± 0.1291` | `+0.0315 ± 0.0090` | `+0.0140 ± 0.0110` | `-0.0247 ± 0.0031` |
| `query_gain(queried_learned_qinf vs dvib_oneshot_q0)` | `-0.0021 ± 0.0005` | `+0.0003 ± 0.0026` | `-0.0001 ± 0.0035` | `-0.0020 ± 0.0009` | `+0.0017 ± 0.0001` |
| `query_gain(trader_transformer_b_qinf vs dvib_oneshot_q0)` | `-0.0146 ± 0.0019` | `-0.0091 ± 0.0005` | `-0.0066 ± 0.0018` | `-0.0086 ± 0.0013` | `-0.0210 ± 0.0029` |

### 2.4 Rule 9.1 Controls & Channel Symbol Count Audit ($b=1, 2$)

| Learned Arm | $b$ | Trained Regret | Untrained Init | Scale-Matched RandomNet | Max Distinct Symbols ($\le 2^b$) |
|---|---:|---:|---:|---:|---:|
| `self_summary_b` | `1` | `0.1209` | `0.4907` | `0.4676` | `2` |
| `self_summary_b` | `2` | `0.0651` | `0.5093` | `0.4299` | `4` |
| `random_encoder_b` | `1` | `0.3134` | `0.4907` | `0.4723` | `2` |
| `random_encoder_b` | `2` | `0.2686` | `0.5093` | `0.4673` | `4` |
| `dvib_oneshot_q0` | `1` | `0.0948` | `0.4907` | `0.4772` | `2` |
| `dvib_oneshot_q0` | `2` | `0.0611` | `0.5093` | `0.4484` | `3` |
| `coa_chain_b` | `1` | `0.2116` | `0.5034` | `0.4965` | `2` |
| `coa_chain_b` | `2` | `0.1096` | `0.4636` | `0.4764` | `4` |
| `fixed_state_nb` | `1` | `0.0588` | `0.4530` | `0.4801` | `10` |
| `fixed_state_nb` | `2` | `0.0511` | `0.4598` | `0.4809` | `66` |
| `queried_learned_qinf` | `1` | `0.0968` | `0.4178` | `0.5269` | `2` |
| `queried_learned_qinf` | `2` | `0.0608` | `0.4572` | `0.4506` | `3` |
| `trader_transformer_b_qinf` | `1` | `0.1094` | `0.5453` | `0.4884` | `2` |
| `trader_transformer_b_qinf` | `2` | `0.0702` | `0.4696` | `0.4541` | `3` |

