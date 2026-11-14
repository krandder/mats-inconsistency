# Qwen corrected-CDF collection — aggregate tranches 02–05, 13 September 2026

Tranches **02–05 completed 32 additional parse-valid EOS readouts**, producing **147 tokens** with unchanged base and adapter weights. All four supervisors exited **0**. At **2026-09-13T14:41:46.536310+00:00**, cumulative corrected-v3 coverage was **34 attempted / 34 parse-valid / 2,926 full propositions** from the original **1,619 training episodes**; **2,892 remained untouched**, with **zero invalid or incomplete attempts**.

This [aggregate evidence manifest](../data/research/qwen-cdf-collection-progress-20260913.json) freezes one continuation snapshot. It references the [first-v3 snapshot](qwen-cdf-collection-20260913.md) and its two observations by immutable hashes. No v1/v2 observations are counted. Coverage remains incomplete: no complete own-base anchor, anchor export, scientific model selection, backward pass, optimizer update, training result or ranking is established.

## Terminal invocations and resources

| Tranche / internal report | Readouts / parse-valid | Tokens | Internal wall s | External wall s | Internal CPU s | Peak RSS bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [tranche-01](../data/research/qwen-cdf-collection-20260913/collection/invocations/d4e87097-92d5-4663-b1c0-22401ff214c3/report.json) | 2 / 2 | 9 | 331.936 | 333.585 | 510.483 | 20,777,639,936 |
| [tranche-02](../data/research/qwen-cdf-collection-progress-20260913/collection/invocations/35fddf66-9d43-4257-805b-076659d32f6e/report.json) | 8 / 8 | 38 | 1007.350 | 1009.041 | 1651.162 | 20,901,392,384 |
| [tranche-03](../data/research/qwen-cdf-collection-progress-20260913/collection/invocations/b670730a-b60e-4248-83df-e4f435840f1d/report.json) | 8 / 8 | 38 | 944.784 | 946.569 | 1581.492 | 20,907,663,360 |
| [tranche-04](../data/research/qwen-cdf-collection-progress-20260913/collection/invocations/cddf9448-bb1f-4adf-a10b-baa113669e17/report.json) | 8 / 8 | 36 | 823.788 | 825.537 | 1405.158 | 20,979,437,568 |
| [tranche-05](../data/research/qwen-cdf-collection-progress-20260913/collection/invocations/b8ce2862-68c0-4e05-bab6-f9e9b0f86b17/report.json) | 8 / 8 | 35 | 882.770 | 884.471 | 1460.178 | 20,917,878,784 |

| Measurement | Continuation 02–05 only | Cumulative v3 01–05 |
| --- | ---: | ---: |
| Invocations | 4 | 5 |
| Readouts / parse-valid | 32 / 32 | 34 / 34 |
| Generated tokens, including EOS | 147 | 156 |
| Internal wall seconds | 3,658.692 | 3,990.628 |
| External wall seconds | 3,665.618 | 3,999.204 |
| Internal CPU seconds | 6,097.991 | 6,608.474 |
| External user CPU seconds | 5,757.100 | 6,233.320 |
| External system CPU seconds | 346.070 | 381.550 |
| Maximum internal RSS, bytes | 20,979,437,568 | 20,979,437,568 |
| Maximum external RSS, bytes | 20,979,437,568 | 20,979,437,568 |
| External swaps | 0 | 0 |

Wall/CPU values sum invocation measurements, excluding gaps between invocations. Peak RSS is the **maximum**, not a sum. GNU time includes shutdown overhead; each linked supervisor result pins its start, output, error and time files. First-v3 resources are counted once; v1/v2 runs and software validation are excluded.

## Every retained v3 readout

All 34 observations at this cutoff appear in catalog order below, including the two earlier observations by reference. Linked receipts preserve full proposition IDs, exact responses and prompts; token-event files retain token IDs, selected-token logprobs, RNG hashes and timing. No observation was filtered, repaired, replaced or retried.

| Catalog index / receipt | Seed | Status | Raw response | Prompt tokens | Completion tokens | Receipt seconds |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| [0](../data/research/qwen-cdf-collection-20260913/collection/attempts/00378ee8bd196abc8a83ad6fd27d65a27f76ae40af5b10b1b957128f5e51c292/receipt.json) | 6678723356247500948 | probability | `0.35` | 277 | 5 | 117.442 |
| [1](../data/research/qwen-cdf-collection-20260913/collection/attempts/0041c4392b30880f8e9b9804eab765b363d2d856d760a5b1f05e91c0ea805828/receipt.json) | 7057277058622107250 | probability | `0.0` | 261 | 4 | 95.977 |
| [2](../data/research/qwen-cdf-collection-progress-20260913/collection/attempts/004ad1f323c8a5f50d2faa8d5e6d645427163827116dd6268c7312707a7f19d7/receipt.json) | 16338427408747832856 | probability | `0.35` | 271 | 5 | 120.195 |
| [3](../data/research/qwen-cdf-collection-progress-20260913/collection/attempts/0073f111bbd0fdaa5aa6c73104e385f881528f51aae01df39336611e75bd9e8e/receipt.json) | 2554828484222619619 | probability | `0.02` | 190 | 5 | 88.791 |
| [4](../data/research/qwen-cdf-collection-progress-20260913/collection/attempts/0077d26746b6914f74339338b22a6a1d66b1aab967916b7dd696e44d60f97bd7/receipt.json) | 5760133503312900059 | probability | `0.55` | 265 | 5 | 117.304 |
| [5](../data/research/qwen-cdf-collection-progress-20260913/collection/attempts/00c2e5f8456393d266e1828ecc34a7cd079732711efb9376e6ede10e5ac0d3d4/receipt.json) | 14817629032893379078 | probability | `0.45` | 266 | 5 | 118.012 |
| [6](../data/research/qwen-cdf-collection-progress-20260913/collection/attempts/00e3d234fa10328adeac37084e7f2622ac187e8c13625f2e82bac960bd83c744/receipt.json) | 8621797841318016432 | probability | `0.15` | 268 | 5 | 118.542 |
| [7](../data/research/qwen-cdf-collection-progress-20260913/collection/attempts/010a08725b061bf503857e9890d6658e580fb6daf53f996e03cc0cd12563ef8c/receipt.json) | 9651072470218642482 | probability | `0.32` | 305 | 5 | 120.223 |
| [8](../data/research/qwen-cdf-collection-progress-20260913/collection/attempts/011faeb917dea27df3b315194ebe006d963178e58b5345273a4ed8413eb847d0/receipt.json) | 11560558147370843313 | probability | `0.2` | 279 | 4 | 97.323 |
| [9](../data/research/qwen-cdf-collection-progress-20260913/collection/attempts/012e0b198a99e660d4efb8ec2638e225f2b3a3ff3f2ac6ef84f3b50bd4073541/receipt.json) | 7244955977783435421 | probability | `0.5` | 279 | 4 | 97.199 |
| [10](../data/research/qwen-cdf-collection-progress-20260913/collection/attempts/0136e72569a48ccd5a6dd71faacc9d6d9e923ab239fb843f797e0600e82b7221/receipt.json) | 3443392906770101221 | probability | `0.1` | 278 | 4 | 96.810 |
| [11](../data/research/qwen-cdf-collection-progress-20260913/collection/attempts/01546cb04ee78862e8f362e8a522718912fb82f721be2674b81b0278c57ef057/receipt.json) | 10779716575264361361 | probability | `0.35` | 281 | 5 | 118.127 |
| [12](../data/research/qwen-cdf-collection-progress-20260913/collection/attempts/016e558e2e4dae2d3095adfd23ee4c89a4b107040eeec51e5be7470d27326724/receipt.json) | 9187623940906487763 | probability | `0.03` | 340 | 5 | 123.578 |
| [13](../data/research/qwen-cdf-collection-progress-20260913/collection/attempts/018f1fd678d2d882b5a70627a9450460592f2886eee78029808cc71e77d69f1d/receipt.json) | 1076517426924830555 | probability | `0.25` | 188 | 5 | 88.623 |
| [14](../data/research/qwen-cdf-collection-progress-20260913/collection/attempts/01a95b230a84eee2418761363996aef1d7908fef3acee5a5bd4f15c56bdf0f1a/receipt.json) | 14957413554031128405 | probability | `0.65` | 190 | 5 | 89.600 |
| [15](../data/research/qwen-cdf-collection-progress-20260913/collection/attempts/01b7a955c4840d5c4f30e6ced76d0535c5eb2a0ba8658cb748660b9a5a6dfcc0/receipt.json) | 2892416165927974017 | probability | `0.15` | 283 | 5 | 116.592 |
| [16](../data/research/qwen-cdf-collection-progress-20260913/collection/attempts/01c88abadef586e414abda9062907031ace70085628a84689507d04c772ad346/receipt.json) | 6979422077869170398 | probability | `0.12` | 193 | 5 | 89.848 |
| [17](../data/research/qwen-cdf-collection-progress-20260913/collection/attempts/01ccffe39d41183e710044e73f03911676be6e5d3f8bd63f4224af613d86af89/receipt.json) | 5983472441203402702 | probability | `0.3` | 300 | 4 | 98.862 |
| [18](../data/research/qwen-cdf-collection-progress-20260913/collection/attempts/01d7a30553d5db146476639f1330b00960a874a89167dc2276273aafecf16e2c/receipt.json) | 15234471068987244621 | probability | `0.35` | 251 | 5 | 97.729 |
| [19](../data/research/qwen-cdf-collection-progress-20260913/collection/attempts/01db5e2353af9204dd6be0f54ed97fb425fcfe8ee71ae157839bec5834747135/receipt.json) | 16271571962515798394 | probability | `0.45` | 225 | 5 | 92.881 |
| [20](../data/research/qwen-cdf-collection-progress-20260913/collection/attempts/01e7900e4ff2f5f8f9acd25ea816f192f56258497646500da6fe2cdca8e13b4c/receipt.json) | 4596610016806844955 | probability | `0.1` | 199 | 4 | 76.004 |
| [21](../data/research/qwen-cdf-collection-progress-20260913/collection/attempts/01ec9b8f4506e62f731a77ea40967c1ca5c2d467412d75bc72d8d016a70296c7/receipt.json) | 10875404535277115299 | probability | `0.15` | 207 | 5 | 90.459 |
| [22](../data/research/qwen-cdf-collection-progress-20260913/collection/attempts/0201bd62ab31f11f9759dea5c71362189d11f582e5eb0fa2d841cd07f7859811/receipt.json) | 5449647253394262183 | probability | `0.3` | 196 | 4 | 75.273 |
| [23](../data/research/qwen-cdf-collection-progress-20260913/collection/attempts/020a91fe59e323e7fed41175ba74dc3ea1dafa7c24867985e93113f37172f29c/receipt.json) | 1769995031194620548 | probability | `0.65` | 228 | 5 | 94.000 |
| [24](../data/research/qwen-cdf-collection-progress-20260913/collection/attempts/020abdaf9001e9adb1d9aab0ba00e9194dde6440622abef7eece2a05c9d24c86/receipt.json) | 9078733988399878730 | probability | `0.0` | 348 | 4 | 102.552 |
| [25](../data/research/qwen-cdf-collection-progress-20260913/collection/attempts/0231eabbb2022570ecfc15033c181f779c961a2ea838fb459d69488c347a73b9/receipt.json) | 3958335529198474027 | probability | `1.0` | 171 | 4 | 72.624 |
| [26](../data/research/qwen-cdf-collection-progress-20260913/collection/attempts/0235b3f9d779015c2bd7e1e473390f0df904e9dd4abe7bc7faaee45b46144a99/receipt.json) | 10730226071144396188 | probability | `0.01` | 275 | 5 | 114.822 |
| [27](../data/research/qwen-cdf-collection-progress-20260913/collection/attempts/024e1767e54eedb083dd2f31a8778a28debdaafe3c3a293fc46a1548af7c2433/receipt.json) | 97455037613691641 | probability | `0.0` | 276 | 4 | 95.673 |
| [28](../data/research/qwen-cdf-collection-progress-20260913/collection/attempts/02740d0979654681068fa21fb3bd6b5df1f056966b1f6867eef2ffafed2fa098/receipt.json) | 4286240988704269401 | probability | `0.0` | 310 | 4 | 97.704 |
| [29](../data/research/qwen-cdf-collection-progress-20260913/collection/attempts/0283b557d89c540c5da339f1d9fc9075f6162ce832f55c139bb5dfe0f118964c/receipt.json) | 17332090255745650985 | probability | `0.85` | 190 | 5 | 89.645 |
| [30](../data/research/qwen-cdf-collection-progress-20260913/collection/attempts/02a52693fd4c9f89dccfae4492e019c6a98c1dfe0c98e5a137dd71f180a40802/receipt.json) | 16201277964908248495 | probability | `0.0` | 176 | 4 | 72.649 |
| [31](../data/research/qwen-cdf-collection-progress-20260913/collection/attempts/02c0f49319dc400c4d27c2b6d0996191ddf0b943c591ad3a4e856947c24aecb9/receipt.json) | 8861086027917325362 | probability | `0.15` | 283 | 5 | 117.323 |
| [32](../data/research/qwen-cdf-collection-progress-20260913/collection/attempts/02c5793f879c4bb9c16068f9ae840098aeab94fa43a56b076c65727c0151b557/receipt.json) | 14143884591434786541 | probability | `0.1` | 257 | 4 | 95.752 |
| [33](../data/research/qwen-cdf-collection-progress-20260913/collection/attempts/02d1dda40919af3e4f418555ef1caee25f2e45d9df8026c4ca5327865635dc2e/receipt.json) | 3743606518108260748 | probability | `0.0` | 220 | 4 | 76.590 |

The `probability` status is the retained parser/receipt category. Every readout here reached EOS **248046** and has recorded runtime-integrity verification. Parser acceptance does not establish calibration, population sampling behavior or complete episode coverage.

## Authorized envelope and unchanged identity

The [continuation envelope](../data/research/qwen-cdf-collection-progress-20260913/tranches-02-05-plan.json), SHA-256 `ab4fa8611d3b9200ea9125fc65d2e1c3ff3b583e241755f5325b85278674bf59`, bounded four sequential invocations at **eight requests / 256 tokens each**, **32 requests / 1,024 tokens total**, with a fixed **32-token cap per proposition**. Each invocation retained internal/external wall limits **1,500/1,800 seconds**, a **10-second kill grace**, CPU soft/hard limits **3,400/3,410 seconds**, and a **36 GiB** address-space limit. A new attempt required the entire fixed cap to fit the remaining token budget. Only one candidate process ran at a time.

The continuation generated **147 / 1,024 tokens**, leaving **877** unused. Every prelaunch gate bound prior terminal successes and cumulative coverage, predecessor report/result hashes, source/config/envelope pins, catalog indices/seeds and an empty candidate-process list. Terminal audits bound the supervisor result, immutable files, exact readout summaries and updated coverage. No stop condition occurred in 02–05; each completed eight requests. Tranche 05 is the exact publication cutoff; 06–13 configurations (unactivated at the stated cutoff), their preparation envelope and future results are excluded.

The sealed source remains `0af9ec3b6fe41b9b76996d318087a52e172f8035`; all **14 source hashes** match. The [first-v3 plan](../data/research/qwen-cdf-collection-20260913/collection/plan.json) SHA-256 is `a0d07bda3b81d880199e62b310caa43e8051f50b5e5da30f912d457f3439a5ce`, and [spec](../data/research/qwen-cdf-collection-20260913/collection/spec.json) is `5afa578df2f430d5add27f7e00c823b1bab722cdb1ef12086b7cabc5bf16491c`. Execution SHA-256 remains `0d2a9d55452046a8d0f71ae3495cc4ac2845c353f200781ed36b6bad71244d71` and initial-policy SHA-256 remains `ffd73988534e4871d94a92ae6f170d0f80d50cb12d5a4d6dadfecc8051e0fe10`.

The sampler remains `softmax-at-least-fp32-fp64-cdf-uniform-right-v1`, helper SHA-256 `24d0e0132a5aa219f74c3146ad5516bbad6d81e0c6ecfe7bf37d7424c68475ba`: FP32 softmax, FP64 normalized CDF, one FP64 uniform draw per token and right-sided search, with the original full vocabulary and no filtering. Finite-precision CDF/uniform limits still apply.

The model is `Qwen/Qwen3.5-9B` at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Training identity, initial base tensors and adapter manifests match the first-v3 invocation byte-for-byte. The run retains **248 rank-16/alpha-32 targets / 43,278,336 FP32 adapter parameters**, LoRA seed **2026091304**, master seed **2026091310**, nonthinking `numeric-v1`, eval/eager attention, no cache, two CPU threads and one interop thread. Each invocation independently verified its initial runtime identity and reported unchanged base/adapters at completion. The original proposition texts, prompts, order and train/valid/heldout splits remain unchanged.

## Archive checks, validation and subscription usage

The new raw archive contains **155 files / 3,454,169 bytes**: the 02–05 envelope, four configurations, prelaunch gates and terminal audits, four invocation identity/report sets, supervisor receipts/logs/resource records, and 32 attempt/receipt/token sets, the final read-only status receipt, and the runtime owner’s aggregate terminal audit. **39 hash-checked referenced files** retain the first-v3 manifest, its 30 raw files, seven byte-identical historical bound inputs and the published CDF-validation manifest. No duplicate catalog or input copy was added, and all earlier archives remain byte-for-byte unchanged.

Publication checks verified every new/reference byte count and SHA-256, all **2,926 catalog rows and seed derivations**, **34 attempt/receipt/token/catalog bindings**, sequential gate and supervisor pins, **408 terminal-audit archived/reference file pins**, token/RNG/timestamp joins, EOS/status categories, runtime identities and final coverage. `AttemptLedger.inspect()` reconstructed the complete state map from a temporary archive copy using the pinned plan/spec and inputs. The owner’s aggregate terminal audit adds **181 archived/reference file-pin checks** and independently matches the counts and resource totals. The four tranche audits and aggregate audit also pin an empty runtime owner-lock file; each matches the SHA-256 of empty bytes, and this synchronization file is not archived. No Torch import, candidate execution, source change or test rerun occurred during publication.

The separately published [CDF software validation](../data/research/qwen-cdf-validation-20260913.json) records **274 unique tests passed / zero unresolved skips**. The original full suite passed 268 and initially skipped six tokenizer tests; those six passed using the pinned artifacts. Software checks and actual candidate receipts are separate evidence; neither establishes a complete own-base anchor.

Preparation no-inference flags describe creation before candidate execution. Read-only status commands retain the label preflight-verified-no-inference because those commands generate no new observations: the first-v3 status recorded two completed outputs, while status-after-tranche-05.json records 34 attempted and 34 valid. The continuation envelope prepared-pending-first-tranche-and-root-review status is its immutable preparation-time state; later prelaunch gates and terminal audits establish execution.

These local CPU v3 invocations made **zero provider inference requests** and consumed **zero provider inference subscription quota**, with no new paid resources allocated. **Agent development/review subscription usage remains unmeasured**; it is separate from provider inference quota.
