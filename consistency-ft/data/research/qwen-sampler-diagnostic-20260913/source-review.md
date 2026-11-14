# Pinned CPU sampler source review

This is a software-source and retained-evidence review, not a new model run or a certification of categorical sampling. It does not change the frozen collection, the original denominator, or any response.

## Independently verified source evidence

`source-manifest.json` records downloaded source URLs, SHA-256 hashes, byte counts, and copies of installed wheel metadata. The installed metadata identifies PyTorch `2.6.0+cpu`, commit `2236df1770800ffea5697b11b0bb0d910b2e59e1`, and `AT_MKL_ENABLED() 1`.

At that exact commit:

- `Distributions.cpp:584-614` implements single-sample multinomial as an exponential race: generate `q = empty_like(probabilities).exponential_(1, generator)`, divide the input probabilities by `q`, and select the argmax.
- `DistributionKernels.cpp:107-112` selects the MKL implementation when MKL is enabled and the build is not FBCODE_CAFFE2.
- `DistributionKernels.cpp:118-125` draws one generator seed with `random()` for FP32, and `random64()` for FP64.
- `DistributionKernels.cpp:145-172` explicitly accounts for MKL exponential variates returning zero by shifting them by `std::numeric_limits<tmp_scalar_t>::min()`. For FP32 this is `2^-126 = 1.1754943508222875e-38`, the smallest positive normal float, not machine epsilon. The FP32 call is `vsRngExponential(VSL_RNG_METHOD_EXPONENTIAL_ICDF, ...)` with `VSL_BRNG_MCG31`.
- `DistributionKernels.cpp:184` partitions the variates with `parallel_for(..., grain_size=800, ...)` and the per-segment implementation uses deterministic skip-ahead.

The frozen runtime, `training/qwen_lora_runtime.py:255-257`, passes FP32 logits through `softmax`, samples with `torch.multinomial(..., 1, generator=generator)`, and reads the selected token's `log_softmax` from the same logits vector. Its receipt/token mapping is faithful. That does not prove the library sampler implements the intended categorical distribution accurately.

## Relation to the suspicious retained token

The retained response is exactly `"0 Regiment"`, IDs `[15, 70215, 248046]`. The second token's nominal model log probability is `-27.26654815673828`, corresponding to approximately `1.4397549985120265e-12`. This response remains `invalid_format`, with no numerical probability assigned.

The execution owner reports a software-only reconstruction using the installed wheel, recorded seed `16432092232865781795`, FP32 vector length `248320`, and the pinned two-thread/one-interop configuration. The reported initial and post-step generator-state hashes all match the retained attempt and token events. At the second token step, the reconstructed exponential noise at index `70215` equals FP32 minimum normal, and uniform multinomial chooses that index with the same generator transition. Those reported diagnostic results have not been independently inspected in this source-only note.

This is a specific, source-supported numerical mechanism, not evidence of a collector indexing error. It does not require loading the model. Its relevance to the recorded draw comes from the matching generator trajectory and vector configuration, not from the synthetic input probabilities being equivalent to the absent original logits.

The selected token's approximate exponential-race score at that noise floor is:

`exp(-27.26654815673828) / 2^-126 = 1.2248080966998116e26`.

For any competing softmax probability `p_j <= 1`, its score is bounded above by `1 / q_j`. Thus, if the smallest *other* reconstructed noise exceeds `8.164544328980626e-27`, no competitor can beat the selected token's race score. That extra noise bound would establish floor domination without recovering the full original logits. Softmax/log-softmax floating-point differences do not plausibly approach the many-order margin expected here, but the original softmax vector was not retained.

## Limits and implications

- A matching RNG hash confirms a generator transition, not categorical-distribution correctness on its own.
- The direct floor reconstruction is stronger evidence than observing an improbable token. The other-noise bound should be retained if a floor-dominated argmax is claimed.
- Floor exposures, parse-invalid responses, and rows passing the parser are distinct quantities. A floor on a parse-valid row still raises a sampling-treatment concern; absence of a floor in a finite retained scan is not broad certification of the implementation.
- Neither these source files nor one retained trajectory estimate the frequency of the defect over arbitrary seeds or its effect on aggregate scores.
- The full original logits and original noise were not recorded. A software-only reconstruction should remain labeled as a reconstruction.
- No new inference, retries, reseeding, replacement outputs, treatment changes, denominator changes, or runtime edits were performed for this review.

## Superseding independent diagnostic verification

The reviewer subsequently inspected and independently reexecuted `scripts/diagnose_qwen_cpu_sampler.py`, SHA-256 `2dffdb363c2c6563cb77bc1f0f7c20c4a3bf97eed09cf53ca846e011d5f8a2fa`, using the already-installed CPU interpreter. This supersedes the earlier owner-reported status of the dynamic results above. The diagnostic only generates exponential-noise vectors; it does not load a model, perform a forward pass, generate a benchmark answer, or edit any collection evidence.

The independent output is `/tmp/consistency-sampler-independent-review-20260913-0wnjzr6m/noise-replay.json` (SHA-256 `c92271e3dacfcf121d5b33609b9b11a10954e3356f100d16e8d5ae9beaa8cca2`). It matches every field of the owner's output except `recorded_at`; no path normalization was necessary. All 66 retained input files were additionally rehashed and remained unchanged.

All 22 attempts and 98 token RNG transitions match. Exactly one reconstructed noise vector has a float32-minimum-normal value, at catalog index 20, token step 1, token ID 70215; this is also the retained selected token. The second-smallest noise is `4.1723342292243615e-06` at token ID 27460. Thus every competing token's race score is at most `239673.99183787353`, whereas the selected token's approximate nominal score is `1.2248080966998116e26`. The dominance margin is approximately `5.1103087460917654e20`.

This independently establishes the reproduced floor-dominated selection mechanism using the retained log probability and a bound valid for every competing softmax probability. It does not establish the absent original full-logit vectors, validate a complete anchor, or measure the defect's population frequency. The machine-readable comparison and bound are retained in the same independent-output directory as `independent-review.json` (SHA-256 `1d483d507c3155a94128a04f689b9628062bcde12b270389393fa936fb1ed04a`).
