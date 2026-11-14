# Training costs: verified sources, September 13, 2026

**The old estimates are superseded. No paid training or GPU rental has been
started or approved by this update.** The subscription benchmark uses separate
billing and is accounted for in [native-evaluation.md](docs/native-evaluation.md).

Three material corrections to the August notes:

- Fireworks' current RFT guide explicitly says **“do not assume a free tier.”**
  The earlier “free for models under 16B” claim was incorrect.
- Fireworks now documents a Training API with `forward_backward_custom()` for
  custom Python loss functions. The old blanket “no custom-loss hook” statement
  is no longer true. Its exposed per-token log-probabilities still do not imply
  arbitrary access to full-vocabulary logits or model internals.
- The current catalog lists **Qwen 3.5 9B on dedicated Training API**, with no
  serverless-training entry and no managed-RFT entry. Its small parameter count
  does not establish a cheap managed-RFT path. In today's estimator, the old
  0.66 / 0.132 / 1.995 / 1.463 token rates appear under **Tinker**, not Fireworks.

## Current published rates and availability

| Option | Evidence | What the number covers |
| --- | --- | --- |
| Fireworks managed SFT, up to 16B | $0.50 per million training tokens | SFT only; not RL rollouts, evaluation, or serving |
| Fireworks managed DPO, up to 16B | $1.00 per million training tokens | DPO only; not the A2 coherence reward experiment |
| Fireworks dedicated B200 | $13 per allocated GPU-hour | Current estimator rate, effective September 1 |
| Qwen 3.5 9B dedicated RL catalog shape | 3 × B200 | Nominal $39/hour if all three listed GPUs use that rate; obtain a current quote and measure runtime |
| Runpod H100 PCIe 80GB | Advertised $2.89/hour | GPU listing; region, capacity, storage, and taxes require a launch-time quote |
| Runpod H100 SXM 80GB | Advertised $3.49/hour | Same limitations |
| Fireworks managed RFT | No defensible fixed total yet | Official estimator sends RL estimates to the training team |

At the advertised SXM rate, a **two-hour engineering smoke test** would have a
$6.98 GPU-only allowance. That is not an estimate for a full pilot, a reservation,
or evidence that GRPO plus rollout sampling fits in one GPU. OpenAI's official
[gpt-oss LoRA tutorial](https://cookbook.openai.com/articles/gpt-oss/fine-tune-transfomers)
uses one H100 80GB for **SFT**; its memory/runtime claims cannot be transferred
to this RL workload without measurement.

The approved research proposal describes an approximately $100 pilot ceiling.
That design number is not an approved invoice, GPU reservation, or account-cap
change. A concrete launch needs the selected model/adapter, finite reward
convention, episode and rollout counts, sequence caps, provider quote, and a
stop mechanism. Include initialization, frozen-base collection, rollout sampling,
training, checkpoints, evaluation, and allocated idle time. Unknown costs are
not zero. Do not convert the native subscriptions' token counters to GPU cost.

## Sources and reproduction evidence

Retrieved public source bytes, exact catalog entries, rates, and SHA-256 hashes
are recorded in [platform evidence](data/research/platform-evidence-20260913.json).
The current [platform and model assessment](docs/training-platform-decision.md)
separates a recommended engineering path from the unresolved scientific model
choice and launch requirements.

- [Fireworks training catalog](https://docs.fireworks.ai/fine-tuning/models)
- [Fireworks RFT guide](https://docs.fireworks.ai/fine-tuning/reinforcement-fine-tuning-models)
- [Fireworks Training API](https://docs.fireworks.ai/fine-tuning/training-api/introduction)
- [Fireworks cost estimator](https://docs.fireworks.ai/fine-tuning/cost-estimator)
- [Fireworks pricing](https://fireworks.ai/pricing)
- [Runpod pricing](https://www.runpod.io/pricing)
