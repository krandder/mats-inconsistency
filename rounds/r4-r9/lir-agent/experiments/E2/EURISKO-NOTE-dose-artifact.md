# eurisko note (2026-10-03 16:20Z): the dose knob (alpha) is ineffective by construction for alpha > 0

harness/protocol/train.py: the reward phase backpropagates `alpha * lbf` ALONE into the belief/forecast
adapters (lines ~262-266), then `clip_grad_norm_` and a per-group `torch.optim.AdamW` step (`_phase`,
lines ~139-155; optimizers created per adapter group at line 120). Scaling a gradient by a constant
alpha leaves an Adam(W) update almost unchanged (Adam normalises by the running RMS of that gradient;
weight_decay=0), and gradient clipping also removes magnitude when the norm exceeds `clip`. So for any
alpha > 0 the belief and forecast adapters receive a full-size step in the reward direction at every
reward phase; only alpha = 0 (no step) differs.

Consequences for the claims: (1) E2's "any alpha > 0 brings sycophancy back" and "gradient/update share
is not a graded predictor" (Spearman -0.90 within alpha > 0) are expected from this, not evidence about
localisation; (2) E1's and E4's protected-vs-fused contrasts are unaffected (alpha = 0 vs fused);
(3) a real dose needs the reward loss and the truth loss summed into ONE backward before ONE Adam step
of the belief adapter (alpha then sets the relative weight), or plain SGD. Skeptic: please mark the
dose claims accordingly. Not verified by rerun; verified by reading the code paths named above.
