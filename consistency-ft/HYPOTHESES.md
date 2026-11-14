# Speculative generalization hypotheses (>5% each, not confident claims)

If consistency/calibration finetuning works, capabilities it might transfer to.
Ranked roughly by (probability × surprise); 3 and 6 are closest to the
training signal's mechanism and cheapest to measure first.

1. **"Are you sure?" robustness / anti-sycophancy** — consistent beliefs leave
   less freedom to flip under pushback. Test: answer-flip rate under challenge.
2. **Long agentic tasks** — calibrated per-step confidence = a working signal
   for backtrack/double-check/ask; could move SWE-bench-style evals without
   code training.
3. **Knowledge editing that propagates** — today edits don't update their
   implications; a coherence-trained model might propagate through the
   implication graph for free. Underrated.
4. **Better search / best-of-n / self-rewarding RL** — model as its own value
   function needs calibrated self-evaluation; miscalibration is what makes
   self-reward hack itself.
5. **Resistance to manipulation and stepwise jailbreaks** — many persuasion
   paths money-pump inconsistency (accept each step, reject the conclusion).
6. **Latent multi-hop reasoning** — "spouse of the director of X" failures look
   like fragmentation; conjunction/implication training may directly train
   composition.
7. **Choosing experiments / active learning** — expected-information-gain needs
   calibrated priors: what to look up, which test to run, what to ask.
8. **Long-document and character coherence** — contradicting page 3 on page 40
   is plausibly the same failure; cheap, surprising benchmark.

Business note: 1, 2, 4, 6 have standard benchmarks labs already pay for — one
hit turns this from nice-to-have into a capabilities story.

Known caveat: consistency and calibration are empirically near-orthogonal
(CoherenceBench), and outcome-RL improved conjunction checks while *regressing*
negation (2512.25070) — measure both, per check type.
