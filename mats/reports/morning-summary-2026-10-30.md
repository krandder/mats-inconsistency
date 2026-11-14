# MATS overnight summary, 29-30 October (written 09:14Z by opus-mats)

Full dated log: `outbox/overnight-report-2026-10-29.md`. Wiki: `wiki/mats-noisy-channel-interp.md` (hub PR #5678).
Repo: futarchy-fi/inconsistency-search PRs #19 and #20 (merged); phase B Updates 47+ and the narrow-deep
follow-up go in the next sync.

## Decisions for Kelvin
1. Proposed next lane, not started: M3 on trained nets. Estimate rare-event probabilities (a sorter's large error,
   an RNN's confident wrong answer) with sampling guided by the net's own algorithm variables, against plain and
   quasi-random sampling at equal compute; CPU first on the narrow trained nets, then AlgZoo.
2. Budget: MATS GPU is roughly $150-165 of the $200 authorisation (list-price ceilings from each lane's GPU-LOG;
   no single ledger). Phase B is at $25.34 of its $40.

## Finished overnight
- Scale-free theory, tests on trained nets (M3-M5): 5 of 7 predictions held. "Layer 1 routes, layer 2 weights" failed
  the causal test (corrected to Kelvin at the time).
- Scale-free experiments: given the routing, a two-round sum tree matches or beats a dense transformer, and the gain
  survives capacity and loss controls. Learning the routing is what dense nets do slowly. Discovering hidden shared
  errors is where dense attention earns its cost (69-77% vs at most 29%).
- Narrow-deep: random plain nets synchronise at the predicted Lyapunov rate. The gate-free bound fails. A Gaussian
  closure wins on random residual nets (median 43x) only because they are nearly linear; on trained nets it loses
  110-1,800x, and splitting a few gates does not fix it (pre-registered refutation held).
- Ngo comparison: an LMSR market over the open latents is exactly our merge rule; hub wiki page updated.
- Explainer page: opened on Kelvin's Mac at 23:52Z (`~/Downloads/mats/scale-free-explainer-v1.html`).

## Phase B (M2), still running (fourth agent)
- E26-E29: about 29% of the nets' distance from Bayes is dependence on how a source's past errors are arranged, mostly
  their signs, which Bayes ignores. A further fifth is dependence on arbitrary source ID codes.
- E30: removing the signs from the input cuts the distance from Bayes to 0.36x (paired seeds), three times what the
  measured sign cost predicted. Fixed ID codes give about 0.5x.
- E37-E47: visible signs keep nets in the worse solution type throughout training (switching late still rescues
  6 of 7). The size test shows no closing of the gap up to width 128. A second task shows the same sign
  dependence (34% of KL) but a smaller retraining gain. The type is set by the initial weights: stable to 0.3%
  perturbations, carried mostly by block-1 attention and the head, invisible at step 0 and readable by 10k steps
  (AUC 0.97). Pretrained LLMs: only Opus clearly moves with the signs when told the errors are unbiased.
- E48-E51 (fifth agent, morning): giving the net the squared or multiplied quantities Bayes needs helps as much
  as removing the signs, and no more. A per-token MLP before attention helps in one task only. The big-miss bias
  survives every format and that architecture change. Revision of E26-E29's reading: the sign dependence is mostly
  specific to each net (correlation between nets 0.14-0.28), and averaging 9 nets trained the same way removes
  36-55% of the KL. So the measured "sign cost" largely reflects run-to-run variation, not a shared bias.

## Corrections made overnight (all told to Kelvin)
- "Layer 1 routes by labels, layer 2 weights" was a correlational reading; the causal test rejected it.
- Plain narrow nets do not forget their input; the deep map is rank one (direction explains 80-92% of the output).
- The better phase B net's late cleanup at 15-20k steps came from one run and does not generalise (most drop the
  marker by 10k).
