Model: GPT-6 (Codex)

Read-only review, checked 2026-10-30; no Python or training/evaluation reruns.

1. **High: prediction misses become excessive mechanistic certainty.** Draft lines 86–89 and item 30 assert ongoing sign costs, release from B, and rejection of selection. `results/e37.json:predictions` correctly records point-estimate failures, but P166–P168 intervals cross their thresholds; Amendment 45’s “no lock-in” interpretation explicitly required P166 to hold. It fails. **Fix:** say the registered prediction bundle misses its bars and most acquired B seeds recover after a late switch under this recipe. Treat the proposed mechanism as a hypothesis; distinguish threshold scoring from statistical rejection.

2. **High: “permanent” stall is contradicted.** Draft:88 places seed 3’s permanent stall between 6k and 10k. `results/e37.json:per_net.c2m10k_s3.gap` is 0.029225, below the registered acquisition threshold 0.10; c2m30k is 0.141216. Finite runs cannot establish permanent commitment, including seed 8’s. **Fix:** report acquired-but-high-error at 10k, unacquired at 30k, and persistence through the tested budget.

3. **High: routing/acquisition claims exceed the diagnostics.** Draft:35,89 infers that routing does not carry the cost and every net enters B “at acquisition.” `e37_traj.py:23,54` first measures rom at 3k; `results/e37.json:trajectories.ctrl.3000.kl` is 0.133. The measurement does not locate each seed’s acquisition. Attention mass is not a causal intervention, and the diagnostic changes meaning across formats (Update 46). **Fix:** report positive rom at the sampled 3k checkpoint and partial retention of attention mass despite improved KL; avoid causal exclusion.

4. **Medium: mag60 does not settle learning speed.** `results/e37.json:R.mag60` is 0.86 [0.74,0.94], versus late-switch 0.70 [0.46,0.97]. `train_sw.py:97,101,119` gives mag60 fresh optimizer state, warmup, and earlier episodes. **Fix:** describe a remaining-step-budget control, not proof against slower adaptation or eventual recovery; retain the paired wins and pooled deficit together.

5. **Medium: summaries overgeneralize.** Draft:32’s “any time” includes 2k, where only 5/7 B seeds become A (`predictions.P169`). Item 30’s reverse-switch claim includes m2c3k, whose R is +0.34. **Fix:** restrict six-of-seven rescue to 3k–30k and worse-than-ctrl reverse-switch point estimates to 10k/30k.

Survive: registered switch/schedule, paired initialization, final-format evaluation, selection, filtering, R/Rlab testability, bootstrap code, tables and P166–P170/P173 scoring. Metadata confirms selection; 18 E30 curves match; saved checks report 96 identical branch points. Binary results were not independently recomputed. Reword selection, acquisition, routing and budget interpretations. Fail: permanent stall by 10k and universal rescue/reverse-harm wording.
