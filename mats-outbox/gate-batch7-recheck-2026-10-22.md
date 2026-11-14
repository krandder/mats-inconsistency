# Gate re-check: parity_learn batch 7 (narrow), 2026-10-22

Opus 5.5 gate subagent for opus-mats. Read-only detached worktree /tmp/gateb7 at origin/main 7b33d6e; harness 17654b4; CPU only, nothing retrained. Full gate: gate-batch6-2026-10-22.md.
Checked at 2026-10-22: recomputed all 69 runs of parity_learn_batch7.json per seed; all 6 VM runs report sha 137a1d90..., matching the commit; `render_parity_learn.py --write` reproduces the README byte for byte.

## Must-fixes: 5 of 5 FIXED
1. Relabels: FIXED in the prose and in the hand-written tables ("STRUCTURE-KNOWN"; "blind at inference, key as pretraining query"). Minor: the generated table still says "keys yes (1000) / Y labels only" and has no pretraining column.
2. Plain controls: FIXED. b7_plain_k{0,1} at 8000 and 10000 steps, b=2/4/6: 36 of 36 runs score 0.267-0.270, against a prior of 0.270.
3. Anchors: FIXED. Transformer: 0.0050 at 10k, 0.0048 at 11k, 0.0059 with masked pretraining (2k+8k). ISAB: 0.0017 at 10k, 0.0014 at 11k. Every anchor's best step is its last step, so the anchors are still improving and the gaps are lower bounds.
4. Renderer: FIXED. key() and stages_key() now include pretrain, pretrain_mode and power_norm, so pretrained arms are compared "best on spec" instead of "same stages".
5. Bugs: FIXED. The warm-up now applies through _embed (use_keys) in decode_masked/decode_recon, and symbols() runs under net.eval(). The batch-6 k1 arms were not rerun (they ran with keys on during pretraining). That is harmless because they are structure-known anyway, and the README records it.

## Fully blind arm (recon): no key in the decoder's input, but keys are in its training target
- Decoder input: for k0, _embed returns inp(m) (pl:244-246), and decode_recon is called with messages only. The Y phase has no key path. TRUE.
- Target: each agent's own readings plus its own var ids, taken from the key (`k.reshape(...)[..., :NV].argmax`, pl:389). The encoder already sees that key, so this autoencodes the agent's own input and leaks nothing external. Still, README "no key reaches the decoder at any point" should read "no key as decoder input; each agent's own var ids are a reconstruction target".

## Headline spot-check: REPRODUCED
- Structure-known b=4: b6_pretrain_k1 (0.0073/0.0111/0.0084, mean 0.0089) minus the transformer at 10k (0.0050) = +0.004. Against ISAB it is +0.007; against the pretrained anchor, +0.003. b=4 is the best of the three budgets (b=2: +0.053).
- The +0.004 compares harness a40442a with 17654b4. The anchor code paths did not change in that diff, so the comparison is valid.
- Fully blind + power + fine-tune, b=6: 0.120/0.071/0.053, mean 0.081, minus the transformer at 11k (0.0048) = +0.076. Spread across seeds is large; 3 seeds.

## Fine-tune steps: NO extra steps
- Arms: 2000 pretraining (no Y) + 8000 Y + 1000 decoder fine-tune = 11000 steps of 512 episodes. Anchor: 11000 Y steps. Arms get the same number of episodes and fewer Y-labelled steps.
- The only asymmetry is three OneCycle restarts against one, a schedule difference rather than a budget difference.

## Verdict
Usable as exploratory: the 5 must-fixes are fixed and the numbers reproduce. Before reporting: soften the "fully blind" wording (var-id target) and state that the anchors are unconverged.
