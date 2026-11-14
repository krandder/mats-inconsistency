Model: GPT-6 (Codex)

Checked 2026-10-30 06:43 UTC using static source/metadata reads, SHA-256 checks and unzip/od scalar reads; no Python or training/evaluation reruns.

1. **High: interpretation contradicts registration.** Draft:18,45 prefers reading (a). `PREREG-B.md:2321-2323` explicitly assigns KL failure with B-type sgp nets to reading (b); `results/e39.json:predictions.P173.holds=false`. Fix: report that registered branch, retain magnitude-helping as descriptive evidence, and label any preference for (a) post hoc. The interval includes 1.1, so failure of the prediction is not decisive evidence of excess KL.

2. **High: stream/initialisation attribution is unsupported.** Draft:17,46 turns overlap into causation. `results/e39.json:per_net.sgnmag_e30_s5.total_steps=50000`; E39 used 90000. Seed 7 acquired in E30 but stalled here. Fix: say stalls overlap at seeds 1/5 and acquired B identities change; distinguish shared stream prefixes from equal exposure. B-changing seeds 0/2/3/4/6 did all receive 90000 steps, which strengthens that descriptive comparison, but E43's ctrl evidence does not establish separate causes of sgnmag stalls/types.

3. **Medium: B count is wrong.** Draft:10,15 says one; JSON `arms.sgp.n_B=2`, seeds 2/3. Seed 3 has KL 0.0205 against mag's 0.0221. Fix: “two acquired B nets, only seed 2 much worse than its mag twin.” The per-seed values and stall counts otherwise match.

4. **Medium: plain words overgeneralise.** Draft:3-5: “makes training less reliable,” “never learned,” and “almost ignore” exceed endpoint evidence. Fix: “more stalls in these 12 runs,” “unacquired at the selected checkpoint,” and “among acquired nets, pooled nine-flip averaging removes 0.8% of family KL.” Also replace “one extra input weight” with “one extra input feature” (`train_sw.py:67-70`).

**Survive:** paired-init implementation, generator 31337+seed, schedules/job flags, acquired-seed ratio of means, paired 2,000-resample percentile bootstrap and E30-style pooled sign share (`e39_summ.py:29-53`); JSON reports 1.35591 [0.97887,2.52711], 0.0078225 versus 0.199886. Relevant code hashes match. Bootstrap endpoints and checkpoint equality were not independently rerun.
**Reword:** mechanistic/reliability claims and E30 comparison.
**Fail:** one-B count and claimed identification of separate stream/initialisation causes; P173 fails as reported.
