Model: GPT-6 (Codex)

Checked 2026-10-31 20:03 UTC: read-only source, hash, jq/awk checks; no Python or model execution.

1. **High: causal interpretation exceeds the decomposition.** Draft `update84_draft.md:3,5,23,80,82` says B “reads around” an unimproved summary and that pool/reading dynamics set the final departure. `e72_detq.py:38-43` detaches queries in every summary-row head; `e72_run.py:82,104-111` uses selected, renormalised own pools and a factorised classifier benchmark. Their difference includes unmatched routes and estimator error, not an isolated reader mechanism or information-loss measure. B's query term also changes, contradicting “only because.” E35 concerns attention in two different replays, not demonstrated reliance here (Update 44). **Fix:** describe operational trajectories; retain bypassing as a hypothesis, remove causal “price,” “sets,” and “never improves” claims from both versions.

2. **High: pair-timing evidence changes the measuring instrument.** `e72_run.py:66` selects heads at P_h ≥ 0.5; `e73_traj.py:33-47` then adds their vectors to M3's classifier inputs. Thus the first two-head checkpoint also expands the benchmark statistic. Example: JSON `e37_ctrl_s0@5000/@10000`, pool [1]→[0,1], M3 .428→.276. **Fix:** report the coincidence with this explicit selection caveat; require fixed-head-set trajectories before attributing the abrupt drop to pair formation.

3. **Medium: “develops late” overgeneralises.** Draft `:7-8,23,81`; JSON `e37_ctrl_s1@5000/@10000` has detq−M3 +.033→−.038, while `s0@5000` already has rom−detq −.073. P296 failure establishes failure of sustained positivity, not late onset; P295 compares endpoints. **Fix:** report median trends and exceptions, and the actual ≥3/4 criterion.

4. **Low: mixed B denominators.** Draft `:17` quotes six eligible B nets but gives the seven-net final M3 median .437; six-net median is .439. Plain words `:5` similarly uses −.07 instead of −.062 at 5k. **Fix:** use matched subsets.

Survive: checkpoint loading, forced-head rule, configured 100,000 episodes, detq, corrected b1h4 block 2, all 256 table rows, and registered scoring/exclusions. P292–P295 pass; P296 and both secondary conjunctions fail. Reword mechanism/timing claims; causal identification fails.
