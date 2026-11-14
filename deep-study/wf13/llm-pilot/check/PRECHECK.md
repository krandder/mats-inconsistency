# Checker's registration (wf13/llm-pilot/check), written before the fresh run is submitted

Model: Claude Fable 5.2. Fresh data: 60 new problems per family (seed 4242, disjoint by text from tuning, shots, pilot,
test and calibration sets), my own sampler and scorer (`code/fresh.py`: float32, a clean copy of M never wrapped by
peft, plain cache sampling, full-forward scoring), my own aggregation code (`code/recheck.py`).
Writers: W1, W3, W3s, W3p, W4 (the work's adapters), plus two new ones:
- W3r: the work's W3 recipe and data, tuning seed 11 instead of 1 (tests "one tuning run").
- W5 "pure bias": LoRA tuned ONLY on the erroneous line for the 1,500 tuning problems per family (the work's W3 was
  tuned on the erroneous line AND one of M's own rollouts per problem, so it writes M's lines about half the time).

Thresholds re-applied as registered in the work's PREREG.md (K = 64, each family): H1, H2, bites, H3, H4, H5.
I count a claim CONFIRMED on fresh data if the registered threshold holds again, with the same sign and within the
bootstrap interval of the work's number.

My expectations (registered now):
- C1. H1, H2, H3, H5 hold again for W3, W3s, W3p in both families. H4 for W3 on variables: undecidable at n = 60
  (the miss was 0.008); I report the number only.
- C2. W3r behaves like W3 (vote accuracy <= W1's - 0.15; dedup and snis accuracy within 0.05 of W1's vote).
- C3. W5 moves BOTH score-based aggregations: acc(W5,dedup,64) and acc(W5,snis,64) <= acc(W1,vote,64) - 0.15 in at
  least one family, with found score < 0.3 on most problems. If this holds, "three biased writers did not move the
  answer" is a property of how those three were built (they keep writing M's lines), not of the aggregation.
- C4. The interval R_F(a) <= T(a) <= 1 - (m_F - R_F(a)) is never violated by more than float rounding against M's
  score in float32, for any writer including W5, and no answer is certified (R_F > 1/2) that is not the reference's
  top answer. Under W5 the certificate fires on < 20% of problems.
- C5. For W1 the sampler's log-probability equals my full-forward score to < 1e-3 nats (float32), and the writers'
  own probabilities agree across my two paths to < 1e-3 on the first 12 problems.
On the stored rollouts (CPU, `code/recheck.py` on the work's `out/main/test.jsonl`): every number in the note's
sections 2-6 that I quote is recomputed with my code; a mismatch above 0.002 is reported.
Simulated hider (CPU): M's rollouts with M's top-scoring found line removed (q renormalised exactly). Expectation:
dedup and snis return the second-best line's answer; top-answer agreement with the reference falls below 0.6 on the
problems where M's top line carries < 0.8 of the found score... reported as measured, no threshold.
