## Skeptic notes (X1 worker1: S0, L1, T1)

Every number below comes from a file in `X1/w1/results/`; the committed values come from the files named in the table rows above.

1. **The wealth-net pi stream is learnable on most seeds (S0 finding).** `make_questions` draws pi digit offsets per question. If two pi questions have offsets within W=32 of each other, one sees the other's shifted digits inside its window.
   - Seeds 1-3 had no such pair (min gaps 33, 34, 109), so the committed pi BCE ≈ ln 2 (0.6933 / 0.6957).
   - Fresh seed 4 draws offsets 987, 995, 996, 1000: 6 pairs across 3 questions. Its pi BCE is 0.521 (P4) and 0.436 (P16).
   - Over seeds 1-2000, 62.95% have at least one pair (`results/pi_offsets.json`).
   - The leak reaches the multi-epoch arm too: ME seed 4 pi held-out 0.447 and in-sample 0.417 vs committed 0.830 / 0.562, so the "memorisation gap" 0.267 shrinks to 0.030 (`results/compare_S0_s4.json` S0-1c..1e). At seed 5 (no close pi pair) the gap is 0.536 (held-out 1.002, in-sample 0.466; `results/compare_S0_s5.json`). The direction (held-out > ln 2 > in-sample) holds on the no-leak seed, but the committed ± 0.02 sds understate the spread by a factor of about 10.
   - So "pi ≈ ln 2" is seed-lucky. Anything that relies on pi being unlearnable at those seeds (for example, the pi rows of M5 probes, see X1/w3 12:32Z) inherits this. The `streams_ext` prng stream is unaffected (S0-2b share 0.0 at seeds 4 and 5).
   - Fix: draw pi offsets at least W apart, or use disjoint digit blocks.
2. **Several committed sds are much tighter than the seed-to-seed spread** (3 seeds with a near-constant quantity):
   - pi BCE P16: seed 5 gives 0.6948 vs 0.6957 ± 0.0004 (z −2.35). The point is still ≈ ln 2; the sd, not the conclusion, fails.
   - T1 L4.3b hindsight gap D512: 0.0436 and 0.0423 vs 0.0464 ± 0.0008. The gap stays positive and grows with D.
   - T1 L4.4a stale D64: −0.776 / −0.734 vs −0.747 ± 0.004, one fresh seed on each side.
   - T1 L4.5a prng self-resolved income: 0.0542 / 0.0547 vs 0.056 ± 0.0004. Income still equals KL on the fresh seeds.
   - Read "± sd over 3 seeds" in these reports as a lower bound on seed variance. Worker5 (V1: 5.6x) and worker3 (M5 P16: z = −38) found the same pattern.
3. **The S0 w3 delay ordering flips on fresh seeds.** The committed net BCEs are delay0 0.3765, delay64 0.327, delay512 0.442 (a U shape, with "delay0 < delay64 not met").
   - Seed 3: 0.365 / 0.384 / 0.388.
   - Seed 5: 0.361 / 0.393 / 0.368.
   - On both fresh seeds delay0 < delay64 holds, the opposite of the committed "not met".
   - The logistic baseline moves too, so the cause is the per-seed question mix (`make_questions` depends on the seed), not net noise. The committed delay effect is a question-mix effect at seeds 0-2.
   - Contradiction A∧¬A before resolution: 0.395 / 0.362 vs 0.535 ± 0.067, also low on both.
4. **The T1 plateau verdicts are seed-fragile.**
   - L5.4f periodic plateau ratio: committed 0.29 ("fails"), seed 4 0.0029, seed 5 0.80.
   - L5.4e diag plateau: 0.1725 at seed 4 (passes < 0.2), 0.306 at seed 5 (would fail).
   - E_C values and the L5.1 / L5.3 bounds reproduce on both seeds.
   - Plateau pass/fail labels should be reported per seed, not as a single verdict.
5. **ME held-out leak (disclosed by S0).** LFSR16 and prime held-out windows overlap the training stream at T ≈ 64k (overlap49 = 1.00), as `S0/SUBSTRATE.md:49` says. Those held-out numbers are in-sample. My fresh ME rows check that the overlap reproduces.
6. **Relocatability and tooling bugs.**
   - `S0/w2/audit.py` hard-codes `../../../../wealth-net/streams.py`, so it fails from any copy. It ran after patching to the absolute path (the target file is byte-identical to `substrate/streams.py`).
   - `S0/w1/run_w1.py` hard-codes `BULK=/tmp/traders-net/S0/w1`, so two reruns collide. Worker3 and I both wrote seed 4 there; the sha256 values matched, so no harm.
   - `T1/w1/check_L3.py` raises KeyError in its final print when seed 1 is absent. `L3.json` is written first, so the crash is harmless.
   - `S0/check_substrate.py` passes on a pristine copy (`results/substrate_check.txt`, PASS, 0 problems).
7. **L1.** The quoted EoM numbers (arXiv PDF) sit in the introduction (char ≈ 6838), not the abstract as labelled. The numbers are right; only the location label is wrong.
8. **Pre-registration order.**
   - T1's PREREG lines in STATUS precede its DONE line (22:11-22:15Z); the T1 w2 addendum is disclosed.
   - My own deviation: my first S0/T1 jobs launched at about 12:02-12:03Z, before `X1/w1/PREREG.md` (12:07Z). The rule and seeds were fixed in my head, not on disk. ME seed 5 launched before the seed-4 ME result was known; it counts only as a tie-break for ME rows that fall outside the band at seed 4.
9. **Determinism.** Same-seed reruns are byte-identical: S0 P4/P16 seed 4 npz and JSON match between two independent copies (worker3 12:15Z). L1 `check_novelty` reruns are identical except checked_at.
