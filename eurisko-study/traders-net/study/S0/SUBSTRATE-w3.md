# S0 worker 3: logical-induction families (D-delay, D-prove, D-expensive)

Code: `substrate/li_families.py`. Tests: `substrate/tests/test_li_families.py` (8 pass; run with
`PYTHONPATH=/home/agent/workspaces/eurisko-study/wealth-net/.pylib python3 -m pytest -q substrate/tests/test_li_families.py`).
Pre-registration: `PREREG-w3.md`. Baseline outputs: `w3/baseline-s{0,1,2}.json`, `w3/baseline-summary.csv`.

## Use

```python
from li_families import make_li_stream, DelayBook, violations, log_score_gain
st = make_li_stream(seed, T=20000)          # 68 questions by default
ps = st.as_positions()                      # plugs into prequential.run_prequential (worker 1)
```
`LIStream` arrays: `bits[Q,T]`, `fam[Q]`/`fam_names` (delay0, delay64, delay512, and, or, prove,
exp64, exp256, exp1024), `kind[Q]` (base, twin, solo, and, or, taut_or, contra, impl, ex3, exp),
`ops[Q,2]` operand question ids, `group[Q]` triple id, `delay[Q]`, `release[Q]`, `sampled[Q]`
(False for D-prove), `cost[Q]`. Methods `window(t)`, `label(t)`, `label_mask(t)`, `reveal(t)`,
`unresolved_history(t)`, in trade-time units (t = position + W).
`violations(p, st, t)` gives AND/OR/D-prove/twin/triple violations, and the same restricted to
questions with no resolved history (`pre:<kind>`). `DelayBook` books at the trade price and
settles at `t + delay` (book, then settle, each step). `log_score_gain` is the LMSR settlement
against a prior.

## Design (what each family is)

- **D-delay.** 10 base questions at D=0 (periodic, LFSR16, CA110, prime, CA30); 10 each at D=64
  and D=512: half **twins** (same bits as a distinct base, so the answer resolves elsewhere
  first), half **solos** (fresh generator). The window only shows labels revealed before the
  trade, with 0.5 for unknown (worker 1's `ArrayStream` now pads with 0.5 too).
- **D-prove.** 4 of each: A|~A, A&~A, (A&B)->A over bases, plus 2 exclusive-exhaustive triples
  [a+b==k]. The deductive process reveals them late: 512 steps for the first two kinds, 768 for
  the implication and 1024 for the triples. Releases are staggered over the first half of the run,
  so each question has a stretch with no resolved history ("before resolution").
- **D-expensive.** CA110 and LFSR16 bits k in {64, 256, 1024} steps ahead, revealed at t+k, with
  cost = k updates. Caveat: this has the same shape as a solo D-delay question (predict k ahead
  from the current window); only the generator differs.
- **AND/OR.** 4 each over bases, at D=0, so AND/OR coherence can be checked in the same stream.

## Baseline (PREREG-w3 arm 5; 3 seeds, T=20000, mean +- sd)

Prequential single pass through worker 1's `run_prequential` (B=4). Net: d=32, 1 layer, Adam
1e-3. Logistic: per question, Adam 1e-2. Source: `w3/baseline-summary.csv`.

| metric (BCE nats) | net | logistic |
|---|---|---|
| delay0 | 0.377 +- 0.006 | 0.424 +- 0.013 |
| delay64 | 0.327 +- 0.003 | 0.368 +- 0.015 |
| delay512 | 0.442 +- 0.013 | 0.475 +- 0.021 |
| twin (all D>0) | 0.354 +- 0.011 | 0.388 +- 0.010 |
| solo (all D>0) | 0.415 +- 0.025 | 0.456 +- 0.017 |
| exp64 / exp256 / exp1024 | 0.520 / 0.537 / 0.533 (sd 0.05-0.07) | 0.609 / 0.604 / 0.585 (sd 0.03-0.07) |

Coherence before resolution (mean absolute violation; 0.5 is an uninformed price):

| kind | net pre | logistic pre | net after history |
|---|---|---|---|
| A\|~A | 0.479 +- 0.184 | 0.500 +- 0.000 | 0.022 +- 0.010 |
| A&~A | 0.535 +- 0.067 | 0.500 +- 0.000 | 0.024 +- 0.004 |
| (A&B)->A | 0.554 +- 0.083 | 0.500 +- 0.000 | 0.033 +- 0.006 |
| triple sum | 0.388 +- 0.148 | 0.500 +- 0.000 | 0.132 +- 0.023 |
| twin vs base | 0.235 +- 0.130 | 0.194 +- 0.017 | 0.070 +- 0.008 |

AND/OR violations over all live steps: net 0.043 +- 0.010 and 0.083 +- 0.007; logistic
0.030 +- 0.016 and 0.078 +- 0.038. AND/OR questions are never unresolved, so they have no
pre-resolution entry.

## Readings and deviations from the pre-registration

- Arms 1-4 (exact labels, no leak, determinism, DelayBook settles once at the booked price):
  **pass** (tests above).
- Arm 5, expectation "delay0 < delay64 <= delay512": **not met as stated.** delay64 scored lower
  than delay0 in both models. The families are not matched on generators (different random
  periods and primes, and twins copy random bases), so family-level BCE confounds delay with
  question mix. For a delay effect, later workstreams should compare a twin with its own base,
  question by question.
- Expectation "exp near ln2 for the logistic model": **not met.** 0.585-0.609 < 0.693, because
  CA110 is biased (mean ~0.56) and partly periodic. The net improves on the logistic model at
  every k and does not degrade with k, so these questions are not yet "expensive" for it in the
  sense intended. If M5/M6 need true compute-graded difficulty, use LFSR only, or report the
  CA110 and LFSR rows separately.
- **The floor for M5/M6.** Neither baseline is coherent before resolution on D-prove: its
  violations are about 0.5, an uninformed price. Once revealed labels arrive, both learn the
  constants (violation below 0.035). The triples stay incoherent at 0.12-0.13 even after history.
  A market or fixed-point layer claiming "coherent before resolution" must push the `pre:`
  numbers well below 0.5.
