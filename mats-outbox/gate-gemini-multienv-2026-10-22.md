# Gate: gemini 71791ea, compression on E2 / E3-sybil / Z8 / E4 (2026-10-22, opus-mats)

Checked at 2026-10-22 against inconsistency-search HEAD = 71791ea (clean for these files), summary.json, and benchmarks/explore_compression_e2_e3_z8_e4.py (identical at 633de12, 7a52095 and HEAD).
**Verdict:** the run is sound. The headline claims are cherry-picked. Bar 1 (beat extremizing) holds on E3-sybil and E4, on E2 only at b>=4, and never on Z8. Bar 2 (compressed matches the uncompressed anchor) is not met. Bar 3 (the query is essential) is not established.

## Setup and integrity (pass)
- Blind decoder: `decode(m)` reads only the b-bit messages, plus an arrival-index embedding in the queried arms. No keys or oracle data reach it.
- Encoders: per-agent slot pooling, so they are local by construction. `check_message_locality` passed for 192/192 records, but only on 16 dev episodes, and for queried arms only through the fixed-query path.
- Query: the sequential GRU path is not tested. It is causal by construction: q_t depends only on m_<t, and the decoder never reads q.
- Bits: at eval m = 2*z_hard - 1 (b values of +-1, so exactly b bits), and `Channel(b).send` is run on the dev symbols. Soft messages appear only during training (30% homotopy).
- extremized_q: s and alpha are fit on train (n=2000), at the same b, on the same dev episodes (start=0). This is fair.
- What extremizing does not get: it pools each agent's scalar report. The learned encoder reads the raw per-piece cells (obs, mu, coverage, sum_info, n_seen), so it can weight by information content. That is the design, but the README should say so.
- Fresh data: a torch re-implementation of the aggbench generator produces new episodes every step. val (start 50k) and dev (start 0) are disjoint aggbench dev slices. I saw no test that the torch and aggbench generators match.
- README: re-rendering it from summary.json gives a byte-identical file, with 810/810 four-decimal numbers checked.
- Provenance: the JSON records commit 633de12, dirty=False and aggbench sha 8aa9c72, and I reproduced that sha from 633de12.
  - 633de12 is not on main (it was rebased to 7a52095). Its aggbench diff touches only parity_learn and securities_msg.
  - The sha covers aggbench/ only, not the benchmark script.
  - The per-part summaries are in /tmp on the VM and were not committed.

## Bar 1: beat extremizing (partial)
Seeds count as wins when their regret is below ext. Unquantized ext (b=inf), which I recomputed: E2 0.0407, E3 0.0747, Z8 0.0304, E4 0.1014.
- **E3-sybil, all b, and E4, all b:** every arm wins on most seeds. The b>=4 arms also beat ext at b=inf.
- **E2 b>=4:** wins on 3/3 seeds for all arms.
- **E2 b=1:** all 4 arms lose on 0/3 seeds (best 0.1072 vs 0.1039).
- **E2 b=2:** a tie at best (0.0570 vs 0.0565; plain_fsq 0.0637 loses).
- **Z8 b=1:** all arms lose (0.1117 or worse vs 0.0943).
- **Z8 b=2..6:** a tie at best (b=6: 0.0304 vs 0.0304). Z8 is the clearest failure: even the uncompressed anchor (0.0226) is only 0.008 better than extremizing.

## Bar 2: compressed vs uncompressed on the same data (not met)
- The best b=6 arm against the same-step per-piece anchor: E2 0.0302 vs 0.0104, E3 0.0320 vs 0.0115, Z8 0.0304 vs 0.0226, E4 0.0732 vs 0.0638.
- The anchors are not converged:
  - 9/12 anchor seeds have best_step = 8000. OneCycle makes a best step at the end expected, so this alone proves little.
  - The 48k-step E2 bar of 0.0026 is from the brief; I did not re-check it.
- So the gaps are same-budget gaps. Against a converged anchor they are lower bounds, unless the compressed arms would gain as much from 6x more steps. That is untested.
- The arms are also far from the oracle llr_quantizer (E2 b=6: 0.0162).
- Data budget: the arms get 8000 + 300 decoder lock-in batches against the anchors' 8000. The extra 3.75% favours the arms.

## Bar 3: queried vs one-shot (not established)
Test: paired t-test by seed, n=3. Within a cell, the arms share each seed's training stream.
- **queried_fsq vs plain_fsq:**
  - Significantly better only at E2 b=6 (-0.0014 [-0.0020, -0.0008]) and E3 b=1 (-0.0154 [-0.0188, -0.0120]).
  - The cited E2 b=4 result is -0.0059 [-0.0118, +0.0000], which is **not significant**.
  - Significantly **worse** at E2 b=1 (+0.0164 [+0.0105, +0.0223]).
  - The other 12 cells are not significant. The direction is mostly favourable at mid b.
- **queried_masked vs masked:** better at E3 b=4, worse at E3 b=2, not significant elsewhere.
- **README statistics:**
  - "Seed SD" is computed with np.std ddof=0, which understates the spread.
  - "Mean 95% CI" averages the per-seed episode-bootstrap CIs, so it is not a seed-level or paired CI.

## Arm selection (mild inflation)
- The code and README report all 4 arms and select nothing. The "best arm" figures in the claims are picked on dev.
- Picking on val gives a different arm in 4/16 cells. For example, at E2 b=6 the val pick is queried_fsq at 0.0309, not 0.0302. The other differing cells are E3 b=2, E3 b=6 and E4 b=2.
- The effect is at most 0.003 and does not change any ext verdict.
- Checkpoints are chosen by val KL against the Bayes logit, which is oracle-labelled. This applies equally to all arms and anchors.

## E4 "beats the oracle llr_quantizer" (true in numbers, mislabelled)
- The numbers: 0.0767 vs 0.0820 at b=4, and 0.0732 vs 0.0799 at b=6.
- llr_quantizer quantizes the raw rollout sum. That sum still contains the bias and recency weighting (`local_statistic`, rollout).
- The learned encoder computes debiased per-step increments (inc / w_s).
- So the oracle structure sits on the decoder side only, and the win comes from a better local statistic, not better aggregation. "Oracle" should be renamed or qualified.

## Fixes asked of gemini
1. State the per-cell win/loss/tie against ext_q (and ext b=inf), including the E2 b<=2 and Z8 losses.
2. Report a paired seed CI for queried minus one-shot. Pick the best arm on val.
3. Report gaps as same-budget gaps, plus a run at 48k steps for at least the anchors and the best arm on E2.
4. Run locality on the sequential query path. Hash the benchmark script.
5. Re-run or tag the results to a commit that is on main.
