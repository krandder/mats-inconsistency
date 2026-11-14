# PREREG R6-E1 (E-INFO-V3): key and attention variants, evidence-floor policy independence, active key-space PUCT

- Written 2026-10-17T06:2xZ, registered 2026-10-17T06:27:53Z (exact time = the `Registered:` line at the end and `PREREG-R6-E1.sha256`), before any GPU launch of
  this experiment. The audit compares it with the first `soup-gpu-r6e1-` line of `logical-soup/GPU-LOG.md`.
- Author: Opus 5.5 subagent of eurisko. Assignment: opus-mats agent-mail #3704 (group `r6`, cap $12 shared with R6-E3; this
  experiment's ceiling $2.50 = its ranked-table estimate, kept by `ledger6.py`; 3 seeds; seed-t 95% CIs; independent audit;
  exact oracles for evaluation only).
- Spec: `/home/agent/workspaces/mats/outbox/experiments-round-6-spec.md` (gemini, 2026-10-17T06:15Z, sha256 8be631f1...170c),
  section 1. No amendments file existed at registration (`experiments-round-6-amendments*.md`: none at 06:26Z). My pre-spend
  ambiguity mail: #3708 (06:16Z); every resolution there is restated below.
- Code (new, `li-net/r6e1/`): `e1v3.py`, `test_r6e1.py`, `analyze6.py`, `ledger6.py`, `stage6.sh`, `remote6.sh`, `driver6.sh`,
  `launch6.sh`. Imported unchanged from `li-net/r5e1/`: `env6.py`, `wm.py`, `rollouts.py`, `run.py`, `partA2.py`; and
  `smallnet/netdef.py`. SHA256 at registration (`/tmp/r6e1/src/MANIFEST.sha256`, appended to `results/stage-manifests.txt`):
  - e1v3.py 88b71cac92e6868901c8e53dbe5818585569bde7b387ecd20007d1ee72565c43
  - analyze6.py 2f76562bbf80c06577c5c87c595693eb62122bf38fafb39eaa3700856e8bd9a9
  - test_r6e1.py 42a70e673022ddaff1bda2a8a2f51136fb6bc47acd8f758c17b4b035ed1ba46e
  - ledger6.py d5a28ea4..., stage6.sh 4fe8b8b5..., remote6.sh e516012c..., driver6.sh 2d9fe50d..., launch6.sh f14c0257...
  - r5e1 imports: env6.py dedcce0d..., wm.py 1e5e448a..., rollouts.py 0e853842..., run.py 0a6a897a..., partA2.py 474c7bbd...;
    smallnet/netdef.py 3c2d246a...
- Unit tests at registration: `cd li-net/r6e1; PYTHONPATH=/tmp/r4e3a/pylib:../r5e1:. python3 -m pytest -q test_r6e1.py` -> 6 passed
  (06:27Z): tiles partition the board; key-variant shapes, unit norm, bag-centring invariance, and padding excluded from the
  bag mean; zero alignment loss when keys equal salience; looped_lin_T4 ignores padding and has fewer than 8k parameters;
  prospective PUCT reads never use the board's real readings; the tau tiers flip at 0 / 0.127 / 0.28.
- Before registration (disclosed): two CPU code-path smokes with a d-32 model, 120 training steps and 40 test boards
  (`/tmp/r6e1/smoke-cpu0`, plus a resume test); their numbers are meaningless. No GPU smoke: the R5-E1 v2 pipeline ran on this
  infrastructure, and a Part A crash triggers one resume from the run's own saved theta_real (`remote6.sh`).

## 1. Design (spec 1.1; resolutions of its open points marked R)

- **World model.** The R5-E1 Part A v2 setup and code (`partA2.py`): 6x6 board, ships 4/3/2, a dense sonar field (one reading per
  cell, flip 0.15) and the commander's note. theta_real v2 (669,633 parameters) is retrained exactly as in R5-E1, because no
  checkpoint was kept: 40,000 steps, batch 512, evidence-subset training, the same seed-derived training stream.
  - Streams as R5-E1 v2: calibration stream 21 (4,000 boards, 3,200 fit and 800 held out), test stream 22 (the same 500 test
    boards). Root belief l0: the note + 2 anchor readings.
  - Exact Bayes P*(Z | all 36 readings + note): test metrics, gates and monitor curves only.
- **Reads.** Bounded-context calls theta_real(note + anchors + the 6 readings of patch A). Tiers:
  - tau 0.3: clean.
  - tau 1.5: each patch reading re-flipped with probability 0.127, fresh per read (effective flip 0.239).
  - tau 2.5: re-flipped with probability 0.28 (effective 0.346).

  Calibration pools hold 48 NW-half and 48 SE-half reads per board, with tau uniform over {0.3, 1.5, 2.5} (R), so the 2.5 tier
  is in distribution. theta_trust is as in R5-E1 (features: hbar, entropy, log tau, reading-level leave-one-out swing; target:
  real excess surprise). The K = 128 reference is one theta_real call on the clean readings of the union of 128 unskewed reads.
- **Key variants** (each with its own key head, trained jointly with its pool; no patch indices are given):
  - K1 raw: normalize(W LN(hbar)).
  - K2: bag-centred hbar (the mean is over the bag's valid reads).
  - K3: normalize(W s), with s = |Delta l| / ||Delta l||, W 36 -> 16 (R).
  - K4: normalize(W [LN(hbar - bag mean) || s]) plus the alignment loss (<k_r,k_b> - <s_r,s_b>)^2 over within-bag pairs,
    weight 1.0 (R).
- **Pools.** All learned pools are trained 6,000 steps x 256 bags with K ~ U{6..32} (6, so that PUCT's 6-read wave-1 bags
  are in range (R)), p_NW ~ Beta(0.5, 0.5), Adam 1e-3 and a 36-cell BCE to the reference.
  - attn_L3 (the R5-E1 architecture, about 106k parameters with its key head) once per key variant.
  - looped_lin_T4 with its own K4 key head (R): x^(t+1) = x^(t) + eta_t P_t * (b - G x^(t)), t = 0..3, x^(0) = 0,
    b_r = beta_r Delta l_r, G_rb = relu(<k_r,k_b>)^2 / sqrt(c_r c_b), learned eta_t and diagonal P_t,
    l* = l0 + s sum_r x_r^(4); 2,789 parameters with its key head.
  - Fitted scalars (1,500 Adam steps on the log scale, BCE to the reference; no oracle (R)):
    - ataraxos_mmd eta;
    - kc_dedup u, on the frozen attn_L3-K4 keys;
    - scaled_mix alpha in l0 + alpha sum_r Delta l_r, on unskewed K = 24 bags.
  - mix: the plain average, with no parameter.
- **P-FullSupport** (both bags cover every cell):
  - Each search reads the 6 partition tiles (patches 0, 3, 8, 11, 16, 19), then 18 clean reads, each in the preferred half
    with probability 0.9. Within a half the patch is drawn by VOI as in R5-E1.
  - The clean union call is therefore identical for the two searches (JSD 0, logged as a check).
  - Pools compared: scaled_mix, ataraxos_mmd, kc_dedup, looped_lin_T4, and attn_L3 with K1-K4.
- **P-ActivePUCT** (reads tau 50/50 over 0.3 / 1.5, as in R5-E1 (R)):
  - **open_U:** 24 open-loop reads, each in the preferred half with probability 0.85.
  - **puct_key_U:** 4 waves of 6 reads. In wave w the score is
    Score_U(p) = (1 + 3 x 1[p in half(U)]) x Var(p) / (1 + sum over earlier waves' reads r' of relu(<k_p,k_r'>)^2)^1.5.
    - Var(p) = sum over p's cells of mu(1 - mu), with mu the attn_L3-K4 merged belief of the reads so far (the root in wave 1).
    - 6 distinct patches are sampled in proportion to the score (R).
    - Crowding counts only reads of earlier waves (R).
    - k_p comes from a prospective read whose readings are imputed from mu (reading = 1 iff mu >= 0.5); it never uses the
      board's real readings (R).
    - half(U): at least 3 of 6 cells; the 4 tie patches count for both halves (R).
- **Key-geometry bags** (E1-D1): all within-bag pairs of the open_U NW and SE bags and an unskewed bag (24 reads, tau 50/50),
  as in R5-E1.
- **Controlled bags** (E1-D3):
  - One patch A* per board (unskewed VOI draw); 16 reads of A* at each of tau 0.3, 1.5, 2.5.
  - 8 background reads on disjoint patches (tau 50/50 over 0.3 / 1.5), shared by every tau and N.
  - N in {1, 2, 4, 8, 16}.
  - W(N) = the sum over the N cluster reads of attn_L3-K4's weight averaged over A*'s 6 cells.
  - MSE to exact Bayes over 36 cells, averaged over N.
- **Seeds** 101, 102, 103: one VM (run id `r6e1-main`, spot first, on-demand after a preemption), 3 seeds concurrently, remote
  100 min, timeout 110 min, through gpu-run.sh with GPU_BUDGET_GROUP=r6 GPU_GROUP_BUDGET_USD=12. Only JSON/CSV/PNG come back.

## 2. Decision rules: spec section 1.2 VERBATIM

(copied programmatically from the spec file; sha256 of this excerpt eb1758455a7d019c076f15b2aa813f1b70851d1657768a76345c376cd59631f6)

### 1.2 Pre-Registered Pass/Fail Rules (`R6-E1`)
- **`E1-D0` (Gate):** World model $\theta_{\text{real}}$ has Pearson $r \ge 0.99$ with exact Bayes on the 6-reading format, AND on `P-FullSupport` bags `scaled_mix` cuts 36-cell KL over $P_0$ by $\ge 40\%$.
- **`E1-D1` (Key-Construction Variants — Graded Partial Overlap Without Supervision):**
  1. `K4_combined` (and/or `K3_salience`) achieves Pearson correlation $r(\langle k_r, k_b \rangle, |A_r \cap A_b|/6) \ge \mathbf{0.60}$ (95% CI $> 0.50$, up from `0.375` in `R5-E1`) **AND** same-patch minus across-sector cosine gap $\ge \mathbf{0.50}$ (up from `0.218`);
  2. Mean key cosine increases monotonically across overlap levels $\{0, 1, 2, 3, 4, 6\}$ shared cells (Spearman rank correlation across the 6 levels $\ge 0.94$).
- **`E1-D2` (Policy Independence under Full-Support Skew & Active Key-Space PUCT):**
  1. **Full-Support Repeat-Count Deduplication (`P-FullSupport`):** Under 90/10 preference skew on top of full support, `attn_L3` (with `K4_combined` keys) cuts between-preference $\mathrm{JSD}\bigl(\mu^*(U_{\text{NW}}), \mu^*(U_{\text{SE}})\bigr)$ by **$\ge 2.5\times$** compared to **each** of `scaled_mix`, `ataraxos_mmd`, and `kc_dedup` (all 3 paired CIs of JSD difference $>0$), while achieving the lowest 36-cell KL to exact Bayes (paired CIs $>0$ vs. all 3 baselines). *(Report `looped_lin_T4` alongside as a secondary architectural comparison: passes if within $15\%$ of `attn_L3`'s KL while using $<1/5$ the parameters.)*
  2. **Active Key-Space PUCT Closes the Support Gap (`P-ActivePUCT`):** Without any pre-given base cover, 4-wave `puct_key_U` raises mean 36-cell board coverage from $\le 87\%$ (`open_U`) to **$\ge 95\%$**, reduces full-board $\mathrm{JSD}\bigl(\mu^*(U_{\text{NW}}), \mu^*(U_{\text{SE}})\bigr)$ under `attn_L3` by **$\ge 1.8\times$** relative to `open_U` (95% CI of difference $>0$), and reduces full-board KL to exact Bayes by **$\ge 20\%$** relative to `open_U` while maintaining $\ge 75\%$ NW share in Wave 1 under $U_{\text{NW}}$.
- **`E1-D3` (Three-Regime Hellwig / Clemen-Winkler Weight Law):**
  On controlled bags ($N \in \{1, 2, 4, 8, 16\}$):
  1. At clean $\tau = 0.3$ (deduplication): $W(16)/W(1) \le \mathbf{1.25}$ and `attn_L3` cuts MSE by $\ge 35\%$ vs. `ataraxos_mmd`;
  2. At high-noise $\tau = 2.5$ (mutual support): $W(8)/W(1) \ge \mathbf{2.20}$ (95% CI lower bound $> 1.90$) and $W(16)/W(1) \ge \mathbf{2.80}$, with `attn_L3` cutting MSE by $\ge 15\%$ vs. `kc_dedup`;
  3. Monotonic noise scaling: $W_{\tau=2.5}(8)/W_{\tau=2.5}(1) > W_{\tau=1.5}(8)/W_{\tau=1.5}(1) > W_{\tau=0.3}(8)/W_{\tau=0.3}(1)$ in every seed.

## 3. How each rule is computed (3-seed means; seed-t 95% CI = mean +- 4.303 sd / sqrt(3); ratios and cuts are means of per-seed values)

- **E1-D0** PASS iff mean clause-1 Pearson (theta_real on 6 standard readings + note vs exact Bayes on the same evidence)
  >= 0.99 AND mean 1 - KL(scaled_mix) / KL(P0) >= 0.40 on P-FullSupport (KL averaged over the U_NW and U_SE bags; P0 = root
  belief).
- **E1-D1** PASS iff K4 or K3 (R) meets both clauses.
  - Clause 1: mean Pearson(<k_r,k_b>, overlap/6) >= 0.60 with CI lower bound > 0.50, AND mean (same-patch cosine minus
    across-sector cosine) >= 0.50.
  - Clause 2: the Spearman correlation between overlap level {0, 1, 2, 3, 4, 6} and the 3-seed mean cosine at that level is
    >= 0.94.
  - The other variants are reported.
- **E1-D2.1** PASS iff, on P-FullSupport, for each of scaled_mix, ataraxos_mmd and kc_dedup all three hold:
  - mean JSD(baseline) / JSD(attn_L3-K4) >= 2.5;
  - the CI of JSD(baseline) - JSD(attn_L3-K4) has lower bound > 0;
  - the CI of KL(baseline) - KL(attn_L3-K4) has lower bound > 0.

  Secondary, reported: looped_lin_T4 "passes" if its mean KL is at most 1.15x attn_L3-K4's and it uses fewer than 1/5 of
  the parameters.
- **E1-D2.2** PASS iff all hold:
  - mean puct_key_U coverage >= 0.95 (open_U coverage is reported (R));
  - mean JSD(open_U) / JSD(puct_key_U) >= 1.8 under attn_L3-K4, with the CI of the difference > 0;
  - mean 1 - KL(puct) / KL(open) >= 0.20;
  - mean wave-1 preferred-half share under U_NW >= 0.75.

  Coverage = the share of the 36 cells read at least once.
- **E1-D3** PASS iff all three clauses hold:
  1. Mean W(16)/W(1) at tau 0.3 <= 1.25, AND attn_L3-K4's MSE cut vs ataraxos_mmd at tau 0.3 >= 0.35.
  2. At tau 2.5: mean W(8)/W(1) >= 2.20 with CI lower bound > 1.90, mean W(16)/W(1) >= 2.80, AND MSE cut vs kc_dedup >= 0.15.
  3. In every seed, W(8)/W(1) at tau 2.5 > at tau 1.5 > at tau 0.3.
- Gate: if E1-D0 fails, E1-D1 to E1-D3 are still computed and reported, labelled "gate failed: descriptive".

## 4. Stop rules and honesty

- No retuning after any test number exists. A rerun only replaces a run lost to infrastructure, with the identical
  configuration; a resume from the run's own saved theta_real (`--load-wm`) only replaces a crashed Part A. All of it is logged.
- Exact Bayes is never a target, a fit, a selection or a stop; PUCT never sees real readings before choosing a read.
- `/home/agent/workspaces/mats/outbox/experiments-round-6-amendments*.md` is checked before launch and before scoring; any change
  after registration goes into the deviations log below, with its UTC time.

## 5. Deviations log (append-only)

(none yet)

Registered: 2026-10-17T06:27:53Z
