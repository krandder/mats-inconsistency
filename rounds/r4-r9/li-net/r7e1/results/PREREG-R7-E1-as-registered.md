# PREREG R7-E1 (E-INFO-V4): exact Jacobi-Richardson pool, weight-tied looped attention, kc_dedup key attribution, coverage-complete key-space PUCT

- Written 2026-10-17T09:3xZ, registered 2026-10-17T09:39:09Z (the exact time is the `Registered:` line at the end and in
  `PREREG-R7-E1.sha256`), before any GPU launch of this experiment. The audit compares it with the first `soup-gpu-r7e1-`
  line of `logical-soup/GPU-LOG.md`.
- Author: Opus 5.5 subagent of eurisko. Assignment: opus-mats agent-mail #3717 (09:15:02Z), relayed by eurisko:
  - group `r7`, cap $6.50 shared with R7-E3; this experiment's ceiling is $2.50, the ranked-table cap, kept by `ledger7.py`;
  - 3 seeds, seed-t 95% CIs, independent audit, exact oracles for evaluation only;
  - every post-hoc edit is logged as a deviation when it happens.
- Spec: `/home/agent/workspaces/mats/outbox/experiments-round-7-spec.md` (gemini, mtime 09:13:12Z, sha256
  cbece88d738601336e5a1fe70a71de7805cfd7087455f39dc83249a35aa24e88), section 3.
- My pre-spend ambiguity mail: #3722 (09:22:01Z), readings 1-10.
- Amendments: `/home/agent/workspaces/mats/outbox/experiments-round-7-amendments.md` (gemini, 09:24Z, sha256
  c377811d71665a998867c198ee946066657d07611951db85ec16c48a48724062). It approves readings 1-8 and 10 as stated and replaces
  reading 9 (the PUCT score in waves 2+). Its R7-E1 part is quoted verbatim in section 2b and implemented as written. Its
  R7-E2 part does not concern this experiment.
- Code (new, `li-net/r7e1/`): `e1v4.py`, `test_r7e1.py`, `analyze7.py`, `ledger7.py`, `stage7.sh`, `remote7.sh`, `driver7.sh`,
  `launch7.sh`. Imported unchanged: `li-net/r6e1/e1v3.py`; from `li-net/r5e1/`: `env6.py`, `wm.py`, `rollouts.py`, `run.py`,
  `partA2.py`; and `smallnet/netdef.py`. SHA256 at registration (`/tmp/r7e1/src/MANIFEST.sha256`, appended to
  `results/stage-manifests.txt`):
  - e1v4.py 02f64c98a345c9d97b04b40c4402afb8c62e7058e4b3b65b5f0ab96d370a2b72
  - analyze7.py 8263475f091e8053166dbd28daa202fe08942133a639716f736e23fb08f2b169
  - test_r7e1.py 6c948cc6e4255f5fc76f57c2d89a255f77b6cf0466ffb7d99f0526389885555f
  - ledger7.py 7e26a143..., stage7.sh 86fe5fea..., remote7.sh bf124159..., driver7.sh b0cf1324..., launch7.sh 9298d6a3...
  - imports: e1v3.py 88b71cac..., env6.py dedcce0d..., wm.py 1e5e448a..., rollouts.py 0e853842..., run.py 0a6a897a..., partA2.py 474c7bbd..., smallnet/netdef.py 3c2d246a...
- Unit tests at registration: `cd li-net/r7e1; PYTHONPATH=/tmp/r4e3a/pylib:. python3 -m pytest -q test_r7e1.py` -> 9 passed
  (09:39Z). They check:
  - jacobi_lin_T4 on N identical reads gives cluster weight N beta / (lambda + N beta) (one step solves it);
  - its first step equals econ_kc;
  - its exact per-read coefficients reproduce its output, padding has no effect, and it has fewer than 3k parameters with
    its K3 head;
  - looped_attn_T3 has one attention block, at most 38% of attn_L3-K3's parameters, and ignores padding;
  - econ_kc weights match the formula;
  - the theta_real .npz round trip is exact (state dict and outputs) and atomic;
  - PUCT never uses unread cells' real readings (per-wave flip test, both selection modes, with a positive control);
  - greedy wave 2 equals the top 6 of the amended score recomputed by hand;
  - the first-24-reads slice.
- Before registration (disclosed): CPU code-path smokes with a d-32 model, 120 training steps, 40 pool-training steps and 40
  test boards (`/tmp/r7e1/smoke-*`):
  - seeds 101-103 trained and saved their .npz; seed 101 reloaded from its .npz (identical test numbers);
  - seed 101 again after the reading-9 amendment;
  - `analyze7.py` ran on the three smoke summaries (`/tmp/r7e1/anatest`).

  Their numbers are meaningless and fed no choice. There was no GPU smoke.

## 1. Design (spec 3.1; my readings as mailed in #3722, approved or amended by gemini)

- **World model and checkpoints** (reading 1).
  - theta_real v2 (669,633 parameters) is retrained per seed exactly as in R5-E1 and R6-E1 (`partA2.train_world_model2`:
    40,000 steps, batch 512, evidence-subset training, seed-derived stream).
  - Right after training it is saved as `ckpt/s<seed>/theta_real_v2.npz` (np.savez_compressed of the float32 state dict,
    written atomically) plus `theta_real_v2.json` (architecture, steps, training curve, sha256). It is reloaded into a fresh
    model and checked; the max |logit difference| on 200 test roots is recorded.
  - The driver syncs the checkpoints off the VM every 4 minutes into `li-net/r7e1/ckpt/`. Later rounds load them with
    `e1v4.load_wm_npz`.
  - Streams as R5-E1 v2 and R6-E1: calibration stream 21 (4,000 boards: 3,200 fit, 800 held out) and test stream 22 (the same
    500 test boards). Root belief: note + 2 anchor readings.
  - Exact Bayes P*(Z | all 36 readings + note): test metrics, the gate and monitor curves only.
- **Reads** as R6-E1: one theta_real call on the note, the anchors and the 6 readings of one patch.
  - Tiers: tau 0.3 clean; tau 1.5 and 2.5 re-flip each patch reading with probability 0.127 and 0.28 (effective flip 0.239
    and 0.346).
  - Calibration pools: 48 NW-half and 48 SE-half reads per board, tau uniform over the 3 tiers.
  - theta_trust as R6-E1: real excess surprise of each calibration read against that board's true occupancy.
  - Reference: one theta_real call on the clean readings of the union of 128 unskewed reads.
- **Keys** (no patch indices are ever given): K1 = normalize(W LN(hbar)); K3 = normalize(W s), s = |Delta l| / ||Delta l||,
  W 36 -> 16 (reading 2); K4 = normalize(W [LN(hbar - bag mean) || s]) plus R6's alignment loss.
- **Pools.** Each learned pool is trained 6,000 steps x 256 bags (K ~ U{6..32}, p_NW ~ Beta(0.5, 0.5), Adam 1e-3, 36-cell BCE
  to the reference), with its own key head trained jointly.
  - attn_L3 (R6 architecture) with K1, K3 and K4 keys: 107,652 / 106,180 / 108,228 parameters.
  - looped_lin_T4: R6-E1's pool, unchanged (K4 head, 2,789 parameters), retrained here as the comparator for E1-D1.2
    (reading 8).
  - jacobi_lin_T4 (K3 head; readings 2 and 3), 812 parameters with its head:
    - g_rb = relu(<k_r,k_b>)^2 v_r v_b with g_rr = v_r (v = validity mask);
    - D_rc = lambda_c + sum_b g_rb beta_b; b_rc = beta_r clamp(Delta l_rc, -8, 8);
    - x^(t+1)_rc = x^(t)_rc + eta_t P_tc (b_rc - (lambda_c x^(t)_rc + sum_b g_rb beta_b x^(t)_bc)) / D_rc, x^(0) = 0,
      t = 0..3;
    - l* = l0 + s_c sum_r x^(4)_rc;
    - learned: lambda_c = softplus(.) (init 1), eta_t (init 1), P_tc (init 1), s_c (init 1).
  - looped_attn_T3 (K3 head; reading 6), 39,428 parameters with its head (37.1% of attn_L3-K3): one shared AttnL3 block
    (pre-LN MHA, d 64, 4 heads, plus FFN) applied 3 times, with a learned step embedding added before each pass; same inputs,
    W_v and output head as AttnL3.
  - Fitted scalars (1,500 Adam steps on the log scale, BCE to the reference; no oracle):
    - ataraxos_mmd eta;
    - kc_dedup u on the frozen K1, K3 and K4 heads of attn_L3-K1 / -K3 / -K4 (kc_K1, kc_K3; kc_K4 = R6's kc_dedup,
      descriptive);
    - econ_kc_K3 (u, lambda) fitted jointly, w_r = u beta_r / (lambda + sum_b beta_b g_rb) on the frozen K3 head (reading 5);
    - scaled_mix alpha, on unskewed K = 24 bags.
  - mix: the plain average, with no parameter.
- **P-FullSupport** as R6-E1: each search reads the 6 partition tiles (patches 0, 3, 8, 11, 16, 19), then 18 clean reads,
  each in the preferred half with probability 0.9. Union-call JSD is 0 by construction (logged as a check).
- **P-ActivePUCT** (reading 9 as amended), under two read settings: (a) mixed, tau 50/50 over 0.3 / 1.5 per read (the R6-E1
  setting); (b) clean, every read at tau 0.3.
  - open_U: 30 open-loop reads, each in the preferred half with probability 0.85; its first 24 reads are the K = 24 bag.
  - puct_key_U: 5 waves of 6 distinct patches. The first 4 waves (K = 24) are scored by the rules; wave 5 (K = 30) is
    descriptive.
    - Wave 1: score = (1 + 3 x 1[p in half(U)]) x Var(p); 6 patches sampled in proportion to it without replacement (as R6).
    - Wave w >= 2, with w0 = w - 1: score = (1 + 3 / (1 + w0) x 1[p in half(U)]) x (Var(p) / (1 + crowd(p))^1.5)^2;
      6 patches sampled in proportion to it without replacement (the amended reading 9, section 2b).
    - Var(p) = sum over p's cells of mu(1 - mu), with mu attn_L3-K3's merged belief of the reads so far (the root in wave 1).
    - crowd(p) = sum over earlier waves' reads r' of relu(<k_p,k_r'>)^2, with K3 keys from attn_L3-K3's head. k_p comes from
      a prospective read of p whose readings are imputed from mu (reading = 1 iff mu >= 0.5), never the board's real readings.
    - half(U): at least 3 of 6 cells; the 4 tie patches count for both halves (R6 convention). The wholly-in-half share
      (>= 4 cells) is reported.
  - Greedy arm (descriptive): the same searches, with waves 2-4 taking the top 6 of the same amended score.
  - Union call (one theta_real call on the clean readings of all cells the bag read): reported for every search and setting,
    as the support-gap part of the JSD decomposition.
- **Key geometry** (descriptive, as R6): within-bag pairs of the mixed open_U NW and SE K = 24 bags and an unskewed bag.
- **Controlled bags** as R6-E1:
  - one patch A* per board; 16 reads of A* at each tier; 8 background reads on disjoint patches (tau 50/50 over 0.3 / 1.5),
    shared across tiers and N; N in {1, 2, 4, 8, 16};
  - W(N) of a pool = the sum over the N cluster reads of the pool's per-read weight averaged over A*'s 6 cells, averaged
    over boards;
  - per-read weight = the coefficient of that read's Delta l in l*: the attention weights for attn_L3 and looped_attn_T3;
    u beta_r / c_r for kc; the econ formula; eta beta_r for mmd; for jacobi_lin_T4 the exact coefficient of its 4-step
    linear map, with keys and beta held fixed (reading 4);
  - MSE = 36-cell MSE to exact Bayes.
- **Seeds** 101, 102, 103 on one L4 (run id `r7e1-main`; spot first, on-demand after a preemption), run concurrently: remote
  100 min, timeout 110 min, through gpu-run.sh with GPU_BUDGET_GROUP=r7 GPU_GROUP_BUDGET_USD=6.5. Only JSON/CSV/PNG and the
  theta_real checkpoints come back.
- **Recovery.** If the VM is lost, the rerun (on-demand, identical configuration) stages the synced checkpoints
  (`STAGE_CKPT=1`) and loads them instead of retraining. A seed whose run crashed after saving its checkpoint is retried
  once from it. Either is logged as a deviation.

## 2. Decision rules

### 2a. Spec section 3.2 VERBATIM

(copied programmatically from the spec file; sha256 of this excerpt 0877c163bf6352a80d5c3207e6119c06f6c84002bf03102a4d97cd69ad8b6f74)

### 3.2 Pre-Registered Decision Rules for `R7-E1`

- **`E1-D0` (Gate):** Same as `R6-E1`: world model Pearson with exact Bayes $\ge 0.99$, and `scaled_mix` on `P-FullSupport` cuts KL vs. root by $\ge 40\%$.
- **`E1-D1` (Iterative Normal-Equation & Weight-Tied Attention Pools):** Across the 3 seeds on `P-FullSupport`:
  1. **Weight-Tied `looped_attn_T3` (`K3` keys, $\sim 36\text{k}$ params, $\le 38\%$ of `attn_L3` params):** Achieves `P-FullSupport` $\mathrm{KL}$ within **$1.10\times$** of `attn_L3-K3` ($\mathrm{KL} \le 0.038$), cuts preference $\mathrm{JSD}$ by **$\ge 4.5\times$** relative to `scaled_mix` and `ataraxos_mmd`, and beats `scaled_mix`, `ataraxos_mmd`, `kc_K1`, and `kc_K3` in both $\mathrm{KL}$ and $\mathrm{JSD}$ (all paired 95% seed-t CIs $> 0$).
  2. **Exact Jacobi-Richardson `jacobi_lin_T4` (`K3` keys, $<3\text{k}$ params):** Strictly beats `R6-E1`'s normalized `looped_lin_T4` (`0.069` KL) and `kc_K1` in `P-FullSupport` $\mathrm{KL}$, and on `D3` controlled bags satisfies **all three regimes of the Econometrica weight law** on its own ($W(16)/W(1) \le 1.25$ at $\tau=0.3$, $W(8)/W(1) \ge 2.0$ at $\tau=2.5$, and $W(8)/W(1)$ strictly monotone across $\tau \in \{0.3, 1.5, 2.5\}$ in every seed!).
- **`E1-D2` (Coverage-Complete Key-Space PUCT `puct_key_U` vs. `open_U`):** Across the 3 seeds at $K=24$ reads (4 waves $\times$ 6 reads):
  1. `puct_key_U` achieves mean board cell coverage **$\ge 0.955$** (with 95% seed-t CI lower bound $> 0.950$, vs. $\sim 0.828$ for `open_U`), while Wave 1 allocates **$\ge 0.75$** of reads to the preferred half (straddling convention as registered in `R6-E1`; wholly-in-half reported alongside).
  2. Under both mixed-noise ($\tau \in \{0.3, 1.5\}$) and clean ($\tau = 0.3$) reads, `puct_key_U` + `attn_L3-K3` cuts preference $\mathrm{JSD}(\mu_{\text{NW}} \Vert \mu_{\text{SE}})$ by **$\ge 2.0\times$** (paired 95% CI $> 0$) and cuts $\mathrm{KL}(P^* \Vert \mu)$ by **$\ge 35\%$** relative to `open_U` + `attn_L3-K3`.
- **`E1-D3` (Three-Regime Weight Law & Key Attribution on Controlled Bags):** Across the 3 seeds on controlled bags ($N \in \{1, 2, 4, 8, 16\}$ at $\tau \in \{0.3, 1.5, 2.5\}$):
  1. **Clean Deduplication ($\tau = 0.3$):** `attn_L3-K3` achieves $W(16)/W(1) \le 1.25$ and cuts 36-cell MSE by **$\ge 35\%$ vs. `ataraxos_mmd`** and **$\ge 20\%$ vs. `kc_K1`** (paired 95% CIs $> 0$).
  2. **High-Noise Mutual Support ($\tau = 2.5$):** `attn_L3-K3` achieves $W(8)/W(1) \ge 2.20$ (95% CI lower bound $> 1.90$) and $W(16)/W(1) \ge 2.80$, cuts 36-cell MSE by **$\ge 20\%$ vs. `kc_K1` (Round-5 key baseline)** and **$\ge 20\%$ vs. `ataraxos_mmd`**, and beats `kc_K3` with paired 95% seed-t CI of $\mathrm{MSE}(\text{kc\_K3}) - \mathrm{MSE}(\text{attn\_L3-K3}) > 0$.
  3. **Strict Monotonicity in Noise:** $W(8)/W(1)\big|_{\tau=0.3} < W(8)/W(1)\big|_{\tau=1.5} < W(8)/W(1)\big|_{\tau=2.5}$ holds in every seed.

### 2b. Amendment, R7-E1 part VERBATIM

(copied programmatically from `experiments-round-7-amendments.md`, from "## Part 2" to the end; sha256 of this excerpt
ba41811550418e8f4267cefe10f8b30825dcc7250a44670455be05cfd5a901c4)

## Part 2: `R7-E1` — Reading 9 Fix (Do Not Square `pref` in Waves 2–4)

In `R7-E1` Reading 9, `score` in Wave 1 ($w = 0$) is:
$$\text{score}_0(p) = \underbrace{(1 + \lambda_U \mathbf{1}[p \in \text{half}(U)])}_{\text{pref}(p) \in \{1, 4\}} \cdot \mathrm{Var}_\mu(p).$$
If Waves 2–4 ($w \in \{1, 2, 3\}$) literally square the entire `score` (`score ** 2`), that squares `pref(p)` from $4\times$ to **$16\times$** in Waves 2–4! A $16\times$ multiplier for the preferred half in Waves 2–4 overpowers the crowding penalty $(1 + \text{crowd})^{1.5}$ on lightly visited preferred-half patches (`16 / 2.5^3 = 1.02 >= 1`), keeping $\sim 75\%$ of reads in the preferred half even in Wave 4 and **reducing** opposite-half coverage below $95\%$.

**Amended Reading 9 for `R7-E1`:**
- In Wave 1 ($w = 0$): `score = pref * var`, sampled proportional to `score` without replacement (identical to `R6-E1`, guaranteeing `wave1_pref_share` $\approx 0.82 \ge 0.75$).
- In Waves $w \ge 1$ ($w \in \{1, 2, 3\}$): sharpen **only the novelty-weighted variance term** (and decay the initial preference nudge as exploration proceeds, $\text{pref}_w = 1 + (\lambda_U / (1 + w)) \mathbf{1}[p \in \text{half}(U)]$, or keep $\text{pref}$ unsquared):
  $$\text{score}_w(p) = \Big(1 + \frac{\lambda_U}{1 + w}\mathbf{1}[p \in \text{half}(U)]\Big) \cdot \left(\frac{\mathrm{Var}_\mu(p)}{(1 + \text{crowd}(p))^{\alpha_{\text{nov}}}}\right)^2,$$
  sampled in proportion to $\text{score}_w(p)$ without replacement (and where prospective candidate keys for `K3` use the imputed prospective read's belief-shift salience $s_p = |\Delta\ell_p| / \|\Delta\ell_p\|_2$, or `KeyNet("K3")` applied to $s_p$). All other readings 1–8 and 10 of `R7-E1` are approved as stated.


## 3. How each rule is computed

Conventions: 3-seed means; seed-t 95% CI = mean +- 4.303 sd / sqrt(3); ratios and cuts are means of per-seed values;
"paired CI > 0" = the CI of the per-seed paired difference has lower bound > 0 (reading 10).

- **E1-D0** PASS iff mean clause-1 Pearson >= 0.99 (theta_real on the 6 standard readings + note vs exact Bayes on the same
  evidence) AND mean 1 - KL(scaled_mix) / KL(P0) >= 0.40 on P-FullSupport (KL averaged over the U_NW and U_SE bags).
- **E1-D1.1** (looped_attn_T3) PASS iff all hold on P-FullSupport:
  - parameters (with key heads) of looped_attn_T3 / attn_L3-K3 <= 0.38;
  - mean KL(looped_attn_T3) / KL(attn_L3-K3) <= 1.10. This ratio governs (reading 7); whether mean KL <= 0.038 is reported;
  - mean JSD(b) / JSD(looped_attn_T3) >= 4.5, for b = scaled_mix and for b = ataraxos_mmd;
  - for each b in scaled_mix, ataraxos_mmd, kc_K1 and kc_K3: paired CI of KL(b) - KL(looped_attn_T3) > 0 AND paired CI of
    JSD(b) - JSD(looped_attn_T3) > 0.
- **E1-D1.2** (jacobi_lin_T4) PASS iff all hold:
  - fewer than 3,000 parameters with its key head;
  - on P-FullSupport, mean KL(jacobi_lin_T4) is below mean KL(looped_lin_T4, retrained here) and below mean KL(kc_K1), and
    below each of them in every seed ("strictly", reading 8). R6-E1's 0.069 is reported alongside;
  - on controlled bags, with jacobi_lin_T4's own W: mean W(16)/W(1) at tau 0.3 <= 1.25; mean W(8)/W(1) at tau 2.5 >= 2.0;
    and in every seed W(8)/W(1) at tau 0.3 < at tau 1.5 < at tau 2.5.
- **E1-D1** PASS iff E1-D1.1 and E1-D1.2 both pass.
- **E1-D2.1** PASS iff, in the mixed setting (reading 9), mean puct_key_U coverage at K = 24 >= 0.955 with CI lower bound
  > 0.950, AND mean wave-1 preferred-half share under U_NW >= 0.75 (tie convention).
  - Coverage = the share of the 36 cells read at least once (anchors not counted), averaged over boards and over U_NW / U_SE.
  - Reported: clean-setting coverage, the wholly-in-half share, open_U coverage, the greedy arm and K = 30.
- **E1-D2.2** PASS iff, in BOTH the mixed and the clean setting, with attn_L3-K3 as the pool:
  - mean JSD(open_U) / JSD(puct_key_U) >= 2.0;
  - paired CI of JSD(open_U) - JSD(puct_key_U) > 0;
  - mean 1 - KL(puct_key_U) / KL(open_U) >= 0.35.

  KL is averaged over U_NW and U_SE; JSD is between the U_NW and U_SE merged beliefs.
- **E1-D2** PASS iff E1-D2.1 and E1-D2.2 both pass.
- **E1-D3** (attn_L3-K3) PASS iff all three clauses hold. MSE per seed and tier = 36-cell MSE to exact Bayes averaged over N;
  cut vs b = 1 - MSE(attn_L3-K3) / MSE(b).
  1. tau 0.3: mean W(16)/W(1) <= 1.25; cut vs ataraxos_mmd >= 0.35 with paired CI of MSE(mmd) - MSE(attn_L3-K3) > 0; cut vs
     kc_K1 >= 0.20 with paired CI of MSE(kc_K1) - MSE(attn_L3-K3) > 0.
  2. tau 2.5: mean W(8)/W(1) >= 2.20 with CI lower bound > 1.90; mean W(16)/W(1) >= 2.80; cut vs kc_K1 >= 0.20 and cut vs
     ataraxos_mmd >= 0.20 (means; reading 10); paired CI of MSE(kc_K3) - MSE(attn_L3-K3) > 0.
  3. In every seed, W(8)/W(1) at tau 0.3 < at tau 1.5 < at tau 2.5.
- Gate: if E1-D0 fails, the other rules are still computed and reported, labelled "gate failed: descriptive".
- `analyze7.py` implements exactly this. It was run on the CPU smoke outputs before registration, which checks only the code
  path.

## 4. Stop rules and honesty

- No retuning after any test number exists. A rerun only replaces a run lost to infrastructure, with the identical
  configuration; loading a seed's own synced checkpoint only replaces its lost or crashed training. All of it is logged.
- Exact Bayes is never a target, a fit, a selection or a stop; PUCT never sees real readings before choosing a read.
- Every change after registration to a registered file or to the analysis, plots included, is logged in section 5 at the
  time it is made, with its UTC time (the round-6 audit lesson).
- `outbox/experiments-round-7-amendments*.md` is checked before launch and before scoring; any later amendment is logged in
  section 5.
- Failures are reported as failures.

## 5. Deviations log (append-only)

(none yet)

Registered: 2026-10-17T09:39:09Z
