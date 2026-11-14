# PREREG R9-E1 (E-INFO-STRONG-ROOT): genuine strong-root anytime safety and the 1-step precision-aware crowding pool econ_prec_K3

- Written 2026-10-17T19:16Z, registered at the time in the `Registered:` line at the end and in `PREREG-R9-E1.sha256`,
  before any GPU launch of this experiment. The audit compares it with the first `soup-gpu-r9e1-` line of
  `logical-soup/GPU-LOG.md`.
- Author: Opus 5.5 subagent of eurisko (the E-INFO owner). Assignment: opus-mats agent-mail #3753 (18:32:25Z), relayed by eurisko:
  - group `r9`, GPU_GROUP_BUDGET_USD = 1.35 shared with R9-E2; a guard deletes any r9 VM older than 90 min and stops launches
    at $1.30 group-estimated; this experiment's cap is $0.60 (the spec's section 3 table), kept by `ledger9.py`;
  - 3 seeds, seed-t 95% CIs, independent audit (only the auditor's final file is installed), exact oracles for evaluation only;
  - every post-hoc edit is logged as a deviation when it happens;
  - the VM is deleted at DONE, after the final sync (gpu-run.sh trap).
- Spec: `/home/agent/workspaces/mats/outbox/experiments-round-9-spec.md` (gemini, mtime 18:31Z, sha256
  28778d6032b990c73f45f07b724060e7dae6ff01a775c0b8ed9bccbd5c367140): section 0, the global protocol, section 1 and section 3.
- My pre-spend ambiguity mail: #3758 (18:42:49Z), readings 1-10, quoted in section 1 where they decide the design.
- Amendments: `/home/agent/workspaces/mats/outbox/experiments-round-9-amendments.md` (gemini, 19:00Z, sha256 87a336c08f18dca6737a3e6be16a130d6c4793c0573c2fc2ba34d00efa9ad34b). Its Part B (lines 43-53, sha256 49db9e10709e2b0db9b1ff8c734148ee776156f64067fb9557dc272dd0b06fc4) approves readings 1, 2, 3, 4, 7, 8,
  9 and 10 as written and amends readings 5 (econ_prec_K3's ridge is softplus(a - b beta)) and 6 (jacobi_lin_T4_prec's
  loss gets R8's standard-bag BCE back: 0.5 x standard + 0.5 x P-FullSupport + 0.25 x D3 MSE). Part B is quoted
  verbatim in section 2b; `PREREG-R9-E1-AMENDMENTS.md` (folded in at 19:07:09Z) maps it to the code. Part A concerns
  R9-E2 only. opus-mats relayed it (#3762); I confirmed with no cost or ETA change (#3765).
- Code (new, `li-net/r9e1/`): `e1v6.py`, `test_r9e1.py`, `analyze9.py`, `numbers9.py`, `ledger9.py`, `stage9.sh`, `remote9.sh`,
  `driver9.sh`, `launch9.sh`. Imported unchanged: `li-net/r8e1/e1v5.py`, `li-net/r7e1/e1v4.py`, `li-net/r6e1/e1v3.py`; from
  `li-net/r5e1/`: `env6.py`, `wm.py`, `rollouts.py`, `run.py`, `partA2.py`; and `smallnet/netdef.py`. Loaded unchanged:
  `li-net/r7e1/ckpt/s{101,102,103}/theta_real_v2.{npz,json}`. SHA256 at registration (`/tmp/r9e1/src/MANIFEST.sha256`,
  appended to `results/stage-manifests.txt`):
  - e1v6.py a06457a372e0b06065318c53722f977bf014a9613fa739700364d0ccaf304683
  - analyze9.py 184e6ee91ea465789198ab4532a7c28ea5eda324a9c4e228a836a1433736df29
  - numbers9.py 9c803b80f94a1eec3fab5d33fa5d53605af24bbfd393a1579760851c5ddf389e
  - test_r9e1.py 6e492bfe5e520f5c4a3dd67737f2ef89a0a3d72a76265429ab8d6fbd51659d7f
  - ledger9.py cbfc0933..., stage9.sh 3e5a1b5c..., remote9.sh 22b75370..., driver9.sh 7b6c6b5e..., launch9.sh 7e8e8bdf..., render_page9.py 7fb2287c..., site_patch9.py 791d779b...
  - imports: e1v5.py b92881b0..., e1v4.py 02f64c98..., e1v3.py 88b71cac..., env6.py dedcce0d..., wm.py 1e5e448a..., rollouts.py 0e853842..., run.py 0a6a897a..., partA2.py 474c7bbd..., smallnet/netdef.py 3c2d246a...
  - checkpoints: ckpt/s101/theta_real_v2.npz aef34272..., ckpt/s102/theta_real_v2.npz abb0fffc..., ckpt/s103/theta_real_v2.npz f9842c1e...
- Unit tests at registration: `cd li-net/r9e1; PYTHONPATH=/tmp/r4e3a/pylib:.:../r8e1 python3 -m pytest -q test_r9e1.py` ->
  10 passed (19:16Z). They check:
  - strong-root selection: the smallest candidate with KL <= 0.035 (inclusive), else 34;
  - econ_prec_K3: the closed form N u beta / (lambda(beta) + N beta) on identical reads, and equality with econ_kc_K3 when the
    ridge is constant;
  - every pool, econ_prec_K3 included, returns the root exactly at K = 0 (M_0 = 2 and 32) and is finite with reads;
  - Mode B: each read is the standard bounded read, its token is Delta l over l0(2), pools add it to l0(M_0), mix ignores
    the root; Mode A: a read inside the strong root's cells changes nothing;
  - anytime training bags: Mode A takes the M_0 root's reads, Mode B the standard reads, never at M_0 = 2; the anchor is
    l0(M_0) with beta_0 = M_0 beta_clean / 6 and the target ref(M_0); the Mode B share is about one half;
  - calibration P-FullSupport bags are prefixes of 1-32 reads;
  - a checkpoint whose bytes do not match its .json is refused;
  - a full calibration fit (tiny model) runs with `env6.bayes_marginals` replaced by a function that raises, so calibration
    never touches exact Bayes; the test protocol then reports every rule input.
- Before registration (disclosed; none of it touched a test board or exact Bayes except the untrained smokes):
  - 18:38-18:40Z, CPU: the calibration-board root KL to ref(36) at M_0 = 18/30/32/34 with the real checkpoints (1,000 and
    4,000 calibration boards), to write reading 1: 0.0300-0.0301 at M_0 = 32 (reference first), 0.0353-0.0355 (root first).
  - 18:51:53Z, CPU: a smoke on the real checkpoints was stopped about 1 min later, still in its standard calibration pool, so
    that no test-board number would exist before registration. It logged only the calibration-board selection (64 boards):
    M_0_strong = 32 in all 3 seeds.
  - From 18:53Z, CPU: full code-path smokes (fit and tests) with untrained, randomly initialised full-size checkpoints in
    `/tmp/r9e1/smoke-ckpt`, 64 calibration boards, 30 training steps and 24 test boards; `analyze9.py` and `numbers9.py` ran
    on those summaries (`/tmp/r9e1/anatest`). Their numbers are meaningless and fed no choice. There was no GPU smoke.

## 1. Design (spec 1.1; my readings as mailed in #3758, approved, with readings 5 and 6 as amended by gemini at 19:00Z)

### 1.1 World model, roots and the strong-root selection

- No world-model training. Each seed loads `li-net/r7e1/ckpt/s<seed>/theta_real_v2.npz` after checking its sha256 against its
  `.json` (669,633 parameters; 6x6 board, ships 4/3/2, the commander's note, a dense sonar field with flip 0.15), as R8.
- Root ladder M_0 in {2, 6, 12, 18, 30, 32, 34}: the root sees the readings of the first M_0 cells of make_boards2's own
  per-board cell permutation (`e1v5.anchor_order`), so the roots are nested and M_0 = 2 is the R5-R8 root. beta_0 =
  M_0 beta_clean / 6, with beta_clean the mean theta_trust precision of tau-0.3 reads on the standard pool.
- **Strong-root selection (reading 1)**, per seed, on all 4,000 calibration boards only (stream 21; no test board, no exact
  Bayes): ref(36) = theta_real on all 36 readings and the note; KL_cal(M_0) = mean over boards and cells of
  KL(ref(36) || root(M_0)), the direction of every test KL; M_0_strong = the smallest of 30, 32, 34 with KL_cal <= 0.035,
  else 34. The other direction is logged.

### 1.2 Reads

- A read is one theta_real call on a root's evidence plus one 2x3 patch; patch cells the root has not seen get the cell's
  reading re-flipped by the read's tier (tau 0.3: never, 1.5: probability 0.127, 2.5: 0.28); cells the root has seen keep
  the root's reading (R8's read function, `e1v3.bag_reads3`, unchanged). Delta l_r = l_r - l0 of the root it was read from.
- **Mode A (reading 2)**, cumulative context, as R8's strong-root test: 24 reads per half (U_NW with p = 0.9, U_SE with 0.1),
  patches VOI-drawn from the strong root's belief (`RO.sample_patches`, gamma 4), read from the strong root's evidence, tau
  three-tier (1/3 each). Pools: out = l0(M_0) + sum_r w_r Delta l_r with Delta l over l0(M_0).
- **Mode B (reading 3)**, bounded context: the same 24 patches per half as Mode A (paired), tau 50/50 over 0.3 / 1.5; each
  read is the standard read (2 base anchor cells + the patch), Delta l_r = l_r_local - l0(2). Every pool adds these to the
  strong root: out = l0(M_0) + sum_r w_r Delta l_r; the anchor-aware pools also get the anchor token l0(M_0) with beta_0 =
  M_0 beta_clean / 6; mix averages the reads' beliefs (so it ignores the root); ataraxos_mmd, scaled_mix and every other pool
  use their standard-root fits.

### 1.3 Pools

- **Anytime pools (anchor-aware), reading 4:** looped_attn_T3 (39,428 parameters with its K3 head) and attn_L3-K3 (106,180),
  each with its own K3 key head, R8's classes unchanged (`e1v5.LoopedAttnT3P`, `e1v5.AttnL3P`: the anchor token r = 0 with
  k_0 from the root's footprint l0 - l_prior, Delta l_0 = 0, beta_0; precision-weighted crowding with the anchor included).
  Training: 6,000 Adam steps (lr 1e-3, batch 256), BCE to ref(M_0) = R8's union call from the M_0 root over the 128 reference
  patches (its evidence contains the root's); M_0 uniform over the 7 strengths; K uniform over {0, 1, 2, 4, 8, 16, 24, 32};
  at M_0 >= 6 half of the bags are Mode A (the M_0 root's reads: 2,000 calibration boards x 64 reads, VOI-drawn from the M_0
  root, three-tier tau) and half Mode B (the standard pool's reads of the same boards, Delta l over l0(2), added to l0(M_0));
  at M_0 = 2 the modes coincide.
- **R8's pools on the standard root, unchanged:** attn_L3-K1, R6's looped_lin_T4, R7's jacobi_lin_T4 (BCE to the reference
  on R8's standard bags); fitted scalars on standard bags: ataraxos_mmd eta, kc_dedup u on K1 and K3 keys, econ_kc_K3 R7-style
  (u, lambda), scaled_mix alpha; econ_kc_K3 (u, lambda) on the mixed-tier D3 calibration probes ("econ_D3", the 2-scalar
  comparator of E1-D2.1 clause 4); mix.
- **econ_prec_K3 (reading 5 as amended):** w_r = u beta_r / (lambda(beta_r) + sum_b beta_b relu(<k_r, k_b>)^2) with
  lambda(beta) = softplus(a - b beta) (gemini's amendment; the spec's 2 sigmoid(a - b beta) capped the ridge at 2), on the
  attn_L3-K3 pool's K3 keys (as econ_kc_K3); out = l0 + sum_r w_r Delta l_r. 3 parameters: u = exp(.), a free,
  b = softplus(.) >= 0; init u = 1, a = log(e - 1) (lambda(0) = 1), b = 0.1. Fit: 1,500 Adam steps (lr 0.02) of BCE to the
  reference on D3 calibration probe bags (R8's procedure for econ_kc_K3; first 80% of 2,000 boards).
- **jacobi_lin_T4_prec (reading 6 as amended):** R8's class (814 parameters with its K3 head), trained 6,000 Adam steps
  (lr 1e-3, one batch of 256 of each kind per step) on loss = 0.5 x BCE to the reference on R8's standard bags (mixed
  tiers, K uniform in 6..32) + 0.5 x BCE to the reference on calibration P-FullSupport bags + 0.25 x MSE(sigmoid(out),
  reference) on D3 calibration probe bags (R8's sampler: tier uniform, N in {1, 2, 4, 8, 16} same-patch reads plus 8
  background reads). Calibration
  P-FullSupport bags: the test's interleaved stream built on the first 2,000 calibration boards (the 6 partition tiles in a
  per-board random order, then 2 of them again, one tile per block of 4, the other 3 slots VOI reads skewed 90/10 per half,
  all tau 0.3, from the standard root); prefix K uniform over 1..32; training on the first 80% of boards.

### 1.4 Tests (the 500 R5-R8 test boards, stream 22; exact Bayes given all 36 readings, for scoring only)

- E1-D0: clause-1 Pearson of the world model with exact Bayes on the standard 6-cell evidence; the K = 0 identity of both
  anytime pools at every M_0 (the rule uses M_0 = 2 and M_0_strong).
- Sparse-root curve: R8's interleaved 32-read P-FullSupport stream from the standard root (one tile per block of 4, the rest
  VOI reads skewed 90/10, all clean), K in {0, 1, 2, 4, 8, 16, 24, 32}, every pool. P-FullSupport KL = KL at K = 24.
- Strong-root curves: Mode A and Mode B at M_0 in {18, 30, 32, 34} (the rule uses each seed's M_0_strong), K in
  {0, 1, 2, 4, 8, 16, 24}, every pool.
- D3 controlled bags (R8's protocol): N in {1, 2, 4, 8, 16} reads of one VOI-drawn patch A* at tau 0.3, 1.5 or 2.5, plus 8
  background reads; W(N) = the sum over the N reads of a pool's per-read weight averaged over A*'s 6 cells (exact linear
  coefficient for the Jacobi pools); D3 MSE to exact Bayes averaged over N and the three tiers.
- Not run (no R9 rule): the 2x2 PUCT; the M_0 = 6 / 12 strong-root curves.
- KL = mean over boards and cells of KL(exact Bayes || pool), averaged over the U_NW and U_SE bags; JSD between them.

### 1.5 Seeds, statistics, compute

- Seeds 101, 102, 103 (one process each, concurrently, on one L4 VM). 3-seed means with seed-t 95% CIs (t = 4.303); ratios
  are means of per-seed ratios; comparisons are paired per seed.
- Sizes: 4,000 calibration boards x 96 standard reads, 128-patch reference, 2,000 boards x 64 reads at each of M_0 = 6 ...
  34, 2,000 D3 probe boards, 2,000 calibration P-FullSupport boards, 500 test boards (R8's values where R8 had them).
- Launch: `launch9.sh r9e1-main main "101 102 103" 28` (spot L4, remote timeout 28 min, gpu-run.sh timeout 31 min; ledger
  worst case 0.87 x (31/60 + 0.05) = $0.49 <= $0.60). Expected about $0.30 at the $0.87/h list-price ceiling (R8-E1's
  equivalent run took 13 min of compute; this one adds 3 root strengths, Mode B and two training terms).
- If the run is lost to infrastructure (preemption, VM failure), one rerun with the identical configuration is allowed if
  the ledger permits; it is logged.

## 2. Decision rules

### 2a. The spec's global protocol and section 1.2, VERBATIM

(copied programmatically from the spec file: lines 36-41, sha256 b9896435f91ce37387e04f352089bceaef5bb7eea7a1f65f031da919a5cf7070, and lines 73-98, sha256 20f8a0ae72591aae5b8e24d6470460c56d5da78868455e2bb940fc02bffe46eb)

## Global Protocol (Mandatory for Both Executors)

1. **Pre-registration before GPU spend:** Commit `PREREG.md` (with SHA256 hashes of scripts and frozen cached checkpoints) before launching the main GPU job.
2. **Seeds & paired CIs:** 3 seeds (`101, 102, 103`), reporting 3-seed mean and paired `seed-t` 95% CIs ($\text{df}=2, t_{0.975,2}=4.303$).
3. **Zero world-model training & zero oracle leakage:** Load frozen world-model checkpoints (`R7-E1` for `R9-E1`, `R4-E2` for `R9-E2`). Never use exact Bayes $P^*$ or ground-truth Sudoku solutions / `valid9` in training, hyperparameter selection, or search control.
4. **Immediate VM teardown & cost log:** Tear down every GPU VM immediately on completion and log exact wall-clock and USD cost in `GPU-LOG.md`.

### 1.2 Pre-Registered Decision Rules (`R9-E1`)

- **`E1-D0` (Checkpoint & $K=0$ Identity Gate — PASS required):**
  1. All 3 seeds load `li-net/r7e1/ckpt/s<seed>/theta_real_v2.npz` matching its SHA256 hash; clause-1 Pearson with exact Bayes $\ge 0.999$.
  2. At $K=0$, `looped_attn_T3` and `attn_L3-K3` return $\ell^{(0)}(M_0)$ identically ($\max |\Delta \mathrm{KL}| < 10^{-7}$) at both $M_0 = 2$ and $M_0^{\text{strong}}$.

- **`E1-D1` (Sparse-Root & Genuine Strong-Root Anytime Monotonicity — PASS if both `D1.1` and `D1.2` hold):**
  - **`E1-D1.1` (Sparse-Root Anytime Curve, $M_0 = 2$, $K \in \{0, 1, 2, 4, 8, 16, 24, 32\}$):**
    1. Across $K \in \{0, 1, 2, 4, 8, 16, 24, 32\}$ on the interleaved `P-FullSupport` stream, `looped_attn_T3` satisfies $\mathrm{KL}(K_{m+1}) \le \mathrm{KL}(K_m) + 0.0005$ at every step in every seed, reaches $\mathrm{KL}(32) \le 0.033$, and achieves $\mathrm{JSD}(24) \le 0.0075$.
  - **`E1-D1.2` (Genuine Strong-Root Safety at $M_0^{\text{strong}} \in \{30, 32, 34\}$, $K \in \{0, 1, 2, 4, 8, 16, 24\}$):**
    1. **Strong-root regime verified:** At $M_0^{\text{strong}}$, the $K=0$ root belief has 3-seed mean $\mathrm{KL}(0) \le 0.050$ nats.
    2. **Naive pooling degrades the strong root:** Under Bounded-Context Local Patch Reads (Mode B, or Mode A if Mode A also degrades), both `mix` and standard-root `ataraxos_mmd` degrade the strong root at $K=16$ by $\mathrm{KL}(16) - \mathrm{KL}(0) \ge +0.015$ nats (with 95% CI lower bound $> 0$).
    3. **Anchor-aware `looped_attn_T3` is strictly safe at every $K$:** Across all $K \in \{1, 2, 4, 8, 16, 24\}$ in both Mode A and Mode B, `looped_attn_T3` never degrades the strong root in any seed ($\max_{K \ge 1}[\mathrm{KL}(K) - \mathrm{KL}(0)] \le +0.0005$ in every seed), strictly improves $\mathrm{KL}(16) < \mathrm{KL}(0)$ (95% CI of $\mathrm{KL}(0) - \mathrm{KL}(16)$ above $0$), and beats `ataraxos_mmd` and `scaled_mix` at $K=16$ and $K=24$ with 95% CIs $> 0$.

- **`E1-D2` (1-Step `econ_prec_K3` & 4-Step `jacobi_lin_T4_prec` Three-Regime Weight Law — PASS if both `D2.1` and `D2.2` hold):**
  - **`E1-D2.1` (`econ_prec_K3` 3-Scalar Three-Regime Law on D3 Controlled Bags):**
    1. `econ_prec_K3` (`3` scalar params $u, a, b$) achieves clean deduplication $W(16)/W(1) \le 1.25$ at $\tau=0.3$ (with 95% CI upper bound $\le 1.30$, vs. `R8-E1`'s `1.417` for tier-independent `econ_kc_K3`),
    2. Strict tier monotonicity $W_{\tau=0.3}(8)/W_{\tau=0.3}(1) < W_{\tau=1.5}(8)/W_{\tau=1.5}(1) < W_{\tau=2.5}(8)/W_{\tau=2.5}(1)$ in every seed,
    3. Mutual support $W(16)/W(1) \ge 2.20$ at $\tau=2.5$, and
    4. Lower D3 all-tier MSE than both `kc_K3` and 2-scalar `econ_kc_K3` (paired 95% CIs $> 0$).
  - **`E1-D2.2` (`jacobi_lin_T4_prec` vs. 1-Step Pools on Partial-Overlap `P-FullSupport`):**
    1. `jacobi_lin_T4_prec` (`814` params) satisfies all three weight-law clauses ($W(16)/W(1) \le 1.25$ at $\tau=0.3$, strict $N=8$ tier monotonicity in every seed, $W(16)/W(1) \ge 2.20$ at $\tau=2.5$),
    2. Achieves `P-FullSupport` ($K=24$) $\mathrm{KL} \le 0.0420$, and
    3. Beats `looped_lin_T4`, `kc_K1`, `kc_K3`, and 1-step `econ_prec_K3` on `P-FullSupport` $\mathrm{KL}$ (all 4 paired 95% CIs $> 0$), demonstrating the exact gap between 1-step diagonal crowding (`econ_prec_K3`) and multi-step normal-equation inversion (`jacobi_lin_T4_prec` / `looped_attn_T3`) under partial overlap $0 < \rho < 1$.

---

### 2b. Gemini's amendments to my readings (Part B), VERBATIM

(copied programmatically: `experiments-round-9-amendments.md` lines 43-53, sha256 49db9e10709e2b0db9b1ff8c734148ee776156f64067fb9557dc272dd0b06fc4)

## Part B: `R9-E1` (`E-INFO-STRONG-ROOT`, `eurisko`) — Readings 1–10

1. **Reading 1 (Strong-root selection on calibration by `KL(ref(36) || root) <= 0.035`, giving $M_0^{\text{strong}} = 32$ in all seeds, with `KL(root || ref(36))` reported) — APPROVED.** Reference-first `KL(ref(36) || root)` matches the forward KL direction used in all test evaluations (`KL(P* || mu)`).
2. **Readings 2, 3, 4, 7, 8, 9, 10 — APPROVED as written.**
3. **Reading 5 (`econ_prec_K3` ridge parameterization) — MINOR AMENDMENT:**
   - As you noted in Reading 5, `lambda(beta) = 2 * sigmoid(a - b * beta)` is bounded above by `2.0` by construction (whereas `beta_by_tau` in `R8-E1` has mean `3.4` at $\tau=0.3$ and `1.5` at $\tau=2.5$, and `jacobi_lin_T4_prec` learned $\lambda = 2.20 > 2.0$ at $\tau=2.5$ because it multiplied $2\sigma(a - b\beta)$ by unconstrained $\text{softplus}(\text{lam\_raw}_c)$).
   - To avoid artificially capping `econ_prec_K3`'s ridge at `2.0` while keeping the **exact same 3 scalar parameters** $(u, a, b)$, define `econ_prec_lambda(beta, a, b)` as:
     $$\lambda(\beta_r) \;=\; \operatorname{softplus}(a - b\beta_r) \quad (\text{or } \exp(a - b\beta_r)), \qquad b = \operatorname{softplus}(b_{\text{raw}}) \ge 0,$$
     fitted by BCE on the D3 calibration probes exactly as in `e1v6.py`. This has the exact same 3 scalar parameters $(u, a, b)$ and allows $\lambda(\beta_{\tau=0.3}) \to 0.1$ and $\lambda(\beta_{\tau=2.5}) \to 2.2+$ without a hard ceiling at $2.0$.
4. **Reading 6 (`jacobi_lin_T4_prec` training loss) — ADOPT YOUR THREE-TERM ALTERNATIVE:**
   - Adopt the three-term alternative noted in Reading 6: keep `R8-E1`'s mixed-tier standard-bag BCE term alongside the calibration `P-FullSupport` BCE term and the D3 MSE term (`0.5 * std_bag_BCE + 0.5 * pfs_bag_BCE + 0.25 * d3_MSE`), so `jacobi_lin_T4_prec` continues to see mixed-tier $\tau \in \{0.3, 1.5, 2.5\}$ standard bags in its BCE loss as well as `P-FullSupport` clean bags and `D3` probes.

## 3. How each rule is computed (`analyze9.py`)

Conventions: 3-seed means; seed-t 95% CI = mean +- 4.303 sd / sqrt(3); differences are paired per seed; "CI > 0" = the CI of
the per-seed paired difference has lower bound > 0. KL is averaged over the U_NW and U_SE bags. Strong-root statistics use
each seed's own M_0_strong (expected 32 in every seed; reading 1).

- **E1-D0** PASS iff all hold:
  - the sha256 of every loaded npz equals its `.json` (3 seeds);
  - mean clause-1 Pearson >= 0.999;
  - in every seed, for looped_attn_T3 and attn_L3-K3, |KL(pool at K = 0) - KL(l0)| < 1e-7 at M_0 = 2 and M_0_strong.
- **E1-D1.1** (looped_attn_T3, sparse root, interleaved stream) PASS iff all hold:
  - in every seed and for every consecutive pair K_m -> K_m+1 of (0, 1, 2, 4, 8, 16, 24, 32): KL(K_m+1) <= KL(K_m) + 0.0005;
  - mean KL(32) <= 0.033;
  - mean JSD(24) <= 0.0075.
- **E1-D1.2** (strong root M_0_strong) PASS iff all hold (reading 7):
  1. mean KL(0) <= 0.050;
  2. Mode B: for mix and for ataraxos_mmd (standard-root fit), mean KL(16) - KL(0) >= +0.015 with CI lower bound > 0 (both
     pools; Mode A's values are reported, and noted if they also degrade);
  3. in Mode A and in Mode B, looped_attn_T3:
     - in every seed, max over K in {1, 2, 4, 8, 16, 24} of KL(K) - KL(0) <= +0.0005;
     - mean KL(0) - KL(16) > 0 with CI > 0;
     - KL_mmd(K) - KL_looped_attn(K) and KL_scaled_mix(K) - KL_looped_attn(K), K = 16 and 24: mean > 0 with CI > 0 (8 CIs).
- **E1-D1** PASS iff E1-D1.1 and E1-D1.2 both pass.
- **E1-D2.1** (econ_prec_K3 on D3 controlled bags) PASS iff all hold:
  - mean W(16)/W(1) at tau 0.3 <= 1.25 with CI upper bound <= 1.30;
  - in every seed, W(8)/W(1) at tau 0.3 < at tau 1.5 < at tau 2.5;
  - mean W(16)/W(1) at tau 2.5 >= 2.20;
  - MSE(kc_K3) - MSE(econ_prec_K3) and MSE(econ_kc_K3 on D3 probes) - MSE(econ_prec_K3), MSE averaged over N and the three
    tiers: CI > 0 for both (reading 8; per-tier reported).
- **E1-D2.2** (jacobi_lin_T4_prec) PASS iff all hold:
  - mean W(16)/W(1) at tau 0.3 <= 1.25; in every seed W(8)/W(1) at tau 0.3 < 1.5 < 2.5; mean W(16)/W(1) at tau 2.5 >= 2.20;
  - mean P-FullSupport KL (the interleaved stream's first 24 reads) <= 0.0420;
  - KL(X) - KL(jacobi_lin_T4_prec) at K = 24 has CI > 0 for each X in looped_lin_T4, kc_K1, kc_K3, econ_prec_K3 (reading 8;
    "lower in every seed" reported).
- **E1-D2** PASS iff E1-D2.1 and E1-D2.2 both pass.
- Gate: if E1-D0 fails, the other rules are still computed and reported, labelled "gate failed: descriptive".
- `analyze9.py` implements exactly this. It ran on CPU smoke outputs (untrained checkpoints) before registration, which
  checks only the code path. `numbers9.py` prints the descriptive numbers RESULTS quotes, each rounded once.

## 4. Stop rules and honesty

- No retuning after any test number exists. A rerun only replaces a run lost to infrastructure, with the identical
  configuration. It is logged.
- Exact Bayes is never a target, a fit, a selection or a stop; the strong-root selection uses calibration boards and the
  model's own ref(36) only.
- Every change after registration to a registered file or to the analysis, plots included, is logged in section 5 at the
  time it is made, with its UTC time.
- `outbox/experiments-round-9-amendments*.md` is checked before launch and before scoring; any later R9-E1 amendment is
  logged in section 5.
- The VM is deleted at DONE, after the final sync (gpu-run.sh trap); zero r9e1 VMs at the end. Ledger: `ledger9.py`
  refuses a launch whose worst case would pass $0.60.
- Failures are reported as failures. Only the auditor's final file is installed as `AUDIT-R9-E1.md`.

## 5. Deviations log (append-only)

(none yet)

Registered: 2026-10-17T19:16:38Z
