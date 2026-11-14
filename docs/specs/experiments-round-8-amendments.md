# Pre-Spend Amendments to Round-8 Pre-Registration Readings (`R8-E1`)

**Timestamp:** 2026-10-17T15:58Z (before the `17:00Z` `R8-E1` launch; zero spend incurred).

We approve `R8-E1` Readings 1, 2, 4, 5, 8, 9, and 10 as written, with **three pre-spend amendments to Readings 3, 6, and 7** to prevent mechanical artifacts:

---

## 1. Amendment to Reading 7 (`e1v5.py` lines 598–606 — `E1-D1.1` Sparse-Root Anytime Curve Ordering: Interleave the 6 Partition Tiles 1-in-4 Instead of Front-Loading All 6 at $r = 1 \dots 6$)

- **Why front-loading `patch = torch.cat([tiles, sk], 1)` (`e1v5.py` line 603) creates an artifact across prefixes $K \in \{0, 1, 2, 4, 8, 16, 24, 32\}$:**
  In `R6/R7`'s static `P-FullSupport` ($K = 24$), placing all 6 disjoint partition tiles (`E3.TILES`) at indices $r = 0 \dots 5$ and all 18 skewed reads `sk` at $r = 6 \dots 23$ did not matter because only the full 24-read bag was scored (and every pool is permutation-invariant).
  However, if `run_prefixes` (`e1v5.py` line 606) takes **prefixes** $K \in \{0, 1, 2, 4, 8, 16, 24, 32\}$ of `torch.cat([tiles, sk], 1)`:
  - By $K = 6$, the first 6 reads (`tiles`) have **already covered all 36 cells of the board cleanly at $\tau = 0.3$ with zero NW/SE skew ($\mathrm{JSD}(6) = 0$)**!
  - At $K = 8$, the prefix has 6 balanced partition tiles + only 2 skewed reads (25% skewed).
  - From $K = 8 \to 16 \to 24 \to 32$, the subsequent reads `sk` add **zero new cell coverage** and **only pile on 90/10-skewed duplicate reads** (raising the skew fraction from 25% at $K=8$ to 75% at $K=24$ and 81.3% at $K=32$), which is a skew-corruption stress test rather than a progressive search curve.
- **Amended Reading 7 (Stationary 1-in-4 Interleaved `P-FullSupport` Stream in `e1v5.py` line 603):**
  - Construct the 32-read `P-FullSupport` sequence in **8 blocks of 4 reads**, where **the first read of each 4-read block (indices $0, 4, 8, 12, 16, 20, 24, 28$) is a clean partition tile and the other 3 reads in each block are 90/10 VoI-skewed clean reads (`sk`)**:
    - Specifically, for blocks $b = 0 \dots 5$ (the first 24 reads, indices $0 \dots 23$), place the 6 partition tiles `E3.TILES` (in a per-board random permutation shared between `NW` and `SE`) at indices $4b \in \{0, 4, 8, 12, 16, 20\}$ and the first 18 skewed reads `sk[:, :18]` at the remaining 18 slots.
    - For blocks $b = 6, 7$ (indices $24 \dots 31$), place 2 additional partition tiles at indices $\{24, 28\}$ and 6 skewed reads `sk[:, 18:24]` at the remaining 6 slots (with `sk` drawn as `24` skewed reads).
  - **Properties:**
    1. Every prefix $K \in \{4, 8, 16, 24, 32\}$ maintains the **exact same stationary $1:3$ ratio** (25% base cover, 75% 90/10-skewed VoI reads).
    2. At $K = 24$, the 24-read bag is **100% identical** (up to permutation, and every pool is permutation-invariant) to `R6/R7`'s `P-FullSupport` (the 6 partition tiles + 18 skewed reads)!
    3. Board cell coverage grows progressively across $K = 0 \to 1 \to 2 \to 4 \to 8 \to 16 \to 24 \to 32$, making `D1.1` a genuine anytime search curve.

---

## 2. Amendment to Reading 3 (`e1v5.py` lines 50, 109–117, 296–336 — Anytime Pool Training Target, $K$-Support, and Precision-Weighted Anchor Crowding)

1. **Train on the model's smooth $K = 128$ union-call reference `ref(M_0)` (matching Reading 4 and `R6/R7`), not binary `Yocc` (`e1v5.py` lines 302, 325):**
   - Training `looped_attn_T3` (`39k` params) and `attn_L3-K3` (`106k` params) on binary `Yocc = C["occ"][:na].float()` across $1{,}600$ calibration boards introduces Bernoulli outcome variance $p(1-p)$ and risks overfitting calibration labels (which can push `KL(32)` above the tight `0.033` bar in `D1.1`), whereas the world model's own $K=128$ union-call reference `ref_M = sigmoid(union_call(model, Cs, evM, ref_patches[:na]))` is smooth, self-supervised purely from $\theta_{\text{real}}$ (zero oracle leakage), and consistent with Reading 4.
2. **Include $K = 1$ and $K = 32$ in `KS_TRAIN` (`e1v5.py` lines 50, 66, 308–314):**
   - Set `KS_TRAIN = (0, 1, 2, 4, 8, 16, 24, 32)` (and `pool_any = 32` with `E3.sample_bags(nb, pa, g_, dev, 32, 32)`) instead of `(0, 2, 4, 8, 16, 24)`, since both `D1.1` and `D1.2` evaluate at $K = 1$ and `D1.1` evaluates at $K = 32$.
3. **Precision-weighted crowding with the anchor token $r = 0$ (`e1v5.py` lines 109–117):**
   - In `wm.crowding(k, valid, gamma)` (`wm.py` line 113), unweighted crowding gives the anchor token $r=0$ weight $1$ regardless of whether $M_0 = 2$ ($\beta_0 = \frac{2}{6}\beta_{\text{clean}}$) or $M_0 = 18$ ($\beta_0 = 3\beta_{\text{clean}}$).
   - For the anchor-aware anytime pools, weight crowding by relative precision $\tilde{\beta}_b = \beta_b / \beta_{\text{clean}}$:
     $$c_r \;=\; \sum_{b=0}^K v_b \, \frac{\beta_b}{\beta_{\text{clean}}} \, \max(0, \langle k_r, k_b \rangle)^\gamma,$$
     so that at $M_0 = 18$ ($\tilde{\beta}_0 = 3.0$), any rollout $r \ge 1$ whose key $k_r$ overlaps the strong root's salience key $k_0$ experiences $3\times$ stronger root crowding than at $M_0 = 2$ ($\tilde{\beta}_0 = 1/3$).

---

## 3. Amendment to Reading 6 (`e1v5.py` lines 620–628 — `E1-D1.2` Strong-Root Test Noise Distribution)

- In `e1v5.py` line 625, the 24 skewed VoI reads on the strong root ($M_0 = 18$) are drawn with `tt = E3.draw_tau3(pt.shape, gen, dev)` ($\tau \in \{0.3, 1.5, 2.5\}$). Note that at $K = 1$ (a single 6-cell read added to an 18-cell clean root), a $\tau = 2.5$ read flips $28\%$ of the patch's sonar bits while $\theta_{\text{real}}$ treats `ev` as clean readings.
- Either draw $\tau_r$ in the `Strong-Root` test from `R7-E1`'s `mixed` search distribution (`RO.draw_tau(pt.shape, gen, dev)`, i.e. $\tau_r \in \{0.3, 1.5\}$ 50/50, matching `puct8` at line 540), OR if keeping `draw_tau3` ($\tau \in \{0.3, 1.5, 2.5\}$), include the precision-weighted anchor crowding from Amendment 2(3) above so $\beta_0 = 3\beta_{\text{clean}}$ damps single high-noise reads at $K=1$. (Report both or use `RO.draw_tau` as primary.)

---
---

# Part B: Pre-Spend Amendments to `R8-E2` Pre-Registration Readings (`calibrated-ai`, `esud9c.py`)

**Timestamp:** 2026-10-17T16:02Z (before the `17:05Z` `R8-E2` GPU launch; zero spend incurred).

We approve `R8-E2` Readings 1, 2, 3, 4, 6, 7, 10, 12, 14, 15, and 16 as written, with **three pre-spend amendments to Readings 5, 8, 9, 11, and 13** referencing `R8-E2/code/esud9c.py`:

## 1. Amendment to Reading 11 (`esud9c.py` lines 65–68 & 419–450 — Budget-Scaled Beam $(L_m, W_m)$ for the Anytime Curve $m \in \{1, 2, 4, 8, 16, 32\}$)

- **Why running unscaled `L = 4, W = 16` at $m = 2$ and $m = 4$ breaks the anytime curve (`esud9c.py` lines 66, 413–426):**
  In `esud9c.py`, `arm(Xs, m)` runs Phase A with the full `m = 16` beam width `W_BEAM = 16` and wave depth `L in {3, 4}` even when $m = 2$ (`budget ≈ 122` passes) or $m = 4$ (`budget ≈ 244` passes).
  - At $m = 1$ (`61` passes), the `line 66` guard (`s.passes[b] + len(A) + emax + 1 > s.budget[b]`) fires at tick 0 of Wave 1, so `wave_passes = 0` and Phase B runs **1 root `'conf'` chain** (`n_leaves = 1`).
  - At $m = 2$ (`122` passes), however, Phase A expands Wave 1 and Wave 2/3 until `s.passes[b] ≈ 75` (when the 1-chain guard `emax + 1` finally fires!), leaving only `~45` passes—which in Phase B still only funds **1 single completion chain** (`n_leaves = 1`), wasting $60\%$ of the pass budget on $7$ beam nodes that receive **zero** completion chains!
  - At $m = 4$ (`244` passes), Phase A spends `150–200` passes building $16$ Landmarks and leaves enough passes for only `2–3` completion chains (leaving $13\text{--}14$ of the $16$ beam Landmarks with zero chains!).
- **Amended Reading 11 (Budget-Proportional Beam Width $W_m$ and Wave Cap $L_m$ + Per-Wave Completion Reserve Guard):**
  Scale the Phase-A beam width $W_m$ and wave depth $L_m$ with the budget multiplier $m$ so Phase A never expands more Landmarks than Phase B can complete:
  - **$m = 1$:** $L_1 = 0$ (skip Phase A $\to$ 1 root `'conf'` chain, `n_leaves = 1`).
  - **$m = 2$:** $L_2 = 1, \; W_2 = 2$ (branch at Stall 1 into top 2 children $\to$ complete both, `n_leaves ≈ 2–3`).
  - **$m = 4$:** $L_4 = 2, \; W_4 = 4$ (2 waves of width 4 $\to$ `n_leaves ≈ 5–6`).
  - **$m = 8$:** $L_8 = 3, \; W_8 = 8$ (3 waves of width 8 $\to$ `n_leaves ≈ 12–14`).
  - **$m = 16$:** $L_{16} = L_*, \; W_{16} = 16$ (the full `D1` run, `n_leaves ≈ 24–28`).
  - **$m = 32$:** $L_{32} = L_*, \; W_{32} = 16$ (`n_leaves ≈ 52–58`).
  - Furthermore, in `EngineA.waves()` (`esud9c.py` line 66), replace the 1-chain guard `emax + 1` with the **beam-completion reserve guard** `sum(n.ne + 1 for n in nodes[b])` (or when `wave[b]` finishes at line 109, truncate `kids` to the largest prefix `kids[:k]` such that `s.passes[b] + sum(c.ne + 1 for c in kids[:k]) <= s.budget[b]`) so **every surviving beam Landmark is guaranteed at least 1 completion chain at every budget $m$**!

## 2. Amendment to Readings 5, 8, and 9 (`esud9c.py` lines 35, 340–342, 400–411 — Balanced `TAUS` and `P8_full` Primary vs. `P8_2stage` Descriptive)

1. **Restrict `TAUS = [2.0, float('inf')]` (`esud9c.py` line 35):**
   Section 2.1 of `experiments-round-8-spec.md` explicitly diagnosed uncalibrated $\tau_{\text{alloc}} = 1.0$ (`w_j = exp(logP_j)`) as Bug 3 of `R7-E2`: because Landmark #0 has `logP` $2\text{--}3$ nats above Landmark #8, $\tau_{\text{alloc}} = 1.0$ gives $8\text{--}10\times$ weight to Landmark #0 (allocating 8 chains to the stiffest node before Landmark #8 gets a 2nd chain) and can inflate `any_accept` via false accepts on seed 103. Using `TAUS = [2.0, float('inf')]` ensures balanced multi-Landmark completion.
2. **Keep `P8_full` as the Primary Arm for `D1/D2`, with `P8_2stage` Reported Descriptively (`esud9c.py` lines 403–411):**
   In `P8_2stage` (`esud9c.py` line 170), `probes()` evaluates the model's `Solvable` head `chk` on half-empty boards (`ne ≈ 25`), which `R6-E2` (`D0`) showed has AUROC $\approx 0.50$ because `Solvable` was trained on completed boards. To prevent noisy partial-board `chk` scores or seed-103 false accepts from overriding `P8_full` on `any_accept`, evaluate `P8_full` as the primary arm in `D1` and `D2` and report `P8_2stage` at $m=16$ as a descriptive ablation (just like `attn_vote_L3` in `R7-E2`).

## 3. Amendment to Reading 13 (`D1.1` & `D1.2` Effect-Size Bars)

- In `D1.1`, `P8_full` passes if it beats `P3`, `P6`, and `R7`'s `P8_beam` in witness rate in every seed with all three paired 95% CIs strictly $> 0$, achieving **$\ge +5.0\text{ pp}$ over `P3`**, **$\ge +6.0\text{ pp}$ over `P6`**, and **$\ge +1.5\text{ pp}$ over `R7`'s `P8_beam`** (mean across 3 seeds).
- In `D1.2`, `P8_full` passes if it beats `P3`, `P6`, and `R7`'s `P8_beam` in empty-cell KL under both `chk_mix` and `vote_chk` with all paired 95% CIs strictly $> 0$, cutting KL by **$\ge 9.0\%$ vs. `P3` `chk_mix`** and **$\ge 41.0\%$ vs. 1-pass `P0`**.
