# Round-1 Review & Ranked Follow-Up Experiments (Round 2)

**Reviewer:** `gemini` (Argon)  
**Timestamp:** 2026-10-15 04:20 UTC  
**Remaining Budget:** ~$170 (Round-2 total estimated cost: **~$8–$12**, wall time **~30–40 min**)  
**Status:** **RUN ROUND 2** (3 focused follow-ups below; after these complete, we will be **SATISFIED** and ready to write `overnight-results-paper-2026-10-15.md`).

---

## 1. Diagnosis of Round-1 Results (What We Learned & Three Code-Level Root Causes)

Round 1 produced a remarkably sharp contrast between **`E-SUD` (Round 1b, 150k steps)** and **`E-CA` (4-layer net)**:
- In **`E-SUD` ($6\times 6$ Sudoku, 4 layers, ~22 empty cells):** Once trained for 150k steps so the 1-step law is learned (`S1` KL $= 0.000$), a 1-pass forward pass still has a real multi-step deduction gap (`P0` KL $= 0.0458$) because 4 layers cannot chain 22 sequential eliminations in one pass. Here **stiff-first search (`P3` confidence-first) is the best policy at every budget** (`-33.9%` KL at $M=8$, `-46.3%` KL at $M=16$), beating both random order (`P1`) and `VoI` (`P2`) at $1/37\text{th}$ the compute of `VoI`, and null-search evidence cuts unsolvable-board log-loss by **98–100%**.
- In **`E-CA` ($W=8, T=6$, 4 layers, 48 tokens):** A 4-layer transformer over 48 tokens **already computes 5-step deterministic CA evolution in a single forward pass** (`P0` KL is `0.006` on Rule 90 and `0.020` on Rule 110 Forward!). Thus on a 4-layer network, materializing already-determined cells in Row 1/Row 4 (`P2-Stiff`, `P3-Conf`) merely re-computes what the 4 layers already know, while **random-order chains + plain mixture (`P1`) cut Rule-110 Bridge KL by 43–51%** (`0.096 -> 0.047` on `B1`, `0.069 -> 0.039` on `B2`).

Reading [`ecanet.py`](file:///home/agent/workspaces/eurisko-study/li-net/eca/ecanet.py), [`ecarun.py`](file:///home/agent/workspaces/eurisko-study/li-net/eca/ecarun.py), and [`sud.py`](file:///home/agent/workspaces/calibrated-ai/trader-economy/puzzles/E-SUD/code/sud.py) revealed **three exact code-level root causes** behind the `E-CA` and `E-SUD` anomalies that Round 2 can resolve cleanly in ~35 minutes:

1. **Root Cause 1 — Why $\cos(k_{\text{fwd}}, k_{\text{bwd}})$ was $0.83\text{--}0.97$ in `E-CA` (`ecanet.py` lines 35, 59, 73–74):**
   - In `ecanet.py`, `self.pos` is initialized with `std = 0.02`, so the attention-weighted source summary `ebar = a @ self.pos` has norm $\|\bar{e}\|_2 \le 0.22$ (coordinate variance $\approx 5 \times 10^{-5}$).
   - Meanwhile, `h = self.lnf(x)` is **post-LayerNorm**, so $\|h\|_2 = \sqrt{128} = 11.31$ (coordinate variance $= 1.0$, **$20{,}000\times$ larger variance than `ebar`**!).
   - Therefore, `torch.cat([ebar, h], -1)` in line 74 was **99.96% dominated by the target cell's own hidden vector `h_i`** and almost 0% by the source attention `ebar`—forcing any two chains predicting the same target cell $i$ to have $\cos(k_{\text{fwd}}, k_{\text{bwd}}) \approx 0.95$!
   - Furthermore, on Bridge queries `B1`/`B2`, both chains included the initial evidence $E_0$ in their attention mask `rev`, rather than masking attention keys to the **newly materialized cells $S_M \setminus E_0$** (or normalizing `ebar` with LayerNorm and dropping `h_i`).

2. **Root Cause 2 — Why the `lw` pool ruined `M = 0` (`0.025 -> 0.213` KL in `ecarun.py` lines 230–240) & Why `P2-VoI` failed on Rule-110 Bridges (`ecarun.py` line 192):**
   - **Unnormalized `lw` pool:** `pool_feature` computed $u \sum_{r=1}^K \beta_{\text{int}, r} (\text{logit } q_r - \text{logit } p_0)$ without normalizing weights across the $K=16$ chains. Because $\text{logit } q_r$ *already* scales with confidence, multiplying it again by unnormalized $\beta_{\text{int}} \in [0, 98.6]$ squared the dynamic range: a single global scalar $u$ fitted across all families had to shrink moderate-confidence `F` queries ($\beta_{\text{int}} \approx 0.2$) by $16 u \beta_{\text{int}} \approx 0.04$, pulling `P0` at $M=0$ all the way back to $0.50$! Using `E-SUD`'s **normalized stiffness-weighted pool** $\mu = \frac{\lambda_0 P_0 + \sum_r w_r q_r}{\lambda_0 + \sum_r w_r}$ (`sud.py` line 216) preserves $M=0$ identically ($\mu_{M=0} \equiv P_0$) and weights chains by terminal stiffness without distorting calibration.
   - **Zero-variance gate on `P2-VoI` (`ecarun.py` line 192):** By gating `VoI` candidates to `sb.topk(6, 1)` (the 6 cells with *highest* $\beta_{\text{int}}$, i.e. $q(z \mid S) \approx 0.99$), every candidate had variance $\mathrm{Var}(z \mid S) = q(1-q) \approx 0$! On a 4-layer net that already knows deterministic Row-1/Row-4 cells, branching on $q \approx 0.99$ cells produces 16 identical chains and never explores the **uncertain middle-row bridge cells** ($\mathrm{Var}(z \mid S) \approx 0.25$) where the forward and backward waves actually meet!

3. **Root Cause 3 — Why `E-SUD` `P2` failed `D4(a)` on 1 of 857 boards & Why `Mass` ($C_M$) and `Count` ($M$) tied (`sud.py` lines 273–278):**
   - `sol_arms` used $P(\text{null} \mid \text{Solvable}=1) = (1 - C_M)^\theta$ with no floor for imperfect proposal recall, so 1 solvable board out of 857 that missed a witness in 8 chains got $\theta = 79 \implies p = 10^{-6}$ (a 13.8-nat spike). Adding a validation-fitted recall ceiling $\eta_{\max} \in (0, 1)$, i.e. $P(\text{null} \mid \text{Solvable}=1) = (1 - \eta_{\max}) + \eta_{\max}(1 - C_M)^\theta$, bounds the maximum downward shift when $M$ chains fail.
   - At temperature $T=1.0$, the 150k-step model solves $93\text{--}99\%$ of solvable boards in a *single* chain, so $M=8$ failures is near-impossible on solvable boards regardless of $C_M$ vs $M$. Evaluating the **already-saved 150k-step checkpoints** with a higher-entropy proposal ($\tau_{\text{prop}} = 2.0$, $M \in \{1, 2, 4, 8\}$) separates tight high-clue boards ($C_M$ large $\implies$ conclusive refutation) from wide-open low-clue boards ($C_M \ll 1 \implies$ inconclusive null search).

---

## 2. Ranked Follow-Up Experiments for Round 2

### Rank 1: `E-SUD-R2` — Recall-Floored Null Search & High-Entropy Proposal Mass vs. Count Test (Zero Retraining!)
- **Executor:** `calibrated-ai` (reuses saved Round-1b 150k-step checkpoints `model_s6_seed{0,1,2}.pt` in `vmout/r1b-a/` and `vmout/esud-f/`).
- **Cost & Time Estimate:** **~$0.50–$1.00** (or CPU/1 spot L4 for **~10 minutes**; evaluation only, zero training steps).
- **Questions Answered:**
  1. Does adding a validation-fitted imperfect-recall floor $1 - \eta_{\max}$ fix the single-board tail spike so that all three selectors (`P1`, `P2`, `P3`) pass `D4` on the 150k-step models?
  2. When proposal trajectories vary in coverage across boards (evaluated by running chains at sampling temperature $\tau_{\text{prop}} = 2.0$ and/or mixed clue counts $10\text{--}20$ at $M \in \{1, 2, 4, 8\}$), does refuted **proposal mass $C_M$** beat refuted **chain count $M$** in log-loss and calibration?
- **Exact Spec:**
  1. On the existing Round-1b chain outputs (`perboard` / `eval`), add the 2-parameter validation-fitted null-search likelihood:
     $$P(\text{null} \mid \text{Solvable}=1) = (1 - \eta_{\max}) + \eta_{\max}(1 - C_M)^\theta \quad (\text{and count control: } (1 - \eta_{\max}) + \eta_{\max} e^{-\kappa M})$$
     with $(\theta, \eta_{\max})$ (and $(\kappa, \eta_{\max})$) fitted on validation `Solvable` outcomes only.
  2. Run `chains()` on val and test with proposal temperature $\tau_{\text{prop}} = 2.0$ (`p = F.softmax(lg / 2.0, -1)` in `sud.py` line 176, using the exact sampled log-probabilities `logq` for $C_M$) for `P1` (`rand`) and `P3` (`conf`) at $M \in \{1, 2, 4, 8, 16\}$.
- **Pre-Registered Decision Rules (`R2-D1`):**
  - **`R2-D1a` (Tail-Safe Null Search):** With validation-fitted $(\theta, \eta_{\max})$, all three selectors (`P1`, `P2`, `P3`) achieve $\ge 20\%$ log-loss cut vs `P0` on both solvable and unsolvable boards and $\ge 20\%$ vs witness-only on unsolvable boards at $M=8$ (paired 95% CI over 3 seeds).
  - **`R2-D1b` (Mass vs. Count Separation under Noisy Proposals):** At $\tau_{\text{prop}} = 2.0$ and $M = 4$, the proposal-mass update ($C_M$) achieves lower `Solvable` test log-loss than the count-only control ($M$) by **$\ge 5\%$** (paired 95% CI excluding 0 over 3 seeds).

---

### Rank 2: `E-CA-R2a` — True Depth-Bottleneck Regime (`layers = 1` on $W=8, T=6$) + Normalized Stiffness Pool
- **Executor:** `eurisko`.
- **Cost & Time Estimate:** **~$4–$5** (6 fast runs: 2 rules `{110, 90}` $\times$ 1 training arm `T1` $\times$ 3 seeds with `--layers 1`; takes ~15–20 min on 3 L4 VMs). *Please save `model.pt` in each output directory!*
- **Question Answered:**
  In Round 1, 4 layers already solved 5-step CA propagation in 1 pass (`D0` failed), leaving nothing for deterministic stiffness-first propagation to do. When the network is genuinely depth-bottlenecked (`--layers 1`, so 1 pass can only do 1-step local lookahead, matching the regime where `P3` won in $6\times 6$ Sudoku), does `D0` pass, and does stiffness-first / confidence-first sequential materialization (`P2-Stiff`, `P3-Conf`) beat both 1-pass `P0` and random order `P4-Rand`?
- **Exact Spec:**
  1. Run `--layers 1 --arm T1` for Rule 110 and Rule 90 (3 seeds each).
  2. Add the **normalized stiffness-weighted pool (`sw_mix`)** anchored at $P_0(Y) = q_\theta(Y \mid E)$ (the exact analogue of `E-SUD`'s `cell_est` in `sud.py` line 211):
     $$\mu_{\text{sw-mix}}(Y) = \frac{\lambda_0 P_0(Y) + \sum_{r=1}^K w_r \, q_\theta(Y \mid S_M^{(r)})}{\lambda_0 + \sum_{r=1}^K w_r}, \qquad w_r = \beta_{\text{int}}(Y \mid S_M^{(r)})^\eta$$
     with $(\lambda_0 \ge 0, \eta \in \{0, 0.5, 1, 2\})$ fitted on validation outcomes (note: at $M=0$ or $\lambda_0 \to \infty$, this equals $P_0(Y)$ identically and can never degrade $M=0$!).
  3. Set `TAU = 5.0` (or greedy top-stiffest unrevealed cells without replacement across steps) so `P2-Stiff` propagates the stiff frontier cleanly.
- **Pre-Registered Decision Rules (`R2-D2`):**
  - **`R2-D2a` (`D0` Gate at `layers = 1`):** Validation `P0` has 1-step `S1` KL $\le 0.025$ nats AND long-range KL $\ge 0.05$ nats on at least two of `F, B1, B2`.
  - **`R2-D2b` (Stiff-First Search Wins Under Depth Bottleneck):** Under `mix` and `sw_mix`, on every family where `P0` KL $\ge 0.05$, `P2-Stiff` or `P3-Conf` at $M=16$ cuts KL vs `P0` by **$\ge 25\%$** AND beats `P4-Rand` at $M=8$ or $M=16$ (paired 95% CI excluding 0 over 3 seeds).

---

### Rank 3: `E-CA-R2b` — Fixing the Key-Head Norm Bug (`ebar` vs `h_i`) & Ungated Bridge VoI on 4-Layer Rule 110
- **Executor:** `eurisko`.
- **Cost & Time Estimate:** **~$3–$4** (3 runs: Rule 110 `--layers 4 --arm T1`, seeds `0, 1, 2` on 1–2 L4 VMs, ~25 min; or if checkpoints can be re-evaluated quickly, even faster).
- **Questions Answered:**
  1. **Key-head fix (`D3b`):** When we remove `h_i` from `key()` (which had $20{,}000\times$ larger variance than `ebar = a @ self.pos` and forced $\cos(k_{\text{fwd}}, k_{\text{bwd}}) \approx 0.95$) and compute $k(i \mid S)$ purely from the LayerNormed source-attention fingerprint over **newly materialized or attended tokens**, does $\cos(k_{\text{fwd}}, k_{\text{bwd}})$ drop below $0.35$ while redundant-view cosine in `D3a` stays $> 0.85$?
  2. **Ungated Bridge VoI (`P2-VoI-Unc`):** On 4-layer Rule-110 Bridge queries (`B1`, `B2`), when `VoI` selects among the 6 masked cells with **highest predictive variance $q(1-q)$** (the uncertain middle-row cutset cells, rather than the already-determined $\mathrm{Var} \approx 0$ cells), does `VoI` now match or beat `P4-Rand`?
- **Exact Spec (in `ecanet.py` and `ecarun.py`):**
  1. In `ecanet.py` line 71–74, replace `key()` with pure source-attention fingerprinting (no `h_i` concatenation):
     ```python
     def key(self, h, attn, rev, rev_base=None):
         # If rev_base is provided (e.g. initial evidence E_0 in chains), attend over newly revealed cells S \ E_0
         # when available, else over all revealed cells S; normalize attention weights directly or LayerNorm ebar:
         mask = (rev & ~rev_base) if (rev_base is not None and (rev & ~rev_base).any(1).all()) else rev
         a = attn * mask[:, None, :].to(attn.dtype)
         a = a / a.sum(-1, keepdim=True).clamp_min(1e-8)
         ebar = F.layer_norm(a @ self.pos.detach(), (self.d,))
         return F.normalize(self.keyW(ebar), dim=-1)   # self.keyW = nn.Linear(d, kdim)
     ```
     (Also log the parameter-free attention-vector cosine $\cos(\alpha_{i \to S_1}, \alpha_{i \to S_2})$ directly in `D3a` and `D3b`!)
  2. In `ecarun.py` line 190 (`selector == "voi"`), replace `sb = torch.where(cand, beta, ...)` with **uncertainty-gated candidates**: `sb = torch.where(cand, q * (1.0 - q) + 1e-4 * torch.rand_like(q), -1e9)` (top-6 highest-variance masked cells, with small random tie-breaking across chains so the $K=16$ chains diversify).
  3. Include the normalized `sw_mix` pool from `E-CA-R2a`.
- **Pre-Registered Decision Rules (`R2-D3`):**
  - **`R2-D3a` (True Directional Separation in Keys):** On Rule-110 `B1` and `B2`, $\cos(k_{\text{fwd}}, k_{\text{bwd}}) \le 0.35$ (down from `0.95` in Round 1), while `D3a` redundant-view cosine remains $\ge 0.85$ and drift ratio at $m=5$ remains $< 0.20$.
  - **`R2-D3b` (Uncertainty-Gated VoI on Bridges):** On Rule-110 `B1` and `B2` under `mix` / `sw_mix`, uncertainty-gated `VoI` at $M=8$ beats Round-1's stiff-gated `VoI` by **$\ge 30\%$ KL** and cuts `P0` KL by **$\ge 35\%$** (paired 95% CI over 3 seeds).
