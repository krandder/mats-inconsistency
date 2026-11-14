# Inconsistency Is All You Need
*coherent beliefs from rollouts that disagree: policy-independent beliefs over a factored world model*

**Research direction note. Kelvin Azevedo Santos, MATS autumn 2026 (mentor: Deger Turan, Metaculus). Drafted with the Gemini Argon agent; edited. Status: proposal with early evidence (section 5).**

---

## Executive Abstract

Current foundation models keep world beliefs, search heuristics, and utility preferences in one parameter set and one serial context window. Appending all reasoning steps into one tape of length $N$ hits context limits, while outcome reinforcement learning (RL) updates beliefs alongside actions—risking belief distortion and search that exploits world-model errors.

We study a factored architecture designed to include (not yet empirically shown across all regimes) **(a) standard autoregressive Transformers**, **(b) AlphaZero / MuZero MCTS**, and **(c) belief-network + update-equivalence search for imperfect-information games (Ataraxos; Sokota et al., *Nature* 2026)**. An **Any-Order World Model** ($q_\theta$) pairs with a **Consensus Memory & Aggregator** running $k$ parallel or tree-forked rollouts of context length $\le n$ ($k \cdot n$ world-model calls with context $\le n$).

For a fixed world model and explicit utility $U$, the search target is the unique minimizer $\mu^*(\theta, U)$ of the world model's **Probabilistic Dependency Graph (PDG) Inconsistency Energy**. Under full support and deduplication—counting each distinct queried conditional once—the asymptotic belief depends only on $(\theta, U)$, not on the search policy $\pi_{\text{pol}}$. **Local Inconsistency Resolution (LIR) control masks** assign converse roles to *distilling search consequences* ($\theta_{\text{amp}}$) and *auditing search trustworthiness against reality* ($\theta_{\text{trust}}$).

---

## 1. Factored Parameter Blocks & Converse LIR Roles

Let $V = \{1, \dots, M\}$ denote domain variables (observed states $O$, latent variables $Z$, actions $A$, and outcomes/goals $Y$). We partition a shared representation into four blocks under **LIR control masks** $\chi = (\chi_{\text{real}}, \chi_{\text{search}}) \in \{0, 1\}^2$:

```mermaid
flowchart LR
  R1["1. Base World Model (θ_real)"] --> S1["2. Consequence Extension (θ_amp)"]
  S1 --> S2["4. Aggregator & Search Policy (θ_agg, θ_pol)"]
  S2 --> R2["3. Self-Trustworthiness Auditor (θ_trust)"]
  R2 --> S2
```

$\theta_{\text{real}}$ and $\theta_{\text{trust}}$ learn strictly from realized data, whereas $\theta_{\text{amp}}$ and $(\theta_{\text{agg}}, \theta_{\text{pol}})$ learn from search:

| Parameter Block | Input Conditioning | LIR Mask $(\chi_{\text{real}}, \chi_{\text{search}})$ | Converse Functional Role |
| :--- | :--- | :---: | :--- |
| **1. Base World Model ($\theta_{\text{real}}$)** | Single-world partial assignment $x_S$ | $(1, 0)$ | Learns local domain laws $q_{\text{real}}(x_i \mid x_S)$ from realized data only. Blind to search trees and utility $U$. |
| **2. Consequence Extension ($\theta_{\text{amp}}$)** | **Single-world state $x_S$** (`stop_grad` from $\theta_{\text{real}}$) | $(0, 1)$ | **Expands base predictions via search distillation:** predicts amortized multi-step consensus from $x_S$ and emits rollout tokens $(k_r, v_r)$ without mutating $\theta_{\text{real}}$. |
| **3. Self-Trustworthiness ($\theta_{\text{trust}}$)** | **State $x_S$ + Search Summary $h_{\text{search}}$** | $(1, 0)$ | **Audits search on reality only:** predicts excess outcome surprise given what the search did, separating calibrated 50/50 risk (high entropy, high trust) from out-of-distribution errors (low entropy, low trust). |
| **4. Modular Utility ($U$)** | Goal/action variables $X_G$ | $\chi_{\text{belief}}=0, \, \chi_{\text{action}}=1$ | Steers actions via $f_U(X_G) \propto \exp(\lambda U(X_G))$ without moving beliefs over uncontrolled variables. |

---

## 2. Policy-Independent Consensus & the Deduplication Principle

### 2.1 The Global Inconsistency Target

For any fixed world model $\theta$, its queried conditionals $\{q_\theta(X_i \mid X_S)\}$ with precisions $\beta_\theta(i \mid S) \ge 0$ form a PDG $M_{\text{world}}(\theta)$. Adding utility factor $f_U(X_G) \propto \exp(\lambda U(X_G))$ defines the **Total Inconsistency Energy**:

$$\mathcal{E}(\mu; \theta, U) \;=\; \underbrace{\sum_{(S \to i)} \beta_\theta(i \mid S) \, \mathbb{E}_{\mu(X_S)}\!\Big[ \mathrm{KL}\big(\mu(X_i \mid X_S) \;\big\|\; q_\theta(X_i \mid X_S)\big) \Big]}_{\text{World-Model Observational Incompatibility}} \;+\; \underbrace{\mathrm{KL}\big(\mu(X_G) \;\big\|\; f_U(X_G)\big)_{\chi_{\text{world}}=0}}_{\text{Modular Goal / Utility Incompatibility}} \;-\; \gamma H(\mu).$$

$$\mu^*(\theta, U) \;=\; \arg\min_{\mu} \, \mathcal{E}(\mu; \theta, U).$$

**Assumptions:** $\mu^*(\theta, U)$ is unique and independent of $\pi_{\text{pol}}$ provided: (i) queried arcs and precisions $\beta_\theta(i \mid S)$ are fixed, (ii) $\gamma > 0$, (iii) each distinct arc is counted once, (iv) mutually exclusive branches are weighted by model probability, (v) $\pi_{\text{pol}}$ has full support, and (vi) the aggregator exactly deduplicates repeated arcs. Across training rounds, updating $\theta_{\text{amp}}$ and $\theta_{\text{trust}}$ on past trajectories couples the effective target to prior search.

### 2.2 Counting Each Distinct Piece of Information Once (The Space of Information)

In practice, we cannot evaluate every conditional in $\mathcal{E}(\mu; \theta, U)$ at once; $\pi_{\text{pol}}$ samples $k$ rollouts, each emitting a key $k_r \in \mathbb{S}^{d-1}$ (which subspace of world variables the rollout inspected, read via `stop_grad` from $\theta_{\text{real}}$'s internals), log-odds shift claim $v_r = \Delta \ell_r$, and trust precision $\beta_r$ (inverse idiosyncratic rollout noise).

If $\pi_{\text{pol}}$ oversamples one flank $9\times$, naive pooling weights that flank $9\times$ more heavily. More generally, in the financial-market and expert-aggregation literature on a continuous **space of private information** (Hellwig 1980; Clemen & Winkler 1985; Ostrovsky 2012; Lambert, Ostrovsky & Panov 2018), $k$ signals with source-overlap Gram matrix $G_{rb} = \langle k_r, k_b \rangle$ and idiosyncratic precisions $B = \mathrm{diag}(\beta_r)$ aggregate via the normal equations $(K^\top B K + \lambda I) w = K^\top B v$. For $N$ rollouts with pairwise source overlap $\rho = \langle k_r, k_b \rangle$ and signal-to-noise ratio $\beta/\lambda$, the combined effective weight is:

$$W_{\text{eff}}(N, \rho, \beta) \;=\; \frac{N \beta}{\lambda + \big(1 + (N-1)\rho\big)\beta}, \qquad \log \mu^*_k(y) \;=\; \log p_0(y) \;+\; \sum_{r=1}^k w_r \, \log \frac{q_r(y)}{p_0(y)} \;+\; \text{const}.$$

This single law unifies two regimes: (i) **mutual support** ($\beta \ll \lambda$): when rollouts carry high independent noise, agreeing rollouts reinforce each other linearly ($W_{\text{eff}} \approx N\beta/\lambda$); and (ii) **deduplication** ($N\rho\beta \gg \lambda$): when rollouts repeatedly query the same deterministic conditional ($\rho \to 1$), total weight saturates at $1$ ($w_r \propto \beta_r / c_r$ with crowding $c_r = \sum_{b=1}^k \max(0, \langle k_r, k_b \rangle)^\gamma$). Across partially overlapping rollouts ($0 < \rho < 1$), $L$ layers of residual self-attention implement sequential market clearing (Ostrovsky 2012; Vladymyrov et al., `arXiv:2402.14180`), pricing in the shared consensus at layer $l$ and trading on the orthogonal residual at layer $l+1$.

---

## 3. Consensus Mechanics, Computation as Evidence & Search Policy

### 3.1 Step-by-Step Consensus & Landmark Memory

The Consensus holds a Blackboard of **Landmark Scenarios** $\{S_0, S_1, \dots, S_M\}$ (cached partial assignments from which rollouts fork):
1. **0-Step Anchor ($k=0$):** $\theta_{\text{amp}}$ emits baseline logits $\ell^{(0)}(S_j)$ at weight $\beta_0$.
2. **Iterative Attention Deduplication ($l = 1 \dots L$):**

$$\text{residual}^{(l)}_r \;=\; v_r \;-\; \mathrm{Attn}_l\!\big(q = k_r, \; K = \{k_{r'}\}, \; V = \{h^{(l-1)}_{r'}\}\big), \qquad \ell^{(l)} \;=\; \ell^{(l-1)} \;+\; \eta_l \sum_{r=1}^k \beta_r \, W_{\text{out}} \, \text{residual}^{(l)}_r.$$

3. **Branching Mixtures ($S_{j, z} = S_j \cup \{Z = z\}$):**

$$\mu^*_k(Y \mid S_j) \;=\; \sum_{z} \mu^*_k(Z = z \mid S_j) \, \mu^*_k(Y \mid S_{j, z}).$$

4. **Computation as Evidence (Null Search):** When all $M$ rollouts are refuted by forward checks, cumulative refuted model mass $C_M \in [0, 1)$ updates $P(\text{sol} \mid \text{null})$ with recall floor $\eta_{\max} \in [0, 1]$:

$$\operatorname{logit} P(\text{sol} \mid \text{null}) \;=\; \operatorname{logit} p_0 \;+\; \log\!\big[(1 - \eta_{\max}) + \eta_{\max}(1 - C_M)^\theta\big].$$

### 3.2 Search Policy (Key-Space PUCT + VoI)

$\pi_{\text{pol}}$ scores candidate variables or moves $v$ by combining value/prior terms with trustworthy stiffness and key-space novelty:

$$\mathrm{Score}_{\text{pol}}(v \mid \text{prefix}, \text{Consensus}) \;=\; \hat{Q}_\mu(v) \;+\; c_{\text{puct}} \, \pi_0(v) \, \underbrace{I_{\text{goal}}(v)}_{\substack{\text{Impact / sensitivity on} \\ \text{goal variables } X_G}} \;\times\; \underbrace{\beta_\theta(v \mid \text{prefix})}_{\substack{\text{Trustworthy Stiffness:} \\ \text{constrained \& reliable in } q_\theta}} \;\times\; \underbrace{\frac{1}{\sqrt{1 + \mathrm{Crowding}(k_v, K_{<r})}}}_{\substack{\text{Key-Space Novelty:} \\ \mathrm{Crowding} = \sum_{r'<r} \beta_{r'} \max(0, \langle k_v, k_{r'} \rangle)}}.$$

---

## 4. Local Credit Assignment (Exact for the $\gamma = 0$ Pool)

For the $\gamma = 0$ log-linear pool (`STUDY-LIR.md` §3c–3f), outcome log-loss equals the **PDG Inconsistency Increment** from clamping reality $\delta_x$:

$$-\beta_0 \log \mu^*_k(x) \;=\; \langle\langle M_k + \delta_x \rangle\rangle \;-\; \langle\langle M_k \rangle\rangle.$$

By the envelope theorem (attending $+\beta_r$ on $\delta_x$ and $-\beta_r$ on a pinned copy of $\mu^*_k$), each rollout's exact local credit is its **Leave-One-Out (LOO) gain**:

$$r_{\mathrm{LOO}}(r) \;=\; \log \mu^*_k(x) - \log \mu^*_{k \setminus \{r\}}(x) \;\approx\; \frac{\beta_r}{\beta_0}\Big(\ell_r(x) - \mathbb{E}_{\mu^*_k}[\ell_r(X)]\Big).$$

Learned multi-layer aggregators ($\theta_{\text{agg}}$) approximate this by training end-to-end on $-\log \mu^*_k(x)$.

---

## 5. Evidence So Far (Rounds 1–8, 3 Seeds) & Paradigm Comparison

Prior work shows adaptive unmasking order lifts Sudoku accuracy from $<7\%$ to $\sim 90\%$ (Kim et al., `arXiv:2502.06768`). Across permuted cellular automata (`E-CA`), $6\times 6$, $9\times 9$, and `Sudoku-Extreme` (`E-SUD`, `R4-E2`, `R5-E3`, `R6-E2`, `R7-E2`, `R8-E2`), $4\times 4$ and $6\times 6$ Micro-Stratego (`R4-E1`, `R5-E2`, `R6-E3`, `R7-E3`, `R8-E3`), $6\times 6$ coupled ShipHunt (`R5-E1`, `R6-E1`, `R7-E1`, `R8-E1`), correlated multi-variable probes (`E-LA`), and RL honesty (`R4-E3a`, `R4-E3b`, `R5-E1` Part B; full reports at mats.ekelvin.com/experiments):
- **Holds:**
  1. *Any-order search, Landmark beam branching & anytime monotonicity:* With the 1-step law learned ($\mathrm{KL} \le 8\times 10^{-5}$ on $9\times 9$ Sudoku naked singles; $0.9999$ battle accuracy on $4\times 4$ and $6\times 6$ Stratego; $r=0.9995$ with exact Bayes on $6\times 6$ ShipHunt), search cuts one-pass/anchor belief KL by **$89.2\%$** ($6\times 6$ ShipHunt 32-read `P-FullSupport` stream under anchor-aware `looped_attn_T3`, `0.280 -> 0.0303` nats, `R8-E1` `D1.1` PASS; **$88.7\%$** at $K=24$, `0.280 -> 0.0317`, `R7-E1` `D1.1` PASS; **$74.0\%$** on open-loop bounded patches, `0.280 -> 0.073`, `R5-E1` `D1.2` PASS), **$48.2\%$ `[46.9%, 49.5%]`** ($6\times 6$ Stratego 16 marginal 4-ply window rollouts vs. the $t-4$ anchor, `0.185 -> 0.096` nats, `R6-E3`/`R7-E3` `D0` PASS; vs. the strong current-turn anchor `0.0488`, naive `raw_count` degrades KL by `+0.591` and fixed-step `ataraxos_mmd` by `+4.73`, whereas `scaled_mix` and early-stopped `looped_attn_T3-K3c` never degrade any $K$ and trim KL to `0.0463`, `R8-E3`), **$46.3\%$** ($6\times 6$ Sudoku), **$39.3\text{--}42.2\%$** ($9\times 9$ Sudoku `P3`/`P6`), and **$45.3\%$** (`Sudoku-Extreme` 17–22 givens unique-solution boards at $32\times(E+1)$ under `P8_full` `chk_mix`, `1.178 -> 0.644` nats and $51\%$ witnesses, `R8-E2`; **$40.8\%$ `[29.6%, 52.0%]`** at $16\times(E+1)$ under `P8_beam` `vote_chk`, `1.178 -> 0.698` nats, `R7-E2`). Crucially, **anytime monotonicity holds strictly across every step and seed** when the anchor and forward checks are respected: on $6\times 6$ ShipHunt (`R8-E1` `D1.1` PASS), anchor-aware `looped_attn_T3` decreases KL at **every step $K \in \{0, 1, 2, 4, 8, 16, 24, 32\}$ in every seed** (`0.2797 -> 0.2305 -> 0.1909 -> 0.1504 -> 0.1026 -> 0.0559 -> 0.0308 -> 0.0303`) and across all four nested root strengths $M_0 \in \{2, 6, 12, 18\}$, while anchor-free `ataraxos_mmd` (`0.1084 -> 0.1661`) and `scaled_mix` (`0.1042 -> 0.1260`) turn back up at large $K$ and `kc_K1` degrades at $K=1$ (`0.2797 -> 0.3523`). On `Sudoku-Extreme` (`R8-E2` `D2.1` PASS), check-gated pooling (`chk_mix` and `vote_chk`) strictly decreases test KL across **all $36/36$ step transitions** $m \in \{0, 1, 2, 4, 8, 16, 32\}$ (`chk_mix`: `1.178 -> 1.036 -> 0.979 -> 0.920 -> 0.809 -> 0.734 -> 0.644`; witnesses `14% -> 43% -> 51%`), whereas unweighted `mix` jumps to `5.98` nats at $m=1$. On `Sudoku-Extreme` (`R7-E2` `E2-G0` PASS), the true solution path hits **$4.89$ stalls** ($H \ge 0.25$) per board with the true digit inside $\{d : q_{z^*, d} > 0.05\}$ on **$96.6\% / 98.0\% / 97.8\%$** of Stalls 1–3; level-synchronous multi-stall beam search (`P8_beam`, $W=16, L \in \{3,4\}$) keeps the true path in the beam across all $L$ waves on **$83.3\%$ (`91% / 71% / 88%`)** of test boards and beats both flat `P3` (**$44.5\%$ vs. $40.2\%$ witnesses, `+4.3 pp [2.0, 6.5]`**) and 1-stall `P6` (**vs. $38.9\%$, `+5.5 pp [0.06, 11.0]`**). Recall-floored null search cuts deep-contradiction Sudoku loss by **$95\text{--}99.6\%$**.
  2. *Salience keys track graded source overlap (`R6-E1` `D1` PASS & `R7-E3` `D1(1)` PASS):* Building keys from a rollout's **belief-shift footprint** $s_r = |\Delta\ell_r| / \|\Delta\ell_r\|_2$ (no patch or flank indices given) resolves the Round-5 common-mode collapse: on $6\times 6$ ShipHunt (`R6-E1` `D1` PASS), `K3` (salience) achieves Pearson **$0.880$ `[0.876, 0.884]`** with true cell overlap, same-vs-across cosine gap **$0.967$ `[0.936, 0.999]`**, and Spearman **$1.00$** across `0, 1, 2, 3, 4, 6` shared cells (`0.01 -> 0.27 -> 0.50 -> 0.67 -> 0.77 -> 0.96`, vs. raw `K1` $0.382$). On $6\times 6$ Stratego (`R7-E3` `D1(1)` PASS), bag-centered salience keys (`K3c`) lift the within-vs-across flank key cosine gap **$165\times$** from $0.004$ (`R5-E2`) to **$0.661$ `[0.642, 0.679]`** against a $0.28$ bar (uncentered `K3` sits at $0.234$ due to $0.56$ physical cone overlap).
  3. *Space of Information — Weight-Tied Iterative Clearing, Policy Independence & Three-Regime Weight Law (`E-LA`, `R5-E1`, `R6-E1`, `R6-E3`, `R7-E1`, `R7-E3`, `R8-E1`):* Consistent with continuous-signal market aggregation (Lambert, Ostrovsky & Panov 2018) and in-context iterative solvers (Vladymyrov et al., `arXiv:2402.14180`), multi-layer residual attention implements $(K^\top B K + \lambda I)^{-1} K^\top B v$:
     - **Weight-tied iterative clearing (`R7-E1` `D1.1` PASS & `R8-E1` `D1.1` PASS):** A single shared Transformer block applied for $T=3$ iterations (`looped_attn_T3`, `39,428` params = **$37.1\%$ of `attn_L3-K3`**) matches untied 3-layer `attn_L3-K3` (**KL ratio `0.967`**, `0.0317` vs. `0.0328` nats at $K=24$; `0.0303` vs. `0.0331` at $K=32$) and cuts preference-skew belief $\mathrm{JSD}$ to **$0.00575$ `[0.00519, 0.00631]`** (`R8-E1`) and **$0.0066$** (`R7-E1`, **$7.05\times$** below `scaled_mix` $0.0462$ and **$7.84\times$** below `ataraxos_mmd` $0.0516$), beating `scaled_mix`, `mmd`, `kc_K1`, and `kc_K3` on both KL and JSD with all 8 paired 95% CIs $> 0$. Meanwhile, Jacobi-preconditioned normal-equation iteration with a 2-parameter precision-aware ridge $\lambda(\beta_r) = \lambda_c \sigma(a - b\beta_r)$ (`jacobi_lin_T4_prec`, **$814$ params**, `R8-E1` `D2.1`) learns tier-specific ridges ($\lambda \approx 0.35, 1.37, 2.20$ across $\tau \in \{0.3, 1.5, 2.5\}$), cuts `R6`'s `looped_lin_T4` KL by **$38.9\%$** (`0.0677 -> 0.0414`), and satisfies all four three-regime weight-law clauses (`W(16)/W(1) = 1.164 [1.147, 1.182]` at $\tau=0.3$, `W(8)/W(1) = 1.150 < 1.765 < 2.501`, and `W(16)/W(1) = 2.816` at $\tau=2.5$).
     - **Full-support & open-loop policy independence (`R6-E1` `D2.1` PASS, `R7-E1` `D2.1` PASS, `R7-E3` `D1(2)` PASS, `R8-E1` `2x2` PUCT):** On $6\times 6$ Stratego (`R7-E3` `D1(2)` PASS), `attn_L3-K3c` cuts Left/Right preference-skew $\mathrm{JSD}$ by **$10.83\times$ vs. `scaled_mix`** and **$10.16\times$ vs. `ataraxos_mmd`** while beating all baselines in exact-posterior $\mathrm{KL}$ (all 95% CIs $> 0$). Under open-ended search on $6\times 6$ ShipHunt (`R7-E1` `D2.1` PASS & `R8-E1` `2x2` factorial), 4-wave key-space PUCT lifts cell coverage from $82.8\%$ (`P-OpenLoop`) and $92.00\%$ (`K1 + R6 score`) to **$95.80\%$ (`K1 + R7 score`), $96.51\%$ (`K3 + R6 score`), and $98.76\%$ `[98.59%, 98.93%]` (`K3 + R7 score`)**, cutting open-loop $\mathrm{JSD}$ to **$0.00819$** and $\mathrm{KL}$ to **$0.0399$** (both key representation `+3.0 to +4.5 pp` and the multi-step novelty/variance score `+2.3 to +3.8 pp` contribute with 95% CIs $> 0$).
     - **Three-regime weight law (`R5-E1` `D2.2`, `R6-E1` `D3`, `R7-E1` `D1.1/D1.2`, `R8-E1` `D2.1`):** Across three sensor noise tiers on $6\times 6$ ShipHunt, 16 identical clean reads ($\tau=0.3$) receive **$1.07\times$** (`looped_attn_T3`), **$1.08\times$** (`attn_L3`), and **$1.16\times$** (`814`-param `jacobi_lin_T4_prec`, down from `1.45x` under tier-independent ridge) the weight of 1 read (**deduplication**, cutting MSE by **$38.5\%$ vs. `ataraxos_mmd`** and **$53.7\%$ vs. `mix`**), while noisy independent reads accumulate weight monotonically (**mutual support**: $W(8)/W(1) = \mathbf{1.06 < 1.99 < 3.44}$ for `looped_attn_T3` and $\mathbf{1.15 < 1.77 < 2.50}$ for `jacobi_lin_T4_prec`, reaching $W(16)/W(1) = \mathbf{2.82\text{--}4.52}$ at $\tau=2.5$).
  4. *Self-trust $\theta_{\text{trust}}$ vs. prediction entropy (`R4-E1`, `R5-E2`, `R6-E3` `D2(1)` PASS, `R7-E3` `D2(1)` PASS):* Training $\theta_{\text{trust}}$ on real-world excess surprise achieves Spearman **$+0.418$** ($4\times 4$ Stratego), **$+0.247$** (`R6-E3` homogeneous bags), and **$+0.347$ `[0.320, 0.375]`** (`R7-E3` `D2(1)` PASS on mixed-reliability bags; **$+0.369$ `[0.346, 0.391]`** within $(s,a)$ pairs) with $-\mathrm{KL}(Q^* \Vert q_r)$, vs. **$-0.395$** overall and **$-0.551$** within pairs for stiffness $-H(q)$ (a **$+0.743$ overall / $+0.920$ within-pair gap**, and **$+0.250$** within entropy strata), while `trust_gate` and `trust_pool` cut action-value MSE by **$63.2\text{--}63.9\%$ vs. `stiff_pool`** (`0.0139 vs. 0.0388`).
- **Negative / Open (Targeted in Round 9):**
  1. *Genuine strong-root regime ($\mathrm{KL}(0) \le 0.035$) & precision-aware 1-step `econ_prec_K3` on $6\times 6$ ShipHunt (`R8-E1` `D1.2`, `D2.1`, `D2.2`):* In `R8-E1`, 18 anchor cells left the root at $\mathrm{KL}(0) = 0.1444$ (above the $\le 0.050$ strong-root bar), where `ataraxos_mmd` still improved the root at $K=16$ (`-0.0665` before losing gain to `-0.0457` at $K=24$, whereas `looped_attn_T3` improved monotonically to `-0.0978`). Meanwhile, 1-step `econ_kc_K3` with a single tier-independent scalar $\lambda$ gave $W(16)/W(1) = 1.417$ at $\tau=0.3$ (since a single $\lambda$ cannot both deduplicate $\tau=0.3$ reads and accumulate $\tau=2.5$ reads), and `jacobi_lin_T4_prec` missed the `0.040` `P-FullSupport` KL bar by `0.0014` (`0.0414`). Round 9 (`R9-E1`) tests a genuinely strong root ($M_0 = 32$ clean cells or a 5-tile clean partition root with $\mathrm{KL}(0) \le 0.035$, where `ataraxos_mmd` and `mix` degrade $\mathrm{KL}(16) > \mathrm{KL}(0) + 0.015$ while `looped_attn_T3` never degrades any $K$) and equips 1-step `econ_prec_K3` with the 2-parameter precision-scaled ridge $\lambda(\beta_r) = \lambda_0 \sigma(a - b\beta_r)$.
  2. *Greedy lead-chain / forced-cell completion in full-budget `Sudoku-Extreme` beam & unresolved-board check sharpness (`R8-E2` `D1` & `D2.2`):* In `R8-E2`, `P8_full` spent $99.5\%$ of the $16(E+1)$ pass budget (`D1.0` PASS), but replacing `R7-E2`'s greedy lead chain ($r=0$) and forced-step `argmax` ($H < 0.25$, `P3_thresh`) with pure multinomial sampling on every empty cell yielded $42.8\%$ witnesses on the standalone $m=16$ run ($45.7\%$ on the $m=32$ 16x prefix) vs. `R7`'s $44.5\%$ (cumulative $1 - 0.985^{28} \approx 35\%$ slip rate on forced cells after Wave 4). On unresolved (no-witness) boards (`D2.2`), seeds 101 and 102 had exact `0.000` KL change vs. `P0`, while seed 103 degraded by `+0.065` due to $7\%$ false accepts at $\text{chk} \ge 0.5$. Round 9 (`R9-E2`) combines full-budget D'Hondt allocation with `R7`'s greedy lead chain ($r=0$) + forced-cell `argmax` ($H < 0.25$) + $q > 0.05$ stall sampling (`P8_hybrid`) and sharp check-threshold / LOO self-consistency gating so unresolved-board safety (`KL(16) - KL(P0) <= +0.005`) holds across all 3 seeds.
  3. *Current-turn anchor saturation & 4-ply horizon bias in $6\times 6$ Stratego (`R8-E3` — Lane Closed):* When anchored at the current-turn one-pass belief $\ell_t$ (`KL = 0.0488`), 4-ply window rollouts (`KL = 0.080`) carry almost no additional information: fitting `108k`-param attention pools to hard piece labels on 2,000 positions without held-out stopping memorised calibration positions, while held-out early stopping kept the pools at their safe initialization (`KL(16) - KL(0) = -0.0025`, matching 1-param `scaled_mix` `0.0488 -> 0.0463`). In `D2`, myopic depth-1 claims (`MSE = 0.040`) beat depth-4 playouts (`0.045`) due to 4-ply horizon bias. The Stratego lane (`E-MS`) is complete and needs no Round-9 spend.

| Paradigm | How It Maps to This Architecture | What This Design Adds |
| :--- | :--- | :--- |
| **1. Next-Token LLM** | Fixed order $v_m = m$, $k=1$ rollout, $0$ landmarks. | $k \cdot n$ calls at context $\le n$; separates $\theta_{\text{real}}$ from search. |
| **2. AlphaZero / MuZero MCTS** | Landmarks = tree nodes; $\pi_{\text{pol}}$ = PUCT; $\theta_{\text{amp}}$ = value/policy head. | Cross-branch key deduplication $(k_r, v_r)$ and $\theta_{\text{trust}}$ hallucination damping. |
| **3. Ataraxos (Sokota et al., *Nature* 2026)** | Unmasks hidden pieces $Z^{(r)}$, rolls out moves, and takes a 1-step MMD log-odds update (using an exact simulator). | Any-order belief unmasking + multi-landmark search over a **learned** world model. |

---

## 6. Alignment and consequences

1. **The separation claim (Parameter Separation vs. Token-Level Masking & Fused RL):** Reward trains actions and search ($\chi_{\text{action}} = 1$), never world beliefs ($\chi_{\text{belief}} = 0$). Fused RL ($\chi = \mathbf{1}$) distorts beliefs along the utility gradient, and **token-level loss masking on a shared trunk (`Token-Masked` / DCPO) still leaks reward pressure via internal representation spillover**, whereas structural parameter separation (`LIR-Masked`) keeps beliefs exact:

$$\mathbb{E}_{\text{fused}}\big[\langle \mu_{\text{belief}} - P^*, \, \nabla U \rangle\big] \;=\; \frac{\lambda}{2}\|\nabla U\|^2 \;\ge\; 0, \qquad \mathbb{E}_{\text{masked}}\big[\langle \mu_{\text{belief}} - P^*, \, \nabla U \rangle\big] \;=\; 0.$$

This is now established cleanly across both small networks with exact Bayesian ground truth and 8B LLMs:
- **Small-Net Exact-Bayes Proof (`R5-E1` Part B, `669k`-param coupled $6\times 6$ ShipHunt, 3 seeds):** With an action-entropy bonus (`0.02`), GRPO reward rose from $0.500$ to **$0.789$ (`Fused`), $0.812$ (`Token-Masked`), and $0.798$ (`LIR-Masked`)** against a $0.830$ exact-Bayes ceiling (**$90.2\%$ normalized reward gain**). Under this rising reward:
  - **`LIR-Masked`:** Mean belief drift **$0.0000$** ($\ell_\infty = 0.0000$ across all $500 \times 36$ cells in every seed), Pearson with exact Bayes **$0.9994 \to 0.9994$**, **unwarranted false-note sycophancy $0.000$** (raw $5.3\%$ equals exact Bayes's $5.5\%$), and **zero-retraining `faith_swap` action flip $100.0\%$ (`1.000`)** across all 3 seeds.
  - **`Fused`:** Desire-projection bias $\Delta_{\text{desire}}$ jumped **$+0.471$ `[0.393, 0.549]`**, false-note sycophancy hit **$83.1\%$** ($77.5\%$ unwarranted), and Bayes correlation fell from $0.9994$ to $0.722$.
  - **`Token-Masked` (DCPO baseline):** Despite zero RL gradient on belief tokens, shared-trunk representation drift pushed beliefs by **$0.0527$ per cell ($14\times$ the truth-only control's $0.0038$)** directly along the goal-utility vector ($\Delta_{\text{desire}} = +0.117$, utility component $+0.100$).
- **LLM Scale (`R4-E3b` on `Qwen3-8B` Manifold forecasting & `R4-E3a` on `1.7B` ShipHunt):** On 2,500 real Manifold markets (`R4-E3b`), `Fused` RL drove $\Delta_{\text{desire}}$ up by **$+0.490$ `[0.487, 0.494]`** and sycophancy to **$99.9\%$**; `Token-Masked` still suffered **$0.200$ belief drift** and $7.1\%$ sycophancy via shared LoRA weights; while `LIR-Masked` held $\Delta_{\text{desire}}$ change at **$-0.0001$** and sycophancy at **$1.2\%$**.

2. **The fixed target:** Under §2.1, search converges to $\mu^*(\theta, U)$—a function of $(\theta, U)$ only. An opaque search heuristic $\pi_{\text{pol}}$ cannot change *what* is believed on covered variables, only *how fast* $\mu_k \to \mu^*(\theta, U)$.

3. **Honest, auditable probabilities:** Beliefs are scored on realized outcomes, failed search is calibrated negative evidence, and inconsistency $\mathrm{Inc}_\theta(\mu)$ is an explicit, readable number.

4. **Consequences, risks, and mitigations:** Risks include reward leaking through $\theta_{\text{amp}}$ or $\theta_{\text{trust}}$, search exploiting world-model errors, false confidence, and **dual use** (better search is a capability). We commit to publishing quantitative audits of **belief drift** ($\Delta_{\text{belief}}$) and **policy sensitivity** ($\Delta_{\text{pol}} = \mathrm{JSD}(\mu^*(\pi_1) \parallel \mu^*(\pi_2))$) on every experiment, and clearing release decisions with the MATS Research Manager.

