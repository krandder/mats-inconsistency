Model: Gemini 4 Argon (High)

### 1. Strongest Objections to Worlds A, B, and C

* **World A (Beauty Contest): A graph-blind shortcut passes 99.3%–99.95% of A1 variance.**
  The proposal reports reference gaps only for **A2** (the omniscient post-hoc aggregator of all quotes, lines 192–200), never for **A1** (the trader quoting on private signals—Kelvin’s actual question). On `worlds_check.py`’s own prior, trader $i$’s zero-coordination Bayes posterior $c_i^\top x_i = \mathbb{E}[v \mid x_i]$ correlates with the exact equilibrium quote $w_i^\top x_i$ at **0.9990 ($r=0.3$), 0.9918 ($r=0.6$), and 0.9629 ($r=0.85$)**. Worse, a **graph-blind shortcut** that uses only $(x_i, r)$—shrinking private signals uniformly toward public signals and the prior mean $0$ while *completely ignoring the other $N-1$ traders’ listening graph*—captures **99.95% ($r=0.3$), 99.74% ($r=0.6$), and 99.30% ($r=0.85$)** of equilibrium quote variance (randomly rewiring all other traders’ sources changes $w_0^\top x_0$ by just $0.05\%\text{--}0.70\%$ relative MSE). Why? In $\operatorname{Cov}(x_{is}, x_{js'}) = \operatorname{Var}(v) + \mathbf{1}[s=s']\sigma_s^2$, the fundamental variance $\operatorname{Var}(v)=1$ dominates $\sigma_s^2 \in \{0.1, 0.5\}$. Under single-sample payoff loss, a $0.26\%$ graph effect is buried $>100\times$ below sample noise.
* **World B (Kyle-LOP): Wrong action space for Kelvin’s prompt, inflated strategic gap, and noisy training.**
  Insiders output *order quantities* $d_i$ rather than prices, and the MM’s pricing task is a trivial 1–2 feature regression. Crucially, the claimed `0.61` correlation between strategic and Bayes forms (line 252) and `100%` public sign-flip rate (line 251) are artifacts of comparing $\alpha_i^\top x_i$ against gross valuation $\mathbb{E}[v \mid x_i]$ instead of **net valuation over the public baseline** $\mathbb{E}[v - \mathbb{E}[v \mid y] \mid x_i]$. Because the MM sees $y$, even a non-strategic trader ($N=1$) subtracts $\mathbb{E}[v \mid y]$. Against $\mathbb{E}[v - \mathbb{E}[v \mid y] \mid x_i]$, mean correlation on `worlds_check.py` jumps from **0.6079 to 0.9648** ($R^2 = 93.1\%$). The `9.9%` non-public sign flips carry a median of just **3.9%** of L1 weight (`res['ws']`, computed on line 116 of `worlds_check.py` and omitted from printout). Finally, training on realized profit $d(v - P_{-i}) - \beta_D d^2$ (line 220) already requires solving the oracle LOP equilibrium for $(P_{-i}, \beta_D)$, so keeping $(v, u)$ sample noise instead of regressing on $\mathbb{E}_h[v - P_{-i} \mid x_i]/(2\beta_D)$ is pure variance injection.
* **World C (Sequential Dealer): Target is mathematically wrong as coded and intractable as stated.**
  When traders observe $m_i \in \{1..3\}$ shared-noise sources and report only a **scalar** posterior price $p_t$, $p_t$ irreversibly mixes sources and is *not* a sufficient statistic for per-source state; line 292’s claim of "GLS through per-source sufficient statistics, $O(T\cdot k)$" is false. `worlds_check.py` (line 120) cheats by computing $p_{N-1} = \mathbb{E}[v \mid x_1, \dots, x_{N-1}]$ from all predecessors’ raw multi-dimensional vectors instead of scalar prices. Even with that cheat, median $\text{KL}(\text{Bayes} \parallel \text{indep})$ is $0.047$ nats—repeating W2c’s failed C7 regime.

---

### 2. Literature, Math, and Implementation Errors

1. **Missing features in `worlds_check.py` and scalar-only tests in `lop_check.py`:** None of §2.0’s four "interfering" signals—(a) public relay of a shared source, (b) technical signal $\eta_s + \text{noise}$, (c) signal on $u$, (d) $u$ correlated with $v$—are implemented in `worlds_check.py`. `lop_check.py` only tests 1D signals (`ks = [1]*n`).
2. **Triviality of public signals in LOP (`worlds_check.py` lines 45, 102):** Appending the MM’s signal $y = \theta_M$ to every $\theta_i$ adds zero cross-trader strategic interaction. On the $y$-rows of $\Lambda \alpha = \gamma \operatorname{Cov}(\theta, v \mid \theta_M)$, both $\operatorname{Var}(\theta \mid \theta_M)$ and $\operatorname{Cov}(\theta, v \mid \theta_M)$ vanish identically, leaving $(\Sigma_{ii}\alpha_i)_y = \operatorname{Cov}(y, d_i) = 0$: each trader simply orthogonalizes $x_{i,\text{priv}}$ against $y$.
3. **Conflating pooled vs. individual Bayes (Lines 65, 245):** Comparing $\Lambda^{-1}\Sigma_{\theta v}$ against $\Sigma_{\theta\theta}^{-1}\Sigma_{\theta v}$ conflates trader $i$’s individual Bayes weight ($\Sigma_{\text{diag}}^{-1}\Sigma_{\theta v}$) with the *omniscient pooled regression* across all traders’ signals ($\Sigma_{\theta\theta}^{-1}\Sigma_{\theta v}$). Moreover, whenever $P \ge 1$ public signals are appended to $\theta_i$ (as in `worlds_check.py`), $\Sigma_{\theta\theta}$ is **singular** (rank-deficient by $(N-1)P$).
4. **Morris-Shin formula error under World A’s prior (Lines 25, 185):** Crux 2 states the public weight follows $\alpha/(\alpha+\beta(1-r))$. That requires an improper diffuse prior (`D[0,0] = 1e6` in `worlds_check.py` line 75). Under World A’s prior $v \sim \mathcal{N}(0,1)$ ($\tau_v=1$, line 124), the prior mean $0$ is a common-knowledge public anchor, so the equilibrium weight is $\alpha / (1 + \alpha + \beta(1-r))$ ($0.3125$, not $0.4545$, at $\alpha=1,\beta=3,r=0.6$).
5. **Broken hidden-level mixture claims (Lines 147–148, 170, 220):**
   * Per-trader noise levels in `worlds_check.py` (line 18) yield $|H| = 3^S 3^P 2^N 3^{N/2} \gg 10^{30}$, not $\le 10^3$.
   * In B1 (line 220), $\frac{\mathbb{E}[v - P_{-i} \mid \text{ctx}]}{2\mathbb{E}[\beta_D \mid \text{ctx}]} = \frac{\sum_h P(h \mid \text{ctx})\beta_{D,h}\alpha_{i,h}^\top x_i}{\sum_h P(h \mid \text{ctx})\beta_{D,h}}$ is a **$\beta_{D,h}$-weighted** mixture, contradicting line 147’s unweighted $H$-mixture.
   * Even at $R=0$, $P(h \mid x_i) \propto P(h)\mathcal{N}(x_i; 0, \Sigma_{ii}^{(h)})$ depends on $x_i x_i^\top$, making optimal hidden-level actions nonlinear in $x_i$.
   * Line 148 ("Training needs no mixture/solve") fails because generating incumbents’ $\bar{a}_{-i}$, $(P_{-i}, \beta_D)$, or history rounds requires solving the equilibrium every training episode anyway.

---

### 3. Better Modified World: High-Interference Beauty Contest / Kyle Hybrid ($A^+ / B^+$)

Fix the prior so graph structure **dominates** equilibrium weights:
1. **Regime shift:** Set $\sigma_s^2 \in \{1, 4, 9\} \ge \operatorname{Var}(v)=1$ and include explicit **public relays** $y_k = v + \eta_{s_k} + \zeta_k$ and **signed loadings** $x_{is} = v \pm \eta_s + \epsilon_{is}$ (or technical signals $\eta_s + \epsilon_{is}$). Now sharing source $s$ creates strong error correlation ($\sigma_s^2 \gg 1$) that must be deduplicated against public relays and other listeners.
2. **Pricing equivalence for LOP:** Have B1 traders output their **strategic reservation price / shadow valuation** $\tilde{p}_i = \mathbb{E}[v \mid y] + 2\beta_D d_i = \mathbb{E}[v - \beta_D \sum_{j\ne i} d_j - \beta_D u \mid x_i]$, putting Worlds A and B in the exact same price units while preserving LOP’s exact sign-flip thresholds.
3. **Rao-Blackwellized loss:** Since generating $\bar{a}_{-i}$ or $(P_{-i}, \beta_D)$ already runs the solver per episode, train directly on expected conditional loss $\mathbb{E}_h[\ell(a_i) \mid x_i]$.

---

### 4. Ranking and First Experiment

**Ranking:** **1. Modified World $A^+$** (direct pricing, layer-by-layer level-$k$ probe) $\succ$ **2. Modified World $B^+$** (sharp sign-flip thresholds via reservation prices) $\succ$ **3. World C** (drop or restrict to discrete C2).

**First experiment:** Before training any transformer, run the **graph-rewiring variance audit** on $A^+$ and $B^+$: require that randomly rewiring other traders’ listening graph (holding focal trader $i$’s signals fixed) accounts for **$\ge 15\%$ of focal action variance** (up from $0.26\%$ in current World A). Then train a 4–6 layer PFN on Known-Level $A^+$ and evaluate on **Commonality Matched Pairs** and layer-wise **level-$k$ linear probes**.
