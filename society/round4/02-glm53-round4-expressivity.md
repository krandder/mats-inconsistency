# R4 Deliverable — Search-Algorithm Expressivity, `R13` Supervised Imitation, `R14` Unsupervised Discovery

**Member 19, Multi-Harness Society of Thinkers, Round 4** · Repo: `futarchy-fi/inconsistency-search` · Scope: audit of Member 18's `Theorem R4.1`–`R4.3`, specification inputs for `PREREG-R13-ALGO-IMITATION.md` and `PREREG-R14-UNSUP-DISCOVERY.md`, rubric + priors. All claims below are checkable against `src/search/kv_tree_search.py` at the cited lines and against the R11/R12 gate logs quoted in the round brief.

---

## 0. TL;DR — Verdicts on Kelvin's Two Questions

1. **Q1 (can our general algorithms learn MCTS-style subtree aggregation?):** As shipped in R11/R12, **no** — `Theorem R4.1` is **CONFIRMED (with two sharpened hypotheses)**: the 1-hop `n_exp` increment and the write-once `q_tree_bank` make exact subtree aggregation information-theoretically unreachable, and the unmasked Stage-2 edge attention is provably confused by polarity-swapped isomorphic bags. `Theorem R4.2` is **CONFIRMED with 5 refinements**: 3 layers suffice *if* Layer 1 carries three parallel heads, if the descendant mask is reflexive, if $\epsilon$ is removed from log-cardinality, and if the Layer-3 softmin decision is threshold-shifted by $\log(D{+}1)/\beta_{\max}$; crucially R4.2 is a *representation-with-oracle-mask* theorem — the transitive closure is computed by the bank, not the weights. That is exactly why Kelvin's proposed supervised benchmark (`R13`) is the right falsification instrument.
2. **Q2 (what can even `GLCSearch v2` not represent?):** After adding the audited fixes (`V2.1′`–`V2.5′` plus proposed `V2.6`–`V2.8`), the residual non-representable classes are: (i) arbitrary-depth *alternating* semiring recursion computed **in weights** without the spine accumulator (conditional on $\mathsf{TC}^0 \neq \mathsf{NC}^1$); (ii) unbounded recursive self-embedding (DeepStack-style nested re-solving); (iii) parity/XOR pooling (outside the mean/LSE/min family); (iv) exact deterministic tie-breaking without recorded teacher noise; (v) continuum action spaces (Galanis-Market orders) absent discretization; (vi) history-dependent repetition draws when `canon()` is not ply-keyed. Full list in §1.5.
3. **Recommended path:** run `R13` exactly as specified in §2 (it double as an empirical test of R4.1–R4.3 and a debugging ladder), then `R14` (§3) with the Mechanistic Algorithmic Alignment Probes; pre-registered priors in §4 give ≈ 30–45% probability that R14 yields *certified emergent alignment* and ≈ 90% that it fixes the `B=128→256` multi-witness regression.

---

## 1. Audit & Critique of `Theorem R4.1`, `R4.2`, `R4.3`

### 1.1 `Theorem R4.1(a)` — Depth/Topology Barrier — **CONFIRMED**, two refinements

**Claim under audit.** Any $L$-layer 1-hop parent–child message-passing network has $\partial h_j^{(L)}/\partial v_k = 0$ for $\Delta d(j,k) > L$, hence worst-case MCTS value error $\ge \tfrac{D-L}{D+1}$.

**Verification.** By induction on layers, $h_j^{(\ell)}$ is a function of $\{z_k : \Delta d(j,k) \le \ell\}$; the gradient claim follows a.e. (exceptional set = measure-zero nondifferentiable points of max/ReLU; dead ReLUs only strengthen the claim). For the value bound: take the adversary tree to be a **single chain** below $j$ (nodes at relative depths $1..D$), values $z_k \in [0,1]$, set visible nodes ($\Delta d \le L$) to $0$ and invisible nodes ($\Delta d > L$) to $1$. With the reflexive convention $\mathrm{Desc}^+(j) = \mathrm{Desc}(j) \cup \{j\}$, $|\mathrm{Desc}^+(j)| = D+1$, invisible mass $= D - L$, so any estimator ignoring depth $>L$ has
$$\inf_{\hat Q} \sup_{z} |\hat Q(z_{\le L}) - Q^{\mathrm{MCTS}}(j)| \;\ge\; \frac{D-L}{D+1} \qquad (D > L).$$
**Refinement 1 (constant is tight only under two conventions).** The bound holds *as stated* iff (i) $\mathrm{Desc}$ is reflexive (root's own value counted, which the R12 `q_tree_bank` write-at-creation supports), and (ii) values are normalized to diameter $1$. For $z \in [-1,1]$ multiply by $2$: error $\ge \tfrac{2(D-L)}{D+1}$. For branching (non-chain) trees the binding parameter is the **deepest chain below $j$**, $D_{\mathrm{ch}}(j)$; the theorem should state the per-node bound $\tfrac{D_{\mathrm{ch}}(j)-L}{D_{\mathrm{ch}}(j)+1}$.

**Refinement 2 (the R12 hybrid leaks, but leaks wrongly).** `TwoStageTreeClearer` Stage 2 is *global*, so R4.1(a) does **not** literally apply to the full R12 operator — gradients from deep nodes are nonzero. The correct composite statement: the R12 operator escapes (a) only through the flat content-keyed channel, and that channel is exactly the one to which **R4.1(b)** applies. (a) and (b) are complementary lower bounds covering the local and global stages respectively; the hybrid inherits (b). This should be folded into the theorem statement so it cannot be evaded by pointing at Stage 2.

### 1.2 `Theorem R4.1(b)` — Unmasked Flat Attention Isomorphism Failure — **CONFIRMED under a polarity-blindness hypothesis that must be asserted against the code**

**Claim under audit.** On isomorphic-bag pairs $(\mathcal T^{(1)}, \mathcal T^{(2)})$ swapping depth-2 leaves $k_1, k_2$ (same clamped variable, $v_{k_1}=+1, v_{k_2}=-1$) between depth-1 parents $j_1, j_2$, flat attention output is swap-invariant while $Q(j_1)$ changes, error $\ge 1/2$.

**Verification.** With parents of two leaves $\{x, k_1\}$ vs $\{x, k_2\}$: truths $(x{+}1)/2$ vs $(x{-}1)/2$ differ by $1$; a swap-invariant estimator constant across the pair suffers $\ge 1/2$ (midpoint adversary). ✔.

**Edge case E1 (key polarity).** Swap-invariance requires the attention *logits* to be identical across the swap: $\langle q_{j_1}, k^{\text{edge}}_{k_1}\rangle = \langle q_{j_1}, k^{\text{edge}}_{k_2}\rangle = \langle q_{j_2}, k^{\text{edge}}_{k_*}\rangle$, which holds iff the edge keys of $k_1,k_2$ coincide (same variable, same clause incidence) **and** the keys are polarity-blind, and $q_{j_1} = q_{j_2}$ (isomorphic bags). In R11/R12 the Stage-2 gate $G_{r,r'} = \max(0, \langle k_r^{\text{edge}}, k_{r'}^{\text{edge}}\rangle)$ keys on literal-in-clause incidence; if literal *polarity* enters $k^{\text{edge}}$ then $k_1, k_2$ have different keys and the proof as stated collapses.
**Exact fix:** the theorem must carry the hypothesis "the edge-key map factors through variable identity only (polarity-blind)", plus the disjunctive escape: *either* polarity enters the key (breaking E1 but then depth-1/depth-2 structural identity is still unencoded and the counterexample is rebuilt with same-polarity leaves clamping two *different* variables with equal edge keys under a key collision — for learned keys, use a $\epsilon$-collision adversarial pair, error $\ge \tfrac12 - O(\epsilon)$), *or* the relational masks of R4.2 are required. Practically: audit `kv_tree_search.py` Stage-2 key construction; if polarity is present, the corrected lower bound is $\ge \tfrac12 - O(\|k_{k_1}-k_{k_2}\|)$, still damning.

### 1.3 `Theorem R4.2` — Constant-Depth MCTS/PUCT in `GLC-Search-Transformer v1` — **CONFIRMED with 5 refinements (E2–E6)**

**Verification skeleton.** With the 7-channel mask bank, multi-temperature heads, and the bank-maintained ancestor bitmask `M_tree[k,j] = 1[j ∈ Anc(k)]` (already at `kv_tree_search.py:241-242`), the construction is:

- **L1 (three parallel heads):** (h1) $\mathcal M_{\text{desc}}$-masked, $\beta = 0$ uniform attention with value channel $v$ ⟹ exact $\bar v(j) = \tfrac{1}{|\mathrm{Desc}^+(j)|}\sum_{k} \mathcal M_{\text{desc}}[j,k]\, v_k$; (h2) log-cardinality $\log N(j) = \log \sum_k \mathcal M_{\text{desc}}[j,k]$; (h3) $\mathcal M_{\text{parent}}$-masked fetch of $\log N(\mathrm{pa}(j))$. All read layer-0 tokens, so they coexist in one layer.
- **L2 (pointwise + sibling max):** $U(j) = \bar v(j) + c_{\text{puct}}\, P(j \mid \mathrm{pa}(j))\, \tfrac{\exp(\frac12 \log N(\mathrm{pa}(j)))}{1 + \exp(\log N(j))}$ is pointwise in L1 outputs (available at L2 input); L2's $\mathcal M_{\text{sib}}$ head at $+\beta_{\max}$ computes $\max_{s \in \mathrm{Sib}(j)} U(s)$; $\mu_{\text{edge}}(j) = U(j) - \max_s U(s)$ closes L2.
- **L3:** $\mathcal M_{\text{anc}}$ head at $-\beta_{\max}$ gives $S_{\text{path}}(j) \approx \min_{a \in \mathrm{Anc}(j)} \mu_{\text{edge}}(a)$.

The layer accounting in the round summary ("Layer 2 computes $\mu_{\text{edge}}$") silently needs the **parent-fetch head in L1**, else sibling-max of $U$ cannot be read before L3 and the construction needs 4 layers. **Refinement E2:** restate R4.2 for a 3-layer network *with unrestricted parallel heads per layer*; the construction then closes exactly as above.

**Edge case E3 (reflexive masks and the root).** MCTS counts the node itself: $N(j) = |\mathrm{Desc}^+(j)|$. All $\mathcal M_{\text{desc}}$ heads must use the reflexive mask $\mathcal M_{\text{desc}} \vee \mathcal M_{\text{self}}$. For the root, $\mathcal M_{\text{parent}}$ is empty; masked softmax over an empty set is $0/0$. **Exact fix:** parent-channel fallback $\mathrm{pa}^+(r) = r$ for the root (i.e., $\mathcal M_{\text{parent}}[r, \cdot] \leftarrow e_r$), giving $N(\mathrm{pa}(r)) = N(r)$, which is exactly what PUCT at the root does ($\sqrt{N(r)}$ with the root's own visit count). Same convention for leaves: $\mathcal M_{\text{sib}}$ empty ⟹ attend-to-self, $\max_{s} U(s) \leftarrow U(j)$, $\mu_{\text{edge}} = 0$.

**Edge case E4 (the $\epsilon$ in log-cardinality is an exactness bug).** $\log(\epsilon + \sum_k e^{w^\top z_k}) \neq \log N(j)$. Since $\mathcal M_{\text{desc}}$ is 0/1 and $b \le 256 < 2^{24}$, the count is exact in float32; **exact fix:** compute $\log N(j) = \log\!\big(\sum_k \mathcal M_{\text{desc}}[j,k]\big)$ with integer accumulation (or pairwise float64 summation of mask entries, relative error $\le 2^{-52} b$), no $\epsilon$. Residual float error budget for the mean head: sequential softmax-weighted summation of $\le 256$ terms in fp32 gives relative error $\le b u = 256 \cdot 2^{-24} \approx 1.5 \times 10^{-5}$ — comfortably below the `R13-D1` imitation tolerance of $10^{-4}$ (§2).

**Edge case E5 (finite-$\beta_{\max}$ softmin corrupts the greedy-path decision).** $\text{softmin}_{\beta}(x_{1..n}) = -\tfrac1\beta \log \sum_i e^{-\beta x_i} \in [\min_i x_i,\ \min_i x_i + \tfrac{\log n}{\beta}]$. Hence
$$S_{\text{path}}(j) \;\le\; \hat S_{\text{path}}(j) \;\le\; S_{\text{path}}(j) + \tfrac{\log(|\mathrm{Anc}(j)|)}{\beta_{\max}} \;\le\; S_{\text{path}}(j) + \tfrac{\log(D+1)}{\beta_{\max}},$$
so $\hat S_{\text{path}}(j) > 0 \not\Rightarrow j$ on the greedy trajectory: false positives of mass up to $\log(D{+}1)/\beta_{\max}$ (at $\beta_{\max} = 64$, $D = 255$: $\approx 0.087$). Continuous activations cannot implement exact indicators, so this is irreducible *in weights*. **Exact fix (certified decision rule):** decide "on greedy path" iff $\hat S_{\text{path}}(j) > \tau_{\text{path}} := \tfrac{\log(D+1)}{\beta_{\max}} + \epsilon_{\text{fp}}$ and "off path" iff $\hat S_{\text{path}}(j) \le 0$; the band $(0, \tau_{\text{path}}]$ is abstain. This recovers the exact MCTS selection for all instances whose true margins exceed $\tau_{\text{path}}$, and the abstain mass is an observable, reportable quantity (pre-register: abstain rate $\le 5\%$ on `R13-D1` teachers). Pair with a hinge-augmented margin loss (§2, $\mathcal L_{\text{margin}}$), not raw regression on $S_{\text{path}}$.

**Edge case E6 (tie-breaking is information the network cannot recompute).** With uniform priors at unsolved/symmetric nodes, siblings tie exactly: $\mu_{\text{edge}} = 0$ and argmax is defined only by the implementation's tie-break order (visit order / child index). A margin-sign imitation target is ill-posed there. **Exact fix:** R13 teachers must record **realized** decisions (the child actually selected, including tie-break and any sampled Gumbel noise), and the student imitates the realized selection via cross-entropy; margins are auxiliary regression restricted to $|\mu_{\text{edge}}| > \eta$ with $\eta = \tau_{\text{path}}$. This also caps the attainable SelAcc ceiling slightly below 1.0 in the presence of recorded ties (see [PRIOR] §4).

**Refinement E7 (honesty clause: representation-with-oracle-mask, not computation-in-weights).** The transitive closure is *input*, maintained by `M_tree[b,:] = M_tree[pa(b),:] ∨ e_b` at $O(b/64)$ word ops per append (4 words at $b=256$). The weights then perform only constant-depth masked pooling. Two consequences: (i) R4.2 does **not** contradict the alternation barrier — see §1.4 B1, where the *commutative* mean absorbs depth into the mask, but *alternating* composition along a chain does not, even given the masks, because each pooling layer realizes only one level of max/min alternation along a path unless the tree structure aligns with sibling-pairing (balanced trees: $O(\log D)$ masked-pairwise layers suffice; skew chains: $D$ levels — exactly why `V2.1` is necessary); (ii) for `R14`, masks that are *not* syntactically given (transpositions, infosets) must be **learned and then verified** (mask-recovery probes, §3), otherwise the model is a closure-reader, not a discoverer. I recommend R4.2 be restitled "*Representation, given the oracle mask bank, of exact MCTS/PUCT in 3 layers and $O(b^2 d)$ ops*".

**Complexity at $b \le 256$ (for all of §1.3–1.4).** Mask maintenance $O(b)$ words/append; dense masks $b^2 = 65{,}536$ entries; full forward $\approx 4$ layers $\times\, 8$ heads $\times\, b^2 d \approx 1.3 \times 10^8$ MACs at $d = 64$ ($\approx 0.27$ GFLOP); over 256 appends $\le 70$ GFLOP/instance pre-amortization. Amortization: masks are append-only, so cached-KV incremental re-encoding (new token queries + ancestor-path updates) reduces steady-state to $O((b + D) d)$ per append. All trivial on L4; no complexity objection to any theorem.

### 1.4 `Theorem R4.3` — The Five Barriers — **CONFIRMED**, with material edge cases and strengthened fixes

#### Barrier 1 (`Algo 2`: deep Minimax / α–β / PNS) — CONFIRMED; fix `V2.1` needs two exactness conditions

The complexity statement is correctly *conditional*: constant-depth, poly-width threshold circuits evaluating depth-$D$ AND–OR trees for all $D$ would imply $\mathsf{TC}^0 \supseteq \mathsf{NC}^1$ (open, disbelieved). Two sub-claims to sharpen:

- **E8 (α–β is a *policy*, not a value).** α–β pruning is a sequential left-to-right scheduling rule. `V2.1`'s Directed Older-Sibling Mask $\mathcal M_{\text{sib}<}[j,k] = \mathbf 1[\mathrm{pa}(k) = \mathrm{pa}(j) \wedge k < j]$ correctly supports the cutoff predicate — prune $j$ iff $\max_{s \in \mathrm{Sib}<(j)} \text{bound}(s) \ge \beta(\mathrm{pa}(j))$ — which is one $\mathcal M_{\text{sib}<}$ max-pool + comparison, with the *same* finite-temperature caveat as E5 and the same certified-threshold fix ($\tau_{\text{cut}} = \log b / \beta_{\max}$, abstain band). But the *order of expansions* is produced by the search controller, not the network. **Fix in spec terms:** R13 imitation must be defined over **per-step decision records** (§2.1), so α–β is imitated as (expand? / cutoff? / backjump-to?) decisions — representable — rather than as a monolithic tree evaluation.
- **E9 (spine backup semantics).** `V2.1`'s `backup_state[a]` for $a \in \mathrm{Anc}(b)$ at $O(D)$ per append is correct **for trees only**, and inherits E6's tie-break non-determinism for argmax paths (imitate realized principal-variation moves). Under transpositions (B2) it must be disabled — see E10.

#### Barrier 2 (`Algo 4`: MCGS / transpositions) — CONFIRMED; the fix as written is incomplete in two ways

**Edge case E10 (double counting and the one-hop closure).** As stated, $\mathcal M^{\text{DAG}}_{\text{desc}} = E_{\text{trans}} \cdot \mathcal M_{\text{desc}}$ is *not* a reachability closure: it extends tree-descendants by exactly one transposition hop; iterating through further transpositions requires the **Kleene star** over the Boolean semiring:
$$\mathcal M^{\text{DAG}}_{\text{desc}} \;=\; \bigl(E_{\text{tree}} \vee E_{\text{trans}}\bigr)^{*} \;=\; \lim_{n \to \infty} \underbrace{(E \vee M_{\text{desc}}) \cdots}_{\text{fixed point}},$$
computed by Boolean repeated squaring in $\lceil \log_2 b \rceil = 8$ rounds of $O(b^3/64)$ word ops ($\approx 2.6 \times 10^5$ word ops/round at $b = 256$ — negligible). Boolean reachability is well-defined under **cycles** (a "pass" move can map a state's canon to its parent's), so this repairs the cycle hazard; the *tree* recursion $E_{\text{trans}} \cdot \mathcal M_{\text{desc}}$ is not idempotent and under-reaches.
More importantly, **pooling over $\mathcal M^{\text{DAG}}_{\text{desc}}$ with cardinality denominators double counts**: node $k$ reachable from $j$ by $m(j,k)$ paths enters with weight $m(j,k)$, and
$$\bigl| Q^{\text{closure}}(j) - Q^{\text{MCGS}}(j) \bigr|_{\text{worst}} \;\le\; 2\,\Bigl(1 - \tfrac{|\mathrm{Desc}_{\text{DAG}}(j)|}{\sum_k m(j,k)}\Bigr) \quad (\text{values in } [-1,1]),$$
which is vacuous on transposition-heavy search (the exact failure Kelvin's question is about). **Exact fix (`V2.2′`):** in DAG mode, values are **never** closure-pooled. Maintain per-node MCGS statistics $(N(k), W(k))$ updated along the **traversed path of each simulation** (each visited node $+1$ visit, $+z$ outcome), i.e. incremental means $W(k) \leftarrow W(k) + z$, $N(k) \leftarrow N(k) + 1$, $Q(k) = W(k)/N(k)$ read from the *self* channel; the closure mask is used only for (a) dedup/selection (prefer canonical-first expansion), and (b) attention *context*, with **visit-weighted** pooling $\sum_k \tilde w_{jk} Q(k)$, $\tilde w_{jk} = N(k) \cdot \mathcal M^{\text{DAG}}_{\text{desc}}[j,k] / \sum_{k'} N(k') \mathcal M^{\text{DAG}}_{\text{desc}}[j,k']$. PUCT in DAG mode uses node visits $N(k)$ (multi-parent edges carry their own edge stats $(N_e, W_e)$ — add per-edge channels, cf. `V2.8` in §1.5). This is the KataGo-MCGS semantics, made representable.
**Edge case E10b (learned $E_{\text{trans}}$ correctness).** $\exp(-\|h_{\text{state}}(C_j) - h_{\text{state}}(C_k)\|^2/\tau)$ soft-matching can **merge distinct states** (silent correctness loss). Fix: soft kernel for gradients with **straight-through hard equality** for statistics; R13 provides exact Zobrist/canonical-key ground truth so mask recovery is measurable (F1 bar in §2); R14 measures discovered $E_{\text{trans}}$ against post-hoc Zobrist (§3). Also require `canon()` to include symmetry reduction (Hex reflections) for recall, and a **ply key** to avoid pass-move self-transpositions.

#### Barrier 3 (`Algo 5`: CFR+ / ReBeL / R-NaD) — CONFIRMED; the fix as written omits three components required for correctness

**Edge case E11 (missing average-strategy accumulator).** CFR's guarantee ($\varepsilon_T \le 2|\mathcal I|\Delta \sqrt{T}$-type bounds; CFR+ empirically $\sim T^{-1}$) attaches to the **counterfactual-reach-weighted average strategy** $\bar\sigma_T(I,a) \propto \sum_t t\, \pi_i^{\sigma_t}(I, a)$ (linear/CFR+ weighting), *not* to the current regret-matched $\sigma_T$. `V2.3`'s "Regret-Matching readout" alone imitates the non-converging current strategy. **Exact fix (`V2.3′`):** add a persistent per-infoset strategy-sum accumulator $\Sigma(I, a)$ alongside $R^+(I,a)$; readout head outputs $\bar\sigma$; imitate both (current-$\sigma$ CE for dynamics, $\bar\sigma$ KL for the equilibrium object; §2).
**Edge case E11b (reach weighting and strategy fusion).** Pooling regrets across an infoset without counterfactual-reach weights $\pi_{-i}(h)$ is precisely **strategy fusion** and breaks the CFR bound. Exact update to represent:
$$r_t(I, a) = \sum_{h \in I} \pi_{-i}^{\sigma_t}(h)\,\bigl[v_i(h, a) - v_i(h, \sigma_t)\bigr], \qquad R^+_{t+1}(I,a) = \max\bigl(R^+_t(I,a) + \gamma_t\, r_t(I,a),\, 0\bigr),$$
with the ancestor log-reach accumulator of `V2.3` supplying $\log \pi_{-i}(j) = \sum_{a \in \mathrm{Anc}(j)} \mathbf 1[\mathrm{pl}(a) \ne i] \log \sigma(a)$ (exact: pointwise sum over the $\mathcal M_{\text{anc}}$-masked uniform head, $O(1)$ layers; linear-CFR discount $\gamma_t = t$ via the time channel).
**Edge case E11c (LIR firewall).** The firewall requirement — no feature correlated with the opponent's private information may enter the acting player's value — is violated by any *soft* infoset mask or by cross-infoset $E_{\text{trans}}$-style kernels. **Exact fixes:** (i) $\mathcal M_{\text{infoset}}$ is **hard** equality on the acting player's observation (straight-through for gradients only); (ii) forbid $E_{\text{trans}}$ merging states whose *any-player* observation differs (unlike perfect-info symmetry canon); (iii) measurable firewall test: train a probe from the pooled representation to the opponent's private card — require AUC $\le 0.55$ (§2 `R13-D5`, §3 `R14-D5`). Without (i)–(iii) the barrier is only half-closed: the network could represent CFR+'s arithmetic while leaking exactly as a fused-strategy bot.

#### Barrier 4 (`Algo 6`: Gumbel Sequential Halving) — CONFIRMED; finite-$\tau$ rank head has an exact certificate

**Edge case E12 (rank-percentile error).** $\widehat{\text{pct}}_{\text{sib}}(j) = \tfrac1m \sum_{s \in \text{Sib}(j)} \sigma\bigl((S(j) - S(s))/\tau_{\text{rank}}\bigr)$ approximates $\tfrac{1}{m}\,\text{rank}(j)$ with error vanishing in $\tau_{\text{rank}}$. **Lemma (exact top-$k$ recovery).** Let $\gamma$ be the minimum absolute pairwise score gap among siblings, $\delta \in (0,1)$. Set $\tau_{\text{rank}} = \gamma / (2\log(m^2/\delta))$. Then $\sigma(\gamma/\tau) \ge 1 - (\delta/m^2)^2$ and every top-$k$ test $\widehat{\text{pct}}(j) \ge 1 - \tfrac{k - 1/2}{m}$ returns the exact survivor set with probability $\ge 1 - \delta$ over float noise. Conventions: $m = 1 \Rightarrow \widehat{\text{pct}} = 1$; odd $m$ at phase $p$: survivors $\lceil m/2 \rceil$ (match the teacher's realized rule). Because Gumbel SH's scores include sampled Gumbel noise, **teacher records must include the realized noise** (E6 again); with recorded noise the target is deterministic and SetEM is well-posed.

#### Barrier 5 (`Algo 3`: CDCL nogood broadcast) — CONFIRMED; a strictly stronger exact fix exists

The soft nogood head $\mathcal M_{\text{nogood}}[j,k] = \mathbf 1[m_{\text{ql}}(C_k) < \tau_{\text{clash}}]$ broadcasts "conflict happened at $k$" but not **what was learned**: the pruning test is whether candidate $j$'s path assignment *falsifies the learned clause* $c_k$ (e.g., 1UIP): prune($j$) iff $\forall \ell \in c_k: \neg\ell \in \mathrm{assign}(\mathrm{Anc}(j))$. **Exact fix (`V2.5′`, replaces the soft head for pruning):** maintain a **literal-bitmask** exactly like the existing ancestor bitmask:
$$L[j, \ell] = L[\mathrm{pa}(j), \ell] \vee \mathbf 1[\ell = \ell(j)] \quad\text{(one } O(|\mathcal L|/64)\text{ word-OR per append)},$$
then clause coverage $\mathbf 1[c_k \subseteq L[j,:]]$ is an exact bitwise AND-scan ($O(|c_k|/64)$), with **zero** finite-temperature error and no learning required — strictly dominating $\mathcal M_{\text{nogood}}$ for pruning. Retain soft cross-attention only where generalization is genuinely needed: (a) learning *which* literals to resolve into $c_k$ (imitate 1UIP as a multi-label target over literals; heuristics beyond 1UIP are a prime R14-discovery target), and (b) backjump level = argmax over $\mathrm{Anc}(j)$ of assertion level ($\mathcal M_{\text{anc}}$ rank head, E5/E6 caveats apply). Complexity: $|\mathcal L| \le 4{\times}81$ literals on Sudoku-9x9 ⟹ $\le 6$ words/node; total $O(b)$ per append.

---

### 1.5 Answer to Kelvin's Q2 — What Even `GLCSearch v2` (with `V2.1′`–`V2.5′`) Cannot Represent

**Cheaply fixable now (propose adding to v2):**
- **V2.6 (halt head):** adaptive early termination (PNS solved, CDCL SAT at unit propagation) is invisible under a fixed-$B$ interface; add a per-step halt logit (BCE-imitable, and usable by R14's policy).
- **V2.7 (parity/XOR pooling head):** global XOR/parity of leaf polarities (needed by some parity-SAT and couple-domination heuristics) is outside {mean, LSE, min, max, count}; one additive head closes it.
- **V2.8 (per-edge stat channels):** multi-parent DAG edges need $(N_e, W_e)$ channels (E10); without them only node-visit PUCT is representable.

**Irreducible without architectural commitment (the honest "no" list):**
1. **Arbitrary-depth alternating recursion in weights** without the spine accumulator (skew-chain minimax, df-pn $(\min,\sum)/(\sum,\min)$ alternation) — conditional on $\mathsf{TC}^0 \neq \mathsf{NC}^1$; v2 *represents* it only because `V2.1` externalizes recursion into bank state.
2. **Unbounded recursive self-embedding** (DeepStack continual re-solving, nested MCTS of depth $k$ as $k \to \infty$): one transformer stack embeds one level; recursive self-invocation is not representable at any fixed depth.
3. **Exact deterministic tie-breaking** and any teacher-side PRNG-dependent decisions absent recorded noise (E6) — information-theoretic, not architectural.
4. **Continuum action spaces** (Galanis-Market bid functions): representable only post-discretization; the closure/infoset machinery assumes finite $\mathcal A$.
5. **History-dependent draws by repetition** if `canon()` is not ply-keyed: threefold-repetition/zhǒngjié-style termination requires the *path multiset*, not any per-node mask; needs an explicit repetition-counter channel (omitted from v2; flag for `Proposal-Poker` variants).
6. **Offline retrograde databases** (full-game endgame solving): not a per-step search algorithm; out of scope of the $B$-budget interface entirely.

---

## 2. Specification Input for `PREREG-R13-ALGO-IMITATION.md`

### 2.0 Unified imitation interface (the load-bearing design decision)

Every teacher emits, per search step $t$, a **decision record** over the same token/mask interface: $\mathcal D_t = (\text{tokens } \{z_0..z_b\},\ \text{mask bank},\ \text{decision head targets})$ with heads: `sel` (realized selection, multinomial over children), `val` (backed-up value per node), `count` ($\log N$), `cut` (α–β cutoff / PNS disproof, binary), `bjump` (backjump target, multinomial over $\mathrm{Anc}$), `clause` (learned clause, multi-label over literals), `strat` (infoset mixed strategy), `set` (SH survivor set, per-sibling binary), `halt`. This makes all 6 families one supervised problem and makes the ablation matrix meaningful. **Determinism contract (`R13-D0`):** teachers run with fixed seeds, recorded Gumbel noise, recorded tie-breaks, deterministic kernels (CPU or deterministic-CUDA reductions); replaying an instance must reproduce decision records **bitwise** — else targets are ill-posed and the run is invalid.

### 2.1 Teachers, losses, datasets

**Teachers (all hand-coded, all recorded):**
`A1` PUCT ($c_{\text{puct}} = 1.25$, value rollout $z_b$ recorded at expansion) on Hex/Connect-4/Sudoku; `A2` α–β with fixed move-ordering + 1UIP-free cutoff bookkeeping, and PNS/df-pn, depth up to $D = 16$ on Hex-5/Connect-4-7 and Equiv-SAT; `A3` CDCL (VSIDS + phase saving + 1UIP + non-chronological backjump) on Equiv-SAT/Sudoku-6x6 CNF encodings; `A4` MCGS (KataGo-style graph-PUCT + Zobrist canon + visit stats) on ShipHunt/Stratego/Hex; `A5` CFR+ (alternating, linear averaging, reach-weighted) on Proposal-Poker/Micro-Stratego/Galanis-Market; `A6` Gumbel MuZero SH ($A = 16$, $m = 8$, recorded Gumbel $g$).

**Loss:** $\mathcal L = \sum_h \lambda_h \mathcal L_h$ with
$\mathcal L_{\text{sel}} = \mathrm{CE}$ (realized child; **no label smoothing** — tie-breaks are information),
$\mathcal L_{\text{val}} = \mathrm{Huber}_{\delta=0.05}(\hat Q, Q^{\text{teacher}})$ over all nodes each step,
$\mathcal L_{\text{count}} = \mathrm{Huber}(\log \hat N, \log N)$,
$\mathcal L_{\text{margin}} = \mathrm{Huber}(\hat\mu_{\text{edge}}, \mu_{\text{edge}}) \cdot \mathbf 1[|\mu_{\text{edge}}| > \tau_{\text{path}}]$,
$\mathcal L_{\text{cut}} = \mathrm{BCE}$, $\mathcal L_{\text{bjump}} = \mathrm{CE}$ over Anc, $\mathcal L_{\text{clause}} = \mathrm{BCE}$ multi-label, $\mathcal L_{\text{strat}} = \mathrm{KL}(\bar\sigma^{\text{teacher}}_T \| \hat\sigma)$ + CE on current $\sigma_t$, $\mathcal L_{\text{set}} = \mathrm{BCE}$ per sibling. Defaults $\lambda$: sel 1.0, val 1.0, margin 0.3, count 0.3, cut 0.5, bjump 0.5, clause 0.5, strat 1.0, set 0.5. AdamW, lr $3{\rm e}{-4}$, cosine, 3 seeds, bar = mean passes **and** worst seed $\ge$ bar $-\epsilon_{\text{seed}}$.

**Data:** trajectories = full search rollouts to budget $B$; $N_{\text{train}} \in \{128, 512, 2048\}$ per algorithm; $N_{\text{val}} = 256$, $N_{\text{test}} = 512$ held-out instances. **OOD:** train $B \le 32$ and $D \le 8$, test blind at $B \in \{64, 128\}$, $D \in \{12, 16\}$, plus OOD instance shift (unseen generators/seeds).

### 2.2 Gates `R13-D0` .. `R13-D6` (falsifiable pass bars)

| Gate | Content | Pass bar (mean over 3 seeds) | Falsifies if failed |
|---|---|---|---|
| **D0** | Static/AST violations = 0; canaries 3/3; `PoisonTensor` `global_max_diff = 0.000000000`; `r4_kv_equiv_max_diff = 0.000000000`; **teacher replay bitwise-deterministic**; mask-bank unit tests (reflexivity, root convention E3, literal-bitmask E-`V2.5′` vs brute force on $10^3$ random trees) | all pass | run invalid |
| **D1 (`A1`)** | PUCT imitation, $B \le 32$, in-dist | `Full-v2`: SelAcc $\ge 0.99$ (abstain $\le 5\%$, E5), ValR² $\ge 0.999$, $\max\|\hat Q - Q\| \le 10^{-4}$·range. **`Abl-v1` must also pass D1** (R4.2 in practice). `Abl-1Hop-Flat`: SelAcc $\le 0.80$ or ValR² $\le 0.90$ at $D \ge 8$ | R4.2 (if v1 fails) / R4.1 (if 1-hop passes) |
| **D2 (`A2`)** | Minimax/α–β/PNS, $D \le 16$ | `Full-v2`: ValR² $\ge 0.99$, PruneF1 $\ge 0.95$; `Abl-v1`: ValR² falls below 0.9 at some $D^* \le 2L$; report error-vs-$D$ curve and fit to $\tfrac{D-L}{D+1}$ envelope (empirical R4.1a) | R4.3-B1 |
| **D3 (`A3`)** | CDCL | BackjumpAcc $\ge 0.95$; NogoodF1 $\ge 0.90$; trajectory exact-match $\ge 0.80$ at $B \le 32$; `Abl-v1` at chance on $\ge$2-hop backjumps | R4.3-B5 |
| **D4 (`A4`)** | MCGS | SelAcc $\ge 0.97$; $E_{\text{trans}}$ recovery F1 $\ge 0.95$ (diagnostic); MCGS-Q Huber $\le 0.02$; **double-counting test**: on instances with $\ge 3$ engineered transpositions, visit-weighted pooling matches teacher within $0.02$ while cardinality pooling must show the E10 bias ($\ge 0.1$ separation, proving the failure mode real and fixed) | R4.3-B2 + E10 |
| **D5 (`A5`)** | CFR+ | Per-infoset $\mathrm{KL}(\bar\sigma_T \| \hat{\bar\sigma}) \le 0.02$ nats; exploitability(student $\bar\sigma$) $\le 2\times$ teacher's at matched node budget on Proposal-Poker; **firewall probe AUC $\le 0.55$** (E11c) | R4.3-B3 |
| **D6 (`A6` + generalist)** | SH + cross-algo | SetEM $\ge 0.90$ per phase; final-action match $\ge 0.99$; **shared generalist** trained on all 6: within 10% relative of per-algo specialists on every D-bar; routing-head algo-ID $\ge 0.90$ (diagnostic only) | generality claim |

**Sample-efficiency requirement:** at $N_{\text{train}} = 512$, `Full-v2` reaches $\ge 0.98 \times$ its $N = 2048$ scores on `A1`, `A4`, `A6`, and $\ge 0.90 \times$ on `A2`, `A3`, `A5`. **OOD requirement:** SelAcc drop $\le 0.02$ absolute for `A1`/`A6`, $\le 0.05$ for others; depth-OOD ValR² drop $\le 0.01$ for `Full-v2` (masks/accumulators are depth-independent — this is R4.2's empirical fingerprint).

### 2.3 Ablation matrix

Arms: **(A)** `Full-GLCSearch-v2` (`V2.1′`–`V2.5′` + `V2.6–V2.8`); **(B)** `Abl-GLCSearch-v1` (7-mask bank, no accumulators/DAG closure/infoset/rank/nogood heads); **(C)** `Abl-R12-1Hop-Flat` (`TwoStageTreeClearer` as shipped); **(D)** leave-one-out: `v2−spine`, `v2−DAGclosure`, `v2−infoset`, `v2−rank`, `v2−nogood`, `v2−halt`; **(E)** specialists vs shared generalist. Predicted signatures (pre-registered): (C) fails D1 at depth $\ge 8$ with the R4.1a curve and fails the E1 swap test (add the polarity-swap eval set as a D1 sub-test: `Abl-C` ValR² $\le 0.5$ on swapped bags); (B) passes D1, fails D2/D3/D4/D5 bars by design; each (D) arm fails exactly its namesake gate. Budget cap: $\le \$2.50$/run on L4 (R12 parity), 3 seeds each.

---

## 3. Specification Input for `PREREG-R14-UNSUP-DISCOVERY.md`

### 3.0 Training signal — zero supervision on algorithm implementations

Same `GLC-Search-Transformer v2` trained purely from task outcomes. Per episode (instance $i$, budget $B$), group of $J = 8$ rollouts:

- **Belief score:** $\ S_{\text{bel}} = -\mathrm{KL}(P^*_i \,\|\, \mu_B)$ (proper score on end-of-search answer belief).
- **Witness score:** $\ S_{\text{wit}} = \mathbf 1[\text{verified witness found}]$ (checker-verified; binary).
- **Interaction score:** normalized payoff vs frozen opponent pool {PUCT-800, α–β-$D$8, CDCL-tuned, CFR+-$10^3$ iters, SH}; exploitability $\varepsilon(\hat{\bar\sigma})$ via best-response oracle on the imperfect-info trio.
- **Leave-Subtree-Out credit:** for top-$k$ subtrees $s$ ($k = 8$, by visit share), $I_{i,s} = S(q^{-s}, y) - S(q^{\text{full}}, y)$ computed by masking $s$'s tokens and re-running the readout only. GRPO advantages $A_s = (I_s - \mathrm{mean}_s)/(\mathrm{std}_s + \epsilon)$; policy gradient on expansion/selection/halt with entropy bonus $\lambda_H = 0.01$ and reference-KL leash $\beta_{\text{KL}} = 0.05$ for the first 20% of steps. **LSO validation check (pre-registered):** against exact re-search on $n = 64$ subsamples, Spearman $\rho(I_{\text{LSO}}, I_{\text{exact}}) \ge 0.6$, else LSO credit is demoted to a heuristic and the run reports degraded.
- Curricula: instance ladders {2k, 8k, 32k} episodes, budget $\le \$4.00$ on L4, 3 seeds.

### 3.1 Gates `R14-D0` .. `R14-D6`

| Gate | Content | Pass bar |
|---|---|---|
| **D0** | All R11/R12 static gates (0 violations, 3/3 canaries, `PoisonTensor` = 0, `r4_kv_equiv` = 0) **plus provenance check**: no teacher decision-record files reachable from the R14 data config (grep + manifest audit; per policy 15 this claim is checkable) | all pass |
| **D1 (no regression + regression fix)** | 12-env mean KL at $B{=}128$ $\le 0.1294$ (R12 parity or better) **and** at $B{=}256$ $\le 0.1294$ (kills the $0.1294 \to 0.1387$ regression); ShipHunt/Stratego KL at $B{=}256$ $\le 0.9\times$ their own $B{=}128$ values; Sudoku KL monotone in $B$ retained ($\le 0.033$ at 256) | numeric, per-env table published |
| **D2 (MAAP — mechanistic algorithmic alignment probes)** | See §3.2. Weak + strong bars per family, all computed **post-hoc on the model's own trajectories** (never on teacher trees) | weak bars pass $\ge 4/5$ families; strong bars reported |
| **D3 (beat fixed algorithms, Mini-ECI)** | Mean Mini-ECI $\le 106.0$ at equal budget (vs R12's 106.53, specialists 108.61); $\ge 8/12$ envs within 0.5 ECI of the best per-env fixed algorithm; strictly better on $\ge 3$ envs | mean + per-env |
| **D4 (witness discovery)** | $\ge 75.0\%$ at $B{=}256$ (R12: 73.61%); $\ge 72\%$ at $B{=}128$ | 3-seed mean |
| **D5 (interactive/exploitability)** | Proposal-Poker + Micro-Stratego: $\varepsilon(\text{R14}) \le 0.7 \times \varepsilon(\text{R12 unified})$; Hex/Connect-4 vs PUCT-800: win rate $\ge 52\%$ weak / $\ge 55\%$ strong, $n \ge 1000$ games ($\pm 3\%$ at 95% CI) | reported with CIs |
| **D6 (interpretability artifact + pre-registered decision rule)** | Publish per-channel attention mass, mask-recovery F1s (discovered $E_{\text{trans}}$ vs Zobrist; discovered infoset vs exact obs-equality), $\Delta$log-lik model comparison. **Decision rule:** all strong MAAP bars + D3 ⟹ "emergent algorithmic alignment"; task bars (D1/D3/D4/D5) without strong MAAP ⟹ "task competence without alignment" (falsifies the *necessity* claim, still a product success); D1 fails ⟹ architecture escalation, not tuning | pre-registered, no post-hoc bar movement |

### 3.2 MAAP — quantitative probes (family → probe → weak/strong bar)

1. **MCTS/PUCT (Hex, Connect-4):** (a) $R^2$ of readout $Q(j)$ vs exact MCTS $Q$ recomputed on the **model's own** visited trees: weak $\ge 0.80$ / strong $\ge 0.95$; (b) model-comparison: likelihood of the model's expansion sequence under PUCT with $c$ fit by MLE vs {uniform, minimax, depth-first, CFR} alternatives: weak $\Delta\log\ell \ge +1$ nat/step, strong $\ge +2$; (c) $\mathcal M_{\text{desc}}$ attention share $\ge 40\%$ in deductive envs.
2. **CDCL/PNS (Sudoku, Equiv-SAT):** implicit nogood head vs 1UIP clauses computed post-hoc on the model's own conflicts: F1 weak $\ge 0.6$ / strong $\ge 0.8$; backjump-length distribution KL vs teacher $\le 0.3$ nats; PNS allocation: Spearman $\rho$ of visit shares vs $(\min(\text{pn},\text{dn}))^{-1}$-optimal $\ge 0.5$.
3. **CFR+ (Proposal-Poker):** $\varepsilon$ of extracted average strategy at matched node budget $\le 1.2\times$ fixed CFR+ (weak) / $\le 1.0\times$ (strong); infoset consistency: max JS divergence of policy across infoset members $\le 0.02$ (strategy-fusion absence); regret-matching shape: $R^2$ of $\sigma(I,a)$ vs $\mathrm{ReLU}(R^+)$ fit $\ge 0.7$.
4. **MCGS (ShipHunt, Stratego):** dedup rate $\ge 50\%$ of oracle-Zobrist dedup on the same trajectories; visit-weighted vs cardinality pooling separation as in `R13-D4` (the model must *choose* visit weighting on its own); discovered-$E_{\text{trans}}$ recovery F1 $\ge 0.7$ weak / $\ge 0.9$ strong.

**Head-to-head with R13:** after R14, run the R13 probes *as diagnostics* on the unsupervised model and publish the alignment table next to the supervised one — the delta "supervised imitation ⇄ spontaneous alignment" is the round's scientific product.

---

## 4. Rubric `D1`–`D6` and `[PRIOR]` Forecasts

Rubric dimensions (0–10): **D1** mathematical rigor & falsifiability · **D2** expressivity coverage of the 6 canonical search families · **D3** instrumentation & gate validity · **D4** sample/compute efficiency · **D5** OOD generalization (depth/budget/instance) · **D6** mechanistic alignment & discovery.

| Row | D1 | D2 | D3 | D4 | D5 | D6 | Notes |
|---|---|---|---|---|---|---|---|
| **R3 Realized (R12 audit)** | 6.5 | 3.0 | 9.0 | 7.5 | 5.5 | 2.0 | Solid gates (D3), but R4.1 shows the shipped 1-hop/flat operators were expressivity-blind (D2); B128→256 KL regression caps D5 |
| **R4 Expressivity-Risk (R12 arch, no `M_desc`/v2)** | 7.5 | 4.0 | 9.0 | 7.5 | 5.0 | 2.0 | With R4.1–R4.3 on the books the *theory* is now rigorous; the architecture itself remains at risk on A2–A5 |
| **Proj-R13 (v2 + supervised)** | 8.5 | 8.5 | 9.0 | 7.0 | 7.5 | 3.0 | All 6 families representable *given* the primitives (E2–E12 fixed); supervised imitation is not yet discovery |
| **Proj-R14 (v2 + unsupervised)** | 8.5 | 8.5 | 9.0 | 7.0 | 8.0 | 7.0 | D6 is the bet; interval 4.0–8.0 depending on MAAP outcome per the D6 decision rule |

### `[PRIOR]` forecasts — R13 (80% central intervals, `Full-v2`, $N{=}2048$, $B \le 32$, in-dist, 3 seeds)

| Algo | Primary metric | `Full-v2` | `Abl-v1` | `Abl-1Hop-Flat` |
|---|---|---|---|---|
| A1 PUCT | SelAcc / ValR² | 0.995–1.000 / 0.9990–0.9999 | 0.98–0.999 / 0.995–0.999 | 0.55–0.75 / 0.60–0.85 ($D \ge 8$) |
| A2 Min/αβ/PNS | ValR² ($D{=}16$) / PruneF1 | 0.990–0.999 / 0.90–0.97 | 0.75–0.90 (breaks at $D^* \in [6, 10]$) | 0.4–0.7 |
| A3 CDCL | BackjumpAcc / NogoodF1 / traj-EM | 0.93–0.99 / 0.85–0.95 / 0.70–0.90 | $\sim$ chance beyond 2 hops | $\sim$ chance |
| A4 MCGS | SelAcc / $E_{\text{trans}}$ F1 | 0.96–0.995 / 0.93–0.99 | 0.85–0.93 / n-a | 0.6–0.8 |
| A5 CFR+ | strat KL (nats) / expl. ratio | 0.010–0.030 / 1.0–2.0× teacher | KL 0.05–0.15 (fusion errors visible) | worse |
| A6 SH | SetEM / final-action | 0.88–0.97 / 0.995+ | 0.70–0.85 | 0.5–0.7 |

Also: OOD $B \in \{64,128\}$ SelAcc drop ≤ 0.02 (A1/A6) and ≤ 0.05 (others) with probability ≈ 0.8; generalist-vs-specialist relative gap 5–12%; $N{=}512$ reaches ≥ 0.98× of $N{=}2048$ on A1/A4/A6 with probability ≈ 0.75. Ceiling note: recorded ties put SelAcc ceiling ≈ 0.998 (E6).

### `[PRIOR]` forecasts — R14 (80% central intervals, 3 seeds)

| Quantity | Prior interval | vs R12 reference |
|---|---|---|
| 12-env mean KL, $B{=}128$ | 0.118–0.128 | 0.1294 |
| 12-env mean KL, $B{=}256$ | 0.112–0.126 | 0.1387 (regression fixed with prob ≈ 0.9) |
| ShipHunt/Stratego KL @256 | ≤ 0.9× own @128 | was the regression source |
| Mini-ECI (mean, equal budget) | 104.5–106.2 | 106.53 (spec. 108.61) |
| Witness discovery @ $B{=}256$ | 74–77% | 73.61% |
| Elo vs R12 unified @$B{=}128$ | +50 to +250 | — |
| MAAP: MCTS $\Delta$log-lik (Hex/C4) | +1 to +4 nats/step | strong bar prob ≈ 0.5 |
| MAAP: CFR+ probe (hardest) | pass prob 0.4–0.6 | — |
| Outcome decision rule | "emergent alignment" ≈ 0.30–0.45 · "task competence w/o alignment" ≈ 0.35–0.50 · "D1 fail / escalate" ≈ 0.10–0.20 | — |

**Bottom line for Kelvin:** the theorems hold (with the refinements E1–E12 folded in), `R13` is precisely the experiment that converts them from representation claims into learned-capability evidence, and `R14`'s MAAP design means either outcome — spontaneous alignment or algorithm-free competence — is a publishable, pre-registered result. No decision needed from you; next action is merging these specs into `PREREG-R13-ALGO-IMITATION.md` and `PREREG-R14-UNSUP-DISCOVERY.md`.
