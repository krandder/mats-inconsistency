# Round 4 Deliverable: Search-Algorithm Expressivity, Supervised Imitation (`R13`), and Unsupervised Search Discovery (`R14`)

**Author:** Society core member, Round 4
**Scope:** Audit of Member 18's `Theorem R4.1`, `Theorem R4.2` and `Theorem R4.3`; pre-registration inputs for `R13` and `R14`; rubric update and forecasts.
**Epistemic status:** This audit uses only the inline summaries of `kv_tree_search.py` (line references 238, 241–242, 290–295, 691–756, 895–1099). I have not independently re-read the code. Every claim below is conditional on those summaries being accurate. `R13-D0` includes a theorem-falsification test that checks this assumption empirically.

---

## 0. TL;DR for Kelvin

### Q1. Can our current algorithms learn MCTS-style subtree aggregation?

**Answer for R11/R12 as built: no. They do not even attempt it.**

- **Selection is hand-coded.** In `R12`, selection is a hand-written Python formula (`ResolvableInconsistencyTreePUCT.select_node`) with zero trainable weights.
- **There is no backup.** `n_exp` counts only immediate children, not subtree visits. `q_tree_bank` is written once per node and never backed up to ancestors.
- **The learned clearer cannot see far enough.** It sees 1 hop of tree structure, plus a flat attention layer that does not know which nodes are descendants of which.
- **Consequence.** The R11/R12 gains came from better evaluation and readout (bilinear lemmas, Platt calibration, FiLM generalist). They did not come from learned search.

**Answer for `GLCSearch v1`: yes, MCTS is representable.** The fix is cheap: one attention head masked by the descendant relation `M_desc`. The bank already maintains this relation, because `M_tree` is its transpose. With that head:

- Uniform attention ($\beta=0$) over $\mathrm{Desc}(j)$ is exactly the AlphaZero $W/N$ subtree mean.
- A log-sum-exp head with zero weights gives $\log N(j)$.

Two caveats from this audit (§1.3):

1. The representation is exact only up to a finite-temperature tie margin. The precise result is "$\varepsilon$-greedy PUCT", made rigorous below.
2. Selecting unexpanded children needs **frontier tokens**, which no current version has.

### Q2. What can even `GLCSearch v2` not represent?

**First, two of the five v2 fixes are themselves incorrect as specified:**

- **V2.2 (MCGS).** $E_{\text{trans}}\cdot\mathcal M_{\text{desc}}$ double-counts and is not transitively closed. KataGo's MCGS value is also *not* a set-mean over DAG descendants.
- **V2.3 (CFR+).** A single forward pass computes one-shot regret matching, not CFR+. CFR+ needs persistent cumulative regrets and averaged strategies across iterations.

**Second, several widely used multi-game search families remain outside v2:**

| Family | Example algorithms | What v2 lacks |
|---|---|---|
| Action-identity statistics across branches | RAVE/GRAVE (General Game Playing), MAST, history/killer heuristics in alpha-beta engines | Cross-branch "same action" pooling |
| Non-append-only search | Iterative deepening, IDA*, MTD(f), CDCL restarts, tree reuse | Ability to re-search, discard or restart (the bank only grows) |
| Unbounded fixed-point computation per step | CFR-D/DeepStack safe re-solving, ReBeL public-belief updates, value iteration on cyclic graphs | More than a constant number of rounds per step |
| In-search parameter adaptation | NRPA, Playout Policy Adaptation | Online weight or memory updates during search |
| Frontier / large-action reasoning | Progressive widening, full Gumbel all-action sets, first-play urgency (FPU) | Tokens for actions that are not yet tree nodes |
| Simultaneous / multiplayer bandits | SM-MCTS with EXP3/regret matching, max^n, Best-Reply Search | Typed vector backups and per-node stochastic bandit state |

§1.6 gives the full table and proposes a minimal `v3` primitive set.

**Your proposed benchmark (supervised imitation as an intermediate test) is right, and should be split in two:**

- **(a) Representability:** hand-constructed weights from the theorems must reproduce each algorithm.
- **(b) Learnability:** gradient descent must find those weights sample-efficiently.

`R13` pre-registers both. `R14` then tests whether the same architecture discovers comparable (or better) search from outcomes alone. It uses causal-abstraction probes to check whether it rediscovers MCTS, CDCL, CFR and so on.

---

## 1. Audit and Critique of Theorems R4.1, R4.2, R4.3

### 1.1 Notation

**Tree and node sets**

- $\mathcal T_b$ is the search tree after $b$ expansions, with nodes $0,\dots,b$ and root $0$.
- $\mathrm{Anc}(j)$ and $\mathrm{Desc}(j)$ are **inclusive** of $j$. Write $\mathrm{Anc}^\circ(j)=\mathrm{Anc}(j)\setminus\{0\}$.
- $\mathrm{Sib}(j)=\mathrm{Ch}(\mathrm{pa}(j))\setminus\{j\}$.
- $d(j)$ is the depth of node $j$.
- $m_j=|\mathrm{Ch}(j)|$ is the number of children of $j$.

**Node quantities**

- $v_k\in[-1,1]$ is the leaf evaluation written to `q_tree_bank` when node $k$ is created.
- **Two-player perspective sign.** Let $s_k=(-1)^{d(k)}$. If $v_k$ is expressed from the perspective of the player to move at $k$, then its value from $j$'s perspective is $s_js_kv_k$.

### 1.2 Theorem R4.1 (Depth and Topology Barrier): verified with two corrections

#### R4.1(a) is correct

**Proof.**

1. Any $L$-layer 1-hop message-passing network has receptive field $\mathcal N_L(j)=\{k:\mathrm{dist}_{\mathcal T}(j,k)\le L\}$. Hence $\partial h_j^{(L)}/\partial v_k\equiv 0$ for every descendant with $\Delta d>L$.
2. Take a chain $0\to1\to\dots\to D$ with $Q(0)=\frac1{D+1}\sum_k v_k$.
3. Fix $v_0,\dots,v_L$. Set the remaining $D-L$ values to all $+1$ in one instance and all $-1$ in another.
4. The network output is identical on both instances, but the targets differ by $\frac{2(D-L)}{D+1}$.
5. So on one instance the error is at least $\frac{D-L}{D+1}$. ∎

**Caveat C1 (scope).** This bound applies only to purely 1-hop stacks. `TwoStageTreeClearer` adds a flat global Stage 2, which gives every token a global receptive field. R4.1(a) therefore does *not* by itself bound R12's clearer. That is exactly why R4.1(b) is needed.

#### R4.1(b) needs a corrected hypothesis

The stated counterexample is: depth-2 leaves $k_1,k_2$ with opposite clamps, swapped between depth-1 parents $j_1,j_2$. It has two gaps.

**Gap 1: the counterexample is too shallow.** Stage 1 mixes parent and child.

- If Stage 1 mixes in the child-to-parent direction, $j_1$ directly absorbs $v_{k_1}$, so depth-2 leaves are visible.
- If Stage 1 mixes parent-to-child, then $k_1$'s Stage-1 feature contains $j_1$'s content. Stage-2 attention from $j_1$ can then match "keys that contain my content" whenever $j_1$'s content is unique.

**Gap 2 (more important): the barrier is about token content, not attention per se.** If a node token encodes its *full partial assignment* $C_k$ (as a sum of literal embeddings $e(C_k)=\sum_{\ell\in C_k}e_\ell$ with near-orthogonal $e_\ell$), then one bilinear head computes

$$
\langle e(C_j),e(C_k)\rangle-|C_j| \;=\; |C_j\cap C_k|-|C_j| \;=\; \begin{cases}0 & C_j\subseteq C_k\iff j\in\mathrm{Anc}(k)\\ <0&\text{otherwise.}\end{cases}
$$

With large $\beta$, this yields the descendant mask from content alone. Flat attention can therefore recover ancestry when tokens are full-assignment encodings.

R12's Stage 2 attends over **edge keys**, $G_{r,r'}=\max(0,\langle k_r^{\text{edge}},k_{r'}^{\text{edge}}\rangle)$, and each edge key encodes a single clamp. That is why the barrier bites in R12.

> **Corrected Theorem R4.1(b′).** Consider an operator built from $L_1$ 1-hop mixing layers followed by any number of unmasked, permutation-equivariant attention layers over tokens whose features are functions of the **radius-$L_1$ rooted ego-neighbourhood** (with edge-local clamp content only, and no path-assignment or ancestry encoding).
>
> Then there exist trees $\mathcal T^{(1)},\mathcal T^{(2)}$ with identical multisets of radius-$L_1$ ego-neighbourhood features, but with $|Q^{(1)}(j_1)-Q^{(2)}(j_1)|\ge 1$. The construction is as follows.
>
> - Attach chains of length $\ell>2L_1+1$ under content-identical siblings $j_1,j_2$ (e.g. symmetric moves).
> - Place the opposite-valued leaves at the chain ends.
> - Swap the two chain tails between the trees.
>
> Every radius-$L_1$ ball is preserved under the swap, so the operator cannot distinguish the two trees. The worst-case error is $\ge \tfrac12$, matching Member 18's bound once the depth condition $\ell>2L_1+1$ is used instead of "depth 2".
>
> **Corollary.** Feeding full-path assignment tokens (or the explicit masks) breaks the barrier. That is exactly what v1 does.

### 1.3 Theorem R4.2 (Constant-depth MCTS in v1): verified, with eight edge cases and exact fixes

**Core claim: correct.** Under AlphaZero semantics, each simulation creates exactly one new node $k$ and backs up $v_k$ along $\mathrm{Anc}(k)$. Therefore

$$
N(j)=|\mathrm{Desc}(j)|, \qquad W(j)=\sum_{k\in\mathrm{Desc}(j)}s_js_kv_k .
$$

A $\beta=0$ head masked by $\mathcal M_{\text{desc}}$ with value readout $s_kv_k$ returns $\frac1{N(j)}\sum s_kv_k$. An elementwise multiplication by $s_j$ (via a parity feature) then gives $Q(j)$ exactly.

An LSE head with $w=0$ returns $\log(\epsilon+N(j))$. Neither quantity depends on the tree depth $D$.

The edge cases below are where "represents" needs qualification.

| # | Edge case | Failure mode | Exact fix |
|---|---|---|---|
| **E1** | **Two-player sign** | A plain subtree mean of $v_k$ mixes perspectives. | Value readout $s_kv_k$, then multiply by $s_j$ after pooling. The sign factor separates as a product of a $k$-term and a $j$-term, so one head suffices. Same for $\log N$ (no sign). |
| **E2** | **Terminal revisits** | A simulation that reaches an already-terminal node creates no node but must increment $N$ and add the terminal value. Then $N(j)\ne|\mathrm{Desc}(j)|$. | Give each node a multiplicity counter $c_k$ (incremented on revisit, stored per node in $O(1)$) and an accumulated value $w_k$. Use LSE with $w=0$ over value $\log c_k$, i.e. $\log\sum_{k\in\mathrm{Desc}}c_k$, and the ratio $Q=\frac{\sum_{\mathrm{Desc}}w_k}{\sum_{\mathrm{Desc}}c_k}$ (two $\beta{=}0$ heads plus an MLP divide). |
| **E3** | **Unexpanded children / FPU** | PUCT's argmax at the selected leaf ranges over *legal unexpanded actions* with $N=0$. These are not tokens, so v1 can only choose among existing nodes. | Add **frontier tokens**: per expanded node $j$, the top-$K$ unexpanded actions by prior, with $Q=\mathrm{FPU}(j)$ and $N=0$, plus an "other" token with the residual prior mass. Cost is $O(Kb)$ tokens. Without this, Theorem R4.2 covers only *which existing node to descend to*, not *which action to expand*. |
| **E4** | **Parent count convention** | AlphaZero uses $\sqrt{\sum_{c\in\mathrm{Ch}(p)}N(c)}=\sqrt{N(p)-1}$, not $\sqrt{N(p)}$. | MLP input $\log N(p)$; the MLP computes $\sqrt{e^{\log N(p)}-1}$. This is a smooth function on $[\log 1,\log B]$, so it is approximable to any $\varepsilon$. |
| **E5** | **Empty masks giving NaN** | An only child has $\mathrm{Sib}(j)=\emptyset$ and the root has $\mathrm{Anc}^\circ=\emptyset$. Masked softmax over an empty set is $0/0$. | Add a per-row **null key** with fixed logit $-C$ and value $-C$, so $\max\emptyset:=-C$ and $\mu_{\text{edge}}=U+C>0$. Use boolean masks, not additive $-10^9$ (which overflows in fp16/bf16). |
| **E6** | **Ties in $\mu_{\text{edge}}$** | If $U(j)=U(j')$ for siblings, then $\mu=0$ for both. With "$S_{\text{path}}>0$" no child is on the path; with "$\ge 0$" both are. | Lexicographic tie-break matching `np.argmax` (first index). Define $j$ as selected iff $\min_{s\in\mathrm{Sib},\,s<j}(U_j-U_s)>0$ **and** $\min_{s\in\mathrm{Sib},\,s>j}(U_j-U_s)\ge 0$. This needs the **directed older-sibling mask** $\mathcal M_{\text{sib}<}$ (from V2.1) and its complement $\mathcal M_{\text{sib}>}$, with two margins combined by min. Without directed masks, v1 is exact only on tie-free trees. |
| **E7** | **Finite temperature** (see Theorem R4.2′ below) | A softmax-weighted "max" *underestimates* the max by up to $\frac{m-1}{e\beta}$. So $\mu$ is *overestimated*, which causes **false positives**: two near-tied siblings both pass. | Use **conservative LSE bounds** (Theorem R4.2′). Then the soft path is a subset of the true path, and the induced selection is $\varepsilon$-greedy PUCT. |
| **E8** | **Depth embeddings and OOD** | If tokens carry absolute-depth embeddings learned on $D\le 8$, depth-OOD tests fail for trivial reasons. | Allow only parity, relative-depth and mask-derived features. Pre-register this in `R13-D0`. |

**Finite-temperature bounds (proved).** For scores $x_1,\dots,x_m$ with $x^*=\max_i x_i$, let $\mathrm{LSE}_\beta(x)=\beta^{-1}\log\sum_i e^{\beta x_i}$. Then:

$$
0\le \mathrm{LSE}_\beta(x)-x^*\le \frac{\log m}{\beta}.
$$

Writing $p_i\propto e^{\beta x_i}$ and $g_i=x^*-x_i$:

$$
0\le x^*-\textstyle\sum_ip_ix_i=\sum_ip_ig_i\le\sum_i g_ie^{-\beta g_i}\le\frac{m-1}{e\beta}.
$$

The last step uses $\max_{g\ge0}ge^{-\beta g}=1/(e\beta)$.

> **Theorem R4.2′ ($\varepsilon$-PUCT equivalence).**
>
> **Setup.** Assume frontier tokens (E3), multiplicity counters (E2), null keys (E5), and parity features (E1). Replace Layer 2's sibling max by $\mathrm{LSE}_\beta$ over $\mathcal M_{\text{sib}}$ with null key, and Layer 3's ancestor min by $-\mathrm{LSE}_\beta(-\cdot)$ over $\mathcal M_{\text{anc}}$. Then:
>
> $$\hat\mu(j)=\hat U(j)-\mathrm{LSE}_\beta\big(\{\hat U(s)\}_{s\in\mathrm{Sib}(j)}\big),\qquad \hat S(j)=-\mathrm{LSE}_\beta\big(\{-\hat\mu(a)\}_{a\in\mathrm{Anc}^\circ(j)}\big),$$
>
> where $|\hat U-U|\le\eta$ is the MLP approximation error.
>
> **Bounds.** It follows that
>
> $$\hat S(j)\;\le\; S_{\text{path}}(j)+2\eta,\qquad \hat S(j)\;\ge\; S_{\text{path}}(j)-2\eta-\frac{\log m_{\max}+\log D}{\beta}.$$
>
> **Conclusions.**
>
> 1. **No false positives.** If $\hat S(j)>2\eta$ then $j$ lies on the exact greedy PUCT path.
> 2. **$\varepsilon$-PUCT.** Let the network select $\hat\jmath=\arg\max_{\text{frontier}}\hat S$, with depth-major tie-break. The chosen path then satisfies, at every level, $U(\text{chosen child})\ge\max_{\text{siblings}}U-\varepsilon$, with
>    $$\varepsilon=4\eta+\frac{\log m_{\max}+\log D}{\beta}.$$
> 3. **Exact match on margin trees.** On trees whose sibling PUCT gaps all exceed $\varepsilon$, the selection equals AlphaZero's exactly.
>
> **Numerical instance.** For $B=256$, $m_{\max}\le 32$, $D\le 32$ and $\eta=10^{-3}$, exactness for gaps $\ge 0.01$ needs $\beta\ge 6.93/0.006\approx 1.2\times10^{3}$. This is well within bf16 range only if logits are computed as $\beta(x-x^*)$ (max-subtraction is mandatory).

**Precision note.** A $\beta=0$ mean over up to 256 terms in bf16 has relative error about $256\cdot 2^{-8}$ in the worst case. Accumulate attention outputs in fp32. I recommend this for all mask-bank heads.

**Per-step cost.**

- **Full recomputation:** $O(b^2dH)$ per layer.
- **Incremental update:** appending node $b$ changes the $\mathrm{Desc}$-aggregates only for $a\in\mathrm{Anc}(b)$. For $\beta=0$ sums and LSE accumulators this costs $O(d(b)\cdot dH)$, using running sums and running log-sum-exp.
- **Note:** $\beta\ne0$ softmax with a *query-dependent* score is not incrementally updatable, so those heads need full recomputation.

### 1.4 Theorem R4.3: the five barriers, verified and corrected

#### Barrier 1: arbitrary-depth minimax and PNS

**Complexity correction.** The claim "NC¹ depth-$D$ barrier" should be stated conditionally.

- Constant-depth, log-precision transformers lie in uniform $\mathsf{TC}^0$ (Merrill & Sabharwal).
- The Boolean formula value problem is $\mathsf{NC}^1$-complete (Buss 1987).
- Supplying $\mathcal M_{\text{anc}}$ does not help. The ancestor relation of a balanced-parenthesis encoding is itself $\mathsf{TC}^0$-computable (via prefix counts), so giving it as input cannot reduce hardness.

> **Corrected statement.** Unless $\mathsf{TC}^0=\mathsf{NC}^1$, no fixed-$L$ v1 with poly-size and log-precision computes the minimax value of all depth-$D$ alternating trees.
>
> By contrast, MCTS's subtree mean is a threshold/counting computation in $\mathsf{TC}^0$. This is the precise sense in which MCTS is "easy" and minimax is "hard" for v1.

**Edge cases in the V2.1 fix**

- **B1a: non-invertible incremental max.** Appending a child can *decrease* a child's backed-up value (e.g. a refutation is found deep down). $\max$ has no inverse, so ancestor $a$'s state cannot be updated from the delta alone.
  - *Fix:* recompute $\text{backup}[a]=\bigoplus_{c\in\mathrm{Ch}(a)}\text{backup}[c]$ over all children of every spine node.
  - Cost per step: $O\big(\sum_{a\in\mathrm{Anc}(b)}m_a\big)\le O(b)$. Still $O(b)$, but state it.
- **B1b: PNS infinities.** Proof-number search uses $\mathrm{pn}=0/\infty$ for proven or disproven nodes. Here $\sum$ is applied with $\infty$, and $\min$ over sums.
  - *Fix:* saturating arithmetic in $\log(1+\cdot)$ space with a sentinel $\log(1+\infty):=\Lambda$.
  - Example of the backup rules: OR nodes use $\mathrm{pn}=\min_c\mathrm{pn}_c$ and $\mathrm{dn}=\sum_c\mathrm{dn}_c$; AND nodes swap the roles.
  - The **most-proving node** is reached by descending OR→min-pn and AND→min-dn children. That is a path-bottleneck selection of the same form as Layer 3, so it reuses the $\varepsilon$-PUCT machinery.
- **B1c: alpha-beta window composition.** The $(\alpha,\beta)$ window at node $j$ is
  $$\alpha(j)=\max_{a\in\mathrm{Anc}(j),\ \text{MAX}(\mathrm{pa}(a))}\ \max_{s\in\mathrm{Sib}_<(a)}\text{backup}[s],$$
  and dually for $\beta$. This is a two-hop composition $\mathcal M_{\text{anc}}\circ\mathcal M_{\text{sib}<}$, which needs 2 layers on top of V2.1 states.
  - Cutoff test: $\alpha(j)\ge\beta(j)$.
  - Alpha-beta also fixes an expansion *order* (depth-first, left to right). To imitate it, the selection must equal the DFS-next unpruned node. The rule "deepest node on the current DFS path that has an unexpanded, unpruned younger sibling or child" is representable with $\mathcal M_{\text{anc}}$, $\mathcal M_{\text{sib}>}$ and frontier tokens.
- **B1d: is the backup learned or hard-coded?** If `backup_state` applies a hand-coded semiring, then "learns minimax" is partially baked in.
  - *Fix:* make the spine operator a **learned gated aggregator** over children states, $\text{backup}[a]=\mathrm{Agg}_\theta(\{\text{backup}[c]\}_{c\in\mathrm{Ch}(a)},z_a)$, with heads for $\{\text{mean},\mathrm{LSE}_{\pm\beta},\text{sum}\}$ and a learned per-node-type gate.
  - `R13` must report the gate entropy, and whether the gate *learned* to choose max at MAX nodes and min at MIN nodes.

#### Barrier 2: transpositions and MCGS — the V2.2 fix as stated is incorrect

1. **Double-counting.** $(E_{\text{trans}}\mathcal M_{\text{desc}})[j,k]=\sum_\ell E[j,\ell]\mathcal M_{\text{desc}}[\ell,k]$ counts $k$ once for every transposition-mate $\ell$ of $j$ whose subtree contains $k$. Means and counts are inflated.
   - *Fix:* Boolean semiring, $\mathcal M^{\text{DAG}}=\mathbb 1[E\mathcal M>0]$.
2. **Not closed.** One composition captures one transposition hop. Descendants of $\ell$ may themselves be transpositions of other nodes.
   - *Fix:* canonicalise at insertion (KataGo-style). Node identity is $\mathrm{canon}(\text{state})$; a transposed expansion adds an **edge** $(p\to\ell)$ to the existing node $\ell$, not a new node.
   - Maintain DAG reachability incrementally: on a new edge $p\to\ell$, every $a\in\mathrm{Anc}^{\text{DAG}}(p)$ gains $\mathrm{Desc}^{\text{DAG}}(\ell)$.
   - Cost: $O(b\cdot b/64)$ word-ops per transposition event, which is about 1k word-ops at $b=256$.
3. **Cycles.** Repetitions (Go superko, chess threefold repetition, reversible moves in Connect-4 variants with removal) make the graph cyclic. Then $\mathrm{Desc}$ contains ancestors and a "subtree mean" is circular. This is the Graph-History-Interaction (GHI) problem.
   - *Fix:* include history-relevant state in the hash (repetition counters, ko state), which makes the graph a DAG by construction. Alternatively, condense strongly connected components (SCCs) and treat each SCC as a draw-valued super-node.
4. **The MCGS target is not a set-mean.** KataGo/Czech et al. MCGS uses edge visit counts $N(n,a)$:
   $$Q(n)=\frac{U(n)+\sum_aN(n,a)\,Q(c(n,a))}{1+\sum_aN(n,a)}.$$
   Unrolled, this is $Q(n)=\sum_k\omega(n,k)U(k)$, where $\omega(n,k)$ sums over **all DAG paths** the product of edge-visit fractions. That is a matrix power series $(I-W)^{-1}$ restricted to a DAG.
   - On a tree, $N(n,a)=N(c)$ and the formula reduces to the subtree mean.
   - On a DAG it does **not** equal $\frac1{|\mathrm{Desc}^{\text{DAG}}|}\sum U$.
   - *Fix:* implement MCGS through V2.1. Recompute $Q$ from children with an edge-count-weighted mean along the traversed spine. Do not use a reachability-mask mean.
   - Reachability masks remain useful for *counting* and for solver flags.
5. **Soft transpositions leak.** $E\supseteq\exp(-\|h-h'\|^2/\tau)$ merges *distinct* states. That leaks value between them and can break solver soundness.
   - *Fix:* use exact $\mathrm{canon}$ equality for value pooling. Soft similarity may enter only the **prior/selection** channel, never backed-up values or proofs. This is a firewall rule and is pre-registered.
6. **Transposition is not information-set membership.** In partially observable games, $\mathrm{canon}(O_j)=\mathrm{canon}(O_k)$ on *observations* defines information sets, not transpositions.
   - **Single-agent POMDPs vs nature (ShipHunt):** merging nodes with equal *observation-action history* is correct, because the belief state is a sufficient statistic.
   - **Two-player games (Stratego):** the same merge conflates hidden states and causes **strategy fusion**.
   - *Fix:* in multi-agent PO games, $E_{\text{trans}}$ keys on the full state; information-set pooling belongs to V2.3 only.

#### Barrier 3: CFR+, ReBeL and R-NaD — the V2.3 fix is one-shot regret matching, not CFR+

**C3a. Iteration.** CFR+ maintains cumulative clipped regrets and outputs the average strategy:

$$
R^{t}_+(I,a)=\max\big(0,\;R^{t-1}_+(I,a)+r^t(I,a)\big),\qquad \bar\sigma^T(I)\propto\sum_t t\,\pi^{\sigma^t}_i(I)\,\sigma^t(I).
$$

The output must be the **average** strategy $\bar\sigma$; the current strategy $\sigma^t$ does not converge in CFR+. A single forward pass gives $\sigma\propto\mathrm{ReLU}(r)$ from one instantaneous counterfactual value. That is one regret-matching step, not CFR+.

- *Fix:* add a **per-infoset persistent memory** keyed by information set, holding $(R_+,\ \bar\sigma\text{-numerator},\ \text{denominator})$ in $O(|\mathcal I|\cdot|A|)$. Update it at each search step, treating one step as one CFR iteration over the current tree. This is V2.1 generalised from the ancestor spine to infoset keys.

**C3b. Perfect recall.** $\mathcal M_{\text{infoset}}[j,k]=\mathbb 1[\mathrm{Obs}_{\text{pl}}(C_j)=\mathrm{Obs}_{\text{pl}}(C_k)]$ on the *current observation* merges histories the player can distinguish by memory. That violates perfect recall and voids CFR's convergence guarantee.

- *Fix:* key on the player's full **observation-action sequence** hash.

**C3c. Reach underflow and signed values.** The counterfactual value is

$$
v_i(I,a)=\sum_{h\in I}\pi_{-i}(h)\,u_i(h\cdot a).
$$

The opponent reach $\pi_{-i}$ is a product of opponent and chance action probabilities along ancestors, which underflows quickly.

- *Fix:* compute in log space. Get $\log\pi_{-i}(j)$ from a $\beta=0$ $\mathcal M_{\text{anc}}$ head times $|\mathrm{Anc}|$, with value $\mathbb 1[\mathrm{pl}(a)\ne i]\log\sigma(a)$ including chance probabilities.
- Split $u=u^+-u^-$ and compute two infoset-masked LSE heads, $\log\sum_h e^{\log\pi_{-i}(h)+\log u^\pm(h)}$.

**C3d. LIR firewall preservation.** The infoset readout must be invariant to permuting hidden states within an information set.

- If the tree is expanded from the *true* hidden state (determinization by the ground truth), the readout leaks hidden information even with correct masks.
- *Fix:* expand the tree only from public-belief-sampled determinizations.
- *Test (hard bar in `R13-D5` and `R14-D4`):* permute the hidden-state assignments of histories within each infoset; the max difference in output must be $\le10^{-6}$. This is a PoisonTensor-style canary.

**C3e. Category errors in the barrier's algorithm list.**

- **R-NaD (DeepNash)** is a *training-time* learning dynamic. DeepNash uses no test-time search, so R-NaD should not be listed as a search algorithm.
- **ReBeL** requires public-belief-state (PBS) Bayes updates over the joint infostate space of *both* players, plus safe subgame re-solving. That needs an unbounded fixed point per step and is outside v2 (see §1.6).

#### Barrier 4: Gumbel MuZero Sequential Halving — V2.4 is correct up to temperature, with three omissions

**Temperature bound.** Write $g_{js}=S_j-S_s$. Then

$$
\big|\widehat{\mathrm{pct}}(j)-\mathrm{pct}(j)\big|\le\frac1{|\mathrm{Sib}|}\sum_s\sigma(-|g_{js}|/\tau_{\text{rank}})\le e^{-\Delta/\tau_{\text{rank}}},
$$

where $\Delta=\min|g_{js}|$. Survivor thresholds sit at multiples of $1/m$, so correct rounding needs error $<\frac1{2m}$, i.e.

$$
\tau_{\text{rank}}<\frac{\Delta}{\log 2m}.
$$

**Ties** occur with probability zero under Gumbel noise but do occur among unvisited actions with equal logits. Break them by index with $\mathcal M_{\text{sib}<}$, as in E6.

**Three omissions:**

1. **Phase scheduling.** Each phase $p$ gives every surviving action $\lfloor B/(\lceil\log_2 m\rceil\cdot m_p)\rfloor$ visits. The phase index is a function of global counters, so a $\mathcal M_{\text{global}}$ count head plus an MLP is needed. State this explicitly.
2. **Non-root selection.** Gumbel MuZero's non-root rule is deterministic:
   $$\arg\max_a\Big(\pi'(a)-\frac{N(a)}{1+\sum_bN(b)}\Big),\qquad \pi'=\mathrm{softmax}(\text{logits}+\sigma(\text{completedQ})).$$
   Here "completed Q" fills unvisited actions with $v_{\text{mix}}$. This needs frontier tokens (E3).
3. **$\sigma(\hat q)$ transform.** The transform is $(c_{\text{visit}}+\max_bN(b))\cdot c_{\text{scale}}\cdot\hat q$, which needs $\max N$ over siblings. An LSE head over $\log N$ on $\mathcal M_{\text{sib}}\cup\mathcal M_{\text{self}}$ provides it.

#### Barrier 5: CDCL global nogoods — V2.5's mask is necessary but not sufficient

$\mathcal M_{\text{nogood}}[j,k]=\mathbb 1[m_{\text{ql}}(C_k)<\tau_{\text{clash}}]$ marks *which nodes conflicted*. Pruning $j$ requires $C_j\supseteq\text{core}(k)$, a subset test. Fix it as follows.

- **Subset-test head.** Score $\langle e(C_j),e(\text{core}_k)\rangle-|\text{core}_k|\ge0$, masked by $\mathcal M_{\text{nogood}}$. This is the same bilinear trick as in §1.2.
- **Core extraction.** 1-UIP conflict analysis walks the *implication graph* inside a node, which is sequential and not represented. Two options:
  - The **decision nogood** (all decision literals on the root-to-conflict path) is always sound, given complete propagation, and is representable with $\mathcal M_{\text{anc}}$. It is weaker than 1-UIP.
  - A learned sparse attention over path clamps may produce smaller cores, but **soundness is not guaranteed**.
- **Firewall.** Learned (soft) nogoods may only *deprioritise*. Hard pruning requires the core to be verified by the propagator ($O(|\text{core}|\cdot\text{props})$).
- **Hard bar.** Zero pruned subtrees containing a true solution (`R13-D3c`).

### 1.5 Complexity summary per search step ($b\le256$, $d$ = model width, $H$ = heads)

| Component | Per step, full recompute | Per step, incremental |
|---|---|---|
| v1 mask-bank attention (7 channels, 3 layers) | $O(L\,b^2dH)$ | $O(L\,d(b)\,dH)$ for sum/LSE heads; full recompute for query-dependent $\beta\ne0$ heads |
| V2.1 spine semiring backup | — | $O(\sum_{a\in\mathrm{Anc}(b)}m_a)\le O(b)$ |
| V2.2 DAG reachability (bitset) | $O(b^3/64)$ | $O(b^2/64)$ per transposition event |
| V2.3 infoset pooling + log-reach + regret memory | $O(b^2dH)$ | $O(|\mathcal I|\,|A|)$ memory update |
| V2.4 pairwise sibling rank | $O(\sum_jm_j^2)\le O(b^2)$ | same |
| V2.5 nogood subset test | $O(b\cdot n_{\text{ng}}\cdot\ell)$ | $O(n_{\text{ng}}\cdot\ell)$ per new node |
| Frontier tokens (E3) | $\times(1+K)$ tokens | — |

**Full-recompute cost per search.** Summing over steps gives $\sum_{b\le B}b^2\approx B^3/3\approx5.6\times10^6$ at $B=256$. Multiplying by $L\,d\,H\approx 3\cdot256\cdot8$ gives about $3.4\times10^{10}$ FLOPs per search. That is roughly 1–3 ms per instance on an L4 at realistic utilisation, or about 10–40 ms with $K=4$ frontier tokens and v2 heads. This is affordable for the prior rounds' evaluation budgets.

### 1.6 Answer to Q2: what v2, even with all fixes above, still cannot represent

I limit this to algorithms with documented multi-game use.

| Algorithm (games) | Required primitive | v1 | v2 | Minimal `v3` fix |
|---|---|---|---|---|
| **A\*, greedy best-first, beam search** (puzzles, Sokoban, planning) | Global argmin of $f=g+h$ over frontier | ✅ with frontier tokens | ✅ | — (positive control) |
| **MCTS-Solver / score-bounded MCTS** (Winands 2008; Lines of Action, Amazons, Breakthrough) | Minimax over proven flags | ❌ | ✅ (V2.1) | — |
| **Expectimax, \*-minimax / Star1–Star2** (Backgammon, 2048, Can't Stop) | Typed semiring $(\max,\min,\sum p)$ plus probing bounds | ❌ | ✅ for values; ⚠ Star2 probing order | Typed node gate in `Agg_θ` |
| **max^n, paranoid, Best-Reply Search** (Chinese Checkers, Hearts, Blokus) | Vector-valued backups, "maximise own component" | ❌ | ⚠ needs vector state plus player index | Vector backup state |
| **RAVE / GRAVE** (Go before AlphaZero; GRAVE is a General Game Playing standard across dozens of games) | Pool returns of action $a$ played *anywhere later* in simulations through $\mathrm{pa}(j)$ | ❌ | ❌ | **Action-identity mask** $\mathcal M_{\text{amaf}}[j,k]=\mathbb 1[k\in\mathrm{Desc}(\mathrm{pa}(j))\land a(j)\in\text{path}(\mathrm{pa}(j)\to k)]$ |
| **MAST / Playout Policy Adaptation / NST** (GGP, Ludii) | Global per-action statistics across the whole tree | ❌ | ❌ | $\mathcal M_{\text{act}}[j,k]=\mathbb 1[a(j)=a(k)]$ (hash join) |
| **History heuristic, killer moves, countermove** (all alpha-beta chess, checkers and shogi engines) | Cross-branch move-ordering statistics | ❌ | ❌ | Same action-identity masks, keyed by (ply, move) |
| **Iterative deepening, aspiration windows, PVS, MTD(f)** (Stockfish-class engines) | Re-search the same nodes with new windows/depths; transposition-table reuse | ❌ | ❌ (append-only bank; budget counts nodes, not visits) | **Visit-event tokens / re-search op** distinct from node creation |
| **IDA\*, df-pn thresholds** (puzzles, tsume-shogi, Go life-and-death) | Repeated threshold-bounded DFS with re-expansion | ❌ | ⚠ df-pn values yes; re-expansion schedule no | Same as above |
| **CDCL restarts, phase saving** (SAT, Sudoku) | Discard tree, keep learned clauses and phases | ❌ | ❌ | **Reset op** plus persistent clause and phase memory |
| **NRPA / NMCS** (Morpion Solitaire, SameGame, TSP-like) | In-search policy-weight adaptation; nested-level recursion | ❌ | ❌ | **Fast-weight memory** written by search (looped / test-time training (TTT) slot) |
| **CFR-D / DeepStack continual re-solving / ReBeL** (poker variants, Liar's Dice) | Safe re-solving gadget; PBS Bayes over the joint infostate space; inner CFR loop of $T$ iterations per step | ❌ | ❌ (V2.3 gives per-step regret memory only) | **Looped inner block** with $T$ adaptive iterations, plus a gadget-node constructor |
| **SM-MCTS with EXP3 / regret matching; SO-/MO-ISMCTS** (simultaneous-move games, card games) | Per-node stochastic bandit state; availability counts for subset-armed bandits | ❌ | ⚠ (V2.3 analogue per joint action; availability counts missing) | Availability-count head $\sum_k\mathbb 1[a\text{ legal at }k]$ |
| **Progressive widening / unpruning** (large or continuous action spaces, Sampled MuZero) | Children allowed $=\lceil CN^\alpha\rceil$; ordered unpruning | ❌ | ⚠ (rank head plus frontier tokens) | Frontier tokens are the bottleneck |
| **Retrograde analysis / endgame tablebases** (Checkers, Nine Men's Morris) | Backward induction from terminals on a cyclic graph | ❌ | ❌ | Fixed-point loop on a cyclic graph (looped block) |
| **Lambda-search / threat-space search / null-move pruning** (Go-Moku, Hex, chess) | Virtual "pass" nodes and threat-order restrictions | ❌ | ⚠ (if pass nodes are allowed) | Virtual-node creation op |

**Characterisation (informal theorem).** With frontier tokens and the fixes in §1.3–§1.4, v2 represents exactly those algorithms whose per-step decision is computed by:

1. a constant number of masked aggregation rounds over the relations $\{\text{self},\text{parent},\text{child},\text{anc},\text{desc},\text{sib},\text{sib}_<,\text{infoset},\text{DAG-desc},\text{nogood},\text{global}\}$, with sum/mean/LSE/rank pooling; plus
2. $O(1)$ incremental recurrences keyed by ancestor spine or infoset.

It cannot represent algorithms that need any of:

- **(i)** relations keyed by *action identity* across branches;
- **(ii)** non-append-only state (re-search, reset);
- **(iii)** unbounded inner fixed points per step;
- **(iv)** in-search parameter adaptation.

**Proposed minimal v3:**

- `V3.1` action-identity hash masks;
- `V3.2` visit-event / reset ops;
- `V3.3` looped inner block with adaptive iteration count;
- `V3.4` fast-weight slots.

None of these should be built before `R13` confirms that v2's primitives are *learnable*.

---

## 2. Specification Input for `PREREG-R13-ALGO-IMITATION.md`

### 2.1 Purpose and design principle

`R13` asks two separable questions for each algorithm $A\in\{\text{Algo 1..6}\}$ and architecture $\mathcal A\in\{$`Full-GLCSearch-v2`, `Abl-GLCSearch-v1`, `Abl-R12-1Hop-Flat`$\}$:

- **(R) Representability.** Do there exist weights reproducing $A$? This is tested with **constructive weights (CW)**: hand-set weights following §1.3–1.4, with no training.
- **(L) Learnability.** Does supervised training on teacher traces find such weights, how sample-efficiently, and do they generalise out of distribution in budget and depth?

**Why both are needed.** A learnability failure alone is uninterpretable. CW success combined with learned failure isolates an *optimisation* problem. CW failure on an arm that the theorems say should succeed means the theorem or the implementation is wrong.

### 2.2 Teachers, environments and traces

**Shared frozen components (critical).** Every teacher uses the same frozen prior/value network $f_\phi$, i.e. the R12 unified generalist. This makes each teacher's decision a deterministic function of the tree state, plus a seeded RNG for Gumbel.

| Algo | Teacher (reference implementation) | Environments | Labelled per-step targets |
|---|---|---|---|
| 1 | AlphaZero PUCT, $c_{\text{puct}}=1.25$, FPU $=$ parent $Q-0.2$ | Hex-5×5, Connect-4 (6×7), synthetic Pearl P-game trees | Next selected frontier token; $Q(j)$; $\log N(j)$; root visit distribution |
| 2a | Fail-hard alpha-beta, DFS order, static move ordering by prior | Connect-4 endgames, P-games (depth 4–14, branching 2–5) | Next expanded node; backed-up minimax value; cutoff flags |
| 2b | df-pn (proof-number search) | Equiv-SAT (OR/AND framing), Hex-4×4 proving, Connect-4 endgames | $\log(1{+}\mathrm{pn}),\ \log(1{+}\mathrm{dn})$; most-proving node |
| 3 | CDCL with decision nogoods + 1-UIP variant (2 teachers), restarts disabled | Sudoku-6×6, Sudoku-9×9-Extreme, Equiv-SAT | Next decision; prune flags; nogood membership |
| 4 | MCGS (Czech et al. / KataGo semantics), exact-canon hashing, history-augmented hash | Hex-5×5 (rich transpositions), Connect-4, ShipHunt (belief-MDP keying) | Edge-count-weighted $Q$; merge decisions; next selection |
| 5 | Per-step CFR+ over current tree, linear averaging, perfect-recall keys | Kuhn, Leduc, Proposal-Poker, Micro-Stratego (small) | $R_+(I,a)$ normalised; current $\sigma$; average $\bar\sigma$; exploitability of $\bar\sigma$ |
| 6 | Gumbel MuZero root SH plus deterministic non-root selection | Hex-5×5, Connect-4, Sudoku-6×6 | Survivor set per phase; next selection; final action |

**Trace generation**

- Each training *episode* is one search run of budget $B_{\text{train}}\sim\mathrm{Unif}\{8,\dots,32\}$ on a fresh instance, producing about $B$ labelled decisions.
- **DAgger mixing.** The tree is grown by the model with probability $1-\rho_t$ and by the teacher with probability $\rho_t$, with $\rho_t$ annealed from $1\to0.25$. The teacher labels every visited state.
- **Ties.** Teacher ties produce *set-valued* labels $\mathcal A^*_t$ (see E6).

**Data split.** Instances are hashed into train / val / test by instance seed. Test instances are generated only after training checkpoints are frozen (blind).

### 2.3 Losses

$$
\mathcal L_{\text{R13}}=\underbrace{-\log\!\!\sum_{a\in\mathcal A^*_t}\!p_\theta(a\mid\mathcal T_t)}_{\mathcal L_{\text{sel}}}
\;+\;\lambda_{\text{stat}}\!\!\sum_{j\in\mathcal T_t}\sum_{c\in\mathcal C_A}\!w_c\,\mathrm{Huber}_\delta\big(\hat s_c(j),s_c^{A}(j)\big)
\;+\;\lambda_\pi\,\mathrm{KL}\big(\pi^{A}_{\text{root}}\Vert\pi_\theta\big)
\;+\;\lambda_{\text{gate}}\,\mathcal L_{\text{gate-ent}}.
$$

- $\mathcal C_A$ are the algorithm's statistics channels from the table above.
- $\mathcal L_{\text{gate-ent}}$ is a small entropy bonus on the learned backup-aggregator gate (B1d), to avoid premature collapse.

**Supervision ablation (pre-registered).** Answering Kelvin's "learn it efficiently" requires comparing three supervision regimes:

- **`Sup-Dense`:** all terms.
- **`Sup-Behav`:** $\lambda_{\text{stat}}=0$, i.e. imitation from decisions only.
- **`Sup-Final`:** only $\lambda_\pi$, i.e. only the root output.

### 2.4 Experimental matrix

- **Arms:** 3 architectures × 6 algorithms (counting 2a and 2b as Algo 2) × $N_{\text{train}}\in\{128,512,2048\}$ episodes × 3 seeds, with `Sup-Dense`. That is $3\times7\times3\times3=189$ runs.
- **Plus:** `Sup-Behav` and `Sup-Final` for `Full-v2` at $N=2048$ (42 runs).
- **Plus:** CW evaluations (no training).
- **Evaluation:** every run is evaluated in-distribution ($B\le32$) and blind OOD at $B\in\{64,128\}$, plus depth-OOD on P-games (train depth $\le6$, test depth 10 and 14).
- **Statistics:** per-arm means with seed-cluster bootstrap 95% CIs. Holm–Bonferroni correction over the ablation-separation family (§2.5, sub-bar e).
- **Cost forecast `[PRIOR]`:** \$8–\$25 on L4, using models at R12 scale (≈2.6M parameters).

### 2.5 Pass bars

**`R13-D0` — Integrity and theorem falsification (gate; all must pass before D1–D6 are interpreted)**

- **D0a:** 0 static AST violations. The teacher modules are importable only by the trace generator, never by the model or eval code.
- **D0b:** 3/3 canaries pass. PoisonTensor `global_max_diff` $=0$ and `r4_kv_equiv_max_diff` $=0$ (inherited bars).
- **D0c: incremental-vs-recompute equality.** Incremental and full-recompute v2 states (V2.1 backups, DAG bitsets, regret memory) agree to $\le10^{-6}$ max abs diff on 1,000 random traces.
- **D0d: theorem falsification.**
  - On the R4.1(b′) adversarial pair set (512 pairs) and on depth-$>L$ chains, `Abl-R12-1Hop-Flat` must score $\le 60\%$ on pair discrimination (chance is 50%).
  - If it scores higher, the audit's model of R12's code is wrong and Theorem R4.1 must be revised before R13 conclusions are drawn.
  - `Full-v2` CW must score $\ge 99.5\%$ on the same set.
- **D0e:** no absolute-depth embeddings (E8); this is checked by AST rule and config schema.
- **D0f: CW representability.** Constructive weights must meet the following on margin-filtered test trees (sibling gaps $\ge\varepsilon$ as in Theorem R4.2′):

  | Algorithm | Arm | Required |
  |---|---|---|
  | Algo 1 | v1 and v2 | $\ge99.5\%$ selection match |
  | Algos 2–6 | v2 | $\ge99.5\%$ selection match |
  | Algos 2–6 | v1 | Documented to fail on barrier-specific test sets |

  This sub-bar answers "can it represent?"

**Per-algorithm bars `R13-D1`..`R13-D6` (learned, `Full-v2`, $N_{\text{train}}=2048$).**

Each algorithm has the same five sub-bars:

- **(a)** in-distribution primary metric;
- **(b)** blind OOD at $B=128$ (and depth-14 for P-games);
- **(c)** statistics fidelity;
- **(d)** sample efficiency, i.e. performance at $N=512$ relative to $N=2048$;
- **(e)** ablation separation (Holm-corrected, one-sided).

| Bar | Algo | (a) ID primary | (b) OOD $B{=}128$ | (c) Statistics | (d) $N{=}512$ / $N{=}2048$ | (e) Pre-registered ablation pattern |
|---|---|---|---|---|---|---|
| **R13-D1** | MCTS-PUCT | Selection top-1 $\ge0.95$ | $\ge0.88$ | $Q$ $R^2\ge0.98$; $\log N$ $R^2\ge0.99$; root-visit KL $\le0.03$ | $\ge0.90$ | **v1 ≈ v2** (|Δ| $\le0.03$, negative control); **v1 − R12 $\ge0.15$** at OOD |
| **R13-D2** | Minimax / α-β / PNS | Minimax-value sign $\ge0.95$ at depth $\le6$; α-β next-node match $\ge0.92$; PNS most-proving node $\ge0.92$ | Value sign $\ge0.88$ at depth 14; next-node match $\ge0.80$ | pn/dn $R^2\ge0.95$; nodes expanded within 10% of teacher | $\ge0.85$ | **v2 − v1 $\ge0.10$** at depth $\ge10$; v1 degrades with depth (slope test, $p<0.01$) |
| **R13-D3** | CDCL-Global-Nogood | Prune-decision F1 $\ge0.92$ | $\ge0.85$ | **Hard: 0 unsound prunes** (verified); decision count within 15% | $\ge0.85$ | **v2 − v1 $\ge0.10$** on the "disjoint-subtree repeat" test set |
| **R13-D4** | MCGS | Edge-weighted $Q$ $R^2\ge0.95$; merge precision $\ge0.995$ | $Q$ $R^2\ge0.90$ | **Hard: 0 soft-merge value leaks** (firewall test); 0 cycle NaNs | $\ge0.85$ | **v2 − v1 $\ge0.08$** in $Q$ $R^2$ on high-transposition Hex |
| **R13-D5** | CFR+ | Exploitability of $\bar\sigma$ $\le1.25\times$ teacher's at equal $T$ (Kuhn, Leduc) | $\le1.5\times$ at $T=128$ | Regret-sign agreement $\ge0.90$; **Hard: infoset hidden-permutation invariance $\le10^{-6}$** | $\ge0.80$ | **v1 exploitability $\ge2\times$ v2's** |
| **R13-D6** | Gumbel SH | Survivor-set exact match per phase $\ge0.93$; final action $\ge0.95$ | $\ge0.88$ | Rank-percentile MAE $\le0.03$ | $\ge0.85$ | **v2 − v1 $\ge0.10$** in survivor exact-match |

**Cross-cutting secondary bars (reported, non-gating)**

- **`R13-X1` (multi-algorithm single model).** One `Full-v2` with an algorithm-ID token is trained on all 7 teachers (6 algorithms, with 2a and 2b separate). It should lose $\le0.05$ on every primary metric relative to single-algorithm models.
- **`R13-X2` (supervision regimes).** Report the ratio of `Sup-Behav` to `Sup-Dense` at $N=2048$. *Hypothesis:* the ratio is $\ge0.9$ for Algos 1 and 6 and $\le0.8$ for Algos 2 and 5. Deep backups and regrets are hard to infer from decisions alone.
- **`R13-X3` (learning-curve exponent).** Fit $\mathrm{err}(N)=aN^{-\alpha}+c$ per arm, and report $\alpha$ with CI and the asymptote $c$.
  - *Hypothesis:* $c_{\text{v2}}<c_{\text{v1}}$ on Algos 2–6, which is a capacity gap.
  - *Hypothesis:* $\alpha$ is similar across arms on Algo 1, i.e. no optimisation gap.
- **`R13-X4` (backup gate interpretability).** The fraction of MAX-type nodes whose learned aggregation gate puts more than 0.8 weight on $\mathrm{LSE}_{+\beta}$ should be $\ge0.9$ (Algo 2).

**Interpretation rule (pre-registered).** For any algorithm whose learned bars fail, the cause is classified as follows.

| CW pass | Learned fails | Classification |
|---|---|---|
| ✅ | ID | **optimisation** |
| ✅ | OOD only (ID passes) | **generalisation** |
| ❌ | — | **expressivity** |

Each class triggers a different next step: curriculum or initialisation, relative features, or `v3` primitives respectively.

---

## 3. Specification Input for `PREREG-R14-UNSUP-DISCOVERY.md`

### 3.1 Principle

The *same* `GLC-Search-Transformer v2` (architecture, primitives and frontier tokens) is trained **only from task outcomes**.

**Allowed signals:**

- $\mathrm{KL}(P^*\Vert\mu_B)$, the end-of-search belief proper score;
- verified witness discovery;
- game payoff;
- exploitability, computed by an external best-response oracle used only as an evaluator.

**Forbidden signals:** any teacher statistic, teacher selection, teacher-shaped auxiliary loss, or initialisation from `R13` (except in the explicitly separate warm-start arm, `R14-D6`).

### 3.2 Objective and credit assignment

**Readout.** The end-of-search belief $\mu_B$ is differentiable and trained directly with the proper score:

$$
\mathcal L_{\text{read}}=\mathbb E\big[\mathrm{KL}(P^*\Vert\mu_B)\big],
$$

or with log-loss on the realised outcome where $P^*$ is unavailable.

**Selection policy.** Selections $a_t\sim p_\theta(\cdot\mid\mathcal T_t)$ are discrete. They are trained with **GRPO** on groups of $G=8$ search traces per instance.

**Trace-level outcome score.** Lower is better:

$$
S_g = \mathrm{KL}(P^*\Vert\mu_B^{(g)})\;-\;\kappa\,\mathbb 1[\text{verified witness found}]\;-\;\kappa'\,\text{payoff}.
$$

**Leave-Subtree-Out counterfactual credit.** For a node $s$ created at step $t$, define $q^{-s}$ as the readout recomputed with $\mathrm{Desc}(s)$ masked out. This is cheap: it is one masked forward pass, batched over sampled $s$. Then

$$
I_{t}\;=\;S(q^{-s_t},y)-S(q^{\text{full}},y).
$$

With $S$ a loss (log-loss or KL), $I_t>0$ means the subtree helped.

**Per-step advantage:**

$$
\hat A_{g,t}=\frac{S_{\bar g}-S_g}{\mathrm{std}_g(S)}+\eta\,\frac{I_{g,t}}{\mathrm{std}(I)},
\qquad
\mathcal L_{\text{sel}}=-\mathbb E\Big[\textstyle\sum_t\min\big(r_t\hat A,\mathrm{clip}(r_t,1\pm\epsilon)\hat A\big)\Big]+\beta_{\text{KL}}\mathrm{KL}(p_\theta\Vert p_{\text{ref}}).
$$

Here $S_{\bar g}$ is the group-mean outcome score, $r_t$ is the per-step policy ratio, and $p_{\text{ref}}$ is the frozen R12 behaviour policy, i.e. the hand-coded selector softened with $\tau=1$.

**Why LSO addresses R12's multi-witness regression.** I hypothesise that the $B=128\to256$ KL uptick on ShipHunt and Stratego came from repeatedly reinforcing the *modal* witness. Under LSO with a proper score against the multi-witness $P^*$:

- A subtree that finds a *second, distinct* witness gets $I>0$, because removing it removes probability mass that $P^*$ requires.
- A *redundant duplicate* of an already-found witness gets $I\approx0$, because the other copy remains.

So LSO credit directly rewards witness diversity and penalises redundancy, which is the opposite of single-witness reinforcement.

**Caveat (pre-registered).** LSO is not Shapley. For pairwise-complementary subtrees, each alone may have $I\approx0$. To cover this, compute pairwise LSO on 10% of steps, $I_{s,s'}=S(q^{-\{s,s'\}})-S(q^{-s})-S(q^{-s'})+S(q^{\text{full}})$, and report the interaction mass.

### 3.3 Baselines (compute-matched: same $f_\phi$, same $B$)

1. The R12 unified generalist with its hand-coded selector.
2. Each fixed algorithm (Algos 1–6) run on every environment where it applies.
3. A **portfolio oracle**: on each environment, the best fixed algorithm chosen with hindsight on the test set. This is a deliberately strong, unfair-to-us baseline.
4. The R13 `Full-v2` multi-algorithm model with an oracle algorithm-ID token.

### 3.4 Mechanistic algorithmic alignment probes

Measured on held-out instances, on the **model's own trees**.

**P1. Behavioural alignment index.** For teacher $A$ and environment $e$:

$$
\mathrm{Agr}_A^e=\Pr\big[a^{\text{model}}_t\in\mathcal A^{*,A}(\mathcal T_t)\big],\qquad
\mathrm{AI}_A^e=\frac{\mathrm{Agr}_A^e-\mathrm{Agr}_{\text{unif}}^e}{\mathrm{Agr}_{A\leftrightarrow A}^e-\mathrm{Agr}_{\text{unif}}^e}.
$$

$\mathrm{Agr}_{A\leftrightarrow A}$ is teacher self-agreement across RNG seeds; it equals 1 for deterministic teachers.

**P2. Representational alignment.** Linear probes from the residual stream to teacher statistics:

| Algorithm | Probe targets |
|---|---|
| MCTS | $Q$, $\log N$ |
| Minimax | Backed-up minimax value |
| PNS | $\log$ pn/dn |
| CFR | $R_+$, counterfactual values |
| MCGS | Edge-weighted $Q$ |

For each probe, report held-out $R^2$, and **selectivity** $=R^2-R^2_{\text{control}}$, where the control task uses the same probe on shuffled-tree labels (Hewitt & Liang). Report a random-initialisation baseline as well.

**P3. Causal alignment via interchange interventions.** Swap the representation that the probe identifies as $Q(j)$ (or the regret $R_+(I,\cdot)$, etc.) between a base and a source tree. Then measure whether the model's next selection changes **as the teacher's would under the swapped statistic**. The metric is interchange-intervention accuracy (IIA), following causal-abstraction methodology (Geiger et al.).

**P4. Mask-channel attribution.** For each environment, report the fraction of attention mass and gradient-times-activation attribution per mask channel ($\mathcal M_{\text{desc}}$, $\mathcal M_{\text{anc}}$, $\mathcal M_{\text{infoset}}$, $\mathcal M_{\text{nogood}}$, …).

**P5. Beneficial-deviation analysis.** On steps where the model disagrees with the best-aligned teacher, compare the outcome score with that teacher's substitution (one-step counterfactual rollout). This separates "fails to learn $A$" from "learns something better than $A$".

**Pre-registered alignment hypotheses (family → expected best-aligned teacher)**

| Family | Expected teacher(s) | Secondary |
|---|---|---|
| Hex, Connect-4 | MCTS-PUCT; MCGS on Hex (transpositions) | Minimax near terminal positions |
| Sudoku, Equiv-SAT | CDCL (decision nogoods) / PNS | — |
| Proposal-Poker | CFR+ (regret sign, average strategy) | — |
| ShipHunt, Stratego | MCGS-Disjunctive-BP (belief-MDP keying for ShipHunt; infoset pooling for Stratego) | — |

### 3.5 Pass bars

**`R14-D0` — Integrity and zero algorithm supervision (gate)**

- 0 AST violations. No import path exists from R14 training code to teacher modules, R13 checkpoints (outside the warm-start arm) or teacher-trace files. Checkpoint lineage hashes are logged.
- 3/3 canaries pass; PoisonTensor diff $=0$.
- **Reward-hacking audit.** The evaluator's verifier and the best-response oracle are frozen and hashed. Additionally, 100 random witnesses accepted by the verifier are re-verified by an independent second verifier, which must agree 100%.
- LIR/infoset hidden-permutation invariance $\le10^{-6}$.

**`R14-D1` — Mini-ECI**

- Exceed the R12 unified generalist (106.53) by $\ge+1.0$, with the 95% CI lower bound $>106.53$.
- **Stretch:** $\ge108.61$ (the specialist ensemble).

**`R14-D2` — Belief quality and fixing the multi-witness regression**

- 12-environment mean KL at $B=256$ is $\le0.1294$ (R12's $B=128$ value).
- **Monotonicity:** $\mathrm{KL}_{B=256}\le\mathrm{KL}_{B=128}-0.003$ on the 12-environment mean **and** on each of ShipHunt and Stratego (CI excludes 0).
- Deductive environments must not regress beyond R12-V2 by more than $+0.005$ (Sudoku-6×6 $\le0.0380$; Sudoku-9×9-Extreme $\le0.0416$).
- Witness-coverage entropy on multi-witness instances must increase versus R12 (report it).

**`R14-D3` — Play strength and discovery**

- $\ge+75$ Elo over R12-V2 at $B=256$ in round-robin against the fixed baselines.
- Witness discovery $\ge75.0\%$ at $B=256$ (R12 was 73.61%).

**`R14-D4` — Imperfect-information play**

- Proposal-Poker exploitability $\le1.2\times$ that of the CFR+ teacher at equal budget.
- Micro-Stratego win rate $\ge55\%$ versus R12-V2.
- Hard firewall bar as in D0.

**`R14-D5` — Mechanistic alignment**

- In $\ge3$ of the 4 families, the pre-registered teacher has the maximal $\mathrm{AI}$ with a margin of $\ge0.10$.
- In the same families: probe selectivity $\ge0.3$ and IIA $\ge0.70$.
- **Null result is still informative.** Failing D5 while passing D1 is reported as "**non-classical discovered search**", and P5 is mandatory in that case.

**`R14-D6` — Beats fixed algorithms; warm-start ablation**

- Match the portfolio oracle on Mini-ECI within $-0.5$, **and** beat every single fixed algorithm on $\ge8$ of 12 environments.
- **Warm-start arm:** R13-initialised then R14-fine-tuned versus R14 from scratch. Report the outcome-score area under the curve (AUC) over training steps. *Hypothesis:* warm-start reaches the D1 bar in $\le50\%$ of the steps.

---

## 4. Updated Society Rubric and Pre-Registered Forecasts

### 4.1 Dimensions

The inline context does not restate the canonical dimension names. I use the following and assume they map by index.

- **D1** Integrity and firewalls (AST, canaries, LIR)
- **D2** Deductive calibration (Sudoku / Equiv-SAT KL and accuracy)
- **D3** Partial observability and multi-witness belief (ShipHunt / Stratego / Proposal-Poker)
- **D4** Interactive play strength (Elo, exploitability)
- **D5** Generalist efficiency and scaling in $B$ (parameters, monotonicity, Mini-ECI)
- **D6** Search-algorithm expressivity and learnability

### 4.2 Scores (0–10)

| Dim | R3 Realized-R12 | R4 Expressivity-Risk (R12 without `M_desc`/v2) | Proj-R13 | Proj-R14 | Rationale |
|---|---|---|---|---|---|
| **D1** | 9.0 | 8.5 | 9.0 | 8.5 | All 0-diff bars passed. New risk surfaces: soft $E_{\text{trans}}$ (§1.4.2.5), infoset keys (C3d), and RL reward hacking in R14. |
| **D2** | 8.0 | 7.0 | 8.0 | 8.5 | Strong calibration, but obtained from readout, not search. Without nogood or PNS backups, scaling past $B=256$ is uncertain. |
| **D3** | 5.5 | 4.0 | 6.0 | 7.5 | Non-monotone KL in $B$. No infoset or transposition pooling. R13 helps only indirectly; R14's LSO credit targets the regression directly. |
| **D4** | 7.5 | 6.0 | 7.5 | 8.0 | Large Elo gains, but the selector is hand-coded. Ceiling limited by the absence of backups. |
| **D5** | 8.0 | 6.5 | 7.5 | 8.0 | 11.8× parameter efficiency is impressive. The $B$-scaling inversion is a warning sign. |
| **D6** | 3.5 | 2.5 | 7.5 | 6.5 | R12 selection has zero trainable weights and 1-hop receptive field. R13 directly measures representability and learnability. R14's alignment is uncertain by design. |

### 4.3 `[PRIOR]` forecasts for R13 (learned, $N_{\text{train}}=2048$, 80% intervals)

| Algo | Metric | Full-v2 ID | Full-v2 OOD $B{=}128$ | Abl-v1 ID / OOD | Abl-R12 ID / OOD | P(Full-v2 passes bar) |
|---|---|---|---|---|---|---|
| 1 MCTS | Selection top-1 | [0.95, 0.99] | [0.87, 0.97] | [0.94, 0.99] / [0.86, 0.97] | [0.55, 0.75] / [0.35, 0.60] | 0.80 |
| 1 MCTS | $Q$ $R^2$ | [0.97, 0.995] | [0.93, 0.99] | ≈ v2 | [0.40, 0.70] | — |
| 2 Minimax / α-β / PNS | Value sign (depth 14 OOD) | [0.90, 0.98] | [0.75, 0.93] | [0.62, 0.80] / [0.50, 0.70] | [0.50, 0.65] / [0.48, 0.60] | 0.55 |
| 3 CDCL | Prune F1 | [0.86, 0.96] | [0.78, 0.92] | [0.55, 0.75] | [0.40, 0.60] | 0.55 (unsound-prune hard bar: 0.85) |
| 4 MCGS | Edge-weighted $Q$ $R^2$ | [0.90, 0.98] | [0.84, 0.95] | [0.70, 0.88] | [0.40, 0.65] | 0.60 |
| 5 CFR+ | Exploitability ratio | [1.05, 1.6]× | [1.2, 2.2]× | [2, 5]× | [3, 8]× | 0.45 |
| 6 Gumbel SH | Survivor exact-match | [0.90, 0.98] | [0.83, 0.95] | [0.55, 0.80] | [0.40, 0.65] | 0.65 |

**Additional R13 priors**

- P(D0d theorem-falsification passes, i.e. R12 fails as the audit predicts) $=0.85$.
- P(all CW representability bars pass on the first implementation) $=0.40$. *Expectation:* 2–3 implementation bugs are found by CW, most likely in MCGS and CFR+.
- `Sup-Behav`/`Sup-Dense` ratio: Algo 1 [0.90, 0.99]; Algo 2 [0.65, 0.88]; Algo 5 [0.55, 0.85].
- Learning-curve exponent $\alpha$: [0.3, 0.7] for Algo 1; [0.2, 0.5] for Algos 2 and 5.
- Cost: \$8–\$25.

### 4.4 `[PRIOR]` forecasts for R14 (80% intervals)

**Primary outcomes**

| Quantity | Forecast | P(pass) |
|---|---|---|
| Mini-ECI | [106.8, 109.3]; median 107.9 | D1 (≥ +1.0): 0.55; reaching 108.61: 0.30 |
| 12-environment mean KL at $B=256$ | [0.115, 0.136] | ≤ 0.1294: 0.60 |
| Monotonicity bar (D2) | — | 0.55 |
| Elo gain over R12-V2 at $B=256$ | +[40, 220]; median +110 | ≥ +75: 0.60 |
| Witness discovery at $B=256$ | [73.0%, 78.5%] | — |
| Proposal-Poker exploitability ratio vs CFR+ | [0.9, 2.0]× | D4: 0.40 |

**Mechanistic alignment (D5): P(pass) $=0.45$**

| Family | Forecast |
|---|---|
| MCTS (Hex/C4) | IIA [0.55, 0.85] |
| CDCL/PNS (Sudoku/SAT) | AI [0.30, 0.70] |
| CFR+ (Poker) | AI [0.20, 0.60] |
| MCGS-BP (ShipHunt/Stratego) | AI [0.25, 0.65] |

**Portfolio and warm start (D6)**

- P(D6 matches portfolio oracle within −0.5 and beats every single fixed algorithm on ≥ 8/12 environments) $=0.35$.
- P(warm-start reaches the D1 bar in ≤ 50% of steps) $=0.65$.

**Cost:** \$12–\$40. GRPO with $G=8$ plus LSO forwards is the dominant cost.

### 4.5 Bottom line for the Society

1. **R12 did not learn search; it learned evaluation.** Theorem R4.1 (with the corrected b′ hypothesis) explains why its selector could not have learned MCTS. `R13-D0d` will confirm or refute that model of the code empirically.
2. **v1 represents MCTS exactly up to an explicit $\varepsilon$ margin** (Theorem R4.2′). This requires frontier tokens, directed sibling masks for ties, and conservative LSE soft-max/min. Without frontier tokens it cannot choose *which action to expand*.
3. **Two of the five v2 fixes need rewriting before implementation.**
   - V2.2: Boolean closure with canonical DAG nodes and an edge-count-weighted $Q$ via spine recomputation.
   - V2.3: persistent per-infoset CFR+ memory with average-strategy readout and perfect-recall keys.
   - Additionally, V2.5 needs a subset-test head plus propagator verification.
4. **Even a corrected v2 cannot represent** action-identity heuristics (RAVE/GRAVE/MAST/history), re-search/restart algorithms (ID, MTD(f), IDA*, CDCL restarts), inner fixed-point re-solvers (CFR-D, ReBeL, retrograde analysis), or in-search adaptation (NRPA/PPA). These are deferred to `v3` and should be built only after R13 shows the v2 primitives are learnable.
5. **Kelvin's two-track plan is the right experimental design.** R13 is the unit test (representability and learnability, with ablations predicted by theorem). R14 is the integration test (discovery from outcomes, with causal probes and a portfolio-oracle baseline). LSO credit is the principled lever for the $B=128\to256$ multi-witness regression.
