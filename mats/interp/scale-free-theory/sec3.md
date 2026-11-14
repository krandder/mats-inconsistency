## 3. How the definitions relate: separations

Each line is a task on which two definitions disagree.

| Separation | Evidence | Status |
|---|---|---|
| D1 vs D2: keyed aggregation (binding) | Omega(N) bits at a node of every fixed hierarchy; O(1) bits plus log N label bits routed (T5) | PROVEN |
| D1 vs D3: f = sum_i a_i x_i, x in {0,1}^n, weights with distinct subset sums (e.g. a_i = 2^i) | rank 2 across every cut; 2^{\|S\|} classes | PROVEN |
| D3 vs estimation: the same f, integer weights | E f is trivial; the tail P(f > t) is the expectation of the indicator 1[f > t], whose rank across a balanced cut can reach 2^{n/2}; computing 2^n P(f > t) is counting knapsack solutions, #P-complete (classical; stated in Dyer 2003). Low rank does not survive a nonlinear readout | PROVEN |
| D4 vs D1/D3: f = x^T U x mod 2, U the complete graph | the pairwise interaction graph has treewidth n - 1; f depends only on \|x\| mod 4, so 2 bits suffice at every cut (rank-width 1) | PROVEN |
| D4 vs SFU: Bayes with one shared target theta, Gaussian case x_i = theta + e_i | with theta integrated out, every pair of items stays dependent given all the others (the precision matrix of x has no zero off-diagonal entry), so the item graph is complete (treewidth N - 1); kept explicit, theta is a single open latent at every cut | PROVEN (Gaussian case; generic otherwise) |
| D1/D3 vs tractability: x^T U x mod 2 on random graphs | linear rank-width (Lee, Lee, Oum, arXiv:1001.0461), yet sum_x (-1)^{x^T U x} takes polynomial time (Dickson) | PROVEN |
| untangled vs learnable: parity | 1 bit across every cut of every hierarchy; hard for statistical-query learners, which include gradient methods at coarse precision; World 3's XOR chains were never learned | PROVEN; MEASURED (World 3) |
| D5 vs D1: inner product mod 2 across a cut, independent uniform inputs | I(X_S; X_{S^c}) = 0, yet Omega(N) bits across the cut even on average (Chor and Goldreich 1988; for disjointness the hard distribution is not a product one) | PROVEN |

So: D1 counts precision and misses routing; D3 counts dimensions but misses tails; D4 at the item level measures a
graph, not a function, and calls the simplest aggregation maximally tangled; D5 describes the data, not the
computation.
