## 2. Six candidate definitions

**D1. Bits across the cuts of a fixed hierarchy.** Two contents x_S, x'_S of a cut S are equivalent if
f(x_S, z) = f(x'_S, z) for every completion z; K_f(S) is the number of classes (the Myhill-Nerode relation for tree
automata, Thatcher and Wright 1968). Untangled: log2 K_f(S) grows slower than |S| (sub-volume), strongly untangled if
polylog.

- T1, minimal tree computation (**PROVEN**). Every exact tree computation along T has at least K_f(S_v) distinct
  messages at v, and sending the class achieves it. Proof: if two inequivalent contents shared a message, the output
  (a function of m_v and of z) would agree on the completion z that separates them. Conversely the class is a
  function of the children's classes: if x_a ~ x'_a and x_b ~ x'_b, then f(x_a, x_b, z) = f(x'_a, x_b, z) =
  f(x'_a, x'_b, z) for every z. K_f(S) is the number of distinct rows of the communication matrix, and
  ceil(log2 K_f(S)) the deterministic one-way communication complexity from S to the rest.
- With tolerance (**PROVEN**): at each node choose an *internal* epsilon-net (representatives are actual contents of
  that node) in sup norm over completions. The same substitution gives e_v <= e_a + e_b + epsilon, so the root error
  is at most epsilon times the number of quantised nodes (<= 2N - 1); internal covering numbers at
  epsilon = delta / (2N - 1) replace K. With an external net (arbitrary functions) the radius doubles per node.

**D2. Bits across the cuts of the input's own hierarchy (routing).** Each item gets a label lambda(x_i) of b bits; a
fixed **rule** builds the hierarchy from the labels; content is counted as Nerode classes given that both sides know
the labels. Labels may be recomputed between rounds from what was broadcast (as later attention layers do). The rule
must be restricted (group-by-label, sort-by-label, contiguous chunks), and the labels charged: otherwise labels can
carry the whole item and the "router" does the whole computation (**PROVEN**, trivially).

**D3. Bond dimension and min-cut.** The rank of the matricisation M_S[x_S, x_{S^c}] = F(x) across an item cut, for F
a function or a probability table. For a probability table, the nonnegative rank is the smallest number of latent
*states* making the two sides conditionally independent (**PROVEN**, standard). For deep arithmetic-circuit networks
the min-cut of the network's tensor-network graph bounds this rank (Levine et al., arXiv:1704.01552); hierarchical
tensor formats are built on these ranks, with quasi-optimal truncation (Grasedyck 2010). A different rank matters for
messages: the parameter-observation rank of a likelihood family, the rank of [log p(x_S | theta)] over (x_S, theta).
For a d-parameter exponential family it is at most d + 2, and d after dropping the terms that depend on x alone or on
theta alone (they do not change the posterior); as a likelihood rather than a log-likelihood it is typically much
larger, infinite for continuous non-degenerate families (**PROVEN**). This second rank bounds a message's dimension
when theta is the only open latent (section 5.1).

**D4. Treewidth of the dependency structure, and its non-Gaussian part.** Treewidth of (a) the computation's DAG,
(b) the graphical model of the items or activations, (c) the discrete (non-Gaussian) part of a conditional-Gaussian
model. D4(a) implies D1: a Boolean circuit of treewidth w compiles, in time linear in its size and singly exponential
in w, into a structured d-DNNF (Amarilli, Capelli, Monet, Senellart, arXiv:1811.02944), which supports exact model
counting and expectations under product laws. A structured circuit along a tree over the inputs is related to D1's
deterministic messages but not the same object: a node with w gates can carry up to 2^w residual classes. Lower
bounds on DNNF size go through multi-partition communication complexity (Bova, Capelli, Mengel, Slivovsky, IJCAI
2016). **PROVEN** (cited).

**D5. Mutual information across scales; renormalisation.** The block law I(X_B; X_{B^c}) of the data as the block
grows (volume, area, log); renormalisation as the merge map acting on the laws of messages, and its fixed points.

**D6. Hierarchical low rank (H-matrices, fast multipole).** For interactions y_i = sum_j K(p_i, p_j) v_j over a
hierarchy (possibly built from the inputs), every well-separated block has epsilon-rank at most k(s) at scale s.
Scale-free: k independent of s, with the same translation rules at every scale (Greengard and Rokhlin 1987).
