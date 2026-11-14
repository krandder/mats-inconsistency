### 5.4 Routing: what it costs, and when density is needed

**T5, routing separation (PROVEN).** Items (k_i, a_i) with keys k_i and bits a_i; f = 1 iff some key carries two
items with a = 1. (i) Every fixed hierarchy has a node with N/3 <= |S_v| <= 2N/3; placing the keys so that about N/3
of them straddle that cut (other values 0) turns f into the complement of set disjointness on N/3 bits. A tree
computation induces a one-way protocol across that cut, so the message at v needs Omega(N) bits even with public
randomness and error 1/3 (Razborov 1992; Kalyanasundaram and Schnitger 1992). (ii) With labels = keys (log N bits)
and the group-by rule, every content message is at most 2 bits.
The Bayes analogue (World FA: flat tokens in random order, per-trader statistics combined nonlinearly, so a balanced
positional node must pass every straddling trader's partial statistic) is **HEURISTIC**: it needs its own
indistinguishability argument, not given here. The reading of World FA's failure as a learning failure (one head can
route by trader ID, T6, and the binding-given arms learned) is likewise **HEURISTIC**: no full construction of the
trained architecture is given.

**T6, routing cost in attention = sign-rank (PROVEN).** A hard-threshold head (i attends to j iff q_i . k_j >= t)
realises the routing relation R with key dimension d only if the sign pattern of R has sign-rank at most d + 1
(absorb t into one extra coordinate), and sign-rank d suffices (t = 0); this is the definition of sign-rank.
Equality on N labels has sign-rank at most 3 (1/2 - (i - j)^2 = (2i, 1/2 - i^2, 1) . (j, 1, -j^2)), order has 2;
with noise or bounded logits, random +-1 codes of length O(log N) give margin 1/2 (Hoeffding and a union bound).
Hadamard-type relations ("attend iff <i, j> mod 2 = 0") need sign-rank at least sqrt(N) (Forster 2002). Natural
routing (same source, same question, before/after, nearby) is cheap; parity-like routing is not.

**Sharp selection needs the logit range of Alman and Song's hard regime (PROVEN, arithmetic plus arXiv:2302.13214).**
For soft attention to put 1 - delta of its mass on one item out of N, the competitors' weights must sum to at most
delta / (1 - delta), so the logit range max_j L_j - min_j L_j must be at least log((N - 1)(1 - delta) / delta). With
A = exp(Q K^T / d) and entries of Q, K in [-B, B], every logit lies in [-B^2, B^2], so B >= sqrt(log((N-1)(1-delta)/delta) / 2),
of order sqrt(log N). Alman and Song prove (worst case, d = O(log n), additive error 1/poly(n)) that attention can be
approximated in n^{1+o(1)} time when B = o(sqrt(log n)) and not in n^{2 - Omega(1)} time when B = Theta(sqrt(log n)),
assuming SETH. So attention that selects sharply sits in the regime where no generic fast algorithm exists; this does
not make every sharp routing task hard. Label routing (equality) is computed in O(N log N) by sorting or O(N) by
hashing; quadratic cost is unavoidable for generic sharp similarity search (also Rubinstein, arXiv:1803.00904).

**Routing by keys is a join (correspondence; cited theory).** Grouping items whose labels are equal and aggregating
within groups is an equi-join followed by an aggregate. Database theory treats joins, constraint satisfaction and
graphical-model inference as one sum-product problem whose exact cost is governed by width parameters of the query
(FAQ: Abo Khamis, Ngo, Rudra, arXiv:1504.04044): the routing varies with the data, the query structure does not.

**Per query, attention is already scale-free (PROVEN).** For one query, the triples (max logit m, normaliser l,
weighted value sum o) over any set of keys merge associatively: m = max(m_a, m_b), l = l_a e^{m_a - m} + l_b e^{m_b - m},
o likewise (the online-softmax rule used by FlashAttention). So one query's attention is a monoid reduction along any
hierarchy with d + 2 numbers per message. Dense attention runs N such reductions, one per query, per layer; the number
of genuinely different reductions per layer is the number of **routing classes** (distinct attention rows up to a
tolerance). This is the measurable form of "dense attention pays for routing": cost N x N, information about C x N,
with C the number of routing classes.

**T7, correlation-based routing (PROVEN for the promise problem on sign strings; HEURISTIC for iid histories).** Let
items carry T-step sign histories and define correlation as their normalised inner product. Deciding correlation
>= rho versus <= -rho needs Omega(min(T, 1/rho^2)) bits of communication per pair, with public randomness: for
rho >= 2/sqrt(T), gap-Hamming (Chakrabarti and Regev, arXiv:1009.3460) on n = floor(4/rho^2) coordinates, each
repeated about T rho^2 / 4 times (rounding changes constants only); for smaller rho, the unrepeated length-T
gap-Hamming instance already has the larger gap. Upper bound: with shared random coordinates, sending
O(log(N/delta) / rho^2) of them per item (capped at T) lets every pair among N items be decided correctly with
probability 1 - delta (union bound). So per-item keys need about min(T, 1/rho^2) bits, up to the log N factor: cheap
above the statistical detection threshold, linear in T at it.
