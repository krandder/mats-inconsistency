### 5.5 Multi-hop needs rounds; parity is untangled but hard to learn

- k-step pointer chasing (two tables f_A, f_B held on the two sides of a cut): protocols with k - 1 rounds need
  Omega(n - k log n) bits deterministically and Omega(n/k^2 - k log n) randomised, while k rounds need O(k log n)
  (Nisan and Wigderson 1993; constants as reported by later papers, the primary text was not checked). A one-pass tree
  computation induces a one-way protocol across each of its cuts, so at any cut that puts the two tables on opposite
  sides, lookups of 3 or more hops need Omega(n) bits (**PROVEN** for such cuts). It is not a statement about every
  hierarchy: an unbalanced pass whose leaf order follows the hop order can carry the current pointer upward and do two
  hops with log n bits. Transformers: constant attention layers simulate constant rounds of massively parallel
  computation, and log depth suffices for k-hop tasks by pointer doubling (Sanford, Hsu, Telgarsky,
  arXiv:2402.09268), which moves Omega(n log n) bits per layer; one layer needs exponentially more size than two for
  induction heads (arXiv:2408.14332). **PROVEN** (cited).
- Parity is maximally untangled under every definition here (1 bit across every cut; rank 1 in +-1 form), yet a family
  of unknown parities is hard for statistical-query learners. Gradient methods fall in that class when the gradient
  precision is coarse relative to the batch size; with fine enough precision SGD can simulate any PAC learner (Abbe,
  Kamath, Malach, Sandon, Srebro, arXiv:2108.04190), so the barrier is a regime, not an impossibility; gradients carry
  exponentially little signal about which parity is the target (Shalev-Shwartz, Shamir, Shammah, arXiv:1703.07950).
  **PROVEN** (cited). World 3 (XOR chains) is consistent with this: the equality version was learned, the XOR version
  never was. **MEASURED** (outbox/world3-chains-pilot). Lesson: untangledness is about what a solution must carry,
  not whether training finds it.
