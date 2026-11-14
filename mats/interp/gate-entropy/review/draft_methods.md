## 1. What was measured

**Gates.** A gate records which side of a kink a unit is on.
- ReLU: g = 1[z > 0], where z is the pre-activation.
- tanh (the saturation region): the two kinks of hardtanh(z) = z - ReLU(z - 1) + ReLU(-z - 1), up = 1[z > 1] and
  dn = 1[z < -1]. That gives three states per neuron.
- GELU: g = 1[u > 0]. This is the middle of the switching region, where GELU(u) = u Phi(u) has Phi(u) = 1/2.

**Flip.** A flip swaps the linear piece a gate selects. The pre-activation is held fixed.
- ReLU: a = g z, so a flip gives a = (1 - g) z.
- tanh: the hardtanh piece change is added to tanh(z).
- GELU: u Phi(u) becomes u Phi(-u). Far from 0 this equals the ReLU flip; at u = 0 it is 0.

Everything downstream is recomputed by the actual network, with the same input and the same noise draws (common random
numbers). These flips are what a family that assigns the wrong piece to a unit would get wrong.

**Question and tolerance, per sample.**
- AlgZoo: the answer is "is the output the 2nd argmax?". It changes if it flips.
- Random MLPs: the answer is the output vector, a_L for ARC's nets and W_out s_L for the tanh nets. It changes if
  ||delta|| > 0.1 sqrt(sum_j Var y_j). We also report the bias of the mean in ARC's nMSE units.
- PFN: the answer is the predictive (mixture) mean of the query. It changes if it moves by more than 10% of the net's
  RMS deviation from exact Bayes (0.063, so a tolerance of 0.0063 against a predictive-mean SD of 0.92). Since Bayes is
  fixed per episode, this is also the question "the deviation from Bayes, to 10%".

**The three entropies (bits; per layer, or per step for the RNNs; over inputs and noise).**
- E1 = H(G_l): all m gates.
- E2 = H(G_R). Let f_i be the fraction of held-out samples whose answer changes when gate i alone is flipped. Then
  R = {i : f_i >= 1%} and D = |R|. D(0.1%) is also reported.
- E3 = the merged entropy, built as follows.
  - A greedy decision tree over gate values is grown on a flip set of 1,024-8,192 samples.
  - Each leaf is one merged class. It replaces the whole gate pattern of its members by the leaf's majority pattern.
  - The tree splits first to remove samples whose single-flip "care" gates disagree with the leaf pattern. When none
    are left, it splits on flip-rate-weighted mismatches (Amendment 1).
  - Every checkpoint is verified on 8,192-65,536 held-out samples by running the network with the substituted
    pattern.
  - E3(eps) is the held-out entropy of the leaves at the first checkpoint whose verified change rate is <= eps, with
    eps = 1% (and 0.2% for the RNNs, ARC's bar).
  - E3 is an achieved size. It is an upper bound on the discrete part this kind of family needs, not a lower bound.
  - "Not reached" means the tree ran out of samples or budget first. The trivial answer, no merging (E3 = H(G) at
    zero change), always exists.

**Estimators** (each checked on synthetic patterns and on T16 against a 2^21-sample reference):
- Lower bounds:
  - plug-in (biased low for every distribution);
  - Renyi-2 entropy from pair collisions (H >= H_2). With zero collisions this becomes log2(pairs/3), marked *;
    approximate, because pairs are not independent;
  - for layer 1 of the random MLPs, where z is exactly Gaussian, a rigorous chain-rule bound,
    sum_i H(g_i | z_{<i}), computed from the Cholesky factor.
- Upper bounds:
  - the sum of per-unit entropies;
  - the held-out cross-entropy + 2 SE of an autoregressive FVSBN, and of a MADE for m <= 64.
- NSB and Miller-Madow are quoted as point estimates only when the bracket is narrower than 1 bit.

**Subspace ("rotate and concentrate").**
- We ran PCA on the relevant gates' pre-activations over the data.
- k(99% var) is the number of components that hold 99% of the variance.
- k(agree 99%) is the number of components after which the reconstructed pre-activations reproduce 99% of the gate
  values.
- log2 sum_{j<=k} C(D, j) is the largest entropy D hyperplanes in general position can produce in k dimensions.

**Capacity.**
- C_l = (d/2) log2(1 + 1/sigma^2), with noise relative to the unit activation scale.
- C_cov = (1/2) log2 det(I + Cov(signal)/noise variance), computed on the noisy state feeding layer l.
- Both bound I(G_l; X) for layers >= 2. Neither bounds the noise-driven part H(G_l | X), which we measured on 128-512
  inputs x 256 noise draws.
