### 5.8 The renormalisation view, made precise

- Latent-tree data (a complete b-ary tree with q states per latent, leaves in their natural left-to-right order) have
  block mutual information at most (2 log_b L + O(1)) log2 q for a contiguous block of L leaves: the ancestors along
  the block's two boundary paths separate it from the rest. **PROVEN**. Log law, like 1D critical states and MERA,
  against the volume law of generic data and the constant of Markov chains.
- Correlations: in the same tree, two leaves at distance l have tree distance at least 2 log_b l; for a stationary
  binary symmetric channel with second eigenvalue lambda = 1 - 2 epsilon, their correlation is lambda^{tree distance},
  hence at most l^{-2 log_b(1/lambda)}: power-law decay at most. **PROVEN** (bound). Lin and Tegmark
  (arXiv:1606.06737) make the grammar version.
- Whether information about the root survives coarse-graining is a sharp threshold: for a binary symmetric channel on
  a tree, reconstruction is possible if br(T)(1 - 2 epsilon)^2 > 1 and impossible if it is < 1 (Evans, Kenyon, Peres,
  Schulman 2000, the Kesten-Stigum bound); the equality case is not settled by this criterion. **PROVEN** (cited).
  Relevant vs irrelevant directions in the RG sense have this exact meaning on trees.
- For additive merges with product input laws, the rescaled law of the message converges to a Gaussian (finite
  variance) or a stable law (in a domain of attraction); max-merges converge to an extreme-value law when the per-item
  law is in its domain of attraction. **PROVEN** (classical). Under SFU with additive merges the large-scale messages
  are therefore near-Gaussian, the rationale for the latent-hub family keeping its non-Gaussian structure in a few
  latents (**HEURISTIC**).
