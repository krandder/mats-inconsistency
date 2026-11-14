### 5.1 Messages are likelihoods of the open latents (T2)

Take a joint model p(x, u) = prod_c psi_c(u_c) prod_i p(x_i | u_{L(i)}), with items attached to latent sets L(i).
Assign each factor psi_c to the lowest hierarchy node whose subtree contains every item attached to the latents in
its scope (prior factors of latents open everywhere end at the root). At a node S a latent is **closed** if all its
items lie inside S, **open** if it is attached inside and outside. The message m_S(u_open) is the sum over the
latents closed inside S of the product of the item likelihoods and assigned factors inside S; the merge multiplies the
children's messages and the factors assigned to the node, and sums out the latents that close there (variable
elimination along the hierarchy; junction-tree separators). **PROVEN** (standard). The posterior of the target is the
normalised root message.

Minimality. If the outside data can produce likelihoods that separate functions of the open latents, two contents
with non-proportional messages give different joint posteriors of the open latents for some completion, so no smaller
message is exact for that joint posterior (**PROVEN UNDER** this richness condition). For a single target y the
minimal message can be smaller: a direction of m_S that no completion couples to y (an open nuisance latent the
outside data never links to y) can be dropped.

Consequences:
- Tiny-PFN world (`work/tiny-pfn/world.py`): each source's reliability v_s closes inside the source; the hidden answer
  y* is attached to every source's query reading, so it is the only open latent at a cut between sources: messages
  are functions of y*. At a cut inside a source v_s is open, and y* is open only if the cut contains that source's
  query reading. **PROVEN** (from the model).
- Shared-error groups with visible membership: each group's shared errors are open at every cut that splits the
  group, so message size grows with the number of **straddling groups**, not of items; under a hierarchy aligned with
  the groups (routing by the membership label) only y* is open. With membership hidden and T past questions, each
  source's likelihood involves every group's shared errors on every past question, so about G T + 1 latents are open
  at a cut between sources, until a pilot round lets sources be re-routed into group-aligned subtrees. **PROVEN**
  (from the model).
- Correlated errors with an inverse-Wishart covariance (World FA variant (b); N traders, T past questions; E is the
  T x N error history). The full posterior of Sigma depends on the scatter E^T E, whose cross block E_S^T E_{S^c}
  would need the whole history E_S (T |S| numbers) to cross a trader cut. The predictive does not need Sigma. With
  prior scale Psi = diag(Psi_S, Psi_{S^c}), Sylvester's identity gives
  |Psi + E^T E| = |Psi| |I_T + E_S Psi_S^{-1} E_S^T + E_{S^c} Psi_{S^c}^{-1} E_{S^c}^T|, so the marginal likelihood,
  and the query predictive (append the query row o - y* 1 as one more time step), depend on each side only through a
  (T + 2) x (T + 2) weighted Gram matrix of [E_S, o_S, 1_S]. These add across trader cuts: O(T^2) numbers,
  independent of |S|. A prior scale whose inverse is diagonal plus rank k (a common source) adds k summed history
  vectors, still additive. **PROVEN UNDER** the matrix-normal / inverse-Wishart model with such a prior scale (the
  dual-coordinate argument is Gemini's review correction of our first draft, which claimed T |S|). So this apparently
  global coupling is untangled in time coordinates: what is open across a trader cut is the T-dimensional time
  structure, not the N(N+1)/2 entries of Sigma. A volume law in |S| returns only if Sigma itself is the target, or
  the prior scale couples the two sides densely.
