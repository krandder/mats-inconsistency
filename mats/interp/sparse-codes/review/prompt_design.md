You are a rigorous critic reviewing a research design (read-only; do not modify any file). Write your critique as your
final answer (it will be saved to a file). Be concrete and terse; number your findings and mark each critical / major /
minor. First ask whether each experiment can answer the question it claims to answer; then audit the details.

Context: MATS noisy-channel interpretability program. Target "M1": a sample-free (deductive) estimator that matches or
beats sampling at equal compute on ARC's deep random He-ReLU MLPs (x ~ N(0, I), W iid N(0, 2/n), no bias, small
activation noise sigma relative to each layer's RMS) at depths that grow with width (e.g. 256 x 32). Estimand: per-neuron
means of the last post-ReLU layer. Bars at 256 x 32: ARC's factorized K=3 cumulant propagation (K3f) ~2-3e-4 nMSE; the
exact Gaussian closure (K=2) ~6-8e-4. The program's "leading family" was k-sparse codes h = D a + noise with sparse
transcoders as layer maps; it was never really tested. This lane tests it on random nets (ARC He-ReLU; tanh+LayerNorm
ordered and chaotic). The coordinator retargeted twice before pre-registration: (1) an independent-feature product is
ruled out analytically (dropping off-diagonal covariances changes next-layer variances by a relative O(rho_rms) that
does not shrink with width), so every family must carry the full covariance and the question is the higher-order
structure beyond it; (2) frame the sparse code as a latent-conditional Gaussian (exact sparse latent + Gaussian
residual with full covariance given the latent), the sparse analogue of a sister lane's dense collapse-mode latent;
compare against a dense k-dim latent at matched size and against K=2 and K3f; test whether the residual is
conditionally Gaussian given the features.

Files (read them):
- /home/agent/workspaces/mats/work/sparse-codes/PREREG.md  (the design under review: definitions, kill tests
  KT1-KT5, predictions)
- /home/agent/workspaces/mats/work/sparse-codes/gpu/run_main.py (implementation of E1 dictionaries in bits, E2
  one-step higher-order structure, E3 representation at a cut pushed through the real layers)
- /home/agent/workspaces/mats/work/sparse-codes/lib/sae.py (batched TopK SAEs, evaluation in bits)
- /home/agent/workspaces/mats/work/sparse-codes/lib/nets.py (net wrappers)
- Context only: /home/agent/workspaces/mats/work/boundary-family/DESIGN.md (sister lane),
  /home/agent/workspaces/mats/outbox/gate-entropy-2026-10-29.md (gate census),
  /home/agent/workspaces/mats/work/sparse-codes/logs/explore1_*_64.log (dev-net exploration).

Questions to answer specifically:
1. Can E1 (TopK SAEs, F <= 16n, error in bits relative to the noise, KT1 with L0 <= n/4) answer "do random layers have
   sparse structure at the noise precision?" Is the TopK SAE with a learned b_dec (initialised at the mean) a fair test
   of sparse structure, given that ReLU layers are exactly sparse in the neuron basis around zero? Is 30 epochs on 2^18
   samples likely to converge to errors near the noise (relative MSE ~1e-4)?
2. Are the latent-conditional models (G, DENSE(k), DENSEQ(k), SPARSE(F,k)) correctly constructed to carry the full
   covariance, and is the comparison sparse vs dense "at matched size" fair? Is it fair that SPARSE's latent is a
   nonlinear function of the full state (encoder sees all of a) while DENSE's is linear?
3. Do the E2 metrics (cap3/cap4 of per-neuron next-layer cumulants, one-step mean nMSE with common random numbers,
   Edgeworth oracle readouts) measure "higher-order structure carried"? Sampling noise? Bias from fitting B, Sigma_e on
   the same samples?
4. E3: is "representation error at a cut with the exact latent taken from true samples" a valid necessary condition
   (KT4 against K3f's end-to-end error)? Any leakage that makes the sparse models look better than any deductive family
   could be?
5. Kill tests KT1-KT5: are thresholds and "fires if" rules well posed, and do they test what they claim?
6. Bugs in run_main.py / sae.py that would invalidate numbers.
