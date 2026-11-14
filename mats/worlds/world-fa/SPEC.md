# World FA: forecast aggregation with hidden correlated information structure (spec v0, 2026-10-28 ~13:20Z)

Kelvin (13:10Z): "Let us get something where the target and equilibria is really the fundamental value, and the difficulty
is just the complex info structure." Approved 13:15Z: "Let us go with that; you can start."

## Generative process (per episode)
- Traders i = 1..N (N varies; train small, test much larger).
- Hidden information structure, drawn once per episode: K noise sources eta_s ~ N(0, s2_s) with variances spanning
  up to several times Var(v) (so ignoring structure is costly); exposure matrix A (N x K): private sources (one
  trader), shared sources (random subsets: "same analyst / same model"), one common source (public-signal analogue);
  optional signed loadings; idiosyncratic noise eps_i ~ N(0, t2_i). Report error e = A eta + eps, so
  Sigma = A diag(s2) A^T + diag(t2).
- History: T resolved rounds (T varies, including 0), same traders and structure, fresh v_t ~ N(0,1) and noise;
  tokens show each trader's report r_{t,i} = v_t + e_{t,i} and the realised v_t.
- Query round: N fresh reports r_i = v + e_i; target: predictive for v.
- Trader identities: random vectors per episode (no memorisable ids).

## Exact Bayes (two prior variants)
(a) Structured: A, s2, t2 drawn from a finite (enumerable) family of hypotheses -> posterior over hypotheses from the
    history errors (Gaussian likelihoods) -> mixture of exact Gaussian GLS posteriors for v.
(b) General: Sigma ~ inverse-Wishart(nu, Psi) -> conjugate posterior from history errors -> predictive for v by a
    1-D integral over v of N(v;0,1) * multivariate-t(r - v1) (quadrature, exact to numerical precision).

## References
prior-level (T ignored; best fixed rule given N and prior); equal average; precision-weighted ignoring correlations
(per-trader variances estimated from history); GLS with sample covariance (plug-in); GLS with true Sigma (oracle,
cheating bound); full Bayes.

## Tests / cruxes
- History gate (GPT-6): identical query reports, different histories (shared vs independent error structure) ->
  Bayes differs by >= 0.3 nats; model must track it; shuffled-history control (permute trader ids between history
  and query) removes the gain.
- Rewiring audit (Gemini): before training, the structure must explain a substantial share of Bayes's answer
  (e.g. >= 15% of predictive-mean variance vs a structure-blind rule) on the training prior.
- Dedup: k traders on one shared source vs k independent traders, same marginal accuracies.
- Binding (astra's result): permute which history column belongs to which query trader -> Bayes changes; model should.
- Scaling: gap to Bayes vs N (up to 64+) and T (up to 128+) beyond training.
- Prior-level vs in-context: performance at T=0 vs T>0; gain must vanish under the shuffled-history control.
