Model: GPT-6 (Codex).

Checked 2026-10-28: reran the script, reproduced output.txt, checked the formulas, and inspected DESIGN-v1 and the original atom/GLS implementations. The numerical output reproduces; several interpretations fail.

1. c2’s continuum of global optima is false for the implemented softening. With eta=1e-9, the conditionals are strictly positive and uniquely determine mu=((1−eta)/2, eta/2, eta/2, (1−eta)/2), where F=0. The diagonal candidates have F≈2e-9 and are not global minima. The continuum exists only at eta=0. The 0.01 grid misses the off-diagonal mass: at gamma=1 its true total is approximately 1e-6. The marginal 0.5 conclusion survives, including uniqueness for gamma>0.

2. c2 removes information the actual estimator receives. The original atom implementation includes every conditioning subset, including unconditional marginals; c2 retains only two full conditionals. Supplying the correct unconditional 0.8 marginal removes the exact-copy ambiguity at gamma=0. Moreover, symmetrically softened copy conditionals encode marginal 0.5, not the stated 0.8 world. This demonstrates missing or misspecified inputs, not failure of the full proposed estimator. Replacing entropy with KL to a prior already containing the desired joint is a trivial recovery example.

3. The connection to earlier GLS work is misleading. That implementation explicitly normalises energies over configurations; it does not select one configuration. Its input conditionals can already encode prior information. Calling it “item 1 only” is unsupported. Likewise, atom F already contains conditional-entropy terms inside its KLs. Gamma=1e-4 is not simply a Bayesian temperature near zero, and error precisions for noisy conditional logits are not automatically calibrated evidence likelihoods.

4. W1 is a valid, strong MAP-versus-marginalisation example. The 120 hypotheses, 119 opposite predictions, 77 within two nats, posterior masses, predictive probabilities, means, temperature results and free-energy value check out. The predictive KL≈3.504 is numerical quadrature, not an exact calculation. The determinant identity uses E_joint evaluated at its conditional minimum. These results establish failure of selecting one hypothesis, not failure of distribution-level inconsistency minimisation.

5. W2’s numbers and nullspace are correct, but “sd does not shrink” is false: it decreases from 0.263 toward 1/sqrt(35)=0.16903. The prior-selected residual optimum 0.914 is not the finite-data MAP/mean 0.878. “Prior log-odds” means log density ratio; 4.63 nats corresponds to odds about 102.5, so comparing it directly with e² mixes scales. Assigning probability zero to a point estimate is an additional plug-in convention. The prior and curvature conclusions survive.

6. W3’s weighted example checks out: 0.871948 at LR=81 and 0.866111 at 76.95. Its star graph is not the specified chain benchmark. Its energies equal negative log joint probabilities only up to a state-independent constant, which does not affect these results.

7. The final necessity claim is too strong. Even an identifiable, unimodal Gaussian posterior differs from a point prediction; multimodality or non-identifiability is unnecessary. The Gibbs identity is correct, but T=1 presumes fixed log-probability units and exact recovery requires an unrestricted distribution family. Generic atom KL penalties cannot simply be relabelled likelihood terms.

Survives: point MAP can badly misrepresent predictive probabilities; residual fitting cannot recover missing prior information; correctly specified variational free energy recovers Bayes. A general refutation of the program’s existing distribution-level objectives does not survive.
