You are an adversarial reviewer of a theory note. Read outbox/scale-free-theory-2026-10-29.md (path relative to the
working directory /home/agent/workspaces/mats). Supporting code and outputs are in work/scale-free-theory/ (scripts
n1c_rounds.py, n1d_small_messages.py, n2_open_latents.py, n3_tanglegram_gf2.py, n5_interaction.py and their .json
outputs) and the world definition in work/tiny-pfn/world.py.

Every statement carries a status tag: PROVEN (proof in the note or a cited theorem), PROVEN UNDER X, HEURISTIC,
CONJECTURE, MEASURED (our own numbers). Your job is to find errors, not to praise.

Check, item by item:
1. Every PROVEN or PROVEN UNDER claim: is the proof or the cited theorem correct as stated, are hypotheses missing,
   is the status too strong? In particular: T1 (Nerode argument, tolerance bound (2N - 1) epsilon); T2 (open latents,
   minimality under richness); T2b (inverse-Wishart: the message across a trader cut is the whole error history);
   T3 (Koopman-Pitman-Darmois as stated) and T3a (KL <= 2 N epsilon, polylog rank); T4 (two-round error
   N (log N / N)^{(p+1)/2}, one-step estimators, citations); T5 (reduction from disjointness at a balanced node,
   randomized lower bound applies to tree computations); T6 (sign-rank statement, the equality construction
   1/2 - (i - j)^2, order with 2, Forster); T6a (the logit-gap argument against Alman-Song's B = Theta(sqrt(log n))
   regime: is the normalisation A = exp(Q K^T / d) handled correctly?); T6b (online-softmax associativity);
   T7 (the gap-Hamming padding: n = 4/rho^2, repetition T rho^2 / 4, validity range); T10 (GF(2) cut classes,
   rank-width, complete graph, Lee-Lee-Oum); T11a (cost formula); T11c (knapsack); T11d (capacity bound
   derivation: rate-distortion + capacity, the (1 + 2 delta) factor); T12 (latent-tree block MI bound with the
   covering-subtree separator; the correlation bound via tree distance >= 2 log_b l); A1 (attention-sink identity);
   A2 (construction sketch).
2. Citations: is each cited paper real and does it say what the note says (Alman-Song 2302.13214, Keles et al.,
   Chakrabarti-Regev, Sanford-Hsu-Telgarsky x3, Amarilli et al., FAQ, Markov-Shi, FMMformer, H-Transformer-1D,
   Levine et al., Wu-Yang, Hahn, Rubinstein, Lee-Lee-Oum, Lerner-Parr, Oum 2008, Dyer 2003, Forster 2002,
   Nisan-Wigderson 1993, Evans-Kenyon-Peres-Schulman 2000, Liu-Ihler 1410.2653, Jordan-Lee-Yang 1605.07689,
   Huang-Huo 1511.01443, Duchi et al. 1405.0782, Abbe et al. 2108.04190, Shalev-Shwartz et al. 1703.07950)?
   Say "cannot verify" rather than guessing.
3. Novelty: flag any place where the note presents a known result as new.
4. Numbers: do the MEASURED numbers in the note match the JSON outputs? Is the measurement design able to show what
   is claimed (e.g. does the one-round vs two-round comparison compare like with like; does the M1 interaction
   design give exactly zero for Bayes on source-aligned cuts, given world.py)?
5. The verdict (section 4): is the proposed definition coherent, and is any of the "separations" in section 3 wrong?

Output: a numbered list of findings, each with severity (high / medium / low), the exact text at issue, and the
correction you propose. Do not edit any file. Be concise.
