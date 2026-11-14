# Round 1 pre-registration (written 2026-10-28 14:45Z, before any ladder result)

Only plumbing has been run before this file: forward/resume equality and the G quadrature against `pfn.kl_grid`
(max diff 6e-16, 2L s0 / 3L s10 / plateau). No q has been fitted to a real network yet.

Code: `r1run.py` (one model per process), `lib/tp.py` (cuts, exact conditional sampler, features, G), `lib/r1est.py`
(estimators, q wrapper), `cmlib/` (validated toolkit). Python `/home/agent/.venvs/aggext/bin/python`, CPU, 4 processes,
OMP_NUM_THREADS=1.

## Models
- **2L near-Bayes:** `tiny-pfn/runs/w32_h2_L2_mix4_ws_s{0..4}` (motifs: s1, s4 two pooling heads; s0, s2, s3 one pooling head + query mixer).
- **3L:** `tiny-pfn-interp/heads/hruns/L3_ws_s{10..14}`.
- **Plateau contrast:** `tiny-pfn/runs/w32_h2_L2_mix4_s3` (gap 0.144).

## Data (episode seeds disjoint from val 10000, test 20000, binding 40000, training)
- fit split: 20,000 episodes, seed 60000 (all q fits, the pushed fits of the consistency check).
- eval A: 6,000 cells (seed 61000) × M = 8 exact conditional draws (seed 61001): ladder, ratios, localisation, naming.
- eval B: 6,000 cells (seed 63000) × 8 draws (63001): fresh acceptance of q1, DL, consistency.
- Exact conditional sampler: keep c = (Q, S); resample IDs, residual directions at fixed SS_s, labelled y.

## G and sites
- G = KL(p_B(·|c) ‖ f(h)) per draw, 201-point Bayes grid (exact to 1e-15 vs kl_grid on real outputs).
- Sites = (cut, token type), rest of the same draw real. 2L: h1-{LAB, QQ, QT, ALL}, a2-QT (after attn2), h2-QT.
  3L: h1-{…}, h2-{…}, a3-QT, h3-QT. ALL at h1 (2L) / h1, h2 (3L) is a full cut.
- Quotient coordinates: every token centred; context tokens at the second-to-last post-block cut also scaled to unit sd.

## q families (per token type, tokens independent given phi; LinearGaussian, full residual covariance, ridge λ by 5-fold CV grouped by episode)
- R_marg: constant mean. R_in: linear in the token's raw inputs (sid, qid, o, y; QT: qid*).
- **q0** (Kelvin's Bayes-latent q, as stated in the task): R_in + own source's Bayes latents at LAB/QQ
  [log SS_s, labelled-only variance posterior p_s (3), expected precision E_s, Bayes precision weight β_s, o*_s, o*_s·E_s];
  at QT: [mean Q, Bayes mean, Bayes log var, labelled-only mean and log var, Σ β_s o*_s].
  Deviation from METHOD §7.3: β_s is in q0 (the task text lists precision weights in q0), not in q1a.
- q0U (unordered): R_in + sorted log SS (the multiset U) + mean Q + moments of the (Q, U) "blind" reference (+ own o*_s). Forgets source binding.
- q0noID: q0 without ID terms (control).

## Estimators (all primary closure numbers are direct, per VALIDATION.md)
- Direct Δ = L(q) − L(p) (= Kelvin's L(0) − L(1)), studentised bootstrap over cells; Δ_rms from two independent halves of the 8 paired draws; Hill index.
- Output-space ratio for q0 at every site: classifier (cmlib MLP, Platt) on [11+1 head outputs, c features], 2-fold cross-fit by cell, 2 refits;
  L(α) by the per-cell bridge (α = 0…1 step 0.1), L(0) − L(1), ∫dL/dα, Cov_q(G, ŵ) per cell (plain and SNIS), E_q[G ŵ], Kish ESS,
  T1 (bridge endpoints vs direct), T2 (normalisation), V5 (integral vs direct). SEs = cell bootstrap ⊕ refit sd.
- Regions: share of Δ by quintile of between-component variance, query spread, coupling, 27-way entropy; by decile of out-of-fold log ŵ_out.

## Localisation
- Components at each post-block cut: state = prev + head_0 + head_1 + mlp. Joint Gaussian of the stacked components (same phi and λ as q0; its sum is q0).
  Knock-out (component drawn from q given the other components and phi, rest real) and knock-in, all token types and per type.
- Residual subspaces: h-space MLP classifier real vs q0 on [ĥ_t, phi_t, candidate library features]; U = top-4 eigenvectors of E_p[∇log ŵ ∇log ŵᵀ];
  knock-out / knock-in of U vs a random 4-d subspace.

## Refinement step (one round)
- Candidate groups (library Λ, fixed now): LAB {r2: r², |r|, r³; pool: tilted pools softmax(±1.6r − 0.4r²)[r, r²] + mean r, mean r³ of own source;
  cross: o*_{−s}, log SS_{−s}, full-Bayes p_B,s, E_B,s − E_lab,s; idgram: ID Gram entries; episode: between-component var fraction, 27-way entropy,
  query spread, coupling KL, Bayes mean, log var; quadlat: pairwise products of the own-source latents}; QQ: same minus r2; QT {betaQ, idgram, episode,
  quadlat}; plus **hetero** (residual covariance per quintile of between-component variance) for every type.
- **Naming rule:** for each (cut, type), refit q0 + candidate on the fit split; score = held-out (eval A) gain in log density of the real states projected
  on U. Choose the best; within 10% of the best, take the one with fewer added parameters. q1 = q0 + chosen candidate (per type; ALL uses each type's q1).
- **Acceptance (fresh B):** Δ_rms(q0) − Δ_rms(q1) ≥ max(0.1·Δ_rms(q0), 3 SE). Add-one Δ of every candidate is also reported on B (validates the rule).
- **DL:** struct bits (feature groups × log2 48 + map + noise type; hetero + 3 bits) + param bits (uniform quantisation, smallest common b that moves Δ by
  ≤ 0.02·KL_model with common random numbers on 1,500 cells of B) and BIC. Anchor: 16 bits × block parameters.

## Cross-layer consistency (METHOD §6, on B)
- Tier i: G computed from cut l equals G after pushing through the real block l+1 (max abs diff).
- Tier ii: q̃_{l+1} = q0 family refitted on F#q_l (q_l = q1 if accepted at the ALL site, else q0); q^fit_{l+1} = q0 fitted on real h_{l+1}.
  Γ_fit = Δ_rms(q̃) − Δ_rms(q^fit), Γ_proj = Δ(q̃) − Δ(F#q_l), signed Δ(q̃) − Δ(q^fit), AUC(q̃ vs q^fit) per type.
  Pass (METHOD): Γ_fit ≤ max(2 SE, 0.1·KL_model) and AUC ≤ 0.55; also the two-sided rule. Pairs: 2L h1→h2; 3L h1→h2, h2→h3.

## Predictions (ε = 0.1·KL_model)
- **P1, q0 does not close.** Near-Bayes nets (2L, 3L): Δ_rms(q0) > ε at every site in ≥ 4/5 seeds, and Δ(q0) at the full cut (2L h1-ALL, 3L h2-ALL) exceeds KL_model in ≥ 4/5 seeds.
- **P2, stuck model = unordered reliabilities.** Plateau: Δ_rms(q0U) ≤ 1.2·Δ_rms(q0) + 2 SE at h1-QQ, h1-ALL and h2-QT, and Δ_rms(q0U) ≤ ε at h1-ALL.
  Near-Bayes 2L: Δ_rms(q0U) ≥ 2·Δ_rms(q0) at the same three sites in ≥ 4/5 seeds.
- **P3, 3-layer smaller residual.** Seed-mean Δ_rms(q0) at the last QT cut is smaller for 3L (h3-QT) than for 2L (h2-QT), and at the main full cut
  3L h2-ALL < 2L h1-ALL, in absolute nats.
- **P4, motif.** At h1-QQ, Δ(q0) in one-pool seeds (s0, s2, s3) ≥ 2× the two-pool seeds' (s1, s4) mean, and the naming step picks `cross` at h1-QQ in ≥ 2/3 one-pool seeds.
- **P5, ratio attenuation.** Wherever output-space AUC < 0.6, the bridge L(0) − L(1) is below the direct Δ (ratio < 1) at ≥ 70% of such sites.
- **P6, IDs.** Δ(q0noID) ≥ 2·Δ(q0) at h1-ALL in ≥ 4/5 2L seeds.
- **P7, consistency.** Tier i ≤ 1e-5 everywhere. Tier ii fails (METHOD rule) for 2L h1→h2 in ≥ 4/5 seeds, because q0 at h1 is not near-closed.
- **P8, refinement.** q1 is accepted at h1-ALL (2L) in ≥ 3/5 seeds, but Δ_rms(q1) still > ε (no closure in one round).

## Not in round 1 (scope cut, stated here)
Smoothing σ rule, DeepSets classifier, uLSIF, quadratic family on outputs, oracle q^R, untrained nets, transfer seeds, seed 8, tier-iii surrogate chains,
M ∈ {8,16,32} SNIS bias check.

## Amendment 1 (2026-10-28 15:00Z, after a quick smoke run of 2L s0 at n = 800 cells; before any full-size run)
- **Change:** the naming directions U come from the *output-space* ratio pulled back to h (top-4 eigenvectors of E_p[g gᵀ], g = ∇_ĥ log ŵ_out(f(ĥ), c)),
  not from the h-space classifier. The h-space classifier is still fitted and reported (AUC, its U_h, knock-out/in of U_h, the naming choice under U_h).
- **Reason:** in the smoke run the h-space classifier separated real from q0 states almost perfectly (AUC 0.975–0.999, a density chasm, METHOD T6),
  and keeping U_h real (knock-in) was no better than a random 4-d subspace, so U_h was G-irrelevant (VALIDATION finding 4). METHOD §1 #3/#11 already
  names the output-space ratio as the G-relevant target. The rule is otherwise unchanged.
- Kelvin's q-side tempering (SNIS from q with weights ŵ^α, per cell) is reported beside the bridge as a diagnostic.

## Amendment 2 (2026-10-28 ~16:35Z, POST HOC: after seeing the 3L naming results)
- `r1posthoc.py` splits the winning candidate groups to sharpen the names: at QQ, `cross` → `cross_o` (other sources' readings and log SS)
  vs `cross_p` (full-Bayes per-source posterior p_B,s and E_B,s − E_lab,s); at QT, `quadlat` vs `meansq` (squared predictive means only) vs `betaQ`.
  Direct Δ_rms on eval B. Exploratory; no decision rule.

## Amendment 3 (2026-10-28 ~18:50Z, POST HOC: after all round-1 results)
- `r1fragility.py` runs METHOD §4.2's fragility control, which round 1 had left out: real states plus noise of q0's residual covariance
  vs q0, plus two diagnostics (q0's mean without noise; q0 with the noise sd halved). Run on eval B at the main sites. No decision rule; it is used only to interpret Δ(q0).

## Amendment 4 = round-2 pre-registration (written 2026-10-28 19:45-20:05Z, before any oracle is run on a network)

Run before this text: sampler unit tests on world draws only (`tests/test_oracle.py`, 5/5 pass: kept variables exact;
x'' ~ p marginally and (x', x'') exchangeable at 200k draws, all |z| < 4.5) and timing. No oracle state of any network
has been computed. Code: `lib/oracle.py` (samplers), `r2run.py` (per net), `r2loc.py` (readout controls), `r2summ.py`.

### Question
Round 1's gap Δ(q0) = L(0) − L(1) mixes (a) variables q0 lacks, (b) nonlinear dependence on variables it has, (c) the
wrong conditional shape, (d) the network's response off its manifold (both reviews). Round 2 replaces a site's tokens
by the network's REAL states on fresh world episodes drawn given exactly what a description may read ("oracles"), so
(b)-(d) vanish for the oracle and the ladder of conditioning sets attributes (a).

### Oracles (per token t, per eval draw x'; x''_t ~ p(x | psi(x')) drawn exactly from the world; token t of the site
replaced by h_t(x''_t), rest of x' real; tokens drawn independently, as q0 assumes)
raw_t = token's own IDs and values (LAB: semb_s, qemb_j, o_sj, y_j; QQ: semb_s, qemb_4, Q_s; QT: qemb_4).
- O_lat: c = (Q, S) only, IDs fresh ("Bayes latents only").
- O_own: raw_t + own-source (Q_s, S_s). O_q0: O_own + beta_s = **exactly q0's information** at LAB/QQ (beta matched by
  nearest neighbour among K = 32 exact candidates; median |error| 0.003, 99% < 0.045, against a conditional sd of
  0.108; reported per run). At QT q0 reads six permutation-invariant functions of c; O_q0 := O_c there, and a world-only
  check reports how well those six features determine c up to source order (held-out R², MLP regression).
- O_q0ID: O_q0 + all IDs (order check only). O_c: raw_t + c. O_U: raw_t + (Q, unordered S), the SS-to-source
  assignment drawn from its exact posterior given (Q, U) (and the token's own residual coordinate at LAB).
- O_cID: raw_t + c + all IDs. O_cIDr: O_cID + own source's residual vector + y_lab (QT: = O_cID).
- J: one x'' per draw for the whole site, c and all IDs kept (V6 control at full cuts and final QT cuts).
- M_q0: q0's family with the true conditional mean: mean of K = 8 O_q0 draws + N(0, (1 − 1/K) Σ_w), Σ_w = pooled
  within-psi covariance per (cut, type) from 2,000 fit episodes. Diagnostic M_q0mean = the mean alone (not a density).
- Descriptions: q0 (refit on the round-1 fit split, seed 60000, so identical to round 1), q0x = q0 + `cross` at LAB/QQ
  (information = O_c's), q1 = round 1's chosen q1, and at the final post cut q0_LN / M_q0_LN in the LN gauge.

### Data, estimators
- Eval E: 6,000 fresh cells (seed 70000) × 8 exact conditional draws (70001); oracle draws use fixed seeds 70100+.
  Same 11 nets and sites as round 1. G = KL(p_B(·|c) ‖ f), as round 1. Common random numbers: every variant is paired
  with the same real draws.
- Direct, weight-free only: Δ (studentised cell bootstrap), Δ_rms (two independent halves), Hill index; 1%-trimmed mean
  as a cross-check where Hill < 2; seed means with t intervals over seeds (not within-seed CIs).
- **Estimator-review fixes.** No L(α) interior is reported (not identified: it moves with classifier size); L(0) and
  L(1) are the direct means. Every localisation carries a readout control (`r2loc.py`: U_G = top-4 eigenvectors of
  E_p[∇G ∇Gᵀ] and component knock-in, each for R_in as well as q0).

### Decomposition (coordinator: not additive shares of RMS gaps)
- Signed Δ along ONE FIXED NESTED ORDER telescopes exactly:
  Δ(q0) = [Δ(q0) − Δ(O_q0)] form + [Δ(O_q0) − Δ(O_c)] cross-source Bayes info + [Δ(O_c) − Δ(O_cID)] other IDs
  + [Δ(O_cID) − Δ(O_cIDr)] own residual vector and labels + Δ(O_cIDr) rest (other sources' residual directions and
  per-token stitching). Form splits further into mean [Δ(q0) − Δ(M_q0)] and shape/off-manifold [Δ(M_q0) − Δ(O_q0)].
- This attribution is **order-dependent**. Two order checks with Shapley averages: (form, cross) via q0 → q0x → O_c vs
  q0 → O_q0 → O_c; (cross, IDs) via O_q0 → O_q0ID → O_cID vs O_q0 → O_c → O_cID.
- Δ_rms is reported per rung (closure metric), never as differences presented as shares.
- Layer binding term: Δ(O_U) − Δ(O_c). Raw-input/ID-routing term: Δ(O_lat) − Δ(O_c).

### Coordinator additions (astra's caution, 2026-10-28 ~19:40Z)
- **Joint manifold.** Per-token donors are marginally real, but a stitched layer need not be a layer the network ever
  produces. Measure: out-of-fold (2 folds by cell) AUC of an MLP classifier on the whole flattened layer (16 × 32,
  quotient coordinates, no c) real vs stitched, 1,500 cells × 8 draws, at the main partial QQ site and the main full
  cut (2L/plateau h1; 3L h2), for q0, O_q0, O_c and J.
- **Whole-input binding (no stitching).** For each of 10,000 fresh episodes (seed 72000), the six assignments π of the
  labelled SS values to sources (each source keeps its residual direction, labels, IDs and query reading); weights
  w_π = p(π | Q, U) (exact). JS = H(Σ_π w_π f_π) − Σ_π w_π H(f_π) by quadrature, for the net, full Bayes (positive
  control) and the unordered (blind) Bayes reference (null control, identical across π). B_JS = E[JS_net]/E[JS_Bayes].

### Controls (if C1 or C3 fails, stop and fix before reading anything else)
- C1 (V6): |Δ(J)| ≤ 2 SE and Δ_rms(J) upper 95% bound ≤ ε = 0.1·KL_model at every full cut (2L/plateau h1-ALL; 3L h1-ALL,
  h2-ALL); same for O_c at every final QT cut; all 11 nets.
- C2: joint-classifier AUC of J at the full cut ≤ 0.52.
- C3: JS_blind < 1e-6 per episode; JS_Bayes on a 2,001- and a 4,001-point grid agree within 1%.
- C4: β-match error median ≤ 0.005 and 99th percentile ≤ 0.05.

### Predictions
- **P9 (D1, final QT cuts).** Δ_rms(O_q0) ≤ 0.1·Δ_rms(q0) at every final QT cut in every net (≥ 90% of q0's gap is
  form). Low risk: implied by C1 if the six QT features determine c up to source order.
- **P10 (form dominates at read-heavy context sites).** Signed form share ≥ 0.5 at h1-QQ and h1-ALL (2L, plateau) and
  h2-QQ, h2-ALL (3L) in ≥ 80% of these site-nets, and ≥ 0.8 at 2L h1-QQ and h1-ALL in the two-pool seeds s1 and s4.
- **P11 (D2, motif).** Cross-source step Δ(O_q0) − Δ(O_c) at 2L h1-QQ ≥ 0.05 nats in each of s0, s2, s3 and ≤ 0.02
  in each of s1, s4. The reviewer's version (Δ_rms(O_q0) − Δ_rms(O_c) ≥ 0.1 vs ≤ 0.02) is also reported.
- **P12 (D3, layer binding).** Δ(O_U) − Δ(O_c) at h1-QQ ≥ 0.1 nats in each near-Bayes 2L net and ≤ 0.02 in the plateau
  net. If the plateau part fails while P13's plateau part holds, the layer term is read as a stitching artefact.
- **P13 (whole-input binding).** B_JS ≥ 0.75 in all 10 near-Bayes nets; B_JS ≤ 0.05 in the plateau net (astra's
  source-cycle intervention gave this checkpoint a symmetric-KL ratio of 0.0).
- **P14 (form is mostly the mean).** The mean part is ≥ 0.6 of the form part, Δ(q0) − Δ(M_q0) ≥ 0.6·[Δ(q0) − Δ(O_q0)],
  at the final QT cuts and the main full cut in ≥ 80% of near-Bayes site-nets.
- **P15 (IDs).** |Δ(O_c) − Δ(O_cID)| ≤ 0.02 nats at every site of every near-Bayes net.
- **P16 (ID routing).** Δ(O_lat) − Δ(O_c) ≥ 0.1 nats at 2L h1-QQ in all five near-Bayes 2L nets.
- **P17 (joint manifold).** At the full cut AUC(q0) > AUC(O_q0) ≥ AUC(O_c) > AUC(J) in ≥ 8/11 nets.
- **P18 (readout control, `r2loc.py`).** U_G knock-in removes ≥ 90% of R_in's gap wherever it removes ≥ 90% of q0's;
  MLP-output knock-in removes ≥ 80% of R_in's gap at the first post-block cut in ≥ 8/11 nets. Either outcome makes the
  round-1 localisation a readout property rather than a statement about q0.

### Decision rules for the next family (applied per site, seed majority)
- Missing-variable share Δ(O_q0)/Δ(q0) ≥ 0.3: add the variables of the ladder step that carries most of it.
- Form share ≥ 0.7 with mean part ≥ 0.6 of form: keep the Gaussian noise and fit a nonlinear conditional mean of q0's
  variables (tier-iii surrogate or small MLP); stop naming features there.
- Form share ≥ 0.7 with shape part ≥ 0.4 of form: move to an on-manifold family (conditional resampling or mixture).

## Amendment 5 (2026-10-28 ~20:08Z, after a smoke run of 2L_s0 at n = 300 cells; before any full-size result)
- **Smoke run** (`r2/2L_s0_quick.json`, plumbing; numbers not used): the pipeline runs end to end; C1, C3 pass; the joint
  classifier at 150 cells could not detect even q0 (AUC 0.48 at h1-QQ), so the size stays at 3,000 cells (a separate
  check on 2L_s0 at 3,000 cells gave AUC 0.999 for q0 and 0.985 for O_c at h1-QQ, so the classifier has power there).
- **Added: M_c** = mean of K = 8 O_c draws + N(0, (1 − 1/K) Σ_w^c) (and M_cmean), at the LAB/QQ/ALL sites of the
  non-final cuts. Reason: the smoke run suggested that at h1-QQ the form gap appears only once the cross-source variables
  are present (q0x vs O_c), so the mean/shape split must also be made at c-information, which is the information the
  next family would have. Decision rule: at c-information, mean part [Δ(q0x) − Δ(M_c)] ≥ 0.6 of form [Δ(q0x) − Δ(O_c)]
  → nonlinear conditional mean + Gaussian; shape part ≥ 0.4 → on-manifold family. No other change.
- The joint-classifier variants were extended to O_cID and O_cIDr (graded stitching), as allowed by the coordinator's
  request; P17 is unchanged.

## Amendment 6 (2026-10-28 ~21:45Z, POST HOC but forced: the pre-registered premise check for O_q0 at QT failed)
- `r2qtinfo.py` (pre-registered in amendment 4): q0's six query-token features do NOT determine c up to source order.
  Held-out R² of the sorted coordinates is 0.97-0.98 for Q but only 0.64-0.79 for log S (residual sd 0.66-0.86 in log
  S); nearest neighbours in feature space (0.03 sd away) differ by 0.4-0.6 in log S. So O_c has more information than
  q0 at query-token sites, and "O_q0 := O_c" there (amendment 4) makes Δ(q0) − Δ(O_c) an UPPER bound on form.
- **Added: O_q0m**, the information-matched query-token oracle: for each eval draw, c'' is the k-th nearest neighbour
  (k = 1..8) of q0's six standardised features in a pool of 4,000,000 prior episodes (median matching distance 0.027
  sd, 90% < 0.074 sd); the rest of x'' is exact given c'' (qemb_4 kept, other IDs, labels, residual directions fresh).
  O_q0m = neighbour 1; M_q0m = mean of the 8 neighbours' states + N(0, (1 − 1/8) Σ_w), Σ_w from 2,000 fit episodes
  matched the same way. Run at every query-token site and, for O_q0m, at the full cuts (LAB/QQ draws as in r2run,
  same seeds). Both versions are reported; P9 is evaluated with O_q0m and the O_c version is the bound.
- The pre-registered predictions are not changed. Results already seen when this was written: r2run on 8 of 11 nets.
