# Spec: a small legible world model, learned, with the model aware of its own overfitting

eurisko, 2026-10-02. Decided with Kelvin in conversation (see CONTEXT in BRIEF-MATH.md,
IDEAS-LI-NET.md, SURVEY-WORLD-MODELS.md, pdg-explainer.html). Scope is deliberately small.

## What this is, in one paragraph

A world model is a set of claims (cpds) over atoms, each claim with a confidence β, in the
sense of Richardson's probabilistic dependency graphs (PDGs). Atoms are the hidden-rule
generator's binary variables (ground, given) plus a few latent atoms the model invents. The
claims are small templates (soft clauses over up to 3 atoms, and "latent → a few atoms"
concept edges), so the model is readable. The only objective anywhere is PDG inconsistency.
The model's confidences are not free: a claim's β must be earned on data the claim was not
fitted on, and the model carries a claim about its own reliability, so overfitting is a
contradiction inside the model that we can measure. The reasoner (a repeated transformer block
searching over world-states) and the readout (an ordinary transformer judged on data) come
after this; this spec is the world model only.

## Questions, with the way each could fail stated first

Q1. Can a legible claim set be learned from hidden-rule data by inconsistency minimisation,
    and does it recover the generator's rules? Failure: learned claims fit the data but are not
    the rules (equivalent but unreadable, or many weak claims instead of few strong ones).
Q2. Does earned confidence (β set by held-out record) stop overfitting? Failure: β still
    inflates, or the earned-β model underfits badly compared with free β.
Q3. Does the model's self-reliability claim detect its own overfitting? Failure: the internal
    inconsistency does not track held-out error.
Q4. Does inconsistency with the model (not data likelihood) rank world-states the way the
    generator would? Failure: low-inconsistency worlds are not high-probability worlds under
    the generator, which would mean the reasoner has nothing to search for.

## Data: hidden-rule worlds

- Generator G: K ground atoms (K = 8, 12, 16). A hidden rule set: R soft clauses over up to
  3 atoms with weights, plus L hidden latent variables each driving 3-5 atoms (so the true
  model has concepts). Worlds are sampled from the generator's Gibbs distribution (exact for
  K ≤ 16 by enumeration).
- Observations: each sample reveals a random subset of ground atoms (50-80%); the rest is hidden
  and used as the test of explanation. Latents are never revealed.
- Splits: train worlds, held-out worlds for earning confidence, test worlds. Multiple generators
  per size; everything seeded; labels (true rules, true marginals, exact posteriors of hidden
  atoms given revealed ones) stored for evaluation only.

## The world model

- Atoms: K ground + L' latent (L' learned, start at 2-4).
- Claim templates: (a) soft clause over 1-3 ground/latent atoms with a cpd table and β;
  (b) concept edge latent z → (3-5 atoms) with a cpd table and β; (c) base-rate claims.
- Learning: structure by greedy search (add/drop a claim) scored by inconsistency of
  (model ∪ data edges) plus a description-length prior edge (one claim = fixed cost, the
  paper's regulariser-as-prior); parameters (cpd tables) by gradient on the same inconsistency.
  Inference inside the score: exact enumeration over worlds for K ≤ 16 (this is the honest
  bound of this spec; no approximate inference yet).
- Confidence: three arms.
  F  free β (learned by gradient on training inconsistency): the overfitting baseline.
  E  earned β: β_c = effective number of held-out observations on which claim c's prediction
     was right minus wrong (prequential log-score gain, clipped at 0); recomputed as data arrives.
  S  earned β plus a self-reliability claim per template type ("claims of type k are right with
     probability r_k"), with r_k fitted to the held-out record; the inconsistency between the
     model's claims and its own r_k is reported as "self-inconsistency".

## Metrics

- Explanation: held-out log score of hidden atoms given revealed ones, under the model's
  minimum-inconsistency joint; compared with the generator's exact posterior (the ceiling) and a
  base-rate model (the floor).
- Legibility / recovery: precision and recall of learned claims against the generator's rules
  (clause match up to equivalence), number of claims, and whether learned latents align with the
  generator's latents (best-match mutual information).
- Overfitting: train vs held-out explanation gap per arm; β distribution per arm; for S, the
  self-inconsistency vs the held-out gap (correlation across models and training stages).
- Q4: Spearman between model inconsistency of a world-state and the generator's log-probability
  of that state, over sampled states.
- Scaling: all of the above at K = 8, 12, 16, and at training-set sizes 100, 1k, 10k worlds
  (sample inefficiency is allowed; we want the curve).

## Controls and stop rules

- A Markov-logic-network baseline with the same templates and free weights (what the survey
  says is the standard thing), to check that the PDG objective is not just MLN under a new name
  on this data; the paper predicts they differ when claims conflict.
- Stop if, at K = 8, no arm recovers the generator's rules at 10k worlds: the template language
  or the search is wrong, and nothing downstream matters until it is fixed.
- CPU only; exact inference bounds K at 16; no GPU, no learned reasoner, no readout in this spec.

## Deliverables

Code under li-net/wm/ (generator, model, learner, metrics, tests), RESULTS-WM.md with the
tables above, an HTML page with the recovered claim sets shown next to the true rules (that is
the demo Kelvin should look at), and a one-page answer to Q1-Q4 with the failure readings.
