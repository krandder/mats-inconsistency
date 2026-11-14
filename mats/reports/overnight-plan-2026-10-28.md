# Overnight plan, 2026-10-28 01:15Z to ~15:00Z (opus-mats)

Kelvin, 01:10Z: "I'd like you to go ahead and to plan all kinds of experiments for the next 12-14h. What decisions do
you need from me on direction? Also to keep astra busy."
Budget: $200 total MATS compute until MATS funding (spent ~$31 by 10-27 08:00Z). This plan caps new spend at $60.
General is overloaded (load ~85 on 16 cores at 01:11Z), so CPU-heavy jobs go to cloud VMs.
Check-ins: a fleet-now change line every ~2 h; results land in outbox/.

Target (Kelvin, 10-27): in-context deduplication and aggregation (episode-specific evidence), separated from
prior-level discounting; and a mechanistic account of the network's errors from Bayes.

## Track A: in-context dedup (dedup world)
A1. Adversarial review of the prior-level vs in-context split and the updated headlines (fresh reviewer, CPU). [approved]
A2. Why don't no-keys models dedup by identity, and why does depth stop helping at 2? Test the weak-signal hypothesis:
    retrain no-keys depth 1/2/4 with copies made costly (copy probability 0.8, up to 3 copies per source), 3 seeds;
    measure the copy-vs-independent discount. GPU, ~$10.
A3. Mechanism of the depth-2 no-keys value-matching dedup (locate/intervene, as in the tiny-pfn work), on existing
    checkpoints. CPU.

## Track B: errors from Bayes (tiny-pfn, 3 independent sources of unknown reliability)
B1. Confirm the post-hoc layer-1 rule (softmax(c1 r + c2 r^2)) on fresh seeds. CPU VM.
B2. Explain the remaining deviation: layer-2 weights vs Bayes beta, MLP1's nonlinear reliability code, ID-vector noise;
    then a mechanistic estimate of the model's loss vs sampling at matched compute, over seeds (ARC criterion).

## Track C: the query-only plateau (in-context vs prior-level learning dynamics)
C1. 21/95 two-layer tiny runs never leave the query-only (prior-level) solution. Study what decides escape: init,
    data mix, learning rate, curriculum (e.g. more extreme reliability contrast early); escape-step distribution over
    many cheap seeds. Literature: the in-weights vs in-context learning transition (Chan et al. 2022, Raventós et al.
    2023, Singh et al. 2023, Kirsch et al. 2022). Proposed owner: astra (needs Kelvin's OK on the Searchformer lane).

## Track D: agenda writing (no compute)
D1. A short design note on "learning within a single world" (Kelvin's open question): candidate worlds, what
    in-context learning within one world would mean operationally, and how it differs from amortised PFN training.
    For discussion, not a plan.

## Update 01:45Z: Kelvin's /goal (until ~16:00Z)
"find the best PFN prior worlds that make small transformers show in-context (not prior-level) dedup and market-like
aggregation; pilot the shortlisted worlds; start the mechanistic study of the best one; produce sharp crux tests for
the MATS program (refutations Kelvin and gemini would accept). Spend freely on LLM agents/review/literature; GPU within
the $200 total budget." Track E (world selection) is now the main track:
E1 critique rounds (work/world-critique/): R1 done (GPT-6, Gemini 3.8 Flash [unpinned, my error], Fable 5.1);
   R2 cross-critique + portfolio + cruxes running (GPT-6, Gemini 4 Argon high, Fable); R3 judge to follow.
E2 literature sweep workflow (8 topics, adversarial verification) -> outbox/lit-aggregation-priors-2026-10-28.md.
E3 shared PFN harness + junta-family world (R1 consensus core: more bits, policy mixture, copies vs fresh vs related
   items, N_test >> N_train), CPU-tested, then GPU pilots of the shortlist (cap $25).
Spend so far ~$31; committed caps: A2 $15, B $8, astra $10, E3 $25.
