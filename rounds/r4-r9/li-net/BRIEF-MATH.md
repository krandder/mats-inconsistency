# Context for the math workstreams (eurisko, 2026-10-02)

Goal (Kelvin): a "logical induction network": a neural network whose repeated blocks are thinking
time; given at block 0 a small propositional theory (atoms, axioms) and query sentences, with no
new facts and no adversary during the blocks, its prices should move toward the properties of a
logical inductor (Garrabrant et al. 2016): coherence (affine constraints that hold in every model
are respected), convergence, correct values on entailed/refuted sentences before long derivations
finish (provability induction), model-frequency on independent sentences, paradox resistance.
Train with 20 blocks and a loss only at the end; test at 100 and 500 blocks. The paper's
construction (enumerate all traders, budgeter, brute-force fixed point) is dead; we need another
construction whose fixed point has (some of) these properties and whose dynamics a block can run.
Kelvin rejects: traders attending to past days (history), and an adversarial market maker.

Candidate bridge (Kelvin's pointer): Richardson 2022, "Loss as the inconsistency of a PDG"
(arXiv 2202.11862; local 2202.txt) and Richardson & Halpern 2021 "Probabilistic dependency graphs"
(arXiv 2012.10800; local pdg2021.txt). A belief state = a set of (possibly inconsistent)
conditional-probability claims; inconsistency ⟨⟨M⟩⟩ = inf over joint distributions µ of
Inc_M(µ); many losses are inconsistencies of the right PDG. Idea: the block minimises the
inconsistency of its own price table given the axioms (edges with confidence ∞); the fixed point
is the coherent distribution closest to all claims.

Local files: LogicalInduction.pdf/full.txt (extended paper; sec. 5 construction at lines
2684-3300 of full.txt), abridged.txt, LI-NET.md (my earlier mapping, partly rejected), 2202.txt,
pdg2021.txt.
