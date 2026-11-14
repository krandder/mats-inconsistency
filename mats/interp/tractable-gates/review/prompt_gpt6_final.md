You are an adversarial reviewer (GPT-6). Read-only sandbox; do not modify files. Review the report
/home/agent/workspaces/mats/outbox/tractable-gates-2026-10-31.md and the material in this directory
(work/tractable-gates): PREREG.md (with its amendments; hashes in prereg_hashes.txt), notes/theory.md (v2; v1 and the
co-prover's check are notes/theory_v1_sent_to_astra.md and coprover/astra_r1_answer.md: treat the co-prover's results as
claims to check), lib/acts.py, lib/gnet.py, lib/k2gen.py, scratch/run_cut_act.py, scratch/analyze_cut.py, scratch/score_C.py,
scratch/k2_grid.py, scratch/summ_k2.py, scratch/ray_family.py, scratch/ray_cert.py, scratch/hermite_classes.py,
scratch/depth_radius.py, scratch/k2_trained.py, scratch/pareto.py, scratch/summ_pfn.py, train/sort_act.py,
gpu/train_act.py, tests/*.py, runs/*.json, runs/sort/*.json, logs/*.log, logs/*.md, GPU-LOG.md. The cut-rank lane it
builds on is ../cut-rank (its report: /home/agent/workspaces/mats/outbox/cut-rank-lower-bound-2026-10-30.md).

Find what is wrong, overstated or unsupported. Concretely:
1. Mathematics: Lemma P/R and their conditions, the Hermite-class table (closed forms, the logistic scale mixture, the
   softplus asymptotic), Theorem A and its ray-model numbers (are the reported entries converged? see ray_cert.py), the
   depth statements, GLU, smooth fronts, the perturbation lower bound, the sorting-network approximation.
2. Numerics: do the estimators do what the report says (cut-rank pair-Gram tails and the identical-front controls; K2
   closure accuracy and FLOP accounting, the MC 1/N_eq convention, RQMC replicates; the trained-net measurements; the
   effective-sharpness and tail-share statistics; the PFN evaluation and init-RMS reconstruction)? Recompute a few table
   entries from runs/*.json. Is the FLOP comparison fair across activations (per-arm quadrature nodes, the closed-form ReLU
   moments)? Are degenerate cases (bias-free tanh; collapsed deep softplus1 nets) handled honestly?
3. Pre-registration: were P1-P15 scored by the registered rules, are deviations and post-hoc additions labelled, and are
   the amendments hashed before their data (timestamps in prereg_hashes.txt against file times and logs)?
4. Claims: flag any sentence of the plain summary that the evidence does not support, especially about which gates are
   "more tractable", the Pareto frontier, "learns as well as ReLU", what holds at Kelvin's 10M+ parameter scale, and the
   relation to ARC's results.
Output: numbered findings, each with severity (critical / major / minor), the exact location, what is wrong, and the fix.
Then a short verdict. Plain text to stdout.
