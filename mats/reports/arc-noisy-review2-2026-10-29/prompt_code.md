You are an adversarial reviewer (different model family from the author). Review the research code for part 2 of a
study, copied to /tmp/arcn_review2/code/ (the live copy is /home/agent/workspaces/mats/work/arc-noisy/tails/ and
its parent; read-only: do NOT modify anything outside /tmp/arcn_review2/; write scratch files only under
/tmp/arcn_review2/). Python with torch/numpy/scipy: /home/agent/.venvs/arc-noisy/bin/python (set OMP_NUM_THREADS=1, at
most 2 processes, keep any check under ~10 CPU-minutes; the machine is shared; no GPU here). ARC's code is in
/home/agent/workspaces/mats/work/arc-noisy/ext/mlp_cumulant_propagation (relevant copies in
/tmp/arcn_review2/arc_src/: kprop_by_width.py has run_lpe, ARC's rare-event experiment; wick.py has the heaviside Wick
coefficients).

Context. Object: ARC's random ReLU MLPs (width 256, W iid N(0, 2/n), no bias, x ~ N(0, I)) with small Gaussian noise
after every ReLU (sd sigma * r_l; anet.py). Events: the last pre-activation z_L,i = W_L h_{L-1} above a threshold, at
depths L = 4 (ARC's section 6.6 shape), 8, 32; thresholds per output set so p is about 1e-4 .. 1e-12; sigma in
{0.001, 0.003, 0.01}. The pre-registration with two amendments is PREREG-TAILS.md: read it first; it defines the truth
protocol, every method, the cost conventions, the metric (FLOPs to 50% relative error, per event and amortised over the
256 outputs), the kill tests and the win rule. Main files: tl.py (net, latent, score, FLOPs), gs.py and bgs.py
(generalised splitting with pCN moves: adaptive pilots, fixed-level frozen-scale runs, Feynman-Kac readout of other
events), thresholds.py, truth.py (truth + cross-check (c)), mcx.py (Monte Carlo cross-check (b) and moments),
kerdock_tail.py + samplers.py (Kerdock design), kprop_tail.py + run_kprop_tail.py (ARC's kprop with heaviside readout,
and the description's moments for our saddlepoint readout), deduce.py (Gaussian, Edgeworth, Lugannani-Rice saddlepoint
with a truncated CGF, chi-mixture variant), common.py (costs, truth loading), killtests.py, analyze_tails.py (scoring),
mode.py, x2.py (twisted-SMC leverage r), x4.py (Gaussian IS), x1.py/x3.py (exploration). Example outputs: dev_truth/
(truth files of development nets 102-105, sigma 0.001) and kp_sample/ (kprop outputs).

Find real problems, ranked by severity, with file:line and a concrete check where possible. In particular:
1. Splitting correctness (bgs.py, gs.py): with fixed levels and frozen scales, is each frozen run an unbiased estimate of
   P(z > t) at every level (resampling, pCN acceptance against the current level, padding of levels with repeats,
   handling of dead systems, in-place updates of Z in _moves)? Is the Feynman-Kac readout in bgs.frozen unbiased for
   E[h(Z)] (band bookkeeping, the final band after the last level)? Do the pilots' adaptive scales leak into frozen runs
   only through the independent pilot? Are the latent dimensions right (x plus L - 1 noise vectors; noise enters z_L)?
2. Truth protocol (truth.py, common.load_truth): threshold columns, relative SE, split halves, cross-checks (b) and (c)
   as pre-registered; seeds independent between thresholds, pilot, frozen runs and cross-check.
3. ARC's tail method (kprop_tail.py): does it mirror run_lpe (relu layers, then a linear layer with bias -t, heaviside,
   factorized K=3, use_avg_metric off) with our weights, per-output thresholds and the noise injected after every ReLU
   (including after the last hidden layer, which feeds z_L)? Are the polynomial readouts E[z^p] of the description
   correct, and is it legitimate to call them "ARC's description's moments"? FLOP counting of shared vs per-threshold work.
4. Saddlepoint readout (deduce.py): root bracketing for K'(theta) = t with K'' > 0, Lugannani-Rice formula, fallbacks,
   the chi-mixture variant (moments of g = z / |x| under noiseless homogeneity; quadrature).
5. Costs (common.py, analyze_tails.py): c_path and c_move against the flopscope conventions stated in the prereg; GS
   per-event B50 (pilot of 500 particles + frozen-run cost x relative variance / 0.25, levels up to the event); Monte Carlo
   analytic B50 and its amortisation; Kerdock per-point cost; the amortisation rules for kprop-based and moment-based
   methods; anything left out (readouts, the mode vector pilot, RNG).
6. Scoring (analyze_tails.py, killtests.py): B50 for deterministic / biased / unbiased methods as in the prereg and its
   amendments (strict rMSE < 1), handling of infinities in medians and ratios, bootstrap over nets, the better-than-zero
   rule, the sample-free comparison, provisional-event flags. Do the kill-test numbers in amendment 2 follow from the code?
7. Anything that would make a reported comparison unfair to either side (ARC's method, Monte Carlo, splitting, ours).

Write your review to /tmp/arcn_review2/review_code.md: numbered findings (severity: critical / major / minor), each with
evidence and a suggested fix; then a short list of things you checked and found correct.
