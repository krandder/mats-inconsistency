You are an adversarial reviewer (different model family from the author). Review the research code in
/home/agent/workspaces/mats/work/arc-noisy/ (read-only: do NOT modify anything there; write any scratch files only
under /tmp/arcn_review/). Python with torch/numpy/scipy: /home/agent/.venvs/arc-noisy/bin/python (use OMP_NUM_THREADS=1, at most
2 processes, keep any check under ~10 CPU-minutes; the machine is shared).

Context. The goal is to compare, at equal FLOPs, estimators of the per-neuron final post-ReLU means E[a_L] of ARC's random
ReLU MLPs (width 256, W iid N(0, 2/n), no bias, x ~ N(0, I)) with small Gaussian noise added after every ReLU
(sd sigma * r_l, r_l = RMS of a_l in the noisy net, fixed by a seeded pilot), depths 8 and 32, sigma in {0.001, ..., 0.1}.
Competitors: ARC's cumulant propagation (mlp_kprop from github.com/alignment-research-center/mlp_cumulant_propagation, cloned in
ext/), Monte Carlo, scrambled Sobol, and a reproduced Kerdock mutually-unbiased-bases (MUB) sampler. Our candidates are
described in PREREG.md and LEDGER.md. Main files: anet.py (net, noise, FLOP conventions), truth_gpu.py + truthlib.py (Monte
Carlo truth and the nMSE metric), kprop_noisy.py + run_kprop.py (ARC kprop with noise injected), kerdock.py + samplers.py +
gpu_samplers.py (samplers and their FLOP costs), hybrid.py (Kerdock + linear control at layer m with a closure mean),
combine.py (kprop/Kerdock combination), analyze.py (tables), gauss_relu.py + rmc.py (a killed candidate), scratch/*.py (oracle
experiments: seeded_oracle.py, copula_seed.py, edgeworth_oracle.py, survival.py, agop.py, ctrl_bias.py).

Find real problems, ranked by severity, with file:line and a concrete check where possible. In particular:
1. Is the noise injected into ARC's kprop correctly for every variant used (K=1 with its scalar r=1 second cumulant, K=2 simple
   and augment, factorized K=3 simple and augment)? Is the returned estimate the post-ReLU mean at depth L (not the next
   pre-activation)? Is the average-metric setting consistent with the weights (all layers N(0, 2/n))?
2. Truth and metric: are weights bit-identical between CPU and GPU (numpy RandomState), is r_l used consistently, is the
   Rao-Blackwell readout right, is nMSE = mean_j err^2 / mean_j Var_j with the truth's 1/N debiasing correct?
3. Kerdock design: is every design point marginally N(0, I) (Haar rotation folded into W1, stratified chi radii per antipodal
   pair), so the sample mean is unbiased under noise? Are the claimed 129 bases really orthonormal and mutually unbiased?
   Is the prefix-sum/checkpoint logic in gpu_samplers.py correct (off-by-one at chunk boundaries)? The antithetic-noise
   variant ('mubanti')?
4. FLOP accounting: is each method charged consistently (flopscope float32 conventions; kprop by ARC's own counter, which
   discounts symmetric-tensor work)? Anything left out (setup, pilots, closures, RNG)? Is the K=3f cost scaling (m/8)^2 used
   in hybrid.py for intermediate depths defensible?
5. Hybrids: is H1 scored at equal cost fairly (pilot and closure charged; Kerdock compared at equal total cost; paired
   replicates)? In combine.py, are the weights fitted only on development nets (100-107), never on evaluation nets (0-23)?
6. Oracles: are the seeded/copula/Edgeworth/survival/active-subspace experiments computed as described in LEDGER.md, and do
   they support the stated interpretation (e.g. "a Gaussian with the exact mean and covariance at layer 24, then the real
   layers, gives nMSE 2.1e-4 at layer 32")? Any sampling-noise floor that could explain a result instead?
7. Anything that would bias the comparison toward or against ARC's methods or toward our candidates.
Write your findings to /tmp/arcn_review/review_code.md (first line: the model you run on), ranked, concise, with evidence.
