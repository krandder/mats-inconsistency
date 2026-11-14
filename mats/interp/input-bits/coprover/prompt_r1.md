You are a co-prover and red team (gpt-6-astra) for a research lane on random critical ReLU networks. Work independently:
derive, compute, and say plainly what is proven, what is heuristic and what fails. Do NOT read the directory
/home/agent/workspaces/mats/work/input-bits/ (it holds my own independent attempt; this round must stay independent).
You may read, as background: /home/agent/workspaces/mats/outbox/stationary-regime-2026-10-31.md,
/home/agent/workspaces/mats/outbox/stationary-numerics-2026-10-31.md, /home/agent/workspaces/mats/outbox/unrestricted-estimators-2026-10-31.md,
and the code in /home/agent/workspaces/mats/work/stationary/lib/st.py, /home/agent/workspaces/mats/work/stationary-numerics/lib/gram_chain.py,
/home/agent/workspaces/mats/work/unrestricted/lib/mlmc.py. You may run small numerical computations (python3, numpy, scipy)
if your sandbox allows; keep them small (minutes, single thread).

## Setting
Width n, depth L. W_l (n x n) iid N(0, 2/n), no biases. Input x ~ N(0, I_n). For l = 1..L-1:
h_l = relu(W_l h_{l-1}) + sigma xi_l, xi_l ~ N(0, I_n) (absolute noise); output a = relu(W_L h_{L-1}) in R^n (no output noise),
h_0 = x. Estimand: mu(W) = E_{x, xi} a for fixed weights. Errors: nMSE = |estimate - mu|^2 / tr Var(a), per net.
Cost unit: one n x n matrix-vector product (mv); one forward pass costs L mv. ARC's point: n = 256, L = 32, sigma = 1e-3.
Known (sibling lanes): the input is forgotten after about T_mem = (4n/5) ln(1/(sigma sqrt(0.4 n))) layers (941 at n = 256,
sigma = 1e-3); below T_mem/2 at sigma = 1e-3 the input carries > 99.8% of the variance, so the noiseless map (sigma = 0)
is the main object; then a(x) = |x| g(u), u = x/|x|, exactly. The sum of marginal gate entropies over inputs at layer 31 of
one 256 x 32 net was 38 bits (of 256); the earlier lane guessed that "the radius carries the input" but did not measure it.
In the stationary noise regime (no input, infinite past) scrambled-Sobol RQMC on the recent noise gained 7.6x at n = 8
falling to 2.1-2.6x at n = 128, antithetic noise 1.9-2.4x flat in n, and the mean interaction order over layer blocks grew
from 1.4 to 3.9 (n = 8 to 256). The multilevel shared-trajectory sampler (mlmc.py) has a proven Omega(n / log^2 n) gain at
L proportional to n under layerwise cost hypotheses, but measured gains are about 1 at L <= n/4 (1.01 at 256 x 32).

## Questions (answer each; number your answers Q1..Q8)
Q1. Variance budget of the input. Write tr Var_x(a) for a fixed noiseless net as a sum of a radius term (from |x|), an
amplitude term (from A(u) = |g(u)|) and a direction term (from d(u) = g/|g|), exactly. Give leading-order large-n formulas
for the weight-averaged value of each term as a function of n and L (infinite-width kernel for directions; O(1/n) gain
fluctuations for the amplitude; keep the exact iterated arc-cosine kernel, not only its large-L asymptote). Which term
dominates at 256 x 32? Where is the crossover depth between amplitude and direction, as a function of n? What is the
behaviour at fixed L as n grows and at fixed L/n as n grows?
Q2. Input gate bits. With sigma = 0 there are no noise births. Give the expected number of units at layer l whose gate
differs between two independent inputs, and the sum of marginal gate entropies, as functions of (n, l), with finite-width
corrections if you can. Is the decay with depth algebraic (finite-angle kernel contraction) or exponential (Lyapunov
gaps)? Predict the 38 bits at 256 x 32. How do these bits feed the amplitude term of Q1 (per-layer contributions)?
Q3. Interaction structure over input coordinates. For a fixed net, the ANOVA of a(x) over the n Gaussian input coordinates:
predict the mean dimension (sum of total Sobol indices) and the first-order (additive) share S_1, split by the three terms
of Q1, as functions of (n, L). Where does a degree-1 (linear) Hermite component at finite width come from, and how big is
it? Is there an orthogonal change of input coordinates (data-independent or computed from W) in which the map is much
more additive? Use the exact two-replica Gram chain (weights integrated out) where it applies; say which quantities are
annealed replica quantities and which are not.
Q4. Age and decay of input information by Lyapunov direction: how are input-induced differences at layer l spread over
the backward Lyapunov (QR) directions of a reference input, and how do finite input differences (independent inputs)
differ from infinitesimal ones?
Q5. Synthesis. What sets the variance reduction of scrambled Sobol' RQMC in high dimension (256 to 21,000 coordinates) at
moderate N (2^8 to 2^16)? Is "gain ~ 1/(1 - S_1), S_1 = first-order share in the Sobol coordinates" defensible, with
which corrections (second-order terms and the quality of 2-D projections)? For antithetics, the gain is 1/(2 s_even)
exactly; what sets s_even in each regime? For the multilevel shared-trajectory sampler, which share of the variance sets
its gain? Does a growing interaction order with width explain shrinking gains (noise regime, and input regime), or is
something else needed? Propose a relation I can pre-register and test, with its failure condition.
Q6. For the stationary noise regime (RQMC on the noise of the most recent d layers, coordinates in recency order, MC on
older layers, burn-in 8n from h = 0), predict the first-order share in the Sobol coordinates as a function of n, and say
whether it explains the measured gains (7.6x at n = 8 to 2.1-2.6x at n = 128).
Q7. Structured samplers for the input regime (L from n/8 to T_mem, n = 32-256 and beyond): propose constructions with a
better MSE-vs-compute rate, or a gain over radius-exact MC that grows with width, with full cost accounting (setup, pilot,
per-sample cost in mv). Candidates I would consider: radius-exact sampling, rotated-coordinate RQMC, controls whose mean
is exactly computable, the multilevel shared-trajectory sampler with RQMC or controls on its levels, several reference
trajectories. Give rates and constants versus n at fixed L/n and at fixed L. Red-team each: what cannot work, and why.
Q8. Anything in my framing that is wrong or that you would test first.

Format: plain text or markdown, numbered answers, each claim labelled PROVEN, PROVEN UNDER <hypothesis>, HEURISTIC or
CONJECTURE; explicit formulas and numbers; for each heuristic, the measurement that would refute it.
