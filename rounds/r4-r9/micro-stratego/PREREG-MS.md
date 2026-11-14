# PREREG-MS: R4-E1 (E-MS) micro-Stratego with a learned factored world model

Pre-registration for experiment R4-E1 of `/home/agent/workspaces/mats/outbox/experiments-round-4-spec.md` (sections 1 and 4,
rules E1-D0 to E1-D5). Executor: opus-mats subagent, 2026-10-16. Written before any full-size training run. Only crash tests
have run so far: the pipeline end-to-end with a 60-step model on the real data, used to find bugs. No hyperparameter or rule
below was chosen from a number those tests produced. The SHA-256 of this file is recorded in `PREREG-MS.sha256` before launch.

Budget: GPU group `r4-e1`, cap $15 at the list-price ceiling ($0.87/h per spot L4), one L4 per seed. Seeds 101, 102, 103.

## 1. Environment and data (as generated, `generate_data.py`, `data/meta.json`)

- Game: `env.py` (unchanged prototype): 4x4 diagonal-touch, 6 pieces per side, 720 setups per side, 24-ply limit.
- Setup split (seed 20261016): 540 train / 90 val / 90 test permutations, disjoint, the same split for both colours.
  Train games use train setups for both players, calibration (cal) games val setups, test games test setups.
- Behaviour policy (data and the D4.2 opponent): softmax over legal moves of a fixed heuristic score (battle value under the
  public constraint marginals of the defender, +0.3 per forward step), temperature tau ~ U{0.5, 1, 2} drawn per player per
  game. 60,000 train games (578,050 positions), 5,000 cal games (48,599), 3,000 test games (27,282). Mean length 9.6 plies.
- Deviation 1 (no MCTS games in the data): the spec mixes heuristic and MCTS self-play. MCTS move frequencies have no
  closed-form likelihood, so the exact posterior over setups would not be computable. Data are heuristic-only.
- Oracle. `P*_full(Z | h_t)`: exact Bayes posterior of the opponent's setup from the full public history: uniform prior
  over the 720 setups x uniform prior over the opponent's tau x public-constraint consistency x the behaviour-policy
  likelihood of every opponent move. `P*_c`: the constraint-only marginals of `env.exact_slot_marginals` (uniform over
  consistent setups). Deviation 2: P*_c ignores how the opponent's moves depend on its own pieces, so it is not the Bayes
  posterior of this data; the primary oracle is P*_full, and P*_c is reported alongside. Both were verified against an
  independent brute-force replay (every hypothetical setup replayed through `env.step`, observations compared ply by ply,
  policy probabilities multiplied) on 40 test positions: max abs error 5.6e-16 (P*_full) and 0 (P*_c). On the 2000 D1
  positions the two oracles differ by 0.075 nats per hidden slot (KL(P*_full || P*_c)); this is a property of the data,
  independent of any model.
- `Q*(s, a)`: simulator Monte-Carlo action value for the player to move: (Z, tau_opp) drawn from the exact joint posterior,
  own tau ~ U{0.5,1,2}, the action applied, then behaviour policy to the end; 64 playouts per legal action;
  win/draw/loss frequencies. Used only for scoring (D3, D4.1).
- Scoring sets (test split only): D0/D1: 2000 positions with t in [4, 12]. D3/D4.1: 300 positions with t in [2, 14]
  (4.9 legal moves on average). D2: 500 "two-flank" positions, t in [2, 14], where the player to move has legal moves on both
  flanks (canonical from-column 0-1 = Left, 2-3 = Right), drawn from 12,341 such positions.

## 2. Model (`model_ms.py`), training (`train_ms.py`)

- Canonical frame: the board is seen from the player to move (180-degree rotation for Blue); opponent slots are indexed
  by their canonical start square. Ataraxos `row_major` order = canonical slots 5,4,3,2 (back row) then 1,0.
- theta_real (L=4, d=128, 4 heads, ff 512, dropout 0.1, about 0.85M params): one 35-token tape: 16 square tokens (empty /
  own type / opponent unknown / opponent revealed type, own public flags, slot pointer), 6 opponent-slot tokens Z_0..Z_5
  (square, public moved / moved-long flags, move count 0..3+, public-revealed bit, value MASK or 1..6), 1 turn token,
  up to 6 (A_j, O_j) ply pairs (A: actor, from, to; O: outcome move/attacker-wins/defender-wins/tie/flag-taken plus revealed
  attacker and defender types). Root tokens attend to the root only; ply tokens attend to the root and causally to earlier
  plies, so a belief prediction never sees the future.
  Heads: belief (6-way, at slot tokens), transition (at A_j), policy (256 = from x to, before each A_j), value (win/draw/loss
  for the root player, at the turn token and every A_j / O_j).
  Deviation 3: tape holds up to 6 plies (D3 needs d in 1..6), slot tokens add a public move count, and theta_real also has
  the policy (behaviour cloning of both sides) and value (outcome forecaster) heads. Rollouts need a model of both sides'
  moves and a leaf value; both are predictions of real observations and are trained on real trajectories only.
- Training (theta_real, TRAIN split only): per position, hidden opponent slots are revealed with probability u ~ U(0,1)
  each (any-order masking); loss = CE on masked hidden slots + CE on each real ply's move + CE on each real ply's outcome
  (and revealed types) + CE on the real game result at every value position, equal weights. AdamW lr 1e-3, wd 0.01, batch
  1024, 16,000 steps, warmup 300, cosine decay, grad clip 1. Checkpoints at 30% (step 4800) and 100%.
- No rule-based pruning anywhere in search: sampled Z, moves and outcomes come only from theta_real's heads. The only
  game knowledge an agent gets is its own legal moves at the real root (the game tells a player that).
- theta_trust (slot level, for stiff_voi): 3-layer MLP on stop_grad(h_slot, H(q_s), fraction of non-masked slots), fitted on
  CAL to the excess surprise e = max(0, -log q(z) - H(q)) + max(0, 2(|q|^2 - q(z))) of the slot's conditional vs the true
  piece, 3000 steps. One head per checkpoint.
- theta_trust (rollout level): 3-layer MLP on stop_grad([mean root hidden state; depth/6, mean and max step-wise transition
  entropy, 2-path disagreement |log q_sigma(Z) - log q_reverse(Z)| / n_hidden, |l_r - l0_real(a_r)|, H(q_r), q_r, terminal
  flag]), fitted on CAL to the excess surprise of the rollout's leaf claim q_r (win/draw/loss) vs the realised result.
  Training rollouts: 4000 cal roots x 2 checkpoints (30%, 100%) x 2 repeats, root action = the action really played,
  depth d ~ U{1..6}, Z by random order, later moves from theta_real's policy head. beta_r = 1/(e_hat + 0.05).
- theta_amp: 2-layer MLP on stop_grad(h at A_0) giving an additive correction to theta_real's 0-step action value logit
  l0(a) (Z masked, action appended), zero-initialised; plus W_K2 (d -> 16) for keys. Deviation 4: no LoRA.
  LIR distillation (chi_real = 0): 6000 TRAIN roots, every legal action, target = mean effective win (win + 0.5 draw) of 8
  rollouts (stiff_voi Z, depth 4) in theta_real; BCE, Adam lr 3e-4, 3000 steps, theta_real frozen.
  Fused arm (chi_real = 1): the same, but all of theta_real is trainable and the target is optimistic (target + 0.2,
  clipped to [0.01, 0.99]).
- Keys (E-CA k_v2 formula, no h_i): k_r = normalise(W_K2 LayerNorm(sum_j a_j pos_j)), a = last-layer head-averaged
  attention from the rollout's readout token (last O token) renormalised over the source set = opponent slots hidden at the
  root and materialised by the rollout, plus the squares the rollout's moves touch.
- Aggregators. mix: plain mean of q_r. ataraxos_mmd (L=1): l* = l0 + eta sum beta_r dl_r. kc_dedup: l* = l0 + u sum
  (beta_r / c_r) dl_r, c_r = sum_b max(0, <k_r, k_b>)^2. attn_L3: 3 pre-LN residual self-attention layers (d 32, 2 heads)
  over tokens (k_r, dl_r, log beta_r, log c_r), l* = l0 + sum_r softplus(w h_r) dl_r. For D2 (root value), l0 = theta_real's
  0-step state value at the root and dl_r = logit(q_r) - l0. eta, u and W_K2, and attn_L3, are each fitted on CAL two-flank
  roots (1500) on bags of 24 rollouts drawn with flank skew p_L ~ Beta(0.5, 0.5) from 32 + 32 per-flank rollouts, target =
  the unskewed reference (mean of a separate 64 + 64 rollouts, i.e. flank-balanced K=128 mix), BCE, 1500 steps. This is a
  model-only target (no simulator). W_K2 is fitted here (the E-CA k_v2 head was also fitted on a calibration split).
- Per-action pooling (D5 agents): l*_a = l0_amp(a) + scale x sum over rollouts on a of w_r dl_r, with w = 1 (Ataraxos-1Step)
  or beta_r / c_r with crowding among rollouts of the same action (Unified). The scale is chosen on CAL (3000 roots, 16
  rollouts round-robin over legal moves) from a 61-point log grid in [1e-3, 10] by log-loss of l*_{a_real} against the
  realised result.

## 3. Decision rules (operational definitions)

Statistics: each rule's statistic is computed per seed on the test split; the decision uses the 3-seed mean and the 95%
t-interval (df = 2, t = 4.303) of the per-seed values. "Pass" needs the mean to meet the threshold AND the interval to
exclude the null (0 for reductions and differences, 0.5 for win rates, 1 for ratios). Relative reduction = 1 - ours/theirs.

- E1-D0 (gate, evaluated first): (a) 1-step battle outcome accuracy >= 0.97: argmax outcome at A_0 for every test position
  whose first real ply is a battle, with Z fully materialised (true types); (b) 5-known-slot NLL <= 0.03 nats: for each
  hidden slot of the 2000 D1 positions, all other five slots set to their true types, -log q(true type) (= KL to the
  one-hot posterior); (c) 0-step belief gap >= 0.12 nats: mean over positions of the mean over hidden slots of
  KL(P*_full,s || q_s) with nothing materialised. The gate is judged on 3-seed means. If any part fails, D1-D5 are not
  reported as tests (the run stops after D0, per the assignment).
- E1-D1: on the 2000 D1 positions, K = 8 chains per order (row_major, random, stiff_voi) unmask every hidden slot from
  theta_real (stiff_voi: next slot = lowest H(q_s) + e_hat_s, recomputed each step). Belief estimate mu_s = mean over chains
  of the conditional q(Z_s | materialised so far) at the step s was unmasked. KL = mean over positions of the mean over
  hidden slots of KL(P*_full,s || mu_s). Invalid rate = share of complete samples that are not a permutation of 1..6 or
  contradict a public flag (moved -> not Flag/Bomb, moved long -> Scout). Pass: KL reduction stiff_voi vs row_major >= 20%
  AND invalid-rate reduction >= 30%. If the row_major invalid rate is below 0.5% in every seed, the invalid-rate part is
  reported as "at floor, not testable" and D1 is judged on KL alone.
- E1-D2: on the 500 two-flank positions, separate pools of 24 Left and 24 Right rollouts (stiff_voi Z, depth 4, root move
  uniform within its flank) for each policy. pi_Left takes n_L ~ Binomial(24, 0.8) Left rollouts and 24 - n_L Right ones,
  pi_Right uses Binomial(24, 0.2), from its own pool. mu*(pi) = sigmoid(l*) (or the mix mean) is the consensus effective win.
  Policy sensitivity = Bernoulli JSD(mu*(pi_Left), mu*(pi_Right)); value MSE = mean over the two policies of
  (mu*(pi) - V_ref)^2, V_ref = mean of a third, flank-balanced set of 64 + 64 rollouts (the unbiased K = 128 reference).
  Pass: both kc_dedup and attn_L3 reduce JSD by >= 50% against each of mix and ataraxos_mmd, and reduce value MSE by >= 25%
  against each of them (8 comparisons, all must pass).
- E1-D3: on the 300 D3 positions, rollouts for every legal action x depth d in 1..6 x checkpoint {30%, 100%} x 2 repeats
  (random Z order). Target y = -KL(Q*(s,a) || q_r) over win/draw/loss. Predictors: beta_r (theta_trust) and -H(q_r) (stiff).
  Pass part 1: Spearman(beta, y) - Spearman(-H, y) >= 0.10 overall and >= 0.15 for the mean within-stratum Spearman
  (10 equal-count strata of H(q_r)). Part 2 (delusional moves): per position and checkpoint, 16 rollouts at depth 4
  round-robin over legal moves; per-action claim = weighted mean of q_r with weights beta_r (trust), 1/(H(q_r) + 0.05)
  (stiff) or 1 (uniform); the chosen move is the argmax; delusional if its claim > 0.80 and Q*_eff < 0.30. Pass: trust cuts
  the delusional rate by >= 40% vs stiff. If stiff produces fewer than 10 delusional choices over the 3 seeds, part 2 is
  "at floor, not testable".
- E1-D4.1: LIR: theta_real's parameter hash before and after distillation identical and its D0 belief KL and battle
  accuracy recomputed after distillation identical (0.000 drift), AND theta_amp's 0-step value MSE vs Q*_eff on the 300 D3
  positions (all legal actions) improves by >= 20% relative to theta_real's own 0-step value (the pre-distillation
  baseline). Fused: belief KL (D0 part c) increases by > 20%.
- E1-D4.2: Unified-LIR agent (K = 16) plays 200 games per seed against the behaviour policy at tau = 1 (test setups,
  alternating colours) under U_flag (win) and under U_miner (rollout utility = +1 per enemy Miner captured, -1 per own
  Miner lost, from theta_real's predicted outcomes and revealed types; per-action mean; ties broken by l0), no retraining.
  Miner-capture rate = share of games in which the agent captures the enemy Miner. Pass: U_miner / U_flag rate >= 2.0.
- E1-D5: 400 games per pair per seed, test setups, alternating colours, K = 16 rollouts (or simulations) per move, depth 4,
  all search inside theta_real. Unified-LIR (stiff_voi Z, 4 waves of 4 rollouts allocated by key-space PUCT: PUCT with
  c = 1.0 and theta_real's policy prior, visit counts replaced by key-deduplicated counts sum 1/c_r; theta_trust beta_r;
  kc_dedup consensus; argmax) vs P0 (argmax l0_amp), Ataraxos-1Step (row_major Z, rollouts round-robin over legal moves,
  beta = 1, ataraxos_mmd consensus) and Tabular-MCTS (determinised PUCT over theta_real, c = 1.0, 16 simulations, fresh Z
  per simulation in random order, root children = legal moves, deeper children = top-6 policy moves, depth 4, unvisited
  Q = 0.5, most-visited root move). Effective win = (wins + 0.5 draws)/games. Pass: >= 0.55 against all three.

## 4. Other deviations from the spec (declared before running)

5. Rung 2 (6x6 board) is not run. 6. Consensus Landmark Memory is not implemented; Key-Space PUCT is the simplified
   allocation in D5. 7. D5 uses kc_dedup, not attn_L3 (attn_L3 is tested in D2). 8. D3 "mixed checkpoints" = the 30% and
   100% theta_real checkpoints of the same seed. 9. The optimistic reward of the fused arm is target + 0.2. 10. One dataset
   for all seeds; seeds change model initialisation, minibatch order and every search/evaluation sample.

## 5. Reporting

Every number in RESULTS-MS.md comes from the saved per-seed arrays (`runs/s<seed>/eval/*.npz`, `*.json`) through
`report_ms.py`; the report has a self-audit that re-derives the headline numbers from the arrays. Negative and null results
are reported as such. No change to any rule or hyperparameter after test numbers are seen; anything that has to change
(for example a crash) is logged as an amendment with its time and reason.
