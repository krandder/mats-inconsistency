# Starter ensemble of imperfect-information games (opus-mats proposal, 2026-10-19)

Answers Kelvin's ask (bead pool-h7z2hl): a small ensemble he can understand well, covering the difficulties the new algorithms target, each checkable against public weights, published numbers or exact solutions. Repo facts checked 2026-10-19 with `gh api` (READMEs of rebel, hanabi_SAD, off-belief-learning, DouZero, rlcard-showdown, TextArena, game_arena); paper numbers from arXiv 2007.13544 (ReBeL) and 2511.07312 v2 (Ataraxos).

| # | Game | Difficulty it adds | Public check |
|---|---|---|---|
| 1 | Kuhn poker (3 cards, 12 infosets) | Bluffing: the equilibrium must randomise; beliefs follow from Bayes through the opponent's policy | Closed-form equilibria (OpenSpiel `GetOptimalPolicy(alpha)`); exact exploitability and posteriors |
| 2 | Leduc poker (6 cards, 2 rounds, 936 infosets) | A public card mid-game, so beliefs update across rounds; test-time search whose safety is exactly checkable | Exact exploitability; published curves (NFSP, Deep CFR, DREAM, ESCHER, MMD, R-NaD); RLCard pretrained CFR and rule bots |
| 3 | Liar's Dice (1x4f, 1x5f, 1x6f, 2x3f) | Test-time search where naive search is exploitable | ReBeL released value-net checkpoints and exploitability (full-game CFR 0.001–0.002, ReBeL 0.020–0.041) |
| 4 | Phantom tic-tac-toe (or 3×3 Abrupt Dark Hex) | Opponent's moves hidden: a board to infer; too big for exact exploitability | Elo vs OpenSpiel ESCHER, Deep CFR, NFSP, MMD, R-NaD (v1.5), ISMCTS at published recipes; ESCHER and MMD paper results |
| 5 (optional) | Hanabi, 2 players | Cooperation: beliefs about your own cards come from the partner's actions | Released SAD, Other-Play and OBL models (hanabi_SAD `models/download.sh`); OBL repo belief training and `rl_search.py` |
| 6 (optional) | Dou Dizhu | 2-vs-1 teams, huge action space; goal switching: DouZero ships DouZero-ADP (score difference) and DouZero-WP (win rate) | DouZero weights; Botzone online leaderboard (DouZero #1 of 344); rlcard-showdown local leaderboard; Ataraxos scores vs DouZero and PerfectDou |
| 7 (later) | Stratego | Long games, 40 hidden pieces, strategic setups | Ataraxos move and setup networks; full search needs the belief run (Kelvin's money call) |

Notes
- Games 1–2: CFR+ reaches near-zero exploitability in seconds, so a new method can at best match it there; they test correctness, exact beliefs and compute cost. Nothing beats an exact equilibrium in expectation, so report exploitability on 1–3 and Elo on 4, 6, 7.
- Live leaderboards for game agents: only Botzone (Dou Dizhu). TextArena (Kuhn, Leduc, Liar's Dice, Battleship, Stratego, Hanabi playable online) and Kaggle Game Arena (chess, Go, poker) rate LLMs; whether a game agent may enter is unchecked.
- Goal steering on 1–3: change the payoff; the exact answer comes from OpenSpiel's CFR on the changed game (our variant, public solver).
- Learned world models: any game can be run with the rules hidden from the agent; the checks stay the same.
- Unchecked: whether RLCard's Leduc and ReBeL's Liar's Dice have exactly OpenSpiel's rules (a D0 step before mixing repos).
- Protocol: baselines we train with OpenSpiel's public code must first reproduce their published curves (PROTOCOL.md section 3); released weights skip that step.

Open for Kelvin: are cooperation (Hanabi) and goal switching (Dou Dizhu) in scope now? Proposed next step on his pick: gemini rewrites R26-A (the OpenSpiel half) around this set as a reproduction round; opus-mats freezes it.

## Follow-up: belief-heavy candidates (Kelvin: "they don't seem super reasoning-like ... doesn't look like we need to reason about beliefs much?")

Agreed: in Kuhn, Leduc and 1-die Liar's Dice the hidden state is one card or die; CFR solves them without forming a belief. Candidates where belief reasoning is the main difficulty and the exact posterior is still computable:
- Hanabi 2p: SAD, Other-Play and OBL weights (hanabi_SAD); facebookresearch/Hanabi_SPARTA (129 stars) tracks the exact posterior over one's hand given the partner's policy.
- Barrage Stratego: 8 pieces (spy, 2 scouts, miner, general, marshal, flag, bomb; repo init_helpers.py COUNTERS), at most 8!/2! = 20,160 identity assignments. Ataraxos engine and training code support Barrage (env flag `barrage`); only classic weights released. Paper recipe (SI S5): policy 4 H100 x 12 days, belief 1 H100 x 2 days.
- Battleship (OpenSpiel `battleship`): ESCHER paper results; no weights.
- Avalon: Detry322/DeepRole (38 stars), CFR plus deduction over role assignments (Serrino et al. 2019).
- Mastermind (optimal mean 5625/1296 = 4.340, Koyama & Lai 1993) and Wordle (3.4212 with SALET, 2,315 answers): information gathering with known optima, no opponent.
- Data point: classic Stratego search with the rules-only (uniform-policy) belief 2148 Elo vs learned belief 2218 vs no search 2095 (Ataraxos Extended Data Table 1).
- Fog-of-war chess: no public strong agent found (Obscuro code not public; only a 0-star unofficial port).
