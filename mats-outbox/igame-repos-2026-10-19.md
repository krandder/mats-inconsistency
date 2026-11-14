# Repos to know for imperfect-information games (opus-mats, 2026-10-19)

Answers Kelvin's question (bead pool-u6yobl): "What are overall the most important (and higher star-ranked) repos on this field? That I should be aware of."
Stars checked 2026-10-19 ~17:35Z with `gh api repos/<owner>/<repo>`; rerun that to refresh.

## Ranked by use for small-game Elo against public baselines

| Repo | Stars | What it gives |
|---|---|---|
| google-deepmind/open_spiel | 5,541 | The standard library: 100+ games; CFR family, MCCFR, Deep CFR, NFSP, PSRO, MMD, ISMCTS, AlphaZero; exploitability and Elo code. No trained weights. |
| datamllab/rlcard | 3,581 | Card games (Leduc, Limit and No-limit Hold'em, Dou Dizhu, Mahjong, UNO) with rule bots and a pretrained Leduc CFR model; rlcard-showdown (413) adds a leaderboard. |
| kwai/DouZero | 4,685 | Dou Dizhu agent with released weights; the opponent PerfectDou and Ataraxos report against. |
| Farama-Foundation/PettingZoo | 3,537 | Common multi-agent API over classic games; Shimmy (937) wraps OpenSpiel games into it. |
| facebookresearch/rebel | 702 | ReBeL code and checkpoints for Liar's Dice, with published exploitability to reproduce. |
| AtaraxosAI/stratego | 111 | Stratego engine, RL, belief-training and search code; released move and setup networks, no belief weights. |
| sotetsuk/pgx | 654 | JAX, GPU-vectorised games incl. Kuhn, Leduc, Sparrow Mahjong, bridge bidding; baseline models for some board games. |
| gambitproject/gambit | 585 | Exact equilibrium solvers; ground truth for small games. |
| EricSteinberger/PokerRL | 546 | Deep CFR, Single Deep CFR, NFSP for poker; companions Deep-CFR (333) and DREAM (123). |
| lifrordi/DeepStack-Leduc | 951 | DeepStack on no-limit Leduc (Lua/Torch, 2018). |
| google-deepmind/hanabi-learning-environment | 669 | Hanabi environment; facebookresearch/hanabi_SAD (103) and off-belief-learning (49) ship trained models. |
| facebookresearch/diplomacy_cicero | 1,432 | Cicero code (Diplomacy); far beyond small games. |
| google-deepmind/game_arena | 117 | Harness for LLM game matches with Elo ratings. |
| liumy2010/LiteEFG | 38 | Fast extensive-form solvers (CFR+, DCFR, MMD and others), Liu, Farina and Ozdaglar 2024. |

## Learned-search references (perfect information)

| Repo | Stars |
|---|---|
| lightvector/KataGo | 5,203 |
| suragnair/alpha-zero-general | 4,519 |
| LeelaChessZero/lc0 | 3,248 |
| werner-duvaud/muzero-general | 2,877 |
| google-deepmind/mctx (JAX MCTS incl. Gumbel MuZero) | 2,678 |
| opendilab/LightZero | 1,652 |
| YeWR/EfficientZero | 944 |

Poker solver outside small games: bupticybee/TexasSolver 2,586 (heads-up no-limit postflop).
