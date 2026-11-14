# Forecasting datasets and environments for small-network training-regime tests

*2026-10-10. Read-only research for Kelvin (MATS). Companion to `shiphunt-smallnet-spec-2026-10-10.md` and `td-gvf-martingale-note-2026-10-10.md`.*

**How this was checked.** Links, sizes and licenses come from web searches and page fetches on 2026-10-10, done by three parallel research passes plus my own spot checks. I fetched arXiv 2512.22471, 2301.13289 and 2606.17832 myself. Local data was checked on disk on 2026-10-10: the Manifold archive index and `/home/agent/data/forecasting-benchmarks/`. **UNVERIFIED** marks anything not confirmed against a primary source. Fit scores (1-5) are my judgement against Kelvin's setting:
- many correlated binary or categorical variables;
- evidence arriving in stages;
- training only on realized outcomes, with one evidence sequence per instance and no resampling;
- single-variable log score or KL as the metric;
- a small net that trains in minutes to hours on one GPU.

## TL;DR

- **(i) Synthetic pick: keep ShipHunt 5x5 as the primary environment, and add bnlearn networks (CHILD, then ALARM) with staged random reveals as the replication environment.** bnlearn gives:
  - exact posteriors from pgmpy variable elimination;
  - a *known true DAG*, so you can check whether the k=1/2 arms' learned parents recover real structure;
  - discrete evidence states that recur across instances, which is the condition under which TD is predicted to help (Cheikhi & Russo, below).
- **(ii) Real-world pick: GHCN-Daily rain/no-rain on a 25-station regional grid.**
  - CC0 license, effectively unlimited instances (station-days), 25 correlated binaries per day.
  - Stages are natural: observations up to D-3, D-2, D-1.
  - Outcomes only; no exact posterior.
  - Runner-up: Grand Slam tennis point-by-point. It has hierarchical correlated targets (sets, match), and an iid-point Markov model gives a near-exact reference posterior.
- **Two literature results should shape the experiment before it runs:**
  1. Dembczyński et al. 2012: per-variable marginals already minimise single-variable log loss. So k=1/2 can win on single-cell KL only through sharing or regularisation at small N, which makes the data-efficiency sweep the right test.
  2. Cheikhi & Russo 2023: TD's gain over Monte Carlo is set by how often trajectories pool through shared states.
- **Rule conflict to fix.** The TD note suggests "draw many probe sequences per training board". That is resampling evidence for the same instance, which Kelvin's rules forbid. Use one probe sequence per board.

## Top-10 ranked

| # | Candidate | Group | Exact posterior for eval? | Real-world valid / outcomes-only | Setup | Fit |
|---|---|---|---|---|---|---|
| 1 | ShipHunt / Battleship 5x5 (own generator; refs Grand et al. 2026, Rothe et al. 2017) | A | Yes, by enumeration | Synthetic; one realized board per instance | Done / hours | 5 |
| 2 | bnlearn networks (CHILD 20, INSURANCE 27, ALARM 37 nodes) with staged random reveals | A | Yes (pgmpy VE / junction tree) | Synthetic; train on sampled full assignments | ~0.5 day | 5 |
| 3 | Factorial HMM (M = 5-8 binary chains, noisy emissions) | A | Yes (forward-backward on the product chain) | Synthetic; time stages are native | ~0.5 day | 4 |
| 4 | GHCN-Daily rain/no-rain, 25-station regional grid | B | No (climatology and persistence baselines) | Yes; CC0; outcomes only | 1-2 days | 4 |
| 5 | Grand Slam tennis point-by-point (Sackmann data via a mirror) | B | Near-exact under an iid-point Markov model | Yes; CC BY-NC-SA | ~1 day | 4 |
| 6 | Minesweeper 5x5 / 6x6 with random reveals | A | Yes (enumeration / CSP solvers) | Synthetic | Hours | 4 |
| 7 | Ising / Potts lattice 5x5 with noisy spin observations | A | Yes (junction tree, treewidth 5; or 2^25 enumeration) | Synthetic; coupling strength is a correlation knob | ~0.5-1 day | 4 |
| 8 | NFL play-by-play (nflverse), multi-target per game | B | No (nflfastR `wp` / `vegas_wp` as strong baselines) | Yes; CC-BY-4.0 | ~1 day | 3 |
| 9 | League seasons (openfootball CC0): all match outcomes and final-table events, revealed weekly | B | No (closing odds as a baseline; see license caveat) | Yes | ~1 day | 3 |
| 10 | Manifold multiple-choice and related markets (local archive) | B | No (market price is the baseline) | Yes; non-commercial license | 2-3 days | 3 |

Below the cut (fit 2-3, details below): 2020 county election-night time series (best structure, but about one timestamped election, so few instances), FluSight, M5 zero/non-zero, CSEP earthquake cells, Polymarket/Kalshi grouped events, brackets, POMDP benchmarks, Mastermind/Wordle, text-question datasets.

---

## Group A: synthetic environments with exact posteriors

### A1. ShipHunt / Battleship. Fit 5
- **Links.**
  - Grand, Pepe, Tenenbaum & Andreas, "Shoot First, Ask Questions Later?", ICLR 2026: https://arxiv.org/abs/2510.20886; code https://github.com/gabegrand/battleship (MIT). 8x8 boards, posterior by SMC particles (the authors call exact sums intractable at that size), noisy answers with eps = 0.1, 126 human games.
  - Earlier "Loose LIPS Sink Ships": https://arxiv.org/abs/2402.19471.
  - Rothe, Lake & Gureckis, NeurIPS 2017: https://arxiv.org/abs/1711.06351. 6x6 board, 3 ships of length 2-4, H/V placement. Data: https://github.com/anselmrothe/question_dataset. EIG code: https://github.com/anselmrothe/EIG (MIT). It appears to enumerate the full hypothesis space; UNVERIFIED.
- **Size and structure.** Unlimited instances. 25 binary cells with strong local correlation (ships are contiguous segments, plus count constraints). Stages: 0/2/4/6 noisy probes, per the spec.
- **Exact posterior.** Yes. Enumerate the legal placements of ships 3,2,2 on 5x5 and filter by the probe likelihood.
- **License and access.** Our own generator; reference repos are MIT.
- **Rule fit.** Real-world-valid only as a synthetic analogue. Satisfies the rules if each board gets one probe sequence (see the TL;DR conflict). The 8 symmetries are allowed by the spec.
- **Effort.** Already specified.
- **Caveat (Cheikhi & Russo).** With noisy probes at random cells, exact evidence states rarely recur across boards, so any TD gain must come through network generalisation. Consider a variant with few probe locations or noiseless probes, which raises trajectory pooling.

### A2. bnlearn Bayesian networks with staged reveals. Fit 5
- **Link.** https://www.bnlearn.com/bnrepository/. BIF/DSC/NET/RDA files; page footer CC BY-SA 3.0 (Marco Scutari).
- **Size.** Nodes / arcs / parameters:

  | Network | Nodes | Arcs | Parameters |
  |---|---|---|---|
  | ASIA | 8 | 8 | 18 |
  | SACHS | 11 | 17 | 178 |
  | CHILD | 20 | 25 | 230 |
  | INSURANCE | 27 | 52 | 1,008 |
  | ALARM | 37 | 46 | 509 |
  | HAILFINDER | 56 | 66 | 2,656 |
  | HEPAR2 | 70 | 123 | 1,453 |
  | WIN95PTS | 76 | 112 | 574 |
  | PATHFINDER | 109 | 195 | 72,079 |

  Arity is mostly 2-4 (UNVERIFIED per network).
- **Structure.** One instance is one ancestral sample of all nodes. Evidence stage t reveals a random subset (e.g. 0/20/40/60% of nodes), cumulatively. Targets are all unrevealed nodes. These are categorical, not only binary: the k=1/2 conditional tables become arity^k-sized.
- **Exact posterior.** Yes: variable elimination or a junction tree in pgmpy, fast up to about WIN95PTS (small treewidth). Load with `pgmpy.example_models.load_model("bnlearn/alarm")` in pgmpy ≥ 1.1; the older `get_example_model` API is UNVERIFIED in current versions. pgmpy is not installed on General (checked 2026-10-10); `pip install pgmpy`.
- **Unique value.**
  - The true DAG is known, so learned parents in the k=1/2 arms can be scored against true Markov blankets.
  - Evidence states are discrete and recur across instances, giving high trajectory pooling for TD.
  - Mixed arities and heterogeneous CPTs make it less "image-like" than ShipHunt, which tests whether conclusions transfer.
- **Optional extra.** UAI 2022 competition networks (https://uaicompetition.github.io/uci-2022/results/benchmarks/, .uai format). Whether exact solutions ship with them is UNVERIFIED, and many are too wide for exact inference.
- **Rule fit.** Yes: training uses sampled full assignments only; VE is for eval only.
- **Effort.** About half a day.

### A3. Factorial / multi-chain HMM. Fit 4
- **Links.**
  - dynamax: https://github.com/probml/dynamax (MIT; exact `filter`/`smoother`).
  - hmmlearn (BSD-3).
  - factorial_hmm: https://github.com/regevs/factorial_hmm (Bioinformatics 2019; license UNVERIFIED).
  - Ghahramani & Jordan 1997: https://mlg.eng.cam.ac.uk/pub/pdf/GhaJor97a.pdf.
- **Structure.** M binary chains, coupled through shared emissions or coupled transitions. Forecast every chain's state (or its state at a fixed future time T) after each observation. Time stages are native and you can set the horizon. Exact cost is O(T·K^(2M)) naively, fine for M ≤ 8.
- **Precedents.**
  - Agarwal, Dalal & Misra, "The Bayesian Geometry of Transformer Attention" ("Bayesian wind tunnels"): https://arxiv.org/abs/2512.22471. I checked the abstract myself. Small transformers match closed-form HMM posteriors to 1e-3 to 1e-4 bits where MLPs fail. No code link.
  - Shai et al., NeurIPS 2024: https://arxiv.org/abs/2405.15943 (belief-state geometry for HMMs).
- **Rule fit.** Yes: one observation sequence and one realized state path per instance.
- **Why it is useful.** It is the cleanest test of TD across many stages (T can be 10-50, unlike ShipHunt's 4).

### A4. Minesweeper. Fit 4
- **Exact solvers.**
  - https://github.com/aijobclub/minesweeper-solver (MIT; handles the global mine count)
  - https://github.com/nabenabe0928/mine-sweeper-solver
  - https://github.com/JohnnyDeuss/minesweeper-solver
  - https://github.com/DavidNHill/Minesweeper
  - Licenses other than the first are UNVERIFIED. On 5x5 or 6x6, brute-force enumeration is enough anyway.
- **Structure.** 25-36 binary mine cells. Evidence is revealed cells with neighbour counts: deterministic, hard-constraint correlations, unlike ShipHunt's noisy probes. Reveal at random safe cells, independent of the model (as in the spec).
- **ML use.**
  - Sajjad 2022: https://arxiv.org/abs/2212.10446 (about 3k-parameter MLP players).
  - Stanford CS229 2015 project, which regressed on solver probabilities: https://cs229.stanford.edu/proj2015/372_report.pdf.
  - **No published curve of small-net KL to exact posteriors vs data budget was found**, which is a possible contribution.
- **Effort.** Hours.

### A5. Ising / Potts lattice with noisy observations. Fit 4
- **Setup.** Sample spins from a 5x5 Ising model at coupling J, then observe noisy spins at random sites in stages. That is ShipHunt with a continuous **correlation-strength knob** (J), which lets you plot "k=1/2 advantage vs dependence strength" (cf. Luaces et al. in C3).
- **Exact posterior.** Junction tree or transfer matrix (treewidth 5); 2^25 enumeration is also feasible. Draw exact samples by forward-filtering backward-sampling over rows.
- **Reference code.** VAN, Wu, Wang & Zhang PRL 2019 (https://arxiv.org/abs/1809.10606, https://github.com/wdphy16/stat-mech-van; license UNVERIFIED).
- **Effort.** Half a day to a day.

### A6. Others (fit 2-3)
- **POMDP belief-update benchmarks.**
  - Tiger, hallway (57 states), hallway2 (89), 4x4: https://pomdp.org/examples/.
  - JuliaPOMDP RockSample.jl and TagPOMDPProblem.jl (MIT).
  - Beliefs are exact, but the latent is usually one state variable, so there are not many correlated targets. RockSample's rock-quality bits are the best multi-variable case. Fit 3.
- **Mastermind / Wordle.**
  - 1,296 codes or 2,315 answers; per-position marginals are exact by enumeration.
  - MastermindEval (LLM-oriented): https://github.com/flairNLP/mastermind.
  - Fit 3. Correlations are combinatorial rather than spatial.
- **BLInD** (https://github.com/HLR/BLInD, no LICENSE file visible) and **BayesBench** (https://arxiv.org/abs/2606.30850; license UNVERIFIED). These are LLM text benchmarks. Their generators are reusable, but bnlearn already covers the idea. Fit 2.

---

## Group B: real-world-ish data (outcomes realized, staged evidence)

### B1. GHCN-Daily rain/no-rain grid. Fit 4 (real-world pick)
- **Link.** https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily. CC0; bulk download via AWS Open Data.
- **Size.** More than 100k stations, 1832 to present. A regional grid of 25 well-reported stations × about 40 years gives about 14k instances (station-days as rows); several regions give more.
- **Structure.** 25 strongly spatially correlated binaries (precipitation > 0.1 mm). Stages are lagged evidence: (D-3), (D-2), (D-1), and optionally same-day early reports from hourly ISD stations. Missing data is common: pick stations with more than 95% coverage.
- **Posterior.** None exact. Baselines are climatology, persistence and logistic regression; NWP is too heavy.
- **Rule fit.** Real-world valid and outcomes only. Avoid overlapping train and test windows (split by year).
- **Effort.** 1-2 days, mostly station selection and QC flags.
- **Lighter alternative.** weatherAUS (Kaggle, 49 Australian stations, 145,460 rows, 2007-2017). The underlying BoM data is CC BY 3.0; the Kaggle label is UNVERIFIED. Drop `RISK_MM` (target leak).

### B2. Grand Slam tennis point-by-point. Fit 4
- **Data.** Jeff Sackmann's `tennis_slam_pointbypoint` (CC BY-NC-SA 4.0), 2011-2024.
  - The primary GitHub repos returned 404 when checked 2026-10-10. `tennis_MatchChartingProject` still resolves.
  - Mirror: https://github.com/Aneeshers/tennis-sackmann-archive (also on Hugging Face).
- **Structure.** Thousands of matches. Correlated targets per match: each set's winner, the match winner, and tie-break occurrence. Stages are points or games.
- **Near-exact reference.** Under an iid-point Markov model with a per-player serve-win probability, P(game/set/match) given the score is exact in closed form. Fit p per match from pre-match stats only (no leak), and you get a "reference posterior" that is wrong only through the model, which makes this a bridge between groups A and B.
- **Rule fit.** Real-world valid. Non-commercial license is fine for research.
- **Effort.** About a day.

### B3. NFL play-by-play (nflverse). Fit 3
- **Link.** https://github.com/nflverse/nflverse-pbp (CC-BY-4.0), 1999 to present, about 7k games.
- **Targets.** Few per game unless you build them: winner, each remaining quarter's scoring, cover, total. They are correlated through game state.
- **Baseline.** `wp` / `vegas_wp`. Known bug: inverted WP on administrative rows (nflfastR issue #606).
- **Staging.** Plays or drives.
- **Rule fit.** Real-world valid.
- **Effort.** About a day.

### B4. League seasons, all remaining results revealed weekly. Fit 3
- **Data.** https://github.com/openfootball/football.json (CC0), major European leagues 2010-11 onward. 380 matches per season, revealed by matchweek.
- **Targets.** Remaining match results plus derived, correlated targets: champion, top-4 and relegation per team.
- **Baseline.** football-data.co.uk closing odds are an excellent per-match baseline, **but its terms forbid "data training products … AI"**. Use it at most as an eval reference, after reading the terms. openfootball covers the outcomes.
- **Limitation.** Few instances (league-seasons number in the hundreds).

### B5. Manifold multiple-choice markets (local archive). Fit 3
- **Local data, checked 2026-10-10.** `/home/agent/data/manifold-archive`:
  - 40,734 MULTIPLE_CHOICE markets, 32,332 resolved;
  - 138,519 BINARY markets, 108,592 resolved;
  - 9.1M bets through 2024-07-04.
  - Count by `outcome_type` in `index.sqlite`.
- **Structure.** Each multiple-choice market is a correlated set of answers (mutually exclusive or independent, depending on the market). The bet and price history gives staged evidence.
- **What the small net learns.** A recalibration and aggregation map from price and trade features to outcomes. That is real-world valid but does not use question semantics.
- **License.** Free dumps are non-commercial; AI training for commercial use needs a paid license.
- **Similar external sources.**
  - Polymarket `negRisk` events: Gamma `/events`, price history from CLOB `/prices-history`. Fine-grain history expires; only 12-hour grain is kept.
  - Kalshi events: public `/historical/markets/{t}/candlesticks` back to 2021-06.
  - Training terms for both are UNVERIFIED.

### B6. Below the cut (fit 2-3)
- **2020 county election-night time series. Fit 3: best structure, fewest instances.**
  - Sources:
    - https://github.com/d0ntw84f8/election2020-timeseries-data (MIT; county, state-race and some precinct series, scraped from NYT/Edison);
    - https://github.com/MEDSL/election_night2020 (15-minute county bins; no license file);
    - MEDSL final county returns 2000-2024: https://doi.org/10.7910/DVN/VOQCHQ (CC0).
  - Instances are states, so only about 50 for one election, too few for a 100-10k sweep. Timestamped 2022/2024 data is UNVERIFIED.
- **FluSight.** https://github.com/cdcepi/FluSight-forecast-hub (MIT). About 53 jurisdictions × weekly 5-class rate-change targets; the FluSight ensemble is a strong baseline. Few seasons. Fit 3.
- **M5.** 30,490 series × 1,941 days; zero vs non-zero sales is correlated within store and department. Kaggle rules: competition, academic and non-commercial use only. Fit 3.
- **CSEP / ComCat earthquakes.** Grid-cell binaries per day. CSEP California has more than 50k daily gridded model forecasts (ETAS, STEP) as baselines: https://www.nature.com/articles/s41597-025-05766-3. Rare events. Fit 3.
- **Brackets** (World Cup via openfootball CC0; Kaggle March Mania is competition-use only). 63 correlated binaries revealed round by round. The bracket posterior given per-game probabilities is exact by dynamic programming (model-relative). One instance per year. Fit 2.
- **Traffic.**
  - METR-LA (207 sensors) and PEMS-BAY (325): DCRNN repo, data license unclear.
  - LargeST (8,600 sensors, CC BY-NC 4.0).
  - Binarising congestion is arbitrary. Fit 2.
- **Conflict.**
  - VIEWS (country-month and PRIO-GRID-month fatalities; mostly zeros, so occurrence binaries are natural).
  - GDELT.
  - **ACLED is excluded: its 2025 EULA bans ML training and testing, including academic use.** Fit 2-3.
- **Solar flares.** SWAN-SF (CC0, 4,098 instances), one target per instance. Fit 2.
- **Text forecasting-question sets.** For small nets these are mostly independent questions; fit 1-2 unless reduced to price features. Several are mirrored locally in `/home/agent/data/forecasting-benchmarks/`:
  - ForecastBench (CC BY-SA)
  - Autocast (MIT; 6,707 questions with crowd time series)
  - Halawi et al. 2024 (5,516 questions)
  - KalshiBench (arXiv 2512.16030; HF dataset, no license tag)
  - OpenForesight (MIT)
  - [removed]

---

## Group C: literature and reusable benchmarks

**C1. TD for prediction**
- **Cheikhi & Russo, "On the Statistical Benefits of TD Learning".** https://arxiv.org/abs/2301.13289. I checked the abstract: arXiv says "submitted to ICML 2023"; acceptance is UNVERIFIED on that page.
  - An inverse trajectory-pooling coefficient fully determines TD's MSE reduction over direct regression, which can be "enormous or nonexistent".
  - **Use it to predict the TD arm's result before running**: TD helps where boards share evidence states.
- **Maystre & Russo, "Temporally-Consistent Survival Analysis", NeurIPS 2022.** https://openreview.net/forum?id=r-CsquKaHvk; code https://github.com/spotify-research/tdsurv.
  - The closest prior art: next-stage predictions as pseudo-targets, anchored on outcomes, give better sample efficiency than outcome regression. Its benchmarks are reusable.
- **Cheng & Weare, NeurIPS 2024.** https://arxiv.org/abs/2405.17638. LSTD is exponentially more data-efficient than MC for rare events, which matters for low-probability cells.
- **Pan et al., "Generalized TD Learning Models" (JAIR).** https://arxiv.org/abs/2404.15518. TD targets for classification built in logit space; a recipe for TD on probabilities.
- **Mou & Qian 2024.** https://arxiv.org/abs/2411.09731. Adaptive mixing of TD and MC (choosing λ).
- **Sutton & Tanner 2004, TD networks.** NeurIPS 17. The multi-prediction version of stage-to-stage targets.
- **Mann et al. 2019, "Learning from Delayed Outcomes via Proxies".** ICML, http://proceedings.mlr.press/v97/mann19a.html.

**C2. Martingale consistency**
- **Fong, Holmes & Walker 2023.** JRSSB, https://doi.org/10.1093/jrsssb/qkad005.
- **Falck, Wang & Holmes 2024.** ICML, https://arxiv.org/abs/2406.00793. Martingale tests reusable as eval diagnostics.
- **Kim, Lee, Fong, Lee & Lee 2026, "From Drift to Coherence".** https://arxiv.org/abs/2606.17832. I checked the abstract: it fine-tunes with a self-consistency loss that reduces martingale violations "without sacrificing accuracy". NLL is not reported in the abstract.
- **Lee et al. 2023, Martingale Posterior Neural Processes.** ICLR, https://arxiv.org/abs/2304.09431.
- **Paleka et al. 2025, consistency checks for LM forecasters.** ICLR, https://arxiv.org/abs/2412.18544.

**C3. Does modelling the joint improve marginals?**
- **Dembczyński, Waegeman, Cheng & Hüllermeier 2012.** Machine Learning 88, https://doi.org/10.1007/s10994-012-5285-8.
  - Per-label log loss is minimised by the marginals, so joint modelling helps marginals only through inductive bias or parameter sharing.
  - **Implication:** k=1/2 gains should shrink as N grows; the 100 → 10k sweep is the right design.
- **Read & Hollmén, "Labels as hidden nodes".** https://arxiv.org/abs/1503.09022. Apparent label dependence is often a weak-base-model artefact.
- **Luaces et al. 2012.** https://doi.org/10.1007/s13748-012-0030-x. A synthetic multi-label generator with controllable dependence; pairs with the A5 Ising knob.
- **Probabilistic classifier chains** (ICML 2010; code https://github.com/multi-label-classification/PCC), MADE (https://arxiv.org/abs/1502.03509) and NADE (https://arxiv.org/abs/1605.02226) are architecture references for the k-parent heads.
- **MULAN/MEKA datasets** (yeast, scene, emotions) are static (no stages), so they fit poorly.

**C4. Trained on samples, evaluated against exact posteriors** (the same protocol as the ShipHunt spec)
- **sbibm**, Lueckmann et al. AISTATS 2021: https://arxiv.org/abs/2101.04653, https://github.com/sbi-benchmark/sbibm (MIT).
  - Ships reference posteriors and plots metric vs simulation budget.
  - **Copy its reporting format for the data-efficiency curves.**
- **PFNs**, Müller et al. ICLR 2022: https://arxiv.org/abs/2112.10510, https://github.com/automl/PFNs (Apache-2.0). Cross-entropy on prior samples converges to the posterior predictive. This is the theoretical justification of the k=0 arm.
- **Simformer**, ICML 2024: https://arxiv.org/abs/2404.09636. A masked transformer for arbitrary conditionals; reference for the masked-variable arm.
- **Shih et al. 2022, "Any-order AR models the right way"**: https://arxiv.org/abs/2205.13554. Training all orderings is redundant and hurts, which bears on random cell order per episode in k=1/2.
- **Orderless NADE**: https://arxiv.org/abs/1310.1757.

**C5. Scoring.**
- Gneiting & Raftery 2007: https://doi.org/10.1198/016214506000001437.
- Marcotte et al. ICML 2023: https://arxiv.org/abs/2304.09836. Energy score misses dependence errors and NLL catches them, which supports joint log-likelihood as the secondary metric.

**C6. Small transformers vs exact belief states.**
- Shai et al. 2024: https://arxiv.org/abs/2405.15943.
- Agarwal et al. ("Bayesian wind tunnels"): https://arxiv.org/abs/2512.22471.
- Othello-GPT: https://arxiv.org/abs/2210.13382.
- None of them reports a data-efficiency comparison of training regimes, which is the gap Kelvin's experiment fills.

## Recommended sequencing

1. Run the ShipHunt spec as is, with one probe sequence per board.
2. Replicate the winning arm and the k=0 arm on bnlearn CHILD → ALARM. This adds a learned-parent vs true-DAG check and a high-pooling TD regime.
3. Add an Ising-J sweep only if the k=1/2 effect is ambiguous; it shows how the effect varies with dependence strength.
4. Take the best regime to GHCN-Daily (25 stations, lagged-day stages), with tennis point-by-point as a second real-world check that has a near-exact reference.
