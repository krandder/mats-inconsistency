# P19 pre-registration: does a fast reader's prequential gain rank candidate intermediates?

Written 2026-09-30 ~01:30Z. It is frozen by `PREREG.sha256` (sha256 of this file, the code and the pilot-rule output)
before any test-seed job (seeds 0 to 4) starts. After the freeze, changes are made only by dated addenda in
`DEVIATIONS.md`. Deadline: 18:30Z on 2026-09-30. Cap: $5 at the list-price ceiling; the watchdog deletes the VM at
$4.50.

## Question

THEORY-3 §3 ("Pricing as a way to find or choose latents", option 3) says the right price for a candidate
intermediate bit z is its prequential code-length gain for a fast reader. It gives a kill rule ("Cheapest decisive
experiment"):

- if (ii) the fast reader does not reach tau ≥ 0.70 at ≤ 5% of the wrapper's compute, then "proposing must be done
  by search";
- if (i) the first-order price against the current reader matches (ii), then "the fast reader is unnecessary".

P19 runs that experiment on the P8 maze ladder plus distractors, including XOR-type candidates.

**Scope.** P19 tests a pricing rule for candidate bits. It tests no economic mechanism. It says nothing about
proposing candidates.

**Related work.**

- **P8.** P8 is the parent. P19 reuses P8's maze task, its pool, its R1/R4/R7b code and its ruler (P8/item3 code,
  unchanged). P19 adds FR, FO and six new items.
- **P16.** P16 is an amortized, meta-trained in-context reader on a synthetic generator with exact CMI labels. P19's
  fast reader is not meta-trained. It is a warm start from the current reader, adapted online, on real maze
  intermediates with a deployed-gain truth.

## Task, items and truth

**Task.** P8's maze task:

- 11×11 grids, wall density 0.3.
- Y = 1[BFS distance to start ≤ 8] per cell.
- Inputs X = wall, start, inside, decoy, plus **key**. Key is new in P19: a public uniform random bit per cell, given
  as input in every context and to every rule.
- Contexts: S = ∅ ("empty") and S = {g2, g4} ("g2g4").
- N = 1024 mazes per bank. Price bank, truth banks and held set are P8's and item3's (`jobs.py`, `P8_N` = 1024).

**Ranked items.** There are 44 ranked items: 23 in empty and 21 in g2g4 (g2 and g4 are dropped from g2g4). They are
P8's pool plus six new items:

| item | definition | role |
|---|---|---|
| g1…g7, m2…m8, d2, d4, d6, reach | P8's oracle ladder and relatives | useful to useless |
| nz05, nz20 | Y with 5% / 20% of the cells flipped (hindsight: functions of Y) | THEORY-3's hindsight distractor |
| x2, x4, x6 | g_k XOR key | XOR type: zero first-order value alone, full value jointly with key |
| xb4 | g4 XOR key2 (key2 is never an input) | matched pure-noise twin of x4 |
| pg4 | g4 of the next maze (roll by one) | permuted: same marginals, unrelated |
| fg4 | 1 − g4 | relabelling: every sane rule prices it like g4 |

**Sanity items.** copyY, random, null, open and startZ are priced in the empty context but not ranked.

**Truth (primary).**

- The truth is `gain_vs_base` on the held set: the bits saved per held maze by the deployed pipeline (a z predictor
  cross-fitted from X (+Ŝ), then Y | X, Ŝ, ẑ) over the same Y net without z.
- It is item3's `job_truthrep`, with P19's items.
- It is the mean over 2 inits × 2 truth replicates (tseed = seed + 1000·rep).

**Truth (secondary).** P8's `truth` column: the gain against the free-latent control.

The truth deploys a *predicted* z, while FR, FO and R1 read the *true* z. R7b predicts z like the truth does. This
asymmetry is by design, because THEORY-3 prices the candidate bit itself. It is expected to cost FR on nz05 and nz20.
Hence the "forward" secondary subset below. It is not a primary.

## Rules (what ranks the 44 items)

| label | rule | task's name |
|---|---|---|
| FR | fast-reader prequential gain | (a) |
| FO | first-order price against the current reader | (b) |
| R4 | P8's MDL price LY − LYZ − LZ, mean over 8 replicates | (c) |
| R1 | P8's importance LY − LYZ, same 8 replicates | (d) importance alone |
| R7b | P8's 80/20 wrapper validation, mean over 12 replicates | (e) plain validation (the wrapper) |

**Current reader.**

- One per context: Y | X + S.
- Model: GConv with hidden width 16 and 2 layers (P8's conv reader).
- Fit with P8's CFG_FIT (lr 3e-3, batch 32, early stopping) on a separate reader bank, drawn with seed
  `seed_int("p19-reader", seed)`.

**FR (fast reader).** `p19.fr_run`.

- A copy of the current reader with one new, zero-initialised input channel. Member j reads z_j. Member 0 reads an
  all-zero channel and is the base.
- Online prequential coding over the 1024-maze price stream in chunks of 16. For each chunk:
  - code the 16 new mazes;
  - then take one Adam step on those 16 plus 16 mazes replayed from the past (rehearsal, as in Bornschein et al.
    2022).
- That gives T = 64 steps (THEORY-3 allows ≤ 100).
- All members share each stream order, so prices are paired.
- FR(z) = (base bits − member bits) summed over the stream, averaged over K stream orders.
- Learning rate LR* and the configuration (K*, T*) come from the pilot rule below. The job always runs 4 orders and
  stores prefix prices at T = 16, 32 and 64.

**FO (first-order).** `p19.fo_prices`.

- The current reader is frozen. Its logit is an offset in a ridge (λ = 1) logistic regression:
  logit = reader + b + w · (3×3 neighbourhood of centred z).
- Fit by damped Newton, 2-fold cross-fitted on the price bank.
- FO(z) = held-fold bits of (reader + b) − held-fold bits of (reader + b + w·z).
- This is the boosting / cascade-correlation / constraint-generation score, exact for a linear read-out.
- FO1 is the centre cell only (secondary).
- z is centred so that the zero padding carries no border signal. This fix was made at the smoke stage, before the
  pilot: uncentred, pure noise scored FO ≈ 70 bits.

**R1, R4, R7b.** P8 code, unchanged (`jobs3.job_pricerep`, `jobs3.job_r7brep`) on P19's item set. The replicate
counts are P8's.

## Compute accounting (P8's convention)

- Work = training forward MACs = stop steps × batch × cells × MACs per cell. FR's coding and training passes are
  counted the same way.
- The **primary ratio** is FR compute per ranked item divided by **one** R7b replicate per ranked item. This is the
  cheapest version of the wrapper, so passing it means ≤ 5% of any replicate count.
- FR compute includes both current readers' training and all members, including the base, for K* orders × (T*/64)
  of the stream.
- Secondary ratios:
  - against the 12-replicate R7b (P8's wrapper as run);
  - excluding the readers (the current reader treated as sunk).

## Pilot rule (seed 99 only; truth-free; applied mechanically by `code/pilot_rule.py`)

- **LR\*** is the argmin over {1e-3, 3e-3, 1e-2, 3e-2} of the total FR codelength: all members, both contexts, mean of
  2 orders.
- **(K\*, T\*)** is the first entry of (4, 64), (2, 64), (1, 64), (1, 32), (1, 16) whose pilot ratio is ≤ 0.045. That
  is a 10% margin under the 0.05 line. If no entry qualifies, the choice is (1, 64) and K1's compute arm is judged on
  the test seeds.
- The pilot also runs truth replicates 0 and 1, one pricerep and one r7brep on seed 99. They are used for timing and
  for a sanity check that is disclosed: the truth A/A and whether the XOR truths are non-degenerate. They are read
  only **after** this freeze. Seed 99 never enters a test statistic.

## Statistics

- **Statistic.** Kendall tau-b between each rule and the truth over the 44 pooled items, per seed.
- **Aggregate.** Mean and 95% t-interval over seeds 0 to 4.
- **Paired differences.** Per-seed tau_FR − tau_rule.
- **Power.** P8's per-seed tau SDs were about 0.01 to 0.05, so with 5 seeds the 95% half-width is about 0.02 to 0.06.

**Gates.** Checked per seed.

| gate | condition | if it fails |
|---|---|---|
| G1 | truth A/A: Spearman between the 2 truth replicates ≥ 0.80 | Inconclusive if it fails on ≥ 2 seeds |
| G2 | null: \|FR(null)\| and \|FO(null)\| < 1e-3 | Inconclusive if it fails on any seed (implementation bug) |
| G3 | positive control: FR(copyY) > every ranked FR in empty | Inconclusive if it fails on ≥ 2 seeds |

**Kill rules.** From THEORY-3, judged on the mean over the 5 test seeds.

- **K1:** mean tau_FR < 0.70, **or** the mean primary FR compute ratio > 0.05.
  - Reading (THEORY-3's): the chicken-and-egg has no cheap answer, and proposing must be done by search.
- **K2:** mean (tau_FR − tau_FO) < 0.05. That is, the first-order price matches or beats the fast reader.
  - Reading (THEORY-3's): the fast reader is unnecessary; boosting's price suffices.

**Status.** One of:

- Pass (neither kill fires)
- Killed (K1)
- Killed (K2)
- Killed (K1 and K2)
- Inconclusive (gate)
- Incomplete (fewer than 5 seeds by the deadline)

**Secondary.** Reported, not decisive.

- FR with all 4 orders.
- FR at T = 16, 32, 64, and tau against compute (replicate curves).
- FO1.
- Per-context taus.
- Tau against the control truth.
- Tau on the "forward" subset (all items except nz05 and nz20).
- XOR items (x2, x4, x6): each rule's values and mean absolute rank error. THEORY-3 predicts FO ≈ 0 on them; does FR
  price them?
- xb4 and pg4 near 0.
- The fg4-against-g4 gap: FO is exactly invariant by construction; FR need not be.
- FR minus each other rule, paired.

## Deviations from THEORY-3's sketch, and disclosures

1. **Size.** P19 uses 11×11 mazes at N = 1024 (P8's sizes), not 9×9 at N = 4096. This reuses P8/item3 code within
   $5. Item3 runs the N = 4096 ruler separately; P19 does not use its results.
2. **Fast reader.** It is a warm start with a zero-initialised channel and plain Adam. It is not meta-initialised and
   not an in-context probe (THEORY-3 allows either). A negative FR result therefore bounds this fast reader, not
   every fast reader.
3. **Sections.** The brief cites "THEORY-3 §6(b) and §4". THEORY-3 has only §§1 to 5. The experiment is §3 option 3
   plus "Cheapest decisive experiment", with the §5 context.
4. **Citation.** Bornschein, Li and Hutter 2022 exists: arXiv 2210.07931, *Sequential Learning of Neural Networks for
   Prequential MDL*. It recommends online learning with rehearsal over block-wise estimation. It does not speak of
   "fast learners". THEORY-3's phrase "prequential MDL with fast learners" is looser than the paper.
5. **What I saw before the freeze.**
   - P8's results, THEORY-3, item3's design (not its results) and P16's design.
   - The seed-99 pilot's FR and FO prices and compute. They are truth-free. FR at LR 3e-3 prices empty:x4 negative
     on the pilot seed.
   - Smoke-size outputs.
6. **R7b's role.** The THEORY-3 sketch calls R7b the ceiling (P8: tau 0.83 against the control truth, 0.91 against
   gain_vs_base). Here it is also the compute denominator.

## Pilot addendum (filled mechanically from `results/pilot/pilot_rule.json` before the freeze)

Applied by `code/pilot_rule.py` to `results/pilot/pilotlr_maze_s99.json` and `results/pilot/r7brep_maze_s99_r0.json`
at 01:30Z. No truth file had been fetched from the VM at that point.

- **LR\* = 3e-3.** Total FR codelength by learning rate: 2.154M bits at 1e-3, **2.099M** at 3e-3, 2.670M at 1e-2 and
  4.618M at 3e-2.
- **(K\*, T\*) = (2, 64).** Pilot ratio 0.0407.
  - For comparison: (4, 64) 0.0625; (1, 64) 0.0297; (1, 32) 0.0243; (1, 16) 0.0215.
  - The current readers alone are 0.0188 of one R7b replicate per item.
- **Frozen values in the code.**
  - `analyze19.py`: defaults P19_KUSE = 2, P19_TUSE = 64.
  - `p19.py`: default P19_LR = 0.003.
  - `queues/main.txt`: 115 jobs on seeds 0 to 4 (2 truthrep, 12 r7brep, 1 frfo with 4 orders, 8 pricerep per seed).
- **Pilot CPU.** pilotlr 301 CPU-s, of which 134 s is reader training. One r7brep replicate: 2,070 CPU-s (6 threads).
