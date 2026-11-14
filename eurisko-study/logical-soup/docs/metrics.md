# metrics.bend: per-epoch metrics (P2.2)

soup.bend prints one CSV row per epoch. The first 13 columns are the core
ledger (docs/soup.md); metrics.bend appends these:

| columns | meaning |
|---|---|
| `brier_<fam>` x 8 | mean Brier score of the market forecast (post-trade marginal quote) vs the outcome, per family (periodic, lfsr, ca30, ca110, prime, pi, and, or), in ppm (250000 = a constant 0.5). Live questions only; `nan` when a family has none this epoch. |
| `log_<fam>` x 8 | mean log loss, -ln p(outcome), micro-nats (693147 = constant 0.5); p floored at 2^-16. |
| `coh_and_ppm` | mean over fully live blocks of max(0, P(A and B) - min(P(A), P(B))), ppm of 1.0 |
| `coh_or_ppm` | mean of max(0, max(P(A), P(B)) - P(A or B)), ppm |
| `sh_*` x 6 | cumulative profit of each shadow trader (docs/shadow.md), Q16 ticks |
| `gini_ppm` | Gini of all cells' wealth (Kahan-summed F32 over sorted wealths), ppm |
| `lineages` | distinct ancestry hashes among active cells (spec'd "ancestry hash"; see note) |
| `genotypes` | distinct code hashes among active cells |
| `active_ppm` | fraction of cells with wealth >= 1 share, ppm |
| `cum_subsidy` | cumulative market-maker subsidy: traders' net P&L plus free income paid (income x cells; saturation ignored), Q16 ticks |
| `cum_ops` | cumulative instructions executed |

Total wealth is the core `wealth` column; per-epoch instructions are `ops`.

Note on `lineages`: a child's ancestry hash is hash(parent anc, mutation), so
every birth starts a new id and this column tracks active cells more than
clades. A founder/clade id would be a change to replication (soup-astra's
region, eurisko's call); `genotypes` is the extra column that shows diversity.

Nat sums stay below 2^48 (the JS lane's Nat limit): per-epoch sums are small,
cumulative ledgers are kept netted (min(gain, loss) removed from both).

## Held-out family, visible radius

- `HELDFAM` (arg 9): 0..5 a base family, 6 all AND compounds, 7 all OR
  compounds. Its questions start held: frozen (no advance), excluded from
  scoring and shadow trading, and every soup demand on them is dropped. A
  compound with a held parent is held too. `RELEASE` (arg 10) is the epoch at
  whose start every held question is released.
- `RADIUS` (arg 11): cell i sits over question i mod F; prices of questions at
  circular distance > RADIUS read as 0.5 and their histories as 0. Default:
  everything visible.

## Snapshots

`SNAPIN`/`SNAPOUT` (args 12, 13; "-" = none). A snapshot holds the state after
an epoch: header (magic "SOUP", version 1, logk, logq, ds, seed, next epoch;
U32 little-endian), then the population in tree order (per cell 2^ds code
bytes, wealth, used, ancestry), the questions (12 U32 each: id, family,
outcome, history bits/head/count/width, held, stream tag and 3 fields), the
price Vec, and the cumulative shadow ledger, subsidy and ops (Nats as two
U32s). Readers walk the byte list once with recursion depth bounded by tree
depth. Size is about K x (2^ds + 12) + 52 F bytes: 1.5 MB at K = 2^14,
F = 2^12. Resuming is bit-exact: `make check` compares a 4-chunk run through
scripts/run.sh with a straight run.

## Running and plotting

    scripts/run.sh --seed 1 --K 2^14 --F 2^12 --epochs 4000 --b 64 --free-income 0 \
      --chunk 100 --out runs/<date>-calib-k14-f12-s1/
    make plot RUN=<date>-calib-k14-f12-s1

run.sh writes config.json, metrics.csv, soup.dump and timing.csv (per chunk
wall seconds and instructions/s); rerunning the same command resumes, a larger
`--epochs` extends. Plots: one PNG per group (calibration, log_score,
coherence, shadow, population, inequality, wealth, flows, compute) plus
overview.png under runs/<id>/plots/.
