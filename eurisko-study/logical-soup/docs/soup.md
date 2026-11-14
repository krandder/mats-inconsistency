# soup.bend: the epoch loop

`bun build/soup.js LOGK EPOCHS SEED FUEL LOGQ SPLIT INCOME W0 LIQ HELDFAM RELEASE RADIUS SNAPIN SNAPOUT`
(defaults 8 10 1 256 2 2097152 0 1048576 64 none never all - -; LOGQ is clamped
to at least 2). Held-out family, visible radius, snapshots and the metrics
columns: docs/metrics.md. Runs normally go through scripts/run.sh.

Per epoch:

1. **History first.** Each question's `Q.history.pack` word (age k in bit k) goes into
   the `hist` Vec *before* the outcome advances. Then the base questions advance and
   the compounds are updated from the advanced parents.
2. **Questions.** 2^LOGQ questions in blocks of four: id 4j is A, 4j+1 is B,
   4j+2 is A and B, 4j+3 is A or B. Base families cycle through the six streams
   (A uses family 2j mod 6, B uses 2j+1 mod 6); the periodic parameter grows with j.
3. **Prices.** Traders see the previous epoch's marginal quote per question (32768 at
   epoch 0). Every question-epoch is a fresh binary event.
4. **Run.** Only active traders run: wealth of at least one share (65536). A trader
   below that is inert (not deleted); it can be displaced by an offspring or revived
   by income.
5. **Budget.** Fuel is paid first (1 tick per instruction used). What remains,
   capped at 2^32-1-w so wins stay in U32, is split evenly over the trader's demands
   (at most 8). Each demand is an order that owns its share; `M.budget` sizes it at
   full collateral.
6. **Market.** A balanced tree reduction over the population sums the buy shares and
   sell shares per question. Each question then gets ONE LMSR update (`LIQ` = b in
   shares): `M.execution.price` for fills and `M.marginal.price` as the next quote.
   Sums use Nat leaves (not U32), preserving values above 2^32 shares.
   No per-chunk clears, so the 4096-order `M.clear` cap does not apply; the reduction
   uses the market's own primitives (`budget`, `difference`, `execution.price`,
   `marginal.price`, `profit`, `wealth.after`), which are the same ones `clear` and
   `settle` compose.
7. **Settle.** Every order is paid by `M.profit` at its question's execution price.
   The sum over traders equals the market's subsidy (`trader_pnl` column).
8. **Replicate** on the torus. A winning parent keeps ceil(w/2), its child
   receives floor(w/2), and the displaced cell's wealth is burned. Losing parents
   keep all their wealth. Replication statistics and CSV balances describe the
   resulting population, including the burn, rather than the pre-replication state.

CSV columns: epoch, alive (at least one share), wealth, wmax, ops, used, demands,
births, deaths (active to inert), volume (shares), trader_pnl (Q16 ticks, signed),
brier_ppm (marginal quote against outcome; 250000 is the constant 0.5 forecast),
burned (displaced wealth, Q16 cash ticks).

Idle cells clear their last-run fuel counter before settlement. Active runs have
a fuel ceiling of min(F, wealth); settlement charges at most the available wealth.
Optional income is credited after market P&L, capped at U32 headroom, and can
revive inert cells. Thus income cannot invalidate the collateral/gain-headroom
budget used to size orders. With INCOME=0, the exact epoch accounting identity is
`wealth_after = wealth_before + trader_pnl - used - burned`.

All Nat totals must remain below 2^48; this is a fail-stop runtime limit. The
planned K<=2^18 at default initial wealth 2^20 fits. Maximal U32 wealth at every
cell does not fit the population-wide total at that scale. Per-trader gross
exposure is at most 32767 shares, so its aggregate at K<=2^18 fits the market
price helper's intermediate bounds. Validated CLI liquidity stays in 1..65535.

## Sparse market aggregation (pool-t257u)

Each trader starts with two `NZ{depth}` empty subtrees, not two full question
vectors. A nonzero budgeted order creates only the path to its question
(O(log Q)); zero-sized orders create nothing. Balanced population merges skip
empty subtrees and combine only populated paths. Buy/sell share sums remain
Nat-valued. The root then materializes one execution-price vector and one
marginal-price vector, with 32768 for untouched questions. Thus there is still
O(Q) output work per epoch, but no O(K*Q) initialization/reduction over cells.
Price and settlement rules, the snapshot format, and the ledger are unchanged.

Measured on General, 2026-09-27, with other calibration jobs running: two epochs,
K=16384, Q=4096, seed=1, fuel=256, other arguments at defaults. Before: 24.91 s;
after: 2.61 s, an observed 9.5x speedup for this pair of runs, including startup.
The complete CSV outputs were byte-identical. This is one timing pair, not a
stable throughput estimate. The ten-epoch smoke run and the two measured epochs
both satisfy the exact zero-income ledger identity.

Tests assert that an empty 4096-question trader uses only two aggregate nodes,
a zero-budget order leaves it empty, and one nonzero order uses 26 nodes across
both sides. They also check high question indices, untouched prices, opposing
cancellation and empty merge identities, alongside the existing market oracle.
`make check` passed in an isolated copy under `build/pool-t257u-check`, with
smoke output redirected inside that copy. The live `runs/` tree was not written.

## Native CPU build on the GPU VM

`scripts/build-native.sh` checks the soup, emits Bend's C lane, and compiles with
clang >=14 using the same CPU flags as Bend 2.0.29 (`-std=c11 -O3 -lpthread -lm`).
It installs nothing and starts no calibration. Copy the same pinned Bend
checkout to the VM and set `BEND_HOME` to its root, for example:

```sh
BEND_HOME=/opt/bend-2.0.29 scripts/build-native.sh --cc clang-19
```

The default executable is `build/native/soup`. The script retains generated C,
checker/compiler logs, compiler versions and source/output hashes alongside it.
An explicitly requested missing or old compiler fails; it does not fall back.
Use `--emit-only` to check C emission without clang (validated on General).
Native compilation/execution remains a VM check because clang is absent here.

The current soup has no `!` GPU entry point. This recipe explicitly builds the
native **CPU** lane on the GPU VM; GPU hardware alone does not turn it into a
CUDA program. A future CUDA entry needs deliberate GPU dispatch, clang >=19,
CUDA 12/NVRTC and Bend's GPU build path. The script rejects such a source rather
than silently compiling it for CPU. Configure `SOUP_CMD` in `scripts/run.sh`
with the native executable and `--threads N` when starting an authorized run.
