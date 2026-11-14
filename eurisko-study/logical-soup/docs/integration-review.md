# Market integration review (pool-4yyyg)

Reviewed `src/soup.bend` from d679479 against the market and question modules.
The balanced reduction is economically equivalent to one `clear` per question:
aggregation and settlement reuse the same post-fuel allocation and budget;
all orders for a question use its single execution price; the separate marginal
quote is exposed next epoch. Multiple orders, including repeated question ids,
share one trader's collateral and gain headroom. History is packed before
advancing outcomes, and compounds update after their base parents.

Fixed findings:

- Child wealth included the displaced neighbour's balance. It now receives only
  floor(parent wealth/2), while the parent retains the remainder. The victim's
  balance is recorded as burned. Post-replication balances, active count and
  maximum wealth now feed the CSV.
- Aggregate share vectors used wrapping U32 sums. They now use bounded Nat
  arithmetic and retain low-bit-first question indexing. A regression crosses
  2^32 shares and verifies the correct clearing price.
- Optional income was credited after budgeting but before P&L, potentially
  consuming reserved gain headroom and wrapping wealth. Income now follows P&L
  with saturation, and can revive inert cells as the existing docs specified.
- An inert cell carried its previous run's `used` count, charging old fuel every
  epoch. Idle execution resets it. Active runs also cannot spend more fuel than
  their wealth; settlement reports the actual charge.
- Replication counted displacement of any nonzero balance as an active death.
  It now uses the same one-share activity threshold as other population stats.

`tests/soup_test.bend` compares execution/marginal prices and signed aggregate
P&L to `M.clear`/`M.settle` on two questions, two traders, opposing orders and
both outcome assignments. It independently sums balances for cash conservation
and checks the subsidy bound. Other regressions cover odd/even splits, funded
and empty victims, repeated idle fuel, income overflow/revival, and wide sums.

Validation: `make check` type-checks all source/tests, runs all four test suites,
and runs 256 cells for ten epochs. The smoke CSV also satisfies, row by row,
`next wealth = previous wealth + trader_pnl - used - burned` (income zero).
UBS was attempted on soup.bend but returned exit 3 because Bend is not a
supported language; it provides no evidence here. The Bend checker, regression
suite and a second manual accounting pass provide the validation.

Scope limits: Nat totals must fit below 2^48 (documented in docs/soup.md).
CLI configuration validation and scientific metrics are separate harness work.
The ancestry hash still changes on a birth; it is not a founder/clade identifier.
