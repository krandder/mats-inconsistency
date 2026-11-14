# Market conventions

`src/market.bend` exposes `clear(b, orders)` and `settle(batch, outcome)`.
An `Order` carries an id, U32 wealth, a Boolean negative sign (short when true),
and an unsigned whole-share demand. Wealth, price, and profit use Q16 cash:
65536 cash units buy one full unit of outcome payoff. Prices include both
endpoints, 0 and 65536. Signed quantities use `Signed{negative, magnitude: Nat}`;
zero is normalized to positive. Nat intermediates avoid U32 multiplication wrap.

Each question **at each epoch is a fresh binary event**, starting at inventory
zero. There is one clearing operation after all traders submit demands, then
one settlement. Repeated calls must not settle the same event again. There is
no persistent inventory or claim that the subsidy bound covers a whole stream
of different outcomes. The maximum subsidy is per question-event, per epoch.

## Cost and batch clearing

Binary LMSR has `C(q)=b ln(1+exp(q/b))`. The normalized charge is
`C(q)-C(0)`. The implementation evaluates it in a stable form using
`discount=b*(ln(2)-ln(1+exp(-abs(q)/b)))`:

- For positive net demand, normalized charge is `q-discount`.
- For negative net demand, normalized charge is `-discount`.
- The common execution price is the normalized charge divided by signed net
  demand. Zero net demand clears at 1/2. This is an average execution price;
  the returned `quote` is the final marginal LMSR price `sigmoid(q/b)`.

All traders get the same execution price, including opposing orders. The
price is rounded toward the maker relative to the sign of aggregate demand.
Gross long and short rounding errors cancel exactly because settlement uses
integer products, with no per-fill division. The total cash charged is at
least the implemented normalized cost. `cost` exposes that normalized cost
for diagnostics; Q16 price quantization can make the actual charge greater.

Only exp/log evaluation uses F32; prices and all accounting are integer.
The small-argument discount uses a sixth-order polynomial to avoid catastrophic
cancellation (`x/2-x²/8+x⁴/192-x⁶/2880`, x<=1/4). The omitted term is below
3.3e-10 at that cutoff. The marginal calculation caps the exp argument at 80.
The discount is clamped to `[0,min(abs(q)*32768,b*45427)]` in Q16 units.
This gives a backend-independent upper bound on positive maker subsidy:

`subsidy <= b*45427 Q16 units < b*ln(2) + b/65536 cash units`.

The stated rounding slack is **one Q16 cash unit per unit of liquidity b**.
The hard integer clamp, rather than an assumption about libm accuracy, enforces
this bound. LMSR cost approximation and price quantization are explicit;
this is not an exact-real arithmetic implementation.

## Budget and settlement

`budget(w,d)=min(d,65535,floor(min(w,2^32-1-w)/65536))`.
It reserves the entire binary payoff per share before the clearing price is
known. This is conservative: it is safe for every possible execution price.
The second term reserves gain headroom, so wealth cannot overflow U32 and
winnings are never burned. Zero or maximal wealth cannot submit exposure.

Settlement computes exactly `d*(outcome*65536-price)`, with the sign of `d`.
`Fill.wealth` is the old wealth plus this signed profit, always in U32 range
for budgeted orders. `Settlement.subsidy` is the signed sum of profits:
positive means maker cash paid out; negative means maker revenue. Thus the
change in total trader wealth equals subsidy exactly, including negative
subsidy. There is no other inflow and no saturating-accounting loss.

Callers must give each order its own wealth allocation. A trader trading
several questions must divide/reserve its wealth across them first; passing
the same full balance to several simultaneous markets double-spends collateral.
The API does not look up accounts or validate unique ids. Construct settlement
inputs through `clear`; manually forged Batch values bypass validation.

## Bounds and validation

`clear` rejects liquidity outside 1..65535 and batches longer than 4096;
it never silently truncates an oversized public batch. Every recursive market
pass has fixed fuel 4096. Amounts are capped at 65535 (the wealth/headroom
budget further limits any actual order to 32767 shares). All aggregate cash
and share intermediates fit below 2^45, inside Bend Nat's 48-bit limit.
Helpers take inputs satisfying these limits; the validated entry point is
`clear`, followed once by `settle`. Rejects produce `Invalid`, not a fake fill.

Run on the checked-out Bend 2.0.29 JS lane:

```sh
bun ../research/bend2-scratch/bend-2.0.29/bend2/main.ts tests/market_test.bend --check-only
bun ../research/bend2-scratch/bend-2.0.29/bend2/main.ts tests/market_test.bend -o build/market_test.js
bun build/market_test.js
```

Tests cover analytic price goldens; 512 seeded random batches settled under
both outcomes; 4096 budget trials; extreme liquidity/demand sweeps; balances
independently summed before/after for conservation; opposing flow; empty and
invalid inputs; and maximum-size/exposure and oversized batches. Tests exit
nonzero on failure. The checker must print exactly `All terms check.`.

## Question generator and harness integration

`src/questions.bend` owns the evolving stream state. Construct a base question
with `new(id,family,seed,parameter,width,held)`, then call `advance` once per
outcome. Construction leaves history empty; the first advance emits the first
bit. Width is clamped to 1..32. Base family codes above 5 normalize to 5.

| Family | Code | Stream and seed meaning |
|---|---|---|
| Periodic | 0 | One pulse per period, parameter clamped to 2..16; seed selects starting phase modulo period. |
| LFSR16 | 1 | Right-shifting Galois LFSR, mask 0xB400; seed's low 16 bits, zero mapped to one; emit low bit before stepping. Period 65535 for nonzero states. |
| Cellular automaton | 2 / 3 | Rule 30 / 110 on a 32-cell ring initialized from seed. Emit columns 0..31 of a row, then compute the next row. Neighborhood index is 4*left+2*center+right. |
| Primality | 4 | Primality of the current 16-bit integer, starting at seed modulo 65536 and wrapping after 65535. At most 254 divisor checks per emission. |
| Constant digits | 5 | Parity of the first 64 decimal digits of pi, including the initial 3. Seed chooses offset modulo 64; the table repeats. |
| AND / OR | 6 / 7 | Created by `new.compound(id,leftId,rightId,conjunction,width,held)`, with true selecting AND. |

This finite table and the 16-bit primality cycle define the actual experiment;
there is no claim to generate an unbounded mathematical constant or unbounded
primes. Every emitter has a fixed work bound; a CA row takes exactly 32 cell
updates. The history ring stores bits, next-write head, filled count and width.
`history.get(h,age)` returns age zero as newest and zero beyond the filled count.
`history.pack(h)` exports the same history with age k in bit k of a U32.

`held.set(q,True{})` freezes stream, outcome and history. Setting it to false
resumes at the same stream position; held-out families do not run invisibly.
The harness enables all questions of a held-out family when introducing it.

For an epoch, show traders `history.pack(history(q))` **before** advancing the
base streams. That word contains only past outcomes relative to the event
being predicted. Keep the full generator `Question` alongside the harness's
shared question snapshot; a packed history alone cannot restore stream state.
After both base questions advance, call `compound.update` once with the matching
parents. It records their AND/OR and advances the compound history. Wrong ids,
held or uninitialized parents, or a held compound leave the compound unchanged.
Compounds accept base parents only; plain `advance` leaves compounds unchanged.
The caller must advance both parents to the same epoch and update a compound
only once; this module does not contain a scheduler.

Market orders use separate wealth allocations for each question. VM instruction
costs can be expressed in Q16 cash ticks; one whole share needs at least 65536
such ticks of collateral. The current market entry point accepts at most 4096
orders per question. Larger population phases need a bounded balanced reduction
of all orders to one global net demand and one common clearing price; clearing
independent chunks would change the economics and multiply the subsidy bound.
No changes to shared `src/types.bend` were required for these modules.

Question validation:

```sh
bun ../research/bend2-scratch/bend-2.0.29/bend2/main.ts tests/questions_test.bend --check-only
bun ../research/bend2-scratch/bend-2.0.29/bend2/main.ts tests/questions_test.bend -o build/questions_test.js
bun build/questions_test.js
```

The suite checks same-seed determinism for all six families, all periods 2..16,
the complete LFSR cycle, independent prime/pi/CA vectors, pi-table wraparound,
ring contents against a list oracle through multiple wraps, width clamps,
packed-history export, held-out pause/resume, and 512 compound updates.
