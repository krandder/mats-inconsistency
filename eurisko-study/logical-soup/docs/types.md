# Shared types (`src/types.bend`)

Kept small on purpose; the market and question modules adapt to these.

Fixed point: prices and probabilities are U32 in [0, 65536] (65536 = 1.0).
Cash is Q16: 65536 cash ticks = one share's payout. Demands are whole shares.

| type | fields | meaning |
|---|---|---|
| `Vec` | `VL{v}` / `VN{l, r}` | complete binary tree of 2^d U32 slots, `Data` (shared with `+`, no copy). Index bits are read low bit first, so an index wraps mod 2^d. `Vec.get/set/new/of_list/sum`. Used for code bytes, prices and histories. |
| `Trader` | `code: Vec, wealth: U32, used: U32, anc: U32` | code: S = 2^ds bytes, one byte (0..255) per slot. wealth: Q16 cash, which is also fuel (1 tick per fuel unit). used: fuel units spent in the last run. anc: ancestry hash, rehashed on every split (`hash2(parent anc, event hash)`), kept on in-place mutation. |
| `Pop` | `PL{t}` / `PN{l, r}` | population as a balanced tree of 2^logk traders; leaf index read low bit first like `Vec`. Every pass over it is a balanced fork. `Pop.get(d, p, i)`. |
| `Demand` | `q: U32, neg: Bool, amt: U32` | amt whole shares on question q, sign + magnitude (neg = sell). Maps 1:1 to market.bend `Order{id, wealth, negative, amount}`. |
| `Question` | `id, family, outcome, hist, price` | what the market sees of a question. hist: last 32 outcomes as bits, bit 0 = the most recent resolved outcome (age 0 in bit 0, as `Q.history.pack`). |
| `Env` | `dq: Nat, ds: Nat, prices: Vec, hist: Vec, wealth: U32, fuel: U32` | what one trader run sees: 2^dq questions, prices and packed histories by question index, own wealth, fixed fuel F. |

Helpers: `b2u`, `pick` (Bool select on U32), `pick_nat`, `prng` (xorshift32),
`hash2`, `sat_add` (saturating U32 add), `args.u32` / `arg` (command line).
