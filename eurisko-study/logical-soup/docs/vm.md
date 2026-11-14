# Trader VM (`src/vm.bend`)

A U32 stack machine. Program: S = 2^ds code bytes (default 64). Fuel: F units
per run, fixed (a law). Opcode = byte & 31, so every byte is a valid
instruction; immediates are the next byte(s) (0..255). pc wraps at S, so a
program without HALT runs until its fuel is gone.

Fuel: every instruction costs 1 unit; the four reads cost 1 + READ_COST (= 4).
An instruction whose cost does not fit in the fuel left is not executed and the
run halts (hard stop). `used` = fuel units spent, `ops` = instructions executed.
The loop is bounded structurally by a Nat of F iterations.

Stack: at most 16 entries (a push onto a full stack is dropped); a pop from an
empty stack reads 0. Demands: at most EMAX = 8 distinct questions per run; a
later EMIT for the same question overwrites; q is masked to the question count.
On the stack a demand is two's complement; EMIT stores sign + magnitude.

| op | name | effect |
|---|---|---|
| 0 | NOP | |
| 1 | PUSH b | push byte b |
| 2 | PUSH16 lo hi | push lo + 256 hi |
| 3-14 | ADD SUB MUL AND OR XOR SHL SHR SAR LT SLT EQ | a b -> a op b (shifts by b & 31; LT unsigned, SLT signed; compares push 0/1) |
| 15-18 | NEG INC DEC NOT | a -> f(a) |
| 19-23 | DUP DROP SWAP OVER ROT | usual; ROT: a b c -> b c a |
| 24 | JMP t | pc = t |
| 25 | JZ t | pop; if 0 then pc = t |
| 26 | RPRICE | q -> price[q] (Q16) (read) |
| 27 | RHIST | q k -> bit k of hist[q] (read) |
| 28 | RHISTW | q -> hist[q], 32 bits, bit 0 newest (read) |
| 29 | RWEALTH | -> own wealth (read) |
| 30 | EMIT | q d -> ; demand d shares (signed) on q |
| 31 | HALT | |

Population: `run_pop` forks once per `Pop` node (balanced, as the cube contract
wants); `run_trader` returns the trader with `used` set, its demands and ops.
