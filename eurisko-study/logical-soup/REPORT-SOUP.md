# The logical-induction soup: first results

eurisko (scientific lead), 2026-09-27. Builders: soup-astra (gpt-6-astra) and
three Opus 5.5 subagents. Spec: `GOAL.md`. Data: `runs/phase3/summary.md`,
`runs/fuel/summary.md` (fuel sweep), `CALIBRATION.md`,
`docs/gpu-phase3.md`, `docs/gpu-utilisation.md`, `docs/gpu-scaling.md`.
Dashboard: [removed] Every number below is
the mean over 3 seeds of the last 50 epochs of a 5000-epoch run unless stated.

## 1. TL;DR

1. **It works as a mechanism.** One Bend2 program holds 4k to 64k fuel-bounded
   trader programs, an LMSR market over up to 4096 synthetic questions, and
   the rules. Wealth is the only currency, the maker's subsidy the only inflow.
   The ledger balances exactly every epoch, runs are byte-identical across the
   JS, native and CUDA lanes, and resume is exact.
2. **The decisive variable is traders per question, K/Q.** Brier scores
   collapse on K/Q, not on K or Q alone. At 64 traders per question the
   periodic family reaches Brier 0.084 and primality 0.091 (a base-rate trader
   scores about 0.25 and 0.10), and the shadow momentum and pattern traders
   lose money against the market. At 4 per question nothing is priced beyond
   base rates and the same shadow traders profit heavily.
3. **The inductor property is partial.** Momentum and pattern-k reference
   traders are beaten at high K/Q. The coherence arbitrageur is never beaten:
   A∧B is priced but A∨B is not, and AND-coherence violations of 0.01-0.09
   persist in every cell. No arbitrage lineage evolved in 5000 epochs.
4. **Never priced, anywhere:** LFSR, cellular automaton rule 110, pi digits
   (all 0.25). Rule 30 is priced worse than chance (0.26-0.29) at low K/Q,
   because rich lineages emit the same demand on every question.
5. **Ablations changed little.** A 50-epoch free-income bootstrap and
   visible-price radius 1 land within noise of the baseline; the held-out
   family (pi) was a poor choice because pi is never priced even when trained.
6. **Bend2 delivered exactness and portability, not yet a GPU multiplier.**
   The CUDA cube runs the soup at 3.4M virtual instructions per second on a
   spot L4 (JS lane 1.1-1.7M, native CPU 2.0-3.2M). An A100 is no faster.
   CUDA is 10-12% of wall time; Bend2's host-side runtime phases dominate,
   and the kernel is register-limited (255 registers per thread). Total GPU
   spend for the whole project: $9.38 plus about $1.20 of smoke tests, of a
   $200 cap.

## 2. What was built

- `src/vm.bend`: U32 stack machine, fixed program size, fixed fuel per run,
  market primitives (read price, read history, read wealth, emit demand).
- `src/market.bend`: LMSR per question in Q16 fixed point, budgeter (worst-case
  loss ≤ wealth), settlement `demand × (outcome − price)`, subsidy as the only
  inflow.
- `src/questions.bend`: families periodic, LFSR, CA rule 30, CA rule 110,
  primality of n, pi-digit parity, plus A∧B and A∨B compounds in blocks of four.
- `src/soup.bend`: the epoch loop as balanced forks over traders and
  questions; grid-local replication (offspring gets half the parent's wealth,
  the displaced neighbour's wealth is burned); death when below one share.
- `src/shadow.bend`: reference traders in a shadow ledger (constant 0.5,
  momentum, pattern-2/4/8, coherence arbitrageur) that never replicate.
- `src/metrics.bend`, `src/snapshot.bend`, `scripts/run.sh`, `scripts/matrix.sh`,
  `scripts/summarize.py`, `infra/` (GPU VMs with 6-hour auto-delete, watchdog,
  preemption resume, cost audit), `infra/grafana/`.
- Fixed constants: initial wealth 2^20 (16 shares), split at 2^21, fuel 256
  per run (all Phase 3 cells; see the fuel sweep below), b = 64, no free
  income, all prices visible.

## 3. Results

### 3.1 The K/Q collapse

| cell | K | Q | K/Q | periodic | prime | A∧B | A∨B | LFSR | CA30 | Δ shadow pattern-2 | Δ shadow arb | takeoff epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| k14-f8  | 16k | 256  | 64 | 0.084±0.004 | 0.091±0.003 | 0.120±0.003 | 0.221 | 0.263 | 0.262 | −9.7 | +3.0 | 2430 (2/3 seeds) |
| k12-f8  | 4k  | 256  | 16 | 0.149±0.021 | 0.120±0.009 | 0.147±0.005 | 0.236 | 0.260 | 0.265 | −0.9 | +8.7 | never ≥50 births |
| k14-f10 | 16k | 1024 | 16 | 0.133±0.015 | 0.108±0.006 | 0.132±0.002 | 0.232 | 0.256 | 0.271 | −15.8 | +24 | 231 |
| k16-f12 | 64k | 4096 | 16 | 0.126±0.007 | 0.104±0.001 | 0.127±0.001 | 0.228 | 0.257 | 0.275 | −116 | +69 | 6 |
| k12-f10 | 4k  | 1024 | 4  | 0.221±0.009 | 0.212±0.017 | 0.210±0.005 | 0.249 | 0.253 | 0.257 | +83 | +30 | 1603 |
| k14-f12 | 16k | 4096 | 4  | 0.204±0.018 | 0.177±0.030 | 0.188±0.021 | 0.253 | 0.256 | 0.290 | +243 | +179 | 205 |
| k12-f12 | 4k  | 4096 | 1  | 0.241±0.001 | 0.238±0.003 | 0.239±0.004 | 0.250 | 0.251 | 0.257 | +451 | +53 | 1348 |

Δ shadow = slope of the reference trader's cumulative profit over the last
1000 epochs, in shares per epoch; negative means the market beats it.

Reading: the three cells with K/Q = 16 (4k/256, 16k/1024, 64k/4096) give the
same Brier on every family within seed noise, although K spans 16x. K/Q is
the variable. More traders per question means more lottery tickets per
question and more replication pressure per question. The takeoff epoch (first
epoch with 50 or more births) falls steeply with K: 1300-1600 at 4k, 200-450
at 16k, 6 at 64k; k12-f8 never reaches 50 births per epoch because 4k traders
cannot, yet it still prices the periodic family.

Primality at 0.09 is below what a base-rate trader can do (about 0.10 at
these n), so the population found structure beyond the base rate, most
likely the even/odd rhythm of the prime sequence; it is not primality.

### 3.2 What is never priced

LFSR (16-bit), CA rule 110 and pi digits stay at 0.25 in every cell. Rule 30
is priced below chance (0.26-0.29) wherever K/Q is low: lineages that win on
biased families emit the same demand on pseudo-random ones and drag them the
wrong way. This is the current frontier of the bitter-lesson curve: none of
K up to 64k, Q down to 256, or 5000 epochs moved these families. A trader
would need to read history and compute a 16-bit recurrence or a CA row in
256 instructions; the DSL and fuel make this possible, but selection never
found it.

### 3.3 Coherence and the arbitrageur

A∧B is priced in every cell where A and B are (down to 0.12); A∨B is not
(0.22-0.25 everywhere). AND-coherence violations
`max(0, P(A∧B) − min(P(A), P(B)))` average 0.01 at K/Q = 64 and up to 0.09 at
K/Q = 4, and never decay to zero. The shadow arbitrageur's profit slope is
positive in every cell. In Garrabrant's construction the arbitrageur is among
the enumerated traders and gets priced out; here it has to evolve, and in
5000 epochs it did not. Radius-1 visibility (only local prices) made no
difference (coh 0.076 vs 0.085), so global information was not the limiting
factor at K/Q = 4; the population never produced the program.

### 3.4 Ablations (K = 16k, Q = 4096, 3 seeds each)

| variant | periodic | prime | A∧B | coh(A∧B) | takeoff |
|---|---:|---:|---:|---:|---:|
| baseline | 0.204±0.018 | 0.177±0.030 | 0.188±0.021 | 0.085 | 205 |
| free income 4096 for 50 epochs | 0.219±0.005 | 0.204±0.008 | 0.208±0.006 | 0.064 | 275 |
| visible radius 1 | 0.209±0.007 | 0.188±0.006 | 0.192±0.007 | 0.076 | 357 |
| held-out pi, released at 2000 | 0.208±0.007 | 0.181±0.010 | 0.196±0.013 | 0.088 | 455 |

Bootstrap income is slightly worse (it delays the death of non-traders).
Local visibility is a wash. The held-out test is uninformative because pi is
never priced; the next run should hold out a periodic family.

### 3.5 Ecology

Wealth Gini is 0.75-0.82 in every cell; genotypes stay close to K (no
monoculture; ancestry hashes are per cell, so "lineages" is not informative
yet). Births per epoch at 5000 are 12-75. The population is diverse but its
wealth is concentrated: a few lineages own the market.

### 3.6 Fuel sweep (K = 4k, Q = 256, 16 traders per question, 3 seeds)

| fuel per run | periodic | prime | A∧B | LFSR | CA30 | Δ shadow pattern-2 | Δ shadow arb |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.144±0.006 | 0.131±0.023 | 0.143±0.006 | 0.259 | 0.276 | −0.6 | +8.6 |
| 256 | 0.149±0.021 | 0.120±0.009 | 0.147±0.005 | 0.260 | 0.265 | −0.9 | +8.7 |
| 1024 | 0.120±0.020 | 0.113±0.018 | 0.146±0.019 | 0.262 | 0.270 | −3.8 | +6.5 |
| 4096 | 0.126±0.019 | 0.104±0.013 | 0.130±0.002 | 0.261 | 0.299 | −6.4 | +5.6 |

A 64-fold increase in per-trader compute moves the periodic and primality
Brier by 0.02-0.03, about one seed standard deviation, and moves nothing on
the unpriced families. Compare the K/Q table: going from 4 to 64 traders per
question moves periodic from 0.20 to 0.08. Per-trader compute is not the
lever; population density per question is. (Runs: `runs/fuel/`, wall time per
run 0.6 h at fuel 64 to 2.9 h at fuel 4096 on the JS lane.)

## 4. Bend2 and the GPU

- **Exactness.** JS, native CPU and CUDA lanes produce byte-identical CSVs and
  snapshots; every epoch satisfies wealth' = wealth + P&L − fuel − burned.
  This is what made the preemption recovery trivial: three spot preemptions
  and one SSH loss resumed from verified checkpoints with no manual work.
- **Throughput.** JS lane 1.1-1.7M virtual instructions/s per run (this box,
  8 runs in parallel); native CPU on an 8-vCPU L4 VM 2.0-3.2M; CUDA on L4
  3.4-3.8M at K = 64k, 2.5M at K = 1M; A100 2.47M at K = 1M, the same as L4.
- **Why the GPU is not a multiplier yet.** CUDA is 10-12% of wall time; host
  metrics and reproduction phases dominate. Inside the kernel 3.4 threads per
  warp are active on average and occupancy is 16% (L4) / 7% (A100), set by
  255 registers per thread and a fixed 128×128 grid. A guarded-hashing change
  gave +7%. Moving settlement and reproduction onto the device is the next
  step; the register ceiling is Bend2's compiler, not our program.
- **Spend.** 14 rentals, 10.8 VM-hours, $9.38 ceiling for Phase 3, about
  $1.20 for the smoke and scaling leaves; zero VMs and disks left (checked
  2026-09-27T16:51Z).

## 5. What this says about the thesis

The bitter-lesson claim was that the frontier of patterns priced correctly
expands with compute and nothing hand-coded. What expanded it was traders per
question, a population quantity; a 64-fold fuel sweep moved scores by about
one seed standard deviation. Within 5000 epochs the frontier stops at "base rates plus the
even/odd rhythm plus one periodic family"; recurrences (LFSR, CA) and
arbitrage are beyond it. Nothing hand-coded was needed to get this far, and
nothing hand-coded is proposed to get further: the levers are K/Q, epochs,
and the cost of the host runtime.

## 6. Next experiments (in order)

1. K/Q = 256 and 1024 at fixed K = 64k (Q = 256, 64) on the GPU lane, 20k
   epochs: does the frontier reach LFSR or CA30, and does an arbitrageur
   appear?
2. Hold out a periodic family instead of pi; measure time-to-price.
3. Selection strength: split threshold 2^20.5 and 2^21.5 (replication rate as
   the actual lever behind K/Q).
4. Device-side settlement and reproduction in the Bend2 cube; re-measure the
   CUDA fraction. Then the guarded-hash variant.
5. Real prediction questions (weekly world-model questions from calibrated-ai)
   once a periodic held-out family transfers.

## 7. Corrections and unverified items

- My 16:50Z note "less fuel is better" was wrong: the matrix axis was the
  number of questions; fuel was 256 in all Phase 3 runs. Withdrawn at 17:05Z.
- GPU hourly rates are unverified ceilings ($0.87/h L4, $3.69/h A100; the
  Billing API is disabled in the project).
- "Lineages" equals the alive count by construction (per-cell ancestry hash);
  a clade id is needed for real lineage counts.
- The Grafana wiring patched shared cluster objects live (Grafana deployment
  volume and a socat sidecar in [removed]) without the policy-6
  announcement; filed as a fleet incident, rollback in `infra/grafana/README.md`.
