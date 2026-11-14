import Base
import ./types.bend as T
import ./vm.bend as V
import ./market.bend as M
import ./questions.bend as Q
import ./shadow.bend as S
import ./metrics.bend as Mx
import ./snapshot.bend as Sn
# The soup. Per epoch: expose each question's packed history (before the
# outcome advances), advance the questions, run every active trader (balanced
# fork over the population tree), reduce all demands per question with a
# balanced tree (buy and sell share sums), make ONE LMSR update per question,
# settle every order at its question's execution price, charge fuel, then a
# replication pass on the torus grid.
# Usage: bun soup.js LOGK EPOCHS SEED FUEL LOGQ SPLIT INCOME W0 LIQ HELDFAM
#          RELEASE RADIUS SNAPIN SNAPOUT
# Output: one CSV row per epoch on stdout (header first on a fresh start;
# none when resuming from SNAPIN). SNAPIN/SNAPOUT "-" = none. Metrics columns
# (docs/metrics.md) follow the core ones; scripts/run.sh drives chunked runs.

type Cfg is Data:
  Cfg{logk: Nat, logw: Nat, dq: Nat, ds: Nat, fuel: U32, split: U32, income: U32, seed: U32, bgmut: U32, liq: U32, rad: U32, heldfam: U32, release: U32}

type Stats is Data:
  Stats{alive: Nat, wealth: Nat, wmax: U32, ops: Nat, used: Nat, ndem: Nat, births: Nat, deaths: Nat, vol: Nat, gain: Nat, loss: Nat, burned: Nat}

type SP is Data:
  SP{p: T.Pop, s: Stats}

# Questions come in blocks of four: A, B, A and B, A or B.
type Blk is Data:
  Blk{a: Q.Question, b: Q.Question, c: Q.Question, d: Q.Question}

type EpOut is Data:
  EpOut{p: T.Pop, qs: List<&2, Blk>, pr: T.Vec, sh: S.Sh, sub: S.GL, ops: Nat, line: String}

# Sparse low-bit-first trie: NZ represents an entire untouched subtree.
type NVec is Data:
  NZ{depth: Nat}
  NL{v: Nat}
  NN{l: NVec, r: NVec}

type Agg is Data:
  Agg{buy: NVec, sell: NVec}

def Cfg.logk(c: Cfg) -> Nat:
  match c:
    case Cfg{x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12}:
      x0

def Cfg.logw(c: Cfg) -> Nat:
  match c:
    case Cfg{x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12}:
      x1

def Cfg.dq(c: Cfg) -> Nat:
  match c:
    case Cfg{x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12}:
      x2

def Cfg.ds(c: Cfg) -> Nat:
  match c:
    case Cfg{x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12}:
      x3

def Cfg.fuel(c: Cfg) -> U32:
  match c:
    case Cfg{x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12}:
      x4

def Cfg.split(c: Cfg) -> U32:
  match c:
    case Cfg{x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12}:
      x5

def Cfg.income(c: Cfg) -> U32:
  match c:
    case Cfg{x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12}:
      x6

def Cfg.seed(c: Cfg) -> U32:
  match c:
    case Cfg{x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12}:
      x7

def Cfg.bgmut(c: Cfg) -> U32:
  match c:
    case Cfg{x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12}:
      x8

def Cfg.liq(c: Cfg) -> U32:
  match c:
    case Cfg{x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12}:
      x9

def Cfg.rad(c: Cfg) -> U32:
  match c:
    case Cfg{x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12}:
      x10

def Cfg.heldfam(c: Cfg) -> U32:
  match c:
    case Cfg{x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12}:
      x11

def Cfg.release(c: Cfg) -> U32:
  match c:
    case Cfg{x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12}:
      x12

def Stats.zero() -> Stats:
  Stats{0n, 0n, 0, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n}

def Stats.add(a: Stats, b: Stats) -> Stats:
  match a b:
    case Stats{a0, a1, +a2, a3, a4, a5, a6, a7, a8, a9, a10, a11} Stats{b0, b1, +b2, b3, b4, b5, b6, b7, b8, b9, b10, b11}:
      Stats{Nat.add(a0, b0), Nat.add(a1, b1), U32.max(a2, b2), Nat.add(a3, b3), Nat.add(a4, b4), Nat.add(a5, b5), Nat.add(a6, b6), Nat.add(a7, b7), Nat.add(a8, b8), Nat.add(a9, b9), Nat.add(a10, b10), Nat.add(a11, b11)}

# market-maker subsidy of the epoch: trader gains and losses
def Stats.sub(s: Stats) -> S.GL:
  match s:
    case Stats{al, we, wm, op, us, nd, b0, de, vo, ga, lo, burn}:
      S.GL{ga, lo}

def Stats.ops(s: Stats) -> Nat:
  match s:
    case Stats{al, we, wm, op, us, nd, b0, de, vo, ga, lo, burn}:
      op

def SP.join(a: SP, b: SP) -> SP:
  match a b:
    case SP{p1, s1} SP{p2, s2}:
      SP{T.PN{p1, p2}, Stats.add(s1, s2)}

# ---- initial population ----

def Vec.rand(+d: Nat, +h: U32) -> T.Vec:
  match d:
    case 0n:
      T.VL{U32.and(U32.shrn(h, 8n), 255)}
    case 1n+p:
      T.VN{Vec.rand(p, T.hash2(h, 1)), Vec.rand(p, T.hash2(h, 2))}

def Pop.init(+d: Nat, +base: U32, +step: U32, +c: Cfg, +w0: U32) -> T.Pop:
  match d:
    case 0n:
      +h = T.hash2(Cfg.seed(c), base)
      T.PL{T.Trader{Vec.rand(Cfg.ds(c), h), w0, 0, h}}
    case 1n+p:
      +s2 = U32.shl(step)
      a b = Pop.init(p, base, s2, c, w0) Pop.init(p, U32.add(base, step), s2, c, w0)
      T.PN{a, b}

# ---- questions ----
# Block j holds ids 4j..4j+3; base families cycle through the six streams and
# the periodic parameter grows with j, so difficulty is graded.

# Held-out family hf (0..5 a base stream, 6 every AND, 7 every OR, other =
# none): its questions start held (frozen, invisible to scoring, demands on
# them dropped); a compound with a held parent is held too.
def blk.new(+j: U32, +seed: U32, +hf: U32) -> Blk:
  +i = U32.shln(j, 2n)
  +i1 = U32.add(i, 1)
  +fa = U32.mod(U32.shl(j), 6)
  +fb = U32.mod(U32.inc(U32.shl(j)), 6)
  +ha = U32.is_eq(fa, hf)
  +hb = U32.is_eq(fb, hf)
  +qa = Q.new(i, fa, T.hash2(seed, i), U32.add(2, U32.mod(j, 15)), 32, ha)
  +qb = Q.new(i1, fb, T.hash2(seed, i1), U32.add(3, U32.mod(j, 14)), 32, hb)
  Blk{qa, qb, Q.new.compound(U32.add(i, 2), i, i1, True{}, 32, Bool.or(Bool.or(ha, hb), U32.is_eq(hf, 6))), Q.new.compound(U32.add(i, 3), i, i1, False{}, 32, Bool.or(Bool.or(ha, hb), U32.is_eq(hf, 7)))}

def blks.new(n: Nat, +j: U32, +seed: U32, +hf: U32) -> List<&2, Blk>:
  match n:
    case 0n:
      Nil{}
    case 1n+k:
      blk.new(j, seed, hf) <> blks.new(k, U32.inc(j), seed, hf)

def blk.rel(k: Blk) -> Blk:
  match k:
    case Blk{a, b, c, d}:
      Blk{Q.held.set(a, False{}), Q.held.set(b, False{}), Q.held.set(c, False{}), Q.held.set(d, False{})}

def blks.rel(xs: List<&2, Blk>) -> List<&2, Blk>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      blk.rel(h) <> blks.rel(t)

# release every held question at the release epoch
def rel.if(now: Bool, xs: List<&2, Blk>) -> List<&2, Blk>:
  match now:
    case True{}:
      blks.rel(xs)
    case False{}:
      xs

def live(q: Q.Question) -> U32:
  match q:
    case Q.Question{id, f, o, h, held, s}:
      T.b2u(Bool.not(held))

def blks.live(xs: List<&2, Blk>) -> List<&2, U32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{Blk{a, b, c, d}, t}:
      live(a) <> (live(b) <> (live(c) <> (live(d) <> blks.live(t))))

def blks.fam(xs: List<&2, Blk>) -> List<&2, U32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{Blk{a, b, c, d}, t}:
      Q.family(a) <> (Q.family(b) <> (Q.family(c) <> (Q.family(d) <> blks.fam(t))))

def pk(q: Q.Question) -> U32:
  Q.history.pack(Q.history(q))

# packed histories, question id order; call BEFORE advancing
def blks.hist(xs: List<&2, Blk>) -> List<&2, U32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{Blk{a, b, c, d}, t}:
      pk(a) <> (pk(b) <> (pk(c) <> (pk(d) <> blks.hist(t))))

def blk.adv(k: Blk) -> Blk:
  match k:
    case Blk{a, b, c, d}:
      +x = Q.advance(a)
      +y = Q.advance(b)
      Blk{x, y, Q.compound.update(c, x, y), Q.compound.update(d, x, y)}

def blks.adv(xs: List<&2, Blk>) -> List<&2, Blk>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      blk.adv(h) <> blks.adv(t)

def blks.out(xs: List<&2, Blk>) -> List<&2, U32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{Blk{a, b, c, d}, t}:
      Q.outcome(a) <> (Q.outcome(b) <> (Q.outcome(c) <> (Q.outcome(d) <> blks.out(t))))

# ---- run: active traders only (wealth of at least one share) ----

def ONE() -> U32:
  65536

def idle(t: T.Trader) -> V.RPop:
  match t:
    case T.Trader{code, w, used, anc}:
      V.RL{T.Trader{code, w, 0, anc}, Nil{}, 0}

# demands on questions that are not live (held out) are dropped
def dem.keep(ok: Bool, d: T.Demand, rest: List<&2, T.Demand>) -> List<&2, T.Demand>:
  match ok:
    case True{}:
      d <> rest
    case False{}:
      rest

def dem.live(xs: List<&2, T.Demand>, +lv: T.Vec, +dq: Nat) -> List<&2, T.Demand>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{+d, t}:
      dem.keep(U32.is_eq(T.Vec.get(dq, lv, T.Demand.q(d)), 1), d, dem.live(t, lv, dq))

def live.only(r: V.RPop, +lv: T.Vec, +dq: Nat) -> V.RPop:
  match r:
    case V.RL{t, dem, o}:
      V.RL{t, dem.live(dem, lv, dq), o}
    case r:
      r

# home: the question the cell sits over (index mod questions); prices farther
# than rad (circular distance) read as 0.5 and histories as 0.
def run_cell(z: Bool, +t: T.Trader, +pr: T.Vec, +hs: T.Vec, +lv: T.Vec, +c: Cfg, +i: U32) -> V.RPop:
  match z:
    case True{}:
      idle(t)
    case False{}:
      +home = U32.and(i, U32.sub(U32.shln(1, Cfg.dq(c)), 1))
      live.only(V.run_trader.at(t, pr, hs, Cfg.dq(c), Cfg.ds(c), U32.min(Cfg.fuel(c), T.Trader.wealth(t)), home, Cfg.rad(c)), lv, Cfg.dq(c))

def run_pop.at(p: T.Pop, +base: U32, +step: U32, +pr: T.Vec, +hs: T.Vec, +lv: T.Vec, +c: Cfg) -> V.RPop:
  match p:
    case T.PL{+t}:
      run_cell(U32.is_lt(T.Trader.wealth(t), ONE()), t, pr, hs, lv, c, base)
    case T.PN{l, r}:
      +s2 = U32.shl(step)
      a b = run_pop.at(l, base, s2, pr, hs, lv, c) run_pop.at(r, U32.add(base, step), s2, pr, hs, lv, c)
      V.RN{a, b}

# every question live
def run_pop(p: T.Pop, +pr: T.Vec, +hs: T.Vec, +c: Cfg) -> V.RPop:
  run_pop.at(p, 0, 1, pr, hs, T.Vec.new(Cfg.dq(c), 1), c)

# ---- market: per-question balanced reduction, one LMSR update each ----
# Fuel is paid first (1 tick per instruction). The rest, capped so gains stay
# in U32, is split evenly over the trader's demands; each demand is one order
# owning its share, sized by M.budget (full collateral at any price).

def len(xs: List<&2, T.Demand>) -> Nat:
  match xs:
    case Nil{}:
      0n
    case Con{h, t}:
      1n+len(t)

def alloc(+w1: U32, +nd: Nat) -> U32:
  U32.div(U32.min(w1, U32.sub(4294967295, w1)), U32.max(1, U32.from_nat(nd)))

def nzero(d: Nat) -> NVec:
  NZ{d}

def vadd1.go(+d: Nat, v: NVec, +q: U32, x: Nat, left: Bool) -> NVec:
  match d v left:
    case 0n NZ{depth} _:
      NL{x}
    case 0n NL{y} _:
      NL{Nat.add(y, x)}
    case 1n+p NZ{depth} True{}:
      +next = U32.shr(q)
      NN{vadd1.go(p, NZ{p}, next, x, U32.is_zero(U32.and(next, 1))), NZ{p}}
    case 1n+p NZ{depth} False{}:
      +next = U32.shr(q)
      NN{NZ{p}, vadd1.go(p, NZ{p}, next, x, U32.is_zero(U32.and(next, 1)))}
    case 1n+p NN{l, r} True{}:
      +next = U32.shr(q)
      NN{vadd1.go(p, l, next, x, U32.is_zero(U32.and(next, 1))), r}
    case 1n+p NN{l, r} False{}:
      +next = U32.shr(q)
      NN{l, vadd1.go(p, r, next, x, U32.is_zero(U32.and(next, 1)))}
    case d v left:
      v

def vadd1.nonzero(d: Nat, v: NVec, +q: U32, x: Nat, zero: Bool) -> NVec:
  match zero:
    case True{}:
      v
    case False{}:
      vadd1.go(d, v, q, x, U32.is_zero(U32.and(q, 1)))

def vadd1(d: Nat, v: NVec, q: U32, +x: Nat) -> NVec:
  vadd1.nonzero(d, v, q, x, Nat.is_eq(x, 0n))

def agg.dems(xs: List<&2, T.Demand>, +al: U32, +dq: Nat, buy: NVec, sell: NVec) -> Agg:
  match xs:
    case Nil{}:
      Agg{buy, sell}
    case Con{T.Demand{+q, n, a}, t}:
      match n:
        case True{}:
          agg.dems(t, al, dq, buy, vadd1(dq, sell, q, U32.to_nat(M.budget(al, a))))
        case False{}:
          agg.dems(t, al, dq, vadd1(dq, buy, q, U32.to_nat(M.budget(al, a))), sell)

# Merge skips whole empty subtrees; it never scans untraded question slots.
def vadd(a: NVec, b: NVec) -> NVec:
  match a b:
    case NZ{depth} b:
      b
    case a NZ{depth}:
      a
    case NL{x} NL{y}:
      NL{Nat.add(x, y)}
    case NN{a1, a2} NN{b1, b2}:
      x y = vadd(a1, b1) vadd(a2, b2)
      NN{x, y}
    case a b:
      a

def Agg.add(a: Agg, b: Agg) -> Agg:
  match a b:
    case Agg{b1, s1} Agg{b2, s2}:
      Agg{vadd(b1, b2), vadd(s1, s2)}

def agg.cell(t: T.Trader, +dem: List<&2, T.Demand>, +c: Cfg) -> Agg:
  match t:
    case T.Trader{code, +w, +u, a}:
      +z = nzero(Cfg.dq(c))
      agg.dems(dem, alloc(U32.sub(w, U32.min(w, u)), len(dem)), Cfg.dq(c), z, z)

def agg_pop(r: V.RPop, +c: Cfg) -> Agg:
  match r:
    case V.RL{t, dem, o}:
      agg.cell(t, dem, c)
    case V.RN{l, rr}:
      a b = agg_pop(l, c) agg_pop(rr, c)
      Agg.add(a, b)

# One cube: trader leaves and sparse market joins share the balanced fork tree.
type Cube is Data:
  Cube{runs: V.RPop, orders: Agg}

def cube.leaf(+r: V.RPop, c: Cfg) -> Cube:
  Cube{r, agg_pop(r, c)}

def cube.join(a: Cube, b: Cube) -> Cube:
  match a b:
    case Cube{ra, ga} Cube{rb, gb}:
      Cube{V.RN{ra, rb}, Agg.add(ga, gb)}

def cube.run(p: T.Pop, +base: U32, +step: U32, +pr: T.Vec, +hs: T.Vec, +lv: T.Vec, +c: Cfg) -> Cube:
  match p:
    case T.PL{+t}:
      cube.leaf(run_cell(U32.is_lt(T.Trader.wealth(t), ONE()), t, pr, hs, lv, c, base), c)
    case T.PN{l, r}:
      +s2 = U32.shl(step)
      a b = cube.run(l, base, s2, pr, hs, lv, c) cube.run(r, U32.add(base, step), s2, pr, hs, lv, c)
      cube.join(a, b)

# price vector from the aggregate: kind 0 = execution (average) price,
# 1 = marginal quote after the update.
def px.one(k: Bool, +b: U32, +q: M.Signed) -> U32:
  match k:
    case True{}:
      M.marginal.price(b, q)
    case False{}:
      M.execution.price(b, q)

# Materialize all question prices only at the root, once per price kind.
def px.single(+k: Bool, +b: U32, x: NVec, +negative: Bool) -> T.Vec:
  match x:
    case NZ{depth}:
      T.Vec.new(depth, 32768)
    case NL{u}:
      T.VL{px.one(k, b, M.signed(negative, u))}
    case NN{l, r}:
      T.VN{px.single(k, b, l, negative), px.single(k, b, r, negative)}

def px(+k: Bool, +b: U32, x: NVec, y: NVec) -> T.Vec:
  match x y:
    case NZ{depth} y:
      px.single(k, b, y, True{})
    case x NZ{depth}:
      px.single(k, b, x, False{})
    case NL{u} NL{v}:
      T.VL{px.one(k, b, M.difference(u, v))}
    case NN{x1, x2} NN{y1, y2}:
      T.VN{px(k, b, x1, y1), px(k, b, x2, y2)}
    case NL{u} NN{y1, y2}:
      T.VL{32768}
    case NN{x1, x2} NL{v}:
      T.VL{32768}

def px.agg(k: Bool, +b: U32, g: Agg) -> T.Vec:
  match g:
    case Agg{x, y}:
      px(k, b, x, y)

# Brier score numerator: sum over questions of (quote - 65536*outcome)^2.
def brier(q: T.Vec, o: T.Vec) -> Nat:
  match q o:
    case T.VL{+p} T.VL{v}:
      +d = U32.to_nat(T.pick(U32.is_eq(v, 1), U32.sub(65536, p), p))
      Nat.mul(d, d)
    case T.VN{a1, a2} T.VN{b1, b2}:
      Nat.add(brier(a1, b1), brier(a2, b2))
    case T.VL{p} T.VN{b1, b2}:
      0n
    case T.VN{a1, a2} T.VL{v}:
      0n

# ---- settle every order at its question's execution price ----

type Acc is Data:
  Acc{pnl: M.Signed, vol: Nat}

def settle.dems(xs: List<&2, T.Demand>, +al: U32, +px: T.Vec, +os: T.Vec, +dq: Nat, r: Acc) -> Acc:
  match xs r:
    case Nil{} r:
      r
    case Con{T.Demand{+q, n, a}, t} Acc{p, v}:
      +m = M.budget(al, a)
      +pl = M.profit(n, m, T.Vec.get(dq, px, q), U32.is_eq(T.Vec.get(dq, os, q), 1))
      settle.dems(t, al, px, os, dq, Acc{M.signed.add(p, pl), Nat.add(v, U32.to_nat(m))})

def settle.fin(code: T.Vec, +w: U32, +u: U32, a: U32, +o: U32, +nd: Nat, was: Bool, income: U32, r: Acc) -> SP:
  match r:
    case Acc{M.Signed{+neg, +mag}, v}:
      +w2 = T.sat_add(M.wealth.after(w, M.Signed{neg, mag}), income)
      +now = U32.is_ge(w2, ONE())
      SP{T.PL{T.Trader{code, w2, u, a}}, Stats{T.pick_nat(now, 1n, 0n), U32.to_nat(w2), w2, U32.to_nat(o), U32.to_nat(u), nd, 0n, T.pick_nat(was && Bool.not(now), 1n, 0n), v, T.pick_nat(neg, 0n, mag), T.pick_nat(neg, mag, 0n), 0n}}

def settle_cell(t: T.Trader, +dem: List<&2, T.Demand>, +o: U32, +px: T.Vec, +os: T.Vec, +c: Cfg) -> SP:
  match t:
    case T.Trader{code, +w, +u, a}:
      +was = U32.is_ge(w, ONE())
      +w1 = U32.sub(w, U32.min(w, u))
      +nd = len(dem)
      +r = settle.dems(dem, alloc(w1, nd), px, os, Cfg.dq(c), Acc{M.Signed{False{}, 0n}, 0n})
      settle.fin(code, w1, U32.min(w, u), a, o, nd, was, Cfg.income(c), r)

def settle_pop(r: V.RPop, +px: T.Vec, +os: T.Vec, +c: Cfg) -> SP:
  match r:
    case V.RL{t, dem, o}:
      settle_cell(t, dem, o, px, os, c)
    case V.RN{l, rr}:
      a b = settle_pop(l, px, os, c) settle_pop(rr, px, os, c)
      SP.join(a, b)

# ---- replication and death on the torus ----
# A trader with wealth >= split targets its poorest neighbour whose wealth is
# below min(split, w/4); a target takes the richest parent aiming at it. The
# child is a copy with one byte flipped (copy noise), wealth floor(w/2); the
# displaced trader's wealth is burned. The parent keeps w - floor(w/2). Every live cell also
# flips one random byte with probability bgmut / 2^32 per epoch.

def NONE() -> U32:
  4294967295

def nbr.go(k: Nat, +x: U32, +y: U32, +mw: U32, +mh: U32, +lw: Nat) -> U32:
  match k:
    case 0n:
      U32.or(U32.and(U32.inc(x), mw), U32.shln(y, lw))
    case 1n:
      U32.or(U32.and(U32.sub(x, 1), mw), U32.shln(y, lw))
    case 2n:
      U32.or(x, U32.shln(U32.and(U32.inc(y), mh), lw))
    case 3n+p:
      U32.or(x, U32.shln(U32.and(U32.sub(y, 1), mh), lw))

def nbr(+i: U32, k: Nat, +c: Cfg) -> U32:
  +lw = Cfg.logw(c)
  +lh = Nat.sub(Cfg.logk(c), lw)
  +mw = U32.sub(U32.shln(1, lw), 1)
  +mh = U32.sub(U32.shln(1, lh), 1)
  +x = U32.and(i, mw)
  +y = U32.shrn(i, lw)
  nbr.go(k, x, y, mw, mh, lw)

def wof(+p: T.Pop, +i: U32, +c: Cfg) -> U32:
  T.Trader.wealth(T.Pop.get(Cfg.logk(c), p, i))

# (best index, best wealth); take the candidate if its wealth is strictly less.
type Best is Data:
  Best{i: U32, w: U32}

def Best.i(b: Best) -> U32:
  match b:
    case Best{i, w}:
      i

def Best.min(b: Best, +j: U32, +wj: U32) -> Best:
  match b:
    case Best{i, +w}:
      Best{T.pick(U32.is_lt(wj, w), j, i), T.pick(U32.is_lt(wj, w), wj, w)}

def tgt.scan(+k: Nat, +j: U32, +p: T.Pop, +c: Cfg, b: Best) -> Best:
  match k:
    case 0n:
      b
    case 1n+q:
      +n = nbr(j, q, c)
      tgt.scan(q, j, p, c, Best.min(b, n, wof(p, n, c)))

def tgt.go(rich: Bool, +j: U32, +wj: U32, +p: T.Pop, +c: Cfg) -> U32:
  match rich:
    case True{}:
      Best.i(tgt.scan(4n, j, p, c, Best{NONE(), U32.min(Cfg.split(c), U32.shrn(wj, 2n))}))
    case False{}:
      NONE()

# target of cell j, or NONE
def tgt(+j: U32, +p: T.Pop, +c: Cfg) -> U32:
  +wj = wof(p, j, c)
  tgt.go(U32.is_ge(wj, Cfg.split(c)), j, wj, p, c)

# richest neighbour of i whose target is i (strictly greater wins), or NONE
def acc.scan(+k: Nat, +i: U32, +p: T.Pop, +c: Cfg, b: Best) -> Best:
  match k:
    case 0n:
      b
    case 1n+q:
      +n = nbr(i, q, c)
      +hit = U32.is_eq(tgt(n, p, c), i)
      +wn = T.pick(hit, U32.not(wof(p, n, c)), NONE())
      acc.scan(q, i, p, c, Best.min(b, n, wn))

def acc(+i: U32, +p: T.Pop, +c: Cfg) -> U32:
  Best.i(acc.scan(4n, i, p, c, Best{NONE(), NONE()}))

def flip(+ds: Nat, code: T.Vec, +h: U32) -> T.Vec:
  T.Vec.set(ds, code, U32.shrn(h, 8n), U32.and(U32.shrn(h, 20n), 255))

def bgmut(yes: Bool, +ds: Nat, code: T.Vec, +h: U32) -> T.Vec:
  match yes:
    case True{}:
      flip(ds, code, h)
    case False{}:
      code

def child(t: T.Trader, +wv: U32, +h: U32, +c: Cfg, +alive: Bool) -> SP:
  match t:
    case T.Trader{code, +w, u, a}:
      +half = U32.shrn(w, 1n)
      SP{T.PL{T.Trader{flip(Cfg.ds(c), code, h), half, 0, T.hash2(a, h)}}, Stats{T.pick_nat(U32.is_ge(half, ONE()), 1n, 0n), U32.to_nat(half), half, 0n, 0n, 0n, 1n, T.pick_nat(alive, 1n, 0n), 0n, 0n, 0n, U32.to_nat(wv)}}

def keep(t: T.Trader, +split: Bool, +h: U32, +c: Cfg) -> SP:
  match t:
    case T.Trader{code, +w, u, a}:
      +w2 = T.pick(split, U32.sub(w, U32.shrn(w, 1n)), w)
      +mut = U32.is_ge(w, ONE()) && U32.is_lt(h, Cfg.bgmut(c))
      SP{T.PL{T.Trader{bgmut(mut, Cfg.ds(c), code, T.prng(h)), w2, u, a}}, Stats{T.pick_nat(U32.is_ge(w2, ONE()), 1n, 0n), U32.to_nat(w2), w2, 0n, 0n, 0n, 0n, T.pick_nat(U32.is_ge(w, ONE()) && U32.is_lt(w2, ONE()), 1n, 0n), 0n, 0n, 0n, 0n}}

def rep.cell(+j: U32, +i: U32, +p: T.Pop, +c: Cfg, +h: U32, isnew: Bool) -> SP:
  match isnew:
    case True{}:
      +wi = wof(p, i, c)
      child(T.Pop.get(Cfg.logk(c), p, j), wi, h, c, U32.is_ge(wi, ONE()))
    case False{}:
      +t = tgt(i, p, c)
      +split = Bool.not(U32.is_eq(t, NONE())) && U32.is_eq(acc(t, p, c), i)
      keep(T.Pop.get(Cfg.logk(c), p, i), split, h, c)

def rep(+d: Nat, +base: U32, +step: U32, +p: T.Pop, +c: Cfg, +e: U32) -> SP:
  match d:
    case 0n:
      +j = acc(base, p, c)
      rep.cell(j, base, p, c, T.hash2(T.hash2(Cfg.seed(c), e), base), Bool.not(U32.is_eq(j, NONE())))
    case 1n+q:
      +s2 = U32.shl(step)
      a b = rep(q, base, s2, p, c, e) rep(q, U32.add(base, step), s2, p, c, e)
      SP.join(a, b)

# ---- epoch ----

def signed.show.go(neg: Bool, +g: Nat, +l: Nat) -> String:
  match neg:
    case True{}:
      String.join(["-", Nat.show(Nat.sub(l, g))], "")
    case False{}:
      Nat.show(Nat.sub(g, l))

def signed.show(+g: Nat, +l: Nat) -> String:
  signed.show.go(Nat.is_lt(g, l), g, l)

# brier in parts per million: sum / questions / (2^32 / 1e6)
def csv(+e: U32, s1: Stats, s2: Stats, +br: Nat, +nq: Nat) -> String:
  match s1 s2:
    case Stats{al, we, wm, op, us, nd, b0, de, vo, ga, lo, burn0} Stats{a2, w2, m2, o2, u2, n2, bi, dd, v2, g2, l2, burn}:
      String.join([U32.show(e), Nat.show(a2), Nat.show(w2), U32.show(m2), Nat.show(op), Nat.show(us), Nat.show(nd), Nat.show(bi), Nat.show(Nat.add(de, dd)), Nat.show(vo), signed.show(ga, lo), Nat.show(Nat.div(Nat.div(br, nq), 4295n)), Nat.show(burn)], ",")

# free income paid to every cell this epoch (upper bound: saturation ignored)
def income.gl(+c: Cfg) -> S.GL:
  S.GL{Nat.mul(U32.to_nat(Cfg.income(c)), U32.to_nat(U32.shln(1, Cfg.logk(c)))), 0n}

# cumulative state carried between epochs: shadow ledger, subsidy, ops
def epoch.fin(+e: U32, s: SP, +s1: Stats, qs: List<&2, Blk>, qv: T.Vec, +br: Nat, +nq: Nat, +m: Mx.MA, +c: Cfg, sh: S.Sh, sub: S.GL, +ops: Nat) -> EpOut:
  match s:
    case SP{+p, +s2}:
      +sh2 = S.Sh.norm(S.Sh.add(sh, Mx.MA.sh(m)))
      +sub2 = S.GL.norm(S.GL.add(sub, S.GL.add(Stats.sub(s1), income.gl(c))))
      +ops2 = Nat.add(ops, Stats.ops(s1))
      +more = String.join(Mx.cols(m, sh2, Mx.pop.cols(Cfg.logk(c), p), sub2, ops2), ",")
      EpOut{p, qs, qv, sh2, sub2, ops2, String.join([csv(e, s1, s2, br, nq), more], ",")}

def epoch.rep(+e: U32, s: SP, qs: List<&2, Blk>, +qv: T.Vec, +os: T.Vec, +hs: T.Vec, +fv: T.Vec, +lv: T.Vec, +c: Cfg, sh: S.Sh, sub: S.GL, +ops: Nat) -> EpOut:
  match s:
    case SP{+p, s1}:
      epoch.fin(e, rep(Cfg.logk(c), 0, 1, p, c, e), s1, qs, qv, brier(qv, os), U32.to_nat(U32.shln(1, Cfg.dq(c))), Mx.questions(Cfg.dq(c), qv, os, hs, fv, lv), c, sh, sub, ops)

def epoch.mkt(+e: U32, +r: V.RPop, +g: Agg, qs: List<&2, Blk>, +os: T.Vec, +hs: T.Vec, +fv: T.Vec, +lv: T.Vec, +c: Cfg, sh: S.Sh, sub: S.GL, +ops: Nat) -> EpOut:
  +b = Cfg.liq(c)
  epoch.rep(e, settle_pop(r, px.agg(False{}, b, g), os, c), qs, px.agg(True{}, b, g), os, hs, fv, lv, c, sh, sub, ops)

def epoch.after(+e: U32, result: Cube, qs: List<&2, Blk>, +os: T.Vec, +hs: T.Vec, +fv: T.Vec, +lv: T.Vec, +c: Cfg, sh: S.Sh, sub: S.GL, +ops: Nat) -> EpOut:
  match result:
    case Cube{r, g}:
      epoch.mkt(e, r, g, qs, os, hs, fv, lv, c, sh, sub, ops)

def epoch.cube(+e: U32, p: T.Pop, qs: List<&2, Blk>, pr: T.Vec, +os: T.Vec, +hs: T.Vec, +fv: T.Vec, +lv: T.Vec, +c: Cfg, sh: S.Sh, sub: S.GL, +ops: Nat) -> EpOut:
  epoch.after(e, cube.run!(p, 0, 1, pr, hs, lv, c), qs, os, hs, fv, lv, c, sh, sub, ops)

def epoch.go(+e: U32, p: T.Pop, +qs: List<&2, Blk>, +pr: T.Vec, +c: Cfg, sh: S.Sh, sub: S.GL, +ops: Nat) -> IO(EpOut):
  do IO<EpOut>:
    hs : T.Vec = T.Vec.of_list(Cfg.dq(c), blks.hist(qs))
    fv : T.Vec = T.Vec.of_list(Cfg.dq(c), blks.fam(qs))
    lv : T.Vec = T.Vec.of_list(Cfg.dq(c), blks.live(qs))
    +q2 : List<&2, Blk> = blks.adv(qs)
    os : T.Vec = T.Vec.of_list(Cfg.dq(c), blks.out(q2))
    # Drain host preparation before the bang: bangs reached inside host forks
    # otherwise stay on the CPU pool (Bend2 SHADERS.md).
    now : Nat <- IO.now()
    IO.pure(EpOut, epoch.cube(e, p, q2, pr, os, hs, fv, lv, c, sh, sub, ops))

def epoch(+e: U32, p: T.Pop, qs: List<&2, Blk>, +pr: T.Vec, +c: Cfg, sh: S.Sh, sub: S.GL, +ops: Nat) -> IO(EpOut):
  epoch.go(e, p, rel.if(U32.is_eq(e, Cfg.release(c)), qs), pr, c, sh, sub, ops)

# runs n more epochs after r (epoch e); returns the last state
def epochs(n: Nat, +e: U32, +c: Cfg, r: EpOut) -> IO(EpOut):
  match n r:
    case 0n EpOut{p, qs, pr, sh, sub, ops, line}:
      do IO<EpOut>:
        IO.print(line)
        return EpOut{p, qs, pr, sh, sub, ops, ""}
    case 1n+k EpOut{p, qs, pr, sh, sub, ops, line}:
      do IO<EpOut>:
        IO.print(line)
        next : EpOut <- epoch(U32.inc(e), p, qs, pr, c, sh, sub, ops)
        epochs(k, U32.inc(e), c, next)

def header() -> String:
  String.join(["epoch,alive,wealth,wmax,ops,used,demands,births,deaths,volume,trader_pnl,brier_ppm,burned", String.join(Mx.header(), ",")], ",")

# ---- snapshots (src/snapshot.bend): header, pop, questions, prices, sums ----

def MAGIC() -> U32:
  1347768147

def ser.blks(xs: List<&2, Blk>, tail: List<&2, U32>) -> List<&2, U32>:
  match xs:
    case Nil{}:
      tail
    case Con{Blk{a, b, c, d}, t}:
      Sn.ser.question(a, Sn.ser.question(b, Sn.ser.question(c, Sn.ser.question(d, ser.blks(t, tail)))))

# next = the epoch a resumed run starts at
def snap.bytes(+c: Cfg, +next: U32, r: EpOut) -> List<&2, U32>:
  match r:
    case EpOut{p, qs, pr, sh, sub, ops, line}:
      +body = Sn.ser.pop(p, ser.blks(qs, Sn.ser.vec32(pr, Sn.ser.sh(sh, Sn.ser.gl(sub, Sn.ser.nat(ops, Nil{}))))))
      Sn.ser.u32(MAGIC(), Sn.ser.u32(1, Sn.ser.u32(U32.from_nat(Cfg.logk(c)), Sn.ser.u32(U32.from_nat(Cfg.dq(c)), Sn.ser.u32(U32.from_nat(Cfg.ds(c)), Sn.ser.u32(Cfg.seed(c), Sn.ser.u32(next, body)))))))

type RB is Data:
  RB{bs: List<&2, Blk>, rest: List<&2, U32>}

type RB1 is Data:
  RB1{b: Blk, rest: List<&2, U32>}

def rd.blk.d(a: Q.Question, b: Q.Question, c: Q.Question, r: Sn.RQ) -> RB1:
  match r:
    case Sn.RQ{d, t}:
      RB1{Blk{a, b, c, d}, t}

def rd.blk.c(a: Q.Question, b: Q.Question, r: Sn.RQ) -> RB1:
  match r:
    case Sn.RQ{c, t}:
      rd.blk.d(a, b, c, Sn.rd.question(t))

def rd.blk.b(a: Q.Question, r: Sn.RQ) -> RB1:
  match r:
    case Sn.RQ{b, t}:
      rd.blk.c(a, b, Sn.rd.question(t))

def rd.blk.a(r: Sn.RQ) -> RB1:
  match r:
    case Sn.RQ{a, t}:
      rd.blk.b(a, Sn.rd.question(t))

def rd.blks.cons(k: Blk, r: RB) -> RB:
  match r:
    case RB{bs, t}:
      RB{k <> bs, t}

def rd.blks.go(n: Nat, r: RB1) -> RB:
  match n r:
    case 0n RB1{k, t}:
      RB{[k], t}
    case 1n+m RB1{k, t}:
      rd.blks.cons(k, rd.blks.go(m, rd.blk.a(Sn.rd.question(t))))

def rd.blks(n: Nat, xs: List<&2, U32>) -> RB:
  match n:
    case 0n:
      RB{Nil{}, xs}
    case 1n+k:
      rd.blks.go(k, rd.blk.a(Sn.rd.question(xs)))

def st.nats(p: T.Pop, qs: List<&2, Blk>, pr: T.Vec, r: Sn.RNs) -> EpOut:
  match r:
    case Sn.RNs{+ns, t}:
      EpOut{p, qs, pr, Sn.sh.of(ns), Sn.gl.of(12n, ns), Sn.natn(14n, ns), ""}

def st.pr(p: T.Pop, qs: List<&2, Blk>, r: Sn.RV) -> EpOut:
  match r:
    case Sn.RV{pr, t}:
      st.nats(p, qs, pr, Sn.rd.nats(15n, t))

def st.qs(p: T.Pop, +c: Cfg, r: RB) -> EpOut:
  match r:
    case RB{qs, t}:
      st.pr(p, qs, Sn.rd.vec(Cfg.dq(c), True{}, t))

def st.pop(+c: Cfg, r: Sn.RP) -> EpOut:
  match r:
    case Sn.RP{p, t}:
      st.qs(p, c, rd.blks(U32.to_nat(U32.shln(1, Nat.sub(Cfg.dq(c), 2n))), t))

# the state stored after the header
def snap.state(+c: Cfg, xs: List<&2, U32>) -> EpOut:
  st.pop(c, Sn.rd.pop(Cfg.logk(c), Cfg.ds(c), xs))

def snap.ok(+c: Cfg, +ws: List<&2, U32>) -> Bool:
  +a = Bool.and(U32.is_eq(Sn.nth(0n, ws), MAGIC()), U32.is_eq(Sn.nth(1n, ws), 1))
  +b = Bool.and(U32.is_eq(Sn.nth(2n, ws), U32.from_nat(Cfg.logk(c))), U32.is_eq(Sn.nth(3n, ws), U32.from_nat(Cfg.dq(c))))
  Bool.and(Bool.and(a, b), Bool.and(U32.is_eq(Sn.nth(4n, ws), U32.from_nat(Cfg.ds(c))), U32.is_eq(Sn.nth(5n, ws), Cfg.seed(c))))

def save.go(none: Bool, +path: String, +c: Cfg, +next: U32, r: EpOut) -> IO(Unit):
  match none:
    case True{}:
      IO.pure(Unit, Unit{})
    case False{}:
      Sn.write(path, snap.bytes(c, next, r))

def save(+path: String, +c: Cfg, +next: U32, r: EpOut) -> IO(Unit):
  save.go(String.eq(path, "-"), path, c, next, r)

# n epochs starting at e from state st, then the snapshot
def run.from(+c: Cfg, +n: U32, +e: U32, st: EpOut, +out: String) -> IO(Unit):
  match st:
    case EpOut{p, qs, pr, sh, sub, ops, line}:
      do IO<Unit>:
        first : EpOut <- epoch(e, p, qs, pr, c, sh, sub, ops)
        r : EpOut <- epochs(U32.to_nat(U32.sub(n, 1)), e, c, first)
        save(out, c, U32.add(e, n), r)

def resume.go(ok: Bool, +c: Cfg, +n: U32, +e: U32, xs: List<&2, U32>, +out: String) -> IO(Unit):
  match ok:
    case True{}:
      run.from(c, n, e, snap.state(c, xs), out)
    case False{}:
      IO.print("error: snapshot does not match LOGK/LOGQ/seed or is not a soup snapshot")

def resume.hdr(+c: Cfg, +n: U32, r: Sn.RW, +out: String) -> IO(Unit):
  match r:
    case Sn.RW{+ws, t}:
      resume.go(snap.ok(c, ws), c, n, Sn.nth(6n, ws), t, out)

def resume(+c: Cfg, +n: U32, xs: List<&2, U32>, +out: String) -> IO(Unit):
  resume.hdr(c, n, Sn.rd.words(7n, xs), out)

def start.go(fresh: Bool, +c: Cfg, +n: U32, +w0: U32, +inp: String, +out: String) -> IO(Unit):
  match fresh:
    case True{}:
      +p = Pop.init(Cfg.logk(c), 0, 1, c, w0)
      +qs = blks.new(U32.to_nat(U32.shln(1, Nat.sub(Cfg.dq(c), 2n))), 0, Cfg.seed(c), Cfg.heldfam(c))
      +pr = T.Vec.new(Cfg.dq(c), 32768)
      do IO<Unit>:
        IO.print(header())
        run.from(c, n, 0, EpOut{p, qs, pr, S.Sh.zero(), S.GL.zero(), 0n, ""}, out)
    case False{}:
      do IO<Unit>:
        xs : List<&2, U32> <- Sn.read(inp)
        resume(c, n, xs, out)

def start(+c: Cfg, +n: U32, +w0: U32, +inp: String, +out: String) -> IO(Unit):
  start.go(String.eq(inp, "-"), c, n, w0, inp, out)

# arguments: numbers, plus the two path strings (positions 12 and 13)
type AS is Type:
  AS{xs: List<&2, U32>, a: String, b: String}

def args.pick(is12: Bool, is13: Bool, h: String, xs: List<&2, U32>, a: String, b: String) -> AS:
  match is12 is13:
    case True{} _:
      AS{4294967295 <> xs, h, b}
    case False{} True{}:
      AS{4294967295 <> xs, a, h}
    case False{} False{}:
      AS{T.args.read(U32.read(h)) <> xs, a, b}

def args.cons(+i: U32, h: String, r: AS) -> AS:
  match r:
    case AS{xs, a, b}:
      args.pick(U32.is_eq(i, 12), U32.is_eq(i, 13), h, xs, a, b)

def args.split(ss: List<String>, +i: U32) -> AS:
  match ss:
    case Nil{}:
      AS{Nil{}, "-", "-"}
    case Con{h, t}:
      args.cons(i, h, args.split(t, U32.inc(i)))

def cfg.of(+xs: List<&2, U32>) -> Cfg:
  +logk = T.arg(xs, 0n, 8)
  +logq = U32.max(2, T.arg(xs, 4n, 2))
  +ds = U32.add(0, 6)
  Cfg{U32.to_nat(logk), U32.to_nat(U32.shr(U32.inc(logk))), U32.to_nat(logq), U32.to_nat(ds), T.arg(xs, 3n, 256), T.arg(xs, 5n, 2097152), T.arg(xs, 6n, 0), T.arg(xs, 2n, 1), U32.mul(U32.shln(1, U32.to_nat(ds)), 429497), U32.max(1, U32.min(65535, T.arg(xs, 8n, 64))), T.arg(xs, 11n, 4294967295), T.arg(xs, 9n, 4294967295), T.arg(xs, 10n, 4294967295)}

def main.go(r: AS) -> IO(Unit):
  match r:
    case AS{+xs, a, b}:
      +c = cfg.of(xs)
      start(c, U32.max(1, T.arg(xs, 1n, 10)), T.arg(xs, 7n, 1048576), a, b)

def main() -> IO(Unit):
  do IO<Unit>:
    ss : List<String> <- IO.args()
    main.go(args.split(ss, 0))
