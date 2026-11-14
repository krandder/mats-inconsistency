import Base
import ../src/types.bend as T
import ../src/market.bend as M
import ../src/soup.bend as S
import ../src/vm.bend as V

def chk(b: Bool) -> U32:
  match b:
    case True{}:
      0
    case False{}:
      1

def cfg(income: U32, dq: Nat) -> S.Cfg:
  S.Cfg{2n, 1n, dq, 0n, 256, 200000, income, 42, 0, 64, 4294967295, 4294967295, 4294967295}

def trader(w: U32, used: U32) -> T.Trader:
  T.Trader{T.VL{0}, w, used, 1}

def pop(+parent: U32, victim: U32) -> T.Pop:
  T.PN{T.PN{T.PL{trader(parent, 0)}, T.PL{trader(100000, 0)}}, T.PN{T.PL{trader(victim, 0)}, T.PL{trader(100000, 0)}}}

def wealth(p: T.Pop) -> Nat:
  match p:
    case T.PL{t}:
      U32.to_nat(T.Trader.wealth(t))
    case T.PN{l, r}:
      Nat.add(wealth(l), wealth(r))

def rep.verify(+old: Nat, +parent: U32, +victim: U32, result: S.SP) -> U32:
  match result:
    case S.SP{+p, S.Stats{alive, total, maxw, ops, used, nd, births, deaths, vol, gain, loss, +burn}}:
      +actual = wealth(p)
      half = chk(U32.is_eq(S.wof(p, 1, cfg(0, 2n)), U32.div(parent, 2)))
      kept = chk(U32.is_eq(S.wof(p, 0, cfg(0, 2n)), U32.sub(parent, U32.div(parent, 2))))
      cash = chk(Bool.and(Nat.is_eq(Nat.add(actual, burn), old), Nat.is_eq(burn, U32.to_nat(victim))))
      stats = chk(Bool.and(Nat.is_eq(total, actual), Bool.and(Nat.is_eq(births, 1n), Nat.is_eq(deaths, 0n))))
      U32.add(U32.add(half, kept), U32.add(cash, stats))

def replication(+parent: U32, +victim: U32) -> U32:
  +p = pop(parent, victim)
  rep.verify(wealth(p), parent, victim, S.rep(2n, 0, 1, p, cfg(0, 2n), 0))

def settled.wealth(s: S.SP) -> Nat:
  match s:
    case S.SP{p, stats}:
      wealth(p)

def inert() -> U32:
  +c = cfg(0, 0n)
  +r = S.run_pop(T.PL{trader(60000, 200)}, T.VL{32768}, T.VL{0}, c)
  s = S.settle_pop(r, T.VL{32768}, T.VL{0}, c)
  chk(Nat.is_eq(settled.wealth(s), 60000n))

def income() -> U32:
  # Old ordering: income exhausted headroom before a winning payout, wrapping.
  s = S.settle_cell(trader(4294901759, 0), [T.Demand{0, False{}, 1}], 0, T.VL{0}, T.VL{1}, cfg(65536, 0n))
  capped = chk(Nat.is_eq(settled.wealth(s), 4294967295n))
  revived = S.settle_cell(trader(60000, 0), Nil{}, 0, T.VL{32768}, T.VL{0}, cfg(6000, 0n))
  U32.add(capped, chk(Nat.is_eq(settled.wealth(revived), 66000n)))

def natleaf(v: S.NVec) -> Nat:
  match v:
    case S.NZ{depth}:
      0n
    case S.NL{n}:
      n
    case S.NN{l, r}:
      0n

def wide() -> U32:
  # A 2^18-cell population can have >2^32 aggregate shares; never wrap to zero.
  +g = S.Agg.add(S.Agg{S.NL{4294967295n}, S.NL{0n}}, S.Agg{S.NL{1n}, S.NL{0n}})
  p = T.Vec.get(0n, S.px.agg(False{}, 64, g), 0)
  n = natleaf(S.vadd(S.NL{4294967295n}, S.NL{1n}))
  U32.add(chk(Nat.is_eq(n, Nat.add(4294967295n, 1n))), chk(U32.is_eq(p, 65536)))

def batch.price(b: M.Batch, quote: Bool) -> U32:
  match b:
    case M.Rejected{}:
      4294967295
    case M.Batch{p, q, net, orders}:
      T.pick(quote, q, p)

def subsidy(s: M.Settlement) -> M.Signed:
  match s:
    case M.Invalid{}:
      M.Signed{False{}, 4294967295n}
    case M.Settlement{p, q, fs, sub}:
      sub

def same(a: M.Signed, b: M.Signed) -> Bool:
  match a b:
    case M.Signed{+s, x} M.Signed{+t, y}:
      Bool.and(Bool.or(s && t, Bool.not(s) && Bool.not(t)), Nat.is_eq(x, y))

def market.stats(initial: Nat, oracle: M.Signed, s: S.SP) -> U32:
  match s:
    case S.SP{p, S.Stats{alive, total, maxw, ops, +used, nd, births, deaths, vol, +gain, +loss, burn}}:
      actual = wealth(p)
      cash = chk(Nat.is_eq(Nat.add(initial, gain), Nat.add(actual, Nat.add(used, loss))))
      pnl = chk(same(M.difference(gain, loss), oracle))
      bound = chk(Nat.is_le(gain, Nat.add(loss, Nat.mul(128n, 45427n))))
      U32.add(cash, U32.add(pnl, bound))

def market(+outcome: Bool) -> U32:
  +c = cfg(0, 1n)
  +r = {V.RN{V.RL{trader(1048596, 20), [T.Demand{0, False{}, 7}, T.Demand{1, True{}, 4}], 20}, V.RL{trader(524298, 10), [T.Demand{0, True{}, 3}, T.Demand{1, False{}, 10}], 10}} : V.RPop}
  +g = S.agg_pop(r, c)
  +px = S.px.agg(False{}, 64, g)
  +qv = S.px.agg(True{}, 64, g)
  +q0 = M.clear(64, [M.Order{0, 524288, False{}, 7}, M.Order{1, 262144, True{}, 3}])
  +q1 = M.clear(64, [M.Order{0, 524288, True{}, 4}, M.Order{1, 262144, False{}, 10}])
  e0 = chk(U32.is_eq(T.Vec.get(1n, px, 0), batch.price(q0, False{})))
  e1 = chk(U32.is_eq(T.Vec.get(1n, px, 1), batch.price(q1, False{})))
  m0 = chk(U32.is_eq(T.Vec.get(1n, qv, 0), batch.price(q0, True{})))
  m1 = chk(U32.is_eq(T.Vec.get(1n, qv, 1), batch.price(q1, True{})))
  oracle = M.signed.add(subsidy(M.settle(q0, outcome)), subsidy(M.settle(q1, Bool.not(outcome))))
  os = {T.VN{T.VL{T.b2u(outcome)}, T.VL{T.b2u(Bool.not(outcome))}} : T.Vec}
  cash = market.stats(1572894n, oracle, S.settle_pop(r, px, os, c))
  U32.add(U32.add(e0, e1), U32.add(m0, U32.add(m1, cash)))

# Sparse structure scales with emitted orders, not question count.
def nodes(v: S.NVec) -> Nat:
  match v:
    case S.NZ{d}:
      1n
    case S.NL{x}:
      1n
    case S.NN{l, r}:
      Nat.add(1n, Nat.add(nodes(l), nodes(r)))

def agg.nodes(g: S.Agg) -> Nat:
  match g:
    case S.Agg{buy, sell}:
      Nat.add(nodes(buy), nodes(sell))

def sparse() -> U32:
  +c = cfg(0, 12n)
  empty = chk(Nat.is_eq(agg.nodes(S.agg.cell(trader(1048576, 0), Nil{}, c)), 2n))
  zero = chk(Nat.is_eq(agg.nodes(S.agg.cell(trader(0, 0), [T.Demand{4095, False{}, 99}], c)), 2n))
  +one = S.agg.cell(trader(1048576, 0), [T.Demand{4095, False{}, 1}], c)
  small = chk(Nat.is_eq(agg.nodes(one), 26n))
  +other = S.agg.cell(trader(1048576, 0), [T.Demand{2048, True{}, 2}, T.Demand{4095, True{}, 1}], c)
  +merged = S.Agg.add(one, other)
  +prices = S.px.agg(False{}, 64, merged)
  # 4095 cancels, 2048 is a short; every untouched slot stays exactly 1/2.
  cancel = chk(U32.is_eq(T.Vec.get(12n, prices, 4095), 32768))
  sell = chk(U32.is_eq(T.Vec.get(12n, prices, 2048), M.execution.price(64, M.Signed{True{}, 2n})))
  untouched = chk(Bool.and(U32.is_eq(T.Vec.get(12n, prices, 0), 32768), U32.is_eq(T.Vec.get(12n, prices, 2049), 32768)))
  # Empty on either side of a merge is an identity, including marginal quotes.
  +left = S.px.agg(True{}, 64, S.Agg.add(S.agg.cell(trader(0, 0), Nil{}, c), merged))
  +right = S.px.agg(True{}, 64, S.Agg.add(merged, S.agg.cell(trader(0, 0), Nil{}, c)))
  identity = chk(U32.is_eq(T.Vec.get(12n, left, 2048), T.Vec.get(12n, right, 2048)))
  U32.add(U32.add(empty, zero), U32.add(small, U32.add(cancel, U32.add(sell, U32.add(untouched, identity)))))

def report(n: U32, ok: Bool) -> IO(Unit):
  match ok:
    case True{}:
      IO.print("soup: PASS (replication burn, odd split, no repeated idle fuel, income headroom, wide/sparse aggregate, market equivalence/conservation)")
    case False{}:
      IO.die(Unit, 1, "soup FAIL: " ++ U32.show(n))

def main() -> IO(Unit):
  +n = U32.add(replication(400001, 10), U32.add(replication(400000, 0), U32.add(inert(), U32.add(income(), U32.add(wide(), U32.add(market(False{}), market(True{})))))))
  +total = U32.add(n, sparse())
  report(total, U32.is_zero(total))
