import Base
import ../src/market.bend as M

def check(ok: Bool) -> U32:
  match ok:
    case True{}:
      0
    case False{}:
      1

def rng(+x: U32) -> U32:
  +a = U32.xor(x, U32.shln(x, 13n))
  +b = U32.xor(a, U32.shrn(a, 17n))
  U32.xor(b, U32.shln(b, 5n))

def equal(a: M.Signed, b: M.Signed) -> Bool:
  match a b:
    case M.Signed{+s, x} M.Signed{+t, y}:
      Bool.and(Bool.or(Bool.and(s, t), Bool.and(Bool.not(s), Bool.not(t))), Nat.is_eq(x, y))

def orders(+n: Nat, +seed: U32) -> List<&2, M.Order>:
  match n:
    case 0n:
      Nil{}
    case 1n+k:
      M.Order{U32.from_nat(k), rng(seed), U32.is_zero(U32.and(seed, 1)), U32.and(U32.shrn(seed, 4n), 131071)} <> orders(k, rng(seed))

def initial(xs: List<&2, M.Order>) -> Nat:
  match xs:
    case Nil{}:
      0n
    case Con{M.Order{id, wealth, negative, amount}, t}:
      Nat.add(U32.to_nat(wealth), initial(t))

def final(xs: List<&2, M.Fill>) -> Nat:
  match xs:
    case Nil{}:
      0n
    case Con{M.Fill{id, wealth, negative, amount, pnl}, t}:
      Nat.add(U32.to_nat(wealth), final(t))

def loss.bounded(s: M.Signed, bound: Nat) -> Bool:
  match s:
    case M.Signed{True{}, n}:
      True{}
    case M.Signed{False{}, n}:
      Nat.is_le(n, bound)

# Independently recover cash changes from the returned balances (not profits).
def balances(os: List<&2, M.Order>, fs: List<&2, M.Fill>) -> U32:
  match os fs:
    case Nil{} Nil{}:
      0
    case Con{M.Order{id, +w, neg, amount}, t} Con{M.Fill{fid, +v, fn, a, pnl}, rest}:
      own = U32.add(check(U32.is_eq(id, fid)), check(equal(M.difference(U32.to_nat(v), U32.to_nat(w)), pnl)))
      U32.add(own, balances(t, rest))
    case os fs:
      1

def verify(+os: List<&2, M.Order>, +b: U32, result: M.Settlement) -> U32:
  match result:
    case M.Invalid{}:
      1
    case M.Settlement{p, q, +fs, +s}:
      ranges = U32.add(check(U32.is_le(p, 65536)), check(U32.is_le(q, 65536)))
      conserved = check(equal(M.difference(final(fs), initial(os)), s))
      bound = check(loss.bounded(s, Nat.mul(U32.to_nat(b), 45427n)))
      U32.add(U32.add(ranges, conserved), U32.add(bound, balances(os, fs)))

def randomized(n: Nat, +seed: U32) -> U32:
  match n:
    case 0n:
      0
    case 1n+k:
      +os = orders(16n, seed)
      +b = U32.add(1, U32.mod(seed, 65535))
      +batch = M.clear(b, os)
      failures = U32.add(verify(os, b, M.settle(batch, False{})), verify(os, b, M.settle(batch, True{})))
      U32.add(failures, randomized(k, rng(seed)))

def budget.tests(n: Nat, +seed: U32) -> U32:
  match n:
    case 0n:
      0
    case 1n+k:
      +w = rng(seed)
      +a = M.budget(w, seed)
      +liability = Nat.mul(U32.to_nat(a), 65536n)
      safe = Bool.and(Nat.is_le(liability, U32.to_nat(w)), Nat.is_le(Nat.add(U32.to_nat(w), liability), 4294967295n))
      limits = Bool.and(U32.is_le(a, seed), U32.is_le(a, 65535))
      U32.add(check(Bool.and(safe, limits)), budget.tests(k, w))

def range(+x: U32, lo: U32, hi: U32) -> Bool:
  Bool.and(U32.is_ge(x, lo), U32.is_le(x, hi))

def examples() -> U32:
  # Independent analytical goldens for b=100, q=100: average=.6201145,
  # marginal=.7310586. One Q16 tick of tolerance for libm/rounding.
  +q = {M.Signed{False{}, 100n} : M.Signed}
  a = check(range(M.execution.price(100, q), 40639, 40641))
  b = check(range(M.marginal.price(100, q), 47910, 47912))
  c = check(U32.is_eq(M.execution.price(100, M.Signed{False{}, 0n}), 32768))
  d = check(U32.is_eq(U32.add(M.execution.price(100, q), M.execution.price(100, M.opposite(q))), 65536))
  e = check(equal(M.cost(100, q), M.difference(6553600n, M.discount(100, 100n))))
  U32.add(U32.add(a, b), U32.add(c, U32.add(d, e)))

def extreme(+amount: U32, +b: U32) -> U32:
  +os = {[M.Order{0, 2147483647, False{}, amount}] : List<&2, M.Order>}
  +ns = {[M.Order{0, 2147483647, True{}, amount}] : List<&2, M.Order>}
  a = verify(os, b, M.settle(M.clear(b, os), True{}))
  c = verify(os, b, M.settle(M.clear(b, os), False{}))
  d = verify(ns, b, M.settle(M.clear(b, ns), True{}))
  e = verify(ns, b, M.settle(M.clear(b, ns), False{}))
  U32.add(U32.add(a, c), U32.add(d, e))

def sweep(+n: Nat) -> U32:
  match n:
    case 0n:
      0
    case 1n+k:
      +d = U32.mul(U32.from_nat(k), 127)
      U32.add(U32.add(extreme(d, 1), extreme(d, 65535)), sweep(k))

def rejected(b: M.Batch) -> U32:
  match b:
    case M.Rejected{}:
      0
    case M.Batch{p, q, s, os}:
      1

def repeat(+n: Nat) -> List<&2, M.Order>:
  match n:
    case 0n:
      Nil{}
    case 1n+k:
      M.Order{U32.from_nat(k), 2147483647, False{}, 4294967295} <> repeat(k)

def edges() -> U32:
  +os = {[M.Order{0, 0, False{}, 4294967295}, M.Order{1, 4294967295, True{}, 4294967295}, M.Order{2, 65535, False{}, 1}] : List<&2, M.Order>}
  a = verify(os, 1, M.settle(M.clear(1, os), True{}))
  b = U32.add(rejected(M.clear(0, Nil{})), rejected(M.clear(65536, Nil{})))
  c = rejected(M.clear(1, repeat(4097n)))
  +maxos = repeat(4096n)
  d = verify(maxos, 1, M.settle(M.clear(1, maxos), True{}))
  +offset = {[M.Order{0, 65536000, False{}, 100}, M.Order{1, 65536000, True{}, 100}] : List<&2, M.Order>}
  e = verify(offset, 100, M.settle(M.clear(100, offset), True{}))
  f = verify(Nil{}, 100, M.settle(M.clear(100, Nil{}), False{}))
  U32.add(U32.add(a, b), U32.add(c, U32.add(d, U32.add(e, f))))

def report(failures: U32, ok: Bool) -> IO(Unit):
  match ok:
    case True{}:
      IO.print("market: PASS (512 random batches x both outcomes, 4096 budgets, 512 extreme cases, bounds and goldens)")
    case False{}:
      IO.die(Unit, 1, "market failures: " ++ U32.show(failures))

def main() -> IO(Unit):
  +n = U32.add(randomized(512n, 42), U32.add(budget.tests(4096n, 7919), U32.add(examples(), U32.add(sweep(256n), edges()))))
  report(n, U32.is_zero(n))
