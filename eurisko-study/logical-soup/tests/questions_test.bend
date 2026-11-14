import Base
import ../src/questions.bend as Q

def check(b: Bool) -> U32:
  match b:
    case True{}:
      0
    case False{}:
      1

def rng(+x: U32) -> U32:
  +a = U32.xor(x, U32.shln(x, 13n))
  +b = U32.xor(a, U32.shrn(a, 17n))
  U32.xor(b, U32.shln(b, 5n))

def histories(a: Q.History, b: Q.History) -> Bool:
  match a b:
    case Q.History{x, h, c, w} Q.History{y, j, d, v}:
      Bool.and(Bool.and(U32.is_eq(x, y), U32.is_eq(h, j)), Bool.and(U32.is_eq(c, d), U32.is_eq(w, v)))

def equal(+a: Q.Question, +b: Q.Question) -> Bool:
  Bool.and(Bool.and(U32.is_eq(Q.id(a), Q.id(b)), U32.is_eq(Q.family(a), Q.family(b))), Bool.and(U32.is_eq(Q.outcome(a), Q.outcome(b)), histories(Q.history(a), Q.history(b))))

def deterministic(n: Nat, a: Q.Question, b: Q.Question) -> U32:
  match n:
    case 0n:
      0
    case 1n+k:
      +x = Q.advance(a)
      +y = Q.advance(b)
      fail = check(Bool.and(equal(x, y), U32.is_le(Q.outcome(x), 1)))
      U32.add(fail, deterministic(k, x, y))

def families(+n: Nat, +seed: U32) -> U32:
  match n:
    case 0n:
      0
    case 1n+k:
      +family = U32.from_nat(k)
      a = Q.new(123, family, seed, 7, 13, False{})
      b = Q.new(123, family, seed, 7, 13, False{})
      U32.add(deterministic(256n, a, b), families(k, rng(seed)))

# Independent history oracle: an ordinary newest-first list, no packed-ring math.
def nth(fuel: Nat, xs: List<&2, U32>, index: Nat) -> U32:
  match fuel xs index:
    case 0n xs index:
      0
    case 1n+k Nil{} index:
      0
    case 1n+k Con{h, t} 0n:
      h
    case 1n+k Con{h, t} 1n+i:
      nth(k, t, i)

def expected(xs: List<&2, U32>, age: Nat, valid: Bool) -> U32:
  match valid:
    case True{}:
      nth(32n, xs, age)
    case False{}:
      0

def ring.compare(+n: Nat, +h: Q.History, +xs: List<&2, U32>, +count: U32) -> U32:
  match n:
    case 0n:
      0
    case 1n+k:
      +age = U32.from_nat(k)
      +e = expected(xs, k, U32.is_lt(age, count))
      packed = Q.bit.at(Q.history.pack(h), age)
      packcheck = Bool.or(U32.is_ge(age, 32), U32.is_eq(packed, e))
      fail = check(Bool.and(U32.is_eq(Q.history.get(h, age), e), packcheck))
      U32.add(fail, ring.compare(k, h, xs, count))

def ring.run(n: Nat, +width: U32, h: Q.History, xs: List<&2, U32>, count: U32, +seed: U32) -> U32:
  match n:
    case 0n:
      0
    case 1n+k:
      +bit = U32.and(seed, 1)
      +hist = Q.history.push(h, bit)
      +list = {bit <> xs : List<&2, U32>}
      +c = U32.min(width, U32.add(count, 1))
      fails = U32.add(ring.compare(34n, hist, list, c), check(U32.is_eq(Q.history.count(hist), c)))
      U32.add(fails, ring.run(k, width, hist, list, c, rng(seed)))

def ring.test(+w: U32) -> U32:
  width = U32.max(1, U32.min(w, 32))
  ring.run(129n, width, Q.history.new(w), Nil{}, 0, 7919)

def steps(n: Nat, q: Q.Question) -> Q.Question:
  match n:
    case 0n:
      q
    case 1n+k:
      steps(k, Q.advance(q))

def held.tests(+n: Nat) -> U32:
  match n:
    case 0n:
      0
    case 1n+k:
      +f = U32.from_nat(k)
      +q = Q.new(0, f, 42, 7, 8, True{})
      +frozen = steps(20n, q)
      nochange = check(equal(q, frozen))
      resumed = Q.held.set(frozen, False{})
      original = Q.new(0, f, 42, 7, 8, False{})
      U32.add(nochange, U32.add(deterministic(128n, resumed, original), held.tests(k)))

def periodic.loop(n: Nat, +p: U32, +phase: U32, q: Q.Question) -> U32:
  match n:
    case 0n:
      0
    case 1n+k:
      +next = Q.advance(q)
      wanted = Q.bool.bit(U32.is_zero(phase))
      fail = check(U32.is_eq(Q.outcome(next), wanted))
      U32.add(fail, periodic.loop(k, p, U32.mod(U32.add(phase, 1), p), next))

def periods(+n: Nat) -> U32:
  match n:
    case 0n:
      0
    case 1n+k:
      +p = U32.add(2, U32.from_nat(k))
      q = Q.new(1, 0, 43, p, 32, False{})
      U32.add(periodic.loop(96n, p, U32.mod(43, p), q), periods(k))

def compound.loop(n: Nat, a: Q.Question, b: Q.Question, c: Q.Question, d: Q.Question) -> U32:
  match n:
    case 0n:
      0
    case 1n+k:
      +x = Q.advance(a)
      +y = Q.advance(b)
      +z = Q.compound.update(c, x, y)
      +w = Q.compound.update(d, x, y)
      fail = check(Bool.and(U32.is_eq(Q.outcome(z), U32.and(Q.outcome(x), Q.outcome(y))), U32.is_eq(Q.outcome(w), U32.or(Q.outcome(x), Q.outcome(y)))))
      hist = check(Bool.and(U32.is_eq(Q.outcome(z), Q.history.get(Q.history(z), 0)), U32.is_eq(Q.outcome(w), Q.history.get(Q.history(w), 0))))
      U32.add(U32.add(fail, hist), compound.loop(k, x, y, z, w))

def compounds() -> U32:
  +a = Q.new(10, 1, 7919, 0, 9, False{})
  +b = Q.new(20, 3, 42, 0, 9, False{})
  +c = Q.new.compound(30, 10, 20, True{}, 9, False{})
  d = Q.new.compound(40, 10, 20, False{}, 9, False{})
  empty = check(equal(c, Q.compound.update(c, a, b)))
  +x = Q.advance(a)
  +y = Q.advance(b)
  wrong = check(equal(c, Q.compound.update(c, y, x)))
  held = check(equal(c, Q.compound.update(c, Q.held.set(x, True{}), y)))
  +hc = Q.held.set(c, True{})
  frozen = check(equal(hc, Q.compound.update(hc, x, y)))
  standalone = check(equal(c, Q.advance(c)))
  resumed = Q.compound.update(Q.held.set(hc, False{}), x, y)
  live = check(U32.is_eq(Q.history.count(Q.history(resumed)), 1))
  U32.add(compound.loop(512n, a, b, c, d), U32.add(empty, U32.add(wrong, U32.add(held, U32.add(frozen, U32.add(standalone, live))))))

# Bits below are offline golden vectors, not generated through the module.
def golden(n: Nat, q: Q.Question, +bits: U32) -> U32:
  match n:
    case 0n:
      0
    case 1n+k:
      +next = Q.advance(q)
      fail = check(U32.is_eq(Q.outcome(next), U32.and(bits, 1)))
      U32.add(fail, golden(k, next, U32.shrn(bits, 1n)))

def lfsr.cycle(+n: Nat, +state: U32, acc: U32) -> U32:
  match n:
    case 0n:
      U32.add(acc, check(U32.is_eq(state, 1)))
    case 1n+k:
      +next = Q.lfsr.next(state)
      premature = Bool.and(U32.is_eq(next, 1), Nat.is_ne(k, 0n))
      fail = check(Bool.and(U32.is_ne(next, 0), Bool.not(premature)))
      lfsr.cycle(k, next, U32.add(acc, fail))

def goldens() -> U32:
  # Primes in 0..31 are 2,3,5,7,11,13,17,19,23,29,31.
  a = golden(32n, Q.new(1, 4, 0, 0, 32, False{}), 2693408940)
  b = golden(32n, Q.new(1, 5, 0, 0, 32, False{}), 3942840123)
  c = golden(32n, Q.new(1, 5, 32, 0, 32, False{}), 1703673312)
  d = golden(32n, steps(32n, Q.new(1, 2, 1, 0, 32, False{})), 2147483651)
  e = golden(32n, steps(32n, Q.new(1, 3, 1, 0, 32, False{})), 2147483649)
  f = check(Bool.and(U32.is_eq(Q.prime(65521), 1), Bool.and(U32.is_eq(Q.prime(65025), 0), U32.is_eq(Q.prime(65535), 0))))
  g = lfsr.cycle(65535n, 1, 0)
  wrap = golden(32n, steps(64n, Q.new(1, 5, 0, 0, 32, False{})), 3942840123)
  U32.add(U32.add(a, b), U32.add(c, U32.add(d, U32.add(e, U32.add(f, U32.add(g, wrap))))) )

def report(failures: U32, ok: Bool) -> IO(Unit):
  match ok:
    case True{}:
      IO.print("questions: PASS (six seeded families, periods 2..16, LFSR full cycle, golden vectors, ring wrap, held-out resume, 512 compounds)")
    case False{}:
      IO.die(Unit, 1, "questions failures: " ++ U32.show(failures))

def main() -> IO(Unit):
  rings = U32.add(ring.test(1), U32.add(ring.test(7), U32.add(ring.test(32), U32.add(ring.test(0), ring.test(99)))))
  streams = U32.add(families(6n, 7919), U32.add(families(6n, 0), U32.add(periods(15n), held.tests(6n))))
  +n = U32.add(rings, U32.add(streams, U32.add(compounds(), goldens())))
  report(n, U32.is_zero(n))
