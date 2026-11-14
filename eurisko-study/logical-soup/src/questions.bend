import Base

# Deterministic streams. Codes: 0 periodic, 1 LFSR16, 2 CA30, 3 CA110,
# 4 primality, 5 pi digit parity, 6 conjunction, 7 disjunction.
# Every history is a packed ring of 1..32 outcomes, age zero = newest.
type History is Data:
  History{bits: U32, head: U32, count: U32, width: U32}

type Stream is Data:
  Periodic{period: U32, phase: U32}
  Lfsr{state: U32}
  Cellular{rule: U32, row: U32, column: U32}
  Prime{n: U32}
  Digits{index: U32}
  Compound{left: U32, right: U32, conjunction: Bool}

type Question is Data:
  Question{id: U32, family: U32, outcome: U32, history: History, held: Bool, stream: Stream}

type Emission is Data:
  Emission{bit: U32, next: Stream}

def bool.bit(b: Bool) -> U32:
  match b:
    case True{}:
      1
    case False{}:
      0

def history.new(w: U32) -> History:
  History{0, 0, 0, U32.max(1, U32.min(w, 32))}

def history.push(h: History, bit: U32) -> History:
  match h:
    case History{bits, +head, count, +width}:
      +mask = U32.shln(1, U32.to_nat(head))
      clean = U32.and(bits, U32.not(mask))
      inserted = U32.shln(U32.and(bit, 1), U32.to_nat(head))
      History{U32.or(clean, inserted), U32.mod(U32.add(head, 1), width), U32.min(U32.add(count, 1), width), width}

def history.get.go(bits: U32, index: U32, valid: Bool) -> U32:
  match valid:
    case True{}:
      U32.and(U32.shrn(bits, U32.to_nat(index)), 1)
    case False{}:
      0

def history.get(h: History, +age: U32) -> U32:
  match h:
    case History{bits, head, count, +width}:
      # Restrict age before subtraction so even an invalid U32 age cannot wrap.
      index = U32.mod(U32.sub(U32.add(head, width), U32.add(1, U32.mod(age, width))), width)
      history.get.go(bits, index, U32.is_lt(age, count))

def history.count(h: History) -> U32:
  match h:
    case History{bits, head, count, width}:
      count

def bit.at(word: U32, index: U32) -> U32:
  U32.and(U32.shrn(word, U32.to_nat(U32.and(index, 31))), 1)

def lfsr.next(+state: U32) -> U32:
  # x^16+x^14+x^13+x^11+1, right-shifting Galois form.
  tap = U32.mul(U32.and(state, 1), 46080)
  U32.xor(U32.shrn(state, 1n), tap)

# 32 cells, periodic boundary, conventional rule index 4*left+2*center+right.
def ca.row(fuel: Nat, +rule: U32, +row: U32, +column: U32, acc: U32) -> U32:
  match fuel:
    case 0n:
      acc
    case 1n+n:
      left = bit.at(row, U32.and(U32.add(column, 31), 31))
      center = bit.at(row, column)
      right = bit.at(row, U32.and(U32.add(column, 1), 31))
      idx = U32.add(U32.mul(left, 4), U32.add(U32.mul(center, 2), right))
      bit = bit.at(rule, idx)
      next = U32.or(acc, U32.shln(bit, U32.to_nat(column)))
      ca.row(n, rule, row, U32.add(column, 1), next)

def ca.next(+rule: U32, row: U32, column: U32, end: Bool) -> Stream:
  match end:
    case True{}:
      Cellular{rule, ca.row(32n, rule, row, 0, 0), 0}
    case False{}:
      Cellular{rule, row, U32.add(column, 1)}

# n is 16-bit, so testing divisors 2..255 suffices. Both loops have fixed fuel.
def prime.test(fuel: Nat, +n: U32, +d: U32) -> Bool:
  match fuel:
    case 0n:
      True{}
    case 1n+k:
      Bool.or(U32.is_gt(U32.mul(d, d), n), Bool.and(U32.is_ne(U32.mod(n, d), 0), prime.test(k, n, U32.add(d, 1))))

def prime(n: U32) -> U32:
  +v = U32.and(n, 65535)
  bool.bit(Bool.and(U32.is_ge(v, 2), prime.test(254n, v, 2)))

# Pi's first 64 decimal digits, including its leading 3, mapped to digit parity:
# 3141592653589793238462643383279502884197169399375105820974944592307
# Two bit tables, least-significant bit first; the 64-digit table repeats.
def digit.bit(index: U32, upper: Bool) -> U32:
  match upper:
    case False{}:
      bit.at(3942840123, index)
    case True{}:
      bit.at(1703673312, index)

def emit(s: Stream) -> Emission:
  match s:
    case Periodic{+period, +phase}:
      Emission{bool.bit(U32.is_zero(phase)), Periodic{period, U32.mod(U32.add(phase, 1), period)}}
    case Lfsr{+state}:
      Emission{U32.and(state, 1), Lfsr{lfsr.next(state)}}
    case Cellular{+rule, +row, +column}:
      Emission{bit.at(row, column), ca.next(rule, row, column, U32.is_eq(column, 31))}
    case Prime{+n}:
      Emission{prime(n), Prime{U32.and(U32.add(n, 1), 65535)}}
    case Digits{+index}:
      Emission{digit.bit(index, U32.is_ge(index, 32)), Digits{U32.and(U32.add(index, 1), 63)}}
    case Compound{left, right, conjunction}:
      Emission{0, Compound{left, right, conjunction}}

# Constructors leave history empty: the first advance produces the first bit.
def base.stream(family: Nat, +seed: U32, parameter: U32) -> Stream:
  match family:
    case 0n:
      +p = U32.max(2, U32.min(parameter, 16))
      Periodic{p, U32.mod(seed, p)}
    case 1n:
      Lfsr{U32.max(1, U32.and(seed, 65535))}
    case 2n:
      Cellular{30, seed, 0}
    case 3n:
      Cellular{110, seed, 0}
    case 4n:
      Prime{U32.and(seed, 65535)}
    case 5n+n:
      Digits{U32.and(seed, 63)}

def new(id: U32, family: U32, seed: U32, parameter: U32, width: U32, held: Bool) -> Question:
  +f = U32.min(family, 5)
  Question{id, f, 0, history.new(width), held, base.stream(U32.to_nat(f), seed, parameter)}

def new.compound(id: U32, left: U32, right: U32, +conjunction: Bool, width: U32, held: Bool) -> Question:
  family = U32.sub(7, bool.bit(conjunction))
  Question{id, family, 0, history.new(width), held, Compound{left, right, conjunction}}

def held.set(q: Question, held: Bool) -> Question:
  match q:
    case Question{id, family, outcome, history, old, stream}:
      Question{id, family, outcome, history, held, stream}

def record(id: U32, family: U32, h: History, e: Emission) -> Question:
  match e:
    case Emission{+bit, next}:
      Question{id, family, bit, history.push(h, bit), False{}, next}

def advance.active(id: U32, family: U32, outcome: U32, h: History, s: Stream) -> Question:
  match s:
    case Compound{left, right, conjunction}:
      Question{id, family, outcome, h, False{}, Compound{left, right, conjunction}}
    case s:
      record(id, family, h, emit(s))

def advance(q: Question) -> Question:
  match q:
    case Question{id, family, outcome, h, True{}, s}:
      Question{id, family, outcome, h, True{}, s}
    case Question{id, family, outcome, h, False{}, s}:
      advance.active(id, family, outcome, h, s)

def id(q: Question) -> U32:
  match q:
    case Question{id, family, outcome, history, held, stream}:
      id

def family(q: Question) -> U32:
  match q:
    case Question{id, family, outcome, history, held, stream}:
      family

def outcome(q: Question) -> U32:
  match q:
    case Question{id, family, outcome, history, held, stream}:
      outcome

def history(q: Question) -> History:
  match q:
    case Question{id, family, outcome, history, held, stream}:
      history

def available(q: Question) -> Bool:
  match q:
    case Question{id, family, outcome, h, held, stream}:
      Bool.and(Bool.not(held), Bool.and(U32.is_lt(family, 6), U32.is_gt(history.count(h), 0)))

def combine(conjunction: Bool, a: U32, b: U32) -> U32:
  match conjunction:
    case True{}:
      U32.and(a, b)
    case False{}:
      U32.or(a, b)

def compound.ready(id: U32, family: U32, outcome: U32, h: History, left: U32, right: U32, +conjunction: Bool, a: U32, b: U32, ready: Bool) -> Question:
  match ready:
    case False{}:
      Question{id, family, outcome, h, False{}, Compound{left, right, conjunction}}
    case True{}:
      record(id, family, h, Emission{combine(conjunction, a, b), Compound{left, right, conjunction}})

# Call once AFTER both base questions advance, using matching parent ids.
# Held/uninitialized parents or wrong ids leave the compound unchanged.
def compound.update(q: Question, +a: Question, +b: Question) -> Question:
  match q:
    case Question{qid, family, bit, h, True{}, s}:
      Question{qid, family, bit, h, True{}, s}
    case Question{qid, family, bit, h, False{}, Compound{+left, +right, conjunction}}:
      ready = Bool.and(Bool.and(available(a), available(b)), Bool.and(U32.is_eq(left, id(a)), U32.is_eq(right, id(b))))
      compound.ready(qid, family, bit, h, left, right, conjunction, outcome(a), outcome(b), ready)
    case Question{qid, family, bit, h, False{}, s}:
      Question{qid, family, bit, h, False{}, s}

# Export a VM-friendly word: bit k is history age k, zero outside valid history.
# Before advance() this is the past visible to a trader predicting the next bit.
def history.pack.go(fuel: Nat, +h: History, +age: U32, acc: U32) -> U32:
  match fuel:
    case 0n:
      acc
    case 1n+n:
      bit = U32.shln(history.get(h, age), U32.to_nat(age))
      history.pack.go(n, h, U32.add(age, 1), U32.or(acc, bit))

def history.pack(h: History) -> U32:
  history.pack.go(32n, h, 0, 0)
