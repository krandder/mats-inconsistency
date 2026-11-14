import Base
import ./types.bend as T
import ./questions.bend as Q
import ./shadow.bend as S
# Binary snapshot pieces (docs/metrics.md, "Snapshots"). Writers prepend
# bytes onto a tail list; U32s are 4 bytes little-endian, Nats (< 2^48) two
# U32s (low, high), code slots one byte each. Trees (Vec, Pop) are written in
# tree order (left subtree first) and rebuilt by halving, so the shape and
# every index are restored exactly. Readers use take/drop on the byte list.

# ---- writers ----

def ser.u32(+x: U32, tail: List<&2, U32>) -> List<&2, U32>:
  U32.and(x, 255) <> (U32.and(U32.shrn(x, 8n), 255) <> (U32.and(U32.shrn(x, 16n), 255) <> (U32.shrn(x, 24n) <> tail)))

def TWO32() -> Nat:
  Nat.mul(65536n, 65536n)

def ser.nat(+n: Nat, tail: List<&2, U32>) -> List<&2, U32>:
  ser.u32(U32.from_nat(Nat.mod(n, TWO32())), ser.u32(U32.from_nat(Nat.div(n, TWO32())), tail))

def ser.bool(b: Bool, tail: List<&2, U32>) -> List<&2, U32>:
  ser.u32(T.b2u(b), tail)

# one byte per slot (code)
def ser.vec8(v: T.Vec, tail: List<&2, U32>) -> List<&2, U32>:
  match v:
    case T.VL{x}:
      U32.and(x, 255) <> tail
    case T.VN{l, r}:
      ser.vec8(l, ser.vec8(r, tail))

# one U32 per slot
def ser.vec32(v: T.Vec, tail: List<&2, U32>) -> List<&2, U32>:
  match v:
    case T.VL{x}:
      ser.u32(x, tail)
    case T.VN{l, r}:
      ser.vec32(l, ser.vec32(r, tail))

def ser.trader(t: T.Trader, tail: List<&2, U32>) -> List<&2, U32>:
  match t:
    case T.Trader{code, w, u, a}:
      ser.vec8(code, ser.u32(w, ser.u32(u, ser.u32(a, tail))))

def ser.pop(p: T.Pop, tail: List<&2, U32>) -> List<&2, U32>:
  match p:
    case T.PL{t}:
      ser.trader(t, tail)
    case T.PN{l, r}:
      ser.pop(l, ser.pop(r, tail))

# stream: tag, f1, f2, f3
def ser.stream(s: Q.Stream, tail: List<&2, U32>) -> List<&2, U32>:
  match s:
    case Q.Periodic{p, ph}:
      ser.u32(0, ser.u32(p, ser.u32(ph, ser.u32(0, tail))))
    case Q.Lfsr{x}:
      ser.u32(1, ser.u32(x, ser.u32(0, ser.u32(0, tail))))
    case Q.Cellular{r, w, c}:
      ser.u32(2, ser.u32(r, ser.u32(w, ser.u32(c, tail))))
    case Q.Prime{n}:
      ser.u32(3, ser.u32(n, ser.u32(0, ser.u32(0, tail))))
    case Q.Digits{i}:
      ser.u32(4, ser.u32(i, ser.u32(0, ser.u32(0, tail))))
    case Q.Compound{l, r, c}:
      ser.u32(5, ser.u32(l, ser.u32(r, ser.bool(c, tail))))

# 12 U32s: id, family, outcome, bits, head, count, width, held, stream(4)
def ser.question(q: Q.Question, tail: List<&2, U32>) -> List<&2, U32>:
  match q:
    case Q.Question{id, f, o, Q.History{b, h, n, w}, held, s}:
      ser.u32(id, ser.u32(f, ser.u32(o, ser.u32(b, ser.u32(h, ser.u32(n, ser.u32(w, ser.bool(held, ser.stream(s, tail)))))))))

def ser.gl(g: S.GL, tail: List<&2, U32>) -> List<&2, U32>:
  match g:
    case S.GL{a, b}:
      ser.nat(a, ser.nat(b, tail))

def ser.sh(x: S.Sh, tail: List<&2, U32>) -> List<&2, U32>:
  match x:
    case S.Sh{a, b, c, d, e, f}:
      ser.gl(a, ser.gl(b, ser.gl(c, ser.gl(d, ser.gl(e, ser.gl(f, tail))))))

# ---- readers ----
# Each reader returns the value and the unread rest, so the byte list is
# walked once, front to back, with recursion depth bounded by tree depth
# (take/drop over the whole list overflowed the JS stack at 2^12 cells).

type RU is Data:
  RU{x: U32, rest: List<&2, U32>}

type RN is Data:
  RN{x: Nat, rest: List<&2, U32>}

type RV is Data:
  RV{v: T.Vec, rest: List<&2, U32>}

type RT is Data:
  RT{t: T.Trader, rest: List<&2, U32>}

type RP is Data:
  RP{p: T.Pop, rest: List<&2, U32>}

type RW is Data:
  RW{ws: List<&2, U32>, rest: List<&2, U32>}

def TWO32N() -> Nat:
  Nat.mul(65536n, 65536n)

def rd.u32(xs: List<&2, U32>) -> RU:
  match xs:
    case Con{a, Con{b, Con{c, Con{d, t}}}}:
      RU{U32.or(U32.or(a, U32.shln(b, 8n)), U32.or(U32.shln(c, 16n), U32.shln(d, 24n))), t}
    case _:
      RU{0, Nil{}}

def rd.byte(xs: List<&2, U32>) -> RU:
  match xs:
    case Con{a, t}:
      RU{a, t}
    case Nil{}:
      RU{0, Nil{}}

def rd.nat.hi(+lo: U32, r: RU) -> RN:
  match r:
    case RU{hi, t}:
      RN{Nat.add(U32.to_nat(lo), Nat.mul(U32.to_nat(hi), TWO32N())), t}

def rd.nat.lo(r: RU) -> RN:
  match r:
    case RU{lo, t}:
      rd.nat.hi(lo, rd.u32(t))

def rd.nat(xs: List<&2, U32>) -> RN:
  rd.nat.lo(rd.u32(xs))

# n words (small n)
def rd.words.k(+x: U32, r: RW) -> RW:
  match r:
    case RW{ws, t}:
      RW{x <> ws, t}

def rd.words.go(n: Nat, r: RU) -> RW:
  match n r:
    case 0n RU{x, t}:
      RW{[x], t}
    case 1n+k RU{+x, t}:
      rd.words.k(x, rd.words.go(k, rd.u32(t)))

def rd.words(n: Nat, xs: List<&2, U32>) -> RW:
  match n:
    case 0n:
      RW{Nil{}, xs}
    case 1n+k:
      rd.words.go(k, rd.u32(xs))

def rd.leaf(w: Bool, xs: List<&2, U32>) -> RU:
  match w:
    case True{}:
      rd.u32(xs)
    case False{}:
      rd.byte(xs)

def rd.vec.leaf(r: RU) -> RV:
  match r:
    case RU{x, t}:
      RV{T.VL{x}, t}

def rd.vec.join(l: T.Vec, r: RV) -> RV:
  match r:
    case RV{v, t}:
      RV{T.VN{l, v}, t}

# One def reads a whole subtree (fresh = True, ignoring r.v) or, with r.v
# the left subtree just read, reads the right one and joins (fresh = False).
def rd.vec.go(d: Nat, fresh: Bool, +w: Bool, r: RV) -> RV:
  match d fresh r:
    case 0n True{} RV{l, t}:
      rd.vec.leaf(rd.leaf(w, t))
    case 0n False{} RV{l, t}:
      rd.vec.join(l, rd.vec.leaf(rd.leaf(w, t)))
    case 1n+j True{} r:
      +k = j
      rd.vec.go(k, False{}, w, rd.vec.go(k, True{}, w, r))
    case 1n+j False{} RV{l, t}:
      +k = j
      rd.vec.join(l, rd.vec.go(k, False{}, w, rd.vec.go(k, True{}, w, RV{T.VL{0}, t})))

# 2^d slots in tree order; w = True: U32 per slot, else one byte
def rd.vec(+d: Nat, +w: Bool, xs: List<&2, U32>) -> RV:
  rd.vec.go(d, True{}, w, RV{T.VL{0}, xs})

def rd.tr.a(code: T.Vec, +w: U32, +u: U32, r: RU) -> RT:
  match r:
    case RU{a, t}:
      RT{T.Trader{code, w, u, a}, t}

def rd.tr.u(code: T.Vec, +w: U32, r: RU) -> RT:
  match r:
    case RU{+u, t}:
      rd.tr.a(code, w, u, rd.u32(t))

def rd.tr.w(code: T.Vec, r: RU) -> RT:
  match r:
    case RU{+w, t}:
      rd.tr.u(code, w, rd.u32(t))

def rd.tr.code(r: RV) -> RT:
  match r:
    case RV{code, t}:
      rd.tr.w(code, rd.u32(t))

def rd.trader(+ds: Nat, xs: List<&2, U32>) -> RT:
  rd.tr.code(rd.vec(ds, False{}, xs))

def rd.pop.leaf(r: RT) -> RP:
  match r:
    case RT{t, rest}:
      RP{T.PL{t}, rest}

def rd.pop.join(l: T.Pop, r: RP) -> RP:
  match r:
    case RP{p, t}:
      RP{T.PN{l, p}, t}

def rd.pop.go(d: Nat, fresh: Bool, +ds: Nat, r: RP) -> RP:
  match d fresh r:
    case 0n True{} RP{l, t}:
      rd.pop.leaf(rd.trader(ds, t))
    case 0n False{} RP{l, t}:
      rd.pop.join(l, rd.pop.leaf(rd.trader(ds, t)))
    case 1n+j True{} r:
      +k = j
      rd.pop.go(k, False{}, ds, rd.pop.go(k, True{}, ds, r))
    case 1n+j False{} RP{l, t}:
      +k = j
      rd.pop.join(l, rd.pop.go(k, False{}, ds, rd.pop.go(k, True{}, ds, RP{T.PL{T.Pop.dead()}, t})))

# 2^d traders in tree order
def rd.pop(+d: Nat, +ds: Nat, xs: List<&2, U32>) -> RP:
  rd.pop.go(d, True{}, ds, RP{T.PL{T.Pop.dead()}, xs})

def head(xs: List<&2, U32>) -> U32:
  match xs:
    case Nil{}:
      0
    case Con{h, t}:
      h

def drop(n: Nat, xs: List<&2, U32>) -> List<&2, U32>:
  match n xs:
    case 0n xs:
      xs
    case 1n+k Nil{}:
      Nil{}
    case 1n+k Con{h, t}:
      drop(k, t)

def nth(n: Nat, xs: List<&2, U32>) -> U32:
  head(drop(n, xs))

def natn(n: Nat, xs: List<&2, Nat>) -> Nat:
  match n xs:
    case _ Nil{}:
      0n
    case 0n Con{h, t}:
      h
    case 1n+k Con{h, t}:
      natn(k, t)

type RNs is Data:
  RNs{ns: List<&2, Nat>, rest: List<&2, U32>}

def rd.nats.cons(+x: Nat, r: RNs) -> RNs:
  match r:
    case RNs{ns, t}:
      RNs{x <> ns, t}

def rd.nats.go(n: Nat, r: RN) -> RNs:
  match n r:
    case 0n RN{x, t}:
      RNs{[x], t}
    case 1n+k RN{+x, t}:
      rd.nats.cons(x, rd.nats.go(k, rd.nat(t)))

def rd.nats(n: Nat, xs: List<&2, U32>) -> RNs:
  match n:
    case 0n:
      RNs{Nil{}, xs}
    case 1n+k:
      rd.nats.go(k, rd.nat(xs))

def gl.of(+i: Nat, +ns: List<&2, Nat>) -> S.GL:
  S.GL{natn(i, ns), natn(Nat.add(i, 1n), ns)}

# 12 Nats: six (gain, loss) pairs in Sh order
def sh.of(+ns: List<&2, Nat>) -> S.Sh:
  S.Sh{gl.of(0n, ns), gl.of(2n, ns), gl.of(4n, ns), gl.of(6n, ns), gl.of(8n, ns), gl.of(10n, ns)}

def stream.of(tag: Nat, +a: U32, +b: U32, +c: U32) -> Q.Stream:
  match tag:
    case 0n:
      Q.Periodic{a, b}
    case 1n:
      Q.Lfsr{a}
    case 2n:
      Q.Cellular{a, b, c}
    case 3n:
      Q.Prime{a}
    case 4n:
      Q.Digits{a}
    case 5n+k:
      Q.Compound{a, b, U32.is_eq(c, 1)}

# from its 12 words
def question.of(+ws: List<&2, U32>) -> Q.Question:
  Q.Question{nth(0n, ws), nth(1n, ws), nth(2n, ws), Q.History{nth(3n, ws), nth(4n, ws), nth(5n, ws), nth(6n, ws)}, U32.is_eq(nth(7n, ws), 1), stream.of(U32.to_nat(U32.min(nth(8n, ws), 5)), nth(9n, ws), nth(10n, ws), nth(11n, ws))}

type RQ is Data:
  RQ{q: Q.Question, rest: List<&2, U32>}

def rd.question.go(r: RW) -> RQ:
  match r:
    case RW{ws, t}:
      RQ{question.of(ws), t}

def rd.question(xs: List<&2, U32>) -> RQ:
  rd.question.go(rd.words(12n, xs))

# ---- file IO ----

law wr.go:
  for w: File & Result<&1, &1, (U32 & String), Unit>
  IO(Unit)

def wr.go(w):
  (f, r) = w
  do IO<Unit>:
    u : Unit <- IO.pass(Unit, r)
    File.close(f)

def write(path: String, bytes: List<&2, U32>) -> IO(Unit):
  do IO<Unit>:
    f : File <- IO.try(File, File.open(path, "w"))
    w : File & Result<&1, &1, (U32 & String), Unit> <- File.write_bytes(f, bytes)
    wr.go(w)

law rd.go:
  for m: File & Result<&1, &1, (U32 & String), List<&2, U32>>
  IO(List<&2, U32>)

def rd.go(m):
  (f, r) = m
  do IO<List<&2, U32>>:
    xs : List<&2, U32> <- IO.pass(List<&2, U32>, r)
    u : Unit <- File.close(f)
    return xs

law rd.size:
  for m: File & Result<&1, &1, (U32 & String), U32>
  IO(List<&2, U32>)

def rd.size(m):
  (f, r) = m
  do IO<List<&2, U32>>:
    n : U32 <- IO.pass(U32, r)
    g : File & Result<&1, &1, (U32 & String), List<&2, U32>> <- File.read_bytes(f, n)
    rd.go(g)

def read(path: String) -> IO(List<&2, U32>):
  do IO<List<&2, U32>>:
    f : File <- IO.try(File, File.open(path, "r"))
    s : File & Result<&1, &1, (U32 & String), U32> <- File.size(f)
    rd.size(s)
