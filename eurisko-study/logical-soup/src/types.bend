import Base
# Shared types of the logical-induction soup. Documented in docs/types.md.
# Fixed point: a price or probability is a U32 in [0, 65536] (65536 = 1.0).
# A demand is sign + magnitude whole shares (maps to market.bend Order).

# Vec: a complete binary tree of 2^d U32 slots, Data (shareable with +).
# Index bits are read low bit first, so an index wraps modulo 2^d.
type Vec is Data:
  VL{v: U32}
  VN{l: Vec, r: Vec}

# One trader: code is a Vec of S = 2^ds bytes (one byte per slot, 0..255);
# wealth is fuel-denominated; used = fuel units spent in the last run;
# anc = ancestry hash (unchanged on copy, rehashed on each split).
type Trader is Data:
  Trader{code: Vec, wealth: U32, used: U32, anc: U32}

# A population: a balanced tree of traders (2^d leaves), forked in halves.
type Pop is Data:
  PL{t: Trader}
  PN{l: Pop, r: Pop}

# A demand for question q: amt whole shares, sign+magnitude (neg = sell),
# the shape market.bend's Order takes.
type Demand is Data:
  Demand{q: U32, neg: Bool, amt: U32}

# A question as the market sees it (the question module owns the streams).
# hist: last 32 outcomes as bits, bit k = outcome k+1 epochs ago.
type Question is Data:
  Question{id: U32, family: U32, outcome: U32, hist: U32, price: U32}

# What one trader run sees. dq, ds: depths of the question and code Vecs;
# fuel: F, the fixed fuel per run; home, rad: the visible-price radius
# ablation (reads of questions farther than rad from home, circularly, see
# price 1/2 and an empty history; rad = 2^32-1 sees everything).
type Env is Data:
  Env{dq: Nat, ds: Nat, prices: Vec, hist: Vec, wealth: U32, fuel: U32, home: U32, rad: U32}

def b2u(b: Bool) -> U32:
  match b:
    case False{}:
      0
    case True{}:
      1

def pick(c: Bool, x: U32, y: U32) -> U32:
  match c:
    case True{}:
      x
    case False{}:
      y

def prng(+x: U32) -> U32:
  +b = U32.xor(x, U32.shln(x, 13n))
  +d = U32.xor(b, U32.shrn(b, 17n))
  U32.xor(d, U32.shln(d, 5n))

def Vec.new(+d: Nat, +x: U32) -> Vec:
  match d:
    case 0n:
      VL{x}
    case 1n+p:
      VN{Vec.new(p, x), Vec.new(p, x)}

def Vec.side(c: Bool, l: Vec, r: Vec) -> Vec:
  match c:
    case True{}:
      l
    case False{}:
      r

def Vec.get(d: Nat, v: Vec, +i: U32) -> U32:
  match d v:
    case 0n VL{x}:
      x
    case 0n VN{l, r}:
      0
    case 1n+p VL{x}:
      x
    case 1n+p VN{l, r}:
      Vec.get(p, Vec.side(U32.is_zero(U32.and(i, 1)), l, r), U32.shr(i))

def Vec.set.go(d: Nat, v: Vec, b: Bool, +j: U32, +x: U32) -> Vec:
  match d v b:
    case 0n VL{y} _:
      VL{x}
    case 0n VN{l, r} _:
      VL{x}
    case 1n+p VL{y} _:
      VL{x}
    case 1n+p VN{l, r} True{}:
      VN{Vec.set.go(p, l, U32.is_zero(U32.and(j, 1)), U32.shr(j), x), r}
    case 1n+p VN{l, r} False{}:
      VN{l, Vec.set.go(p, r, U32.is_zero(U32.and(j, 1)), U32.shr(j), x)}

def Vec.set(d: Nat, v: Vec, +i: U32, +x: U32) -> Vec:
  Vec.set.go(d, v, U32.is_zero(U32.and(i, 1)), U32.shr(i), x)

# Vec from a list, slots 0.. in order; missing slots are 0.
def Vec.of_list.go(+d: Nat, xs: List<&2, U32>, +i: U32, v: Vec) -> Vec:
  match xs:
    case Nil{}:
      v
    case Con{h, t}:
      Vec.of_list.go(d, t, U32.inc(i), Vec.set(d, v, i, h))

def Vec.of_list(+d: Nat, xs: List<&2, U32>) -> Vec:
  Vec.of_list.go(d, xs, 0, Vec.new(d, 0))

def Vec.sum(v: Vec) -> U32:
  match v:
    case VL{x}:
      x
    case VN{l, r}:
      U32.add(Vec.sum(l), Vec.sum(r))

def Demand.q(d: Demand) -> U32:
  match d:
    case Demand{q, n, a}:
      q

def Env.dq(e: Env) -> Nat:
  match e:
    case Env{dq, ds, prices, hist, wealth, fuel, home, rad}:
      dq

def Env.ds(e: Env) -> Nat:
  match e:
    case Env{dq, ds, prices, hist, wealth, fuel, home, rad}:
      ds

def Env.prices(e: Env) -> Vec:
  match e:
    case Env{dq, ds, prices, hist, wealth, fuel, home, rad}:
      prices

def Env.hist(e: Env) -> Vec:
  match e:
    case Env{dq, ds, prices, hist, wealth, fuel, home, rad}:
      hist

def Env.wealth(e: Env) -> U32:
  match e:
    case Env{dq, ds, prices, hist, wealth, fuel, home, rad}:
      wealth

def Env.fuel(e: Env) -> U32:
  match e:
    case Env{dq, ds, prices, hist, wealth, fuel, home, rad}:
      fuel

def Env.home(e: Env) -> U32:
  match e:
    case Env{dq, ds, prices, hist, wealth, fuel, home, rad}:
      home

def Env.rad(e: Env) -> U32:
  match e:
    case Env{dq, ds, prices, hist, wealth, fuel, home, rad}:
      rad

# Command-line arguments as U32 (4294967295 when malformed), and
# argument i or dflt when missing or malformed.
def args.read(m: Maybe<&2, U32>) -> U32:
  match m:
    case None{}:
      4294967295
    case Some{x}:
      x

def args.u32(xs: List<String>) -> List<&2, U32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      args.read(U32.read(h)) <> args.u32(t)

def arg(xs: List<&2, U32>, i: Nat, +dflt: U32) -> U32:
  match xs i:
    case Nil{} _:
      dflt
    case Con{+h, t} 0n:
      pick(U32.is_eq(h, 4294967295), dflt, h)
    case Con{h, t} 1n+p:
      arg(t, p, dflt)

def Pop.side(c: Bool, l: Pop, r: Pop) -> Pop:
  match c:
    case True{}:
      l
    case False{}:
      r

def Pop.dead() -> Trader:
  Trader{VL{0}, 0, 0, 0}

# Trader at index i (low bit first, like Vec.get).
def Pop.get(d: Nat, p: Pop, +i: U32) -> Trader:
  match d p:
    case _ PL{t}:
      t
    case 0n PN{l, r}:
      Pop.dead()
    case 1n+k PN{l, r}:
      Pop.get(k, Pop.side(U32.is_zero(U32.and(i, 1)), l, r), U32.shr(i))

def Trader.wealth(t: Trader) -> U32:
  match t:
    case Trader{c, w, u, a}:
      w

def sat_add(+a: U32, +b: U32) -> U32:
  +s = U32.add(a, b)
  pick(U32.is_lt(s, a), 4294967295, s)

def hash2(+a: U32, +b: U32) -> U32:
  prng(U32.add(U32.mul(prng(U32.add(a, 2654435769)), 2246822519), b))

def pick_nat(c: Bool, x: Nat, y: Nat) -> Nat:
  match c:
    case True{}:
      x
    case False{}:
      y
