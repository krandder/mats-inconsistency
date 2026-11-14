import Base
import ./types.bend as T
# The trader VM: a U32 stack machine over S = 2^ds code bytes, fixed fuel F.
# Spec and opcode table: docs/vm.md. Every instruction costs 1 fuel unit,
# reads cost 1 + READ_COST; an instruction that does not fit in the fuel left
# halts the run (hard stop). The run loop is bounded by a Nat of F steps.

# Constants.
def STACK_CAP() -> U32:
  16

def READ_COST() -> U32:
  3

def EMAX() -> U32:
  8

type Stk is Data:
  Stk{n: U32, xs: List<&2, U32>}

type St is Data:
  St{halt: Bool, pc: U32, stk: Stk, used: U32, ops: U32, dem: List<&2, T.Demand>}

# Stack
def Stk.pop.go(n: U32, xs: List<&2, U32>) -> U32 & Stk:
  match xs:
    case Nil{}:
      (0, Stk{0, Nil{}})
    case Con{h, t}:
      (h, Stk{U32.sub(n, 1), t})

def Stk.pop(s: Stk) -> U32 & Stk:
  match s:
    case Stk{n, xs}:
      Stk.pop.go(n, xs)

def Stk.push.go(full: Bool, x: U32, n: U32, xs: List<&2, U32>) -> Stk:
  match full:
    case True{}:
      Stk{n, xs}
    case False{}:
      Stk{U32.inc(n), x <> xs}

def Stk.push(x: U32, s: Stk) -> Stk:
  match s:
    case Stk{+n, xs}:
      Stk.push.go(U32.is_ge(n, STACK_CAP()), x, n, xs)

def Stk.peek.go(n: U32, xs: List<&2, U32>) -> U32 & Stk:
  match xs:
    case Nil{}:
      (0, Stk{n, Nil{}})
    case Con{+h, t}:
      (h, Stk{n, h <> t})

def Stk.peek(s: Stk) -> U32 & Stk:
  match s:
    case Stk{n, xs}:
      Stk.peek.go(n, xs)

# Signed helpers (two's complement U32)
def slt(a: U32, b: U32) -> Bool:
  U32.is_lt(U32.xor(a, 2147483648), U32.xor(b, 2147483648))

def sar.go(neg: Bool, a: U32, +k: Nat) -> U32:
  match neg:
    case True{}:
      U32.not(U32.shrn(U32.not(a), k))
    case False{}:
      U32.shrn(a, k)

def sar(+a: U32, b: U32) -> U32:
  sar.go(U32.is_ge(a, 2147483648), a, U32.to_nat(U32.and(b, 31)))

# ALU for binary ops: a is below b on the stack; pushes a op b.
def alu(k: Nat, +a: U32, +b: U32) -> U32:
  match k:
    case 0n:
      U32.add(a, b)
    case 1n:
      U32.sub(a, b)
    case 2n:
      U32.mul(a, b)
    case 3n:
      U32.and(a, b)
    case 4n:
      U32.or(a, b)
    case 5n:
      U32.xor(a, b)
    case 6n:
      U32.shln(a, U32.to_nat(U32.and(b, 31)))
    case 7n:
      U32.shrn(a, U32.to_nat(U32.and(b, 31)))
    case 8n:
      sar(a, b)
    case 9n:
      T.b2u(U32.is_lt(a, b))
    case 10n:
      T.b2u(slt(a, b))
    case 11n+p:
      T.b2u(U32.is_eq(a, b))

def bin.go(k: Nat, b: U32, r: U32 & Stk) -> Stk:
  (a, s) = r
  Stk.push(alu(k, a, b), s)

def bin.top(k: Nat, r: U32 & Stk) -> Stk:
  (b, s) = r
  bin.go(k, b, Stk.pop(s))

def bin(k: Nat, s: Stk) -> Stk:
  bin.top(k, Stk.pop(s))

# unary ops: 0 NEG, 1 INC, 2 DEC, 3 NOT
def una.f(k: Nat, a: U32) -> U32:
  match k:
    case 0n:
      U32.sub(0, a)
    case 1n:
      U32.inc(a)
    case 2n:
      U32.sub(a, 1)
    case 3n+p:
      U32.not(a)

def una.go(k: Nat, r: U32 & Stk) -> Stk:
  (a, s) = r
  Stk.push(una.f(k, a), s)

def una(k: Nat, s: Stk) -> Stk:
  una.go(k, Stk.pop(s))

def una.dup(r: U32 & Stk) -> Stk:
  (a, s) = r
  Stk.push(a, s)

def dup(s: Stk) -> Stk:
  una.dup(Stk.peek(s))

def drop.go(r: U32 & Stk) -> Stk:
  (a, s) = r
  s

def drop(s: Stk) -> Stk:
  drop.go(Stk.pop(s))

def swap.b(b: U32, r: U32 & Stk) -> Stk:
  (a, s) = r
  Stk.push(a, Stk.push(b, s))

def swap.a(r: U32 & Stk) -> Stk:
  (b, s) = r
  swap.b(b, Stk.pop(s))

def swap(s: Stk) -> Stk:
  swap.a(Stk.pop(s))

# OVER: a b -> a b a
def over.b(b: U32, r: U32 & Stk) -> Stk:
  (+a, s) = r
  Stk.push(a, Stk.push(b, Stk.push(a, s)))

def over.a(r: U32 & Stk) -> Stk:
  (b, s) = r
  over.b(b, Stk.pop(s))

def over(s: Stk) -> Stk:
  over.a(Stk.pop(s))

# ROT: a b c -> b c a
def rot.c(c: U32, b: U32, r: U32 & Stk) -> Stk:
  (a, s) = r
  Stk.push(a, Stk.push(c, Stk.push(b, s)))

def rot.b(c: U32, r: U32 & Stk) -> Stk:
  (b, s) = r
  rot.c(c, b, Stk.pop(s))

def rot.a(r: U32 & Stk) -> Stk:
  (c, s) = r
  rot.b(c, Stk.pop(s))

def rot(s: Stk) -> Stk:
  rot.a(Stk.pop(s))

# Demands: replace the entry for q, else append if fewer than EMAX entries.
# The VM stack holds a demand as two's complement; EMIT stores sign+magnitude.
def dem.hq(xs: List<&2, T.Demand>) -> U32:
  match xs:
    case Nil{}:
      4294967295
    case Con{T.Demand{q, n, a}, t}:
      q

def dem.new(ok: Bool, d: T.Demand) -> List<&2, T.Demand>:
  match ok:
    case True{}:
      [d]
    case False{}:
      Nil{}

def dem.put.go(xs: List<&2, T.Demand>, hit: Bool, +i: U32, +d: T.Demand) -> List<&2, T.Demand>:
  match xs hit:
    case Nil{} _:
      dem.new(U32.is_lt(i, EMAX()), d)
    case Con{h, t} True{}:
      d <> t
    case Con{h, +t} False{}:
      h <> dem.put.go(t, U32.is_eq(dem.hq(t), T.Demand.q(d)), U32.inc(i), d)

def dem.put(+xs: List<&2, T.Demand>, +d: T.Demand) -> List<&2, T.Demand>:
  +q = T.Demand.q(d)
  dem.put.go(xs, U32.is_eq(dem.hq(xs), q), 0, d)

def mkdem(+q: U32, +v: U32) -> T.Demand:
  +neg = U32.is_ge(v, 2147483648)
  T.Demand{q, neg, T.pick(neg, U32.sub(0, v), v)}

# Opcodes (op = code byte & 31); table in docs/vm.md.
def imm(+code: T.Vec, +env: T.Env, +i: U32) -> U32:
  T.Vec.get(T.Env.ds(env), code, i)

def imm16(+code: T.Vec, +env: T.Env, +pc: U32) -> U32:
  U32.or(imm(code, env, U32.inc(pc)), U32.shln(imm(code, env, U32.add(pc, 2)), 8n))

def qmask(+env: T.Env) -> U32:
  U32.sub(U32.shln(1, T.Env.dq(env)), 1)

def jz.go(z: Bool, t: U32, +pc: U32, s: Stk, u: U32, o: U32, dem: List<&2, T.Demand>) -> St:
  match z:
    case True{}:
      St{False{}, t, s, u, o, dem}
    case False{}:
      St{False{}, U32.add(pc, 2), s, u, o, dem}

def jz(t: U32, +pc: U32, r: U32 & Stk, u: U32, o: U32, dem: List<&2, T.Demand>) -> St:
  (x, s) = r
  jz.go(U32.is_zero(x), t, pc, s, u, o, dem)

# Visible-price radius: question q is visible when its circular distance
# from the trader's home question is at most rad. Hidden reads see price
# 1/2 and an empty history (reads still cost fuel).
def visible(+env: T.Env, +q: U32) -> Bool:
  +m = qmask(env)
  +h = T.Env.home(env)
  +d1 = U32.and(U32.sub(q, h), m)
  +d2 = U32.and(U32.sub(h, q), m)
  U32.is_le(U32.min(d1, d2), T.Env.rad(env))

def rd.price(+env: T.Env, +q: U32) -> U32:
  T.pick(visible(env, q), T.Vec.get(T.Env.dq(env), T.Env.prices(env), q), 32768)

def rd.hist(+env: T.Env, +q: U32) -> U32:
  T.pick(visible(env, q), T.Vec.get(T.Env.dq(env), T.Env.hist(env), q), 0)

def rprice(+env: T.Env, pc: U32, r: U32 & Stk, u: U32, o: U32, dem: List<&2, T.Demand>) -> St:
  (q, s) = r
  St{False{}, U32.inc(pc), Stk.push(rd.price(env, q), s), u, o, dem}

def rhistw(+env: T.Env, pc: U32, r: U32 & Stk, u: U32, o: U32, dem: List<&2, T.Demand>) -> St:
  (q, s) = r
  St{False{}, U32.inc(pc), Stk.push(rd.hist(env, q), s), u, o, dem}

def rhist.q(+env: T.Env, k: U32, pc: U32, r: U32 & Stk, u: U32, o: U32, dem: List<&2, T.Demand>) -> St:
  (q, s) = r
  +w = rd.hist(env, q)
  St{False{}, U32.inc(pc), Stk.push(U32.and(U32.shrn(w, U32.to_nat(U32.and(k, 31))), 1), s), u, o, dem}

def rhist(+env: T.Env, pc: U32, r: U32 & Stk, u: U32, o: U32, dem: List<&2, T.Demand>) -> St:
  (k, s) = r
  rhist.q(env, k, pc, Stk.pop(s), u, o, dem)

def emit.q(+env: T.Env, +d: U32, pc: U32, r: U32 & Stk, u: U32, o: U32, dem: List<&2, T.Demand>) -> St:
  (q, s) = r
  St{False{}, U32.inc(pc), s, u, o, dem.put(dem, mkdem(U32.and(q, qmask(env)), d))}

def emit(+env: T.Env, pc: U32, r: U32 & Stk, u: U32, o: U32, dem: List<&2, T.Demand>) -> St:
  (d, s) = r
  emit.q(env, d, pc, Stk.pop(s), u, o, dem)

def exec(k: Nat, +code: T.Vec, +env: T.Env, +pc: U32, stk: Stk, u: U32, o: U32, dem: List<&2, T.Demand>) -> St:
  match k:
    case 0n:  # NOP
      St{False{}, U32.inc(pc), stk, u, o, dem}
    case 1n:  # PUSH
      St{False{}, U32.add(pc, 2), Stk.push(imm(code, env, U32.inc(pc)), stk), u, o, dem}
    case 2n:  # PUSH16
      St{False{}, U32.add(pc, 3), Stk.push(imm16(code, env, pc), stk), u, o, dem}
    case 3n:  # ADD
      St{False{}, U32.inc(pc), bin(0n, stk), u, o, dem}
    case 4n:  # SUB
      St{False{}, U32.inc(pc), bin(1n, stk), u, o, dem}
    case 5n:  # MUL
      St{False{}, U32.inc(pc), bin(2n, stk), u, o, dem}
    case 6n:  # AND
      St{False{}, U32.inc(pc), bin(3n, stk), u, o, dem}
    case 7n:  # OR
      St{False{}, U32.inc(pc), bin(4n, stk), u, o, dem}
    case 8n:  # XOR
      St{False{}, U32.inc(pc), bin(5n, stk), u, o, dem}
    case 9n:  # SHL
      St{False{}, U32.inc(pc), bin(6n, stk), u, o, dem}
    case 10n:  # SHR
      St{False{}, U32.inc(pc), bin(7n, stk), u, o, dem}
    case 11n:  # SAR
      St{False{}, U32.inc(pc), bin(8n, stk), u, o, dem}
    case 12n:  # LT
      St{False{}, U32.inc(pc), bin(9n, stk), u, o, dem}
    case 13n:  # SLT
      St{False{}, U32.inc(pc), bin(10n, stk), u, o, dem}
    case 14n:  # EQ
      St{False{}, U32.inc(pc), bin(11n, stk), u, o, dem}
    case 15n:  # NEG
      St{False{}, U32.inc(pc), una(0n, stk), u, o, dem}
    case 16n:  # INC
      St{False{}, U32.inc(pc), una(1n, stk), u, o, dem}
    case 17n:  # DEC
      St{False{}, U32.inc(pc), una(2n, stk), u, o, dem}
    case 18n:  # NOT
      St{False{}, U32.inc(pc), una(3n, stk), u, o, dem}
    case 19n:  # DUP
      St{False{}, U32.inc(pc), dup(stk), u, o, dem}
    case 20n:  # DROP
      St{False{}, U32.inc(pc), drop(stk), u, o, dem}
    case 21n:  # SWAP
      St{False{}, U32.inc(pc), swap(stk), u, o, dem}
    case 22n:  # OVER
      St{False{}, U32.inc(pc), over(stk), u, o, dem}
    case 23n:  # ROT
      St{False{}, U32.inc(pc), rot(stk), u, o, dem}
    case 24n:  # JMP
      St{False{}, imm(code, env, U32.inc(pc)), stk, u, o, dem}
    case 25n:  # JZ
      jz(imm(code, env, U32.inc(pc)), pc, Stk.pop(stk), u, o, dem)
    case 26n:  # RPRICE
      rprice(env, pc, Stk.pop(stk), u, o, dem)
    case 27n:  # RHIST
      rhist(env, pc, Stk.pop(stk), u, o, dem)
    case 28n:  # RHISTW
      rhistw(env, pc, Stk.pop(stk), u, o, dem)
    case 29n:  # RWEALTH
      St{False{}, U32.inc(pc), Stk.push(T.Env.wealth(env), stk), u, o, dem}
    case 30n:  # EMIT
      emit(env, pc, Stk.pop(stk), u, o, dem)
    case 31n+p:  # HALT
      St{True{}, pc, stk, u, o, dem}

def cost(+op: U32) -> U32:
  T.pick(U32.is_ge(op, 26) && U32.is_le(op, 29), U32.inc(READ_COST()), 1)

def step.chk(over: Bool, k: Nat, +code: T.Vec, +env: T.Env, pc: U32, stk: Stk, u: U32, nu: U32, o: U32, dem: List<&2, T.Demand>) -> St:
  match over:
    case True{}:
      St{True{}, pc, stk, u, o, dem}
    case False{}:
      exec(k, code, env, pc, stk, nu, U32.inc(o), dem)

# One instruction: halts if its cost does not fit in the fuel left.
def step(+code: T.Vec, +env: T.Env, +pc: U32, stk: Stk, +u: U32, o: U32, dem: List<&2, T.Demand>) -> St:
  +op = U32.and(T.Vec.get(T.Env.ds(env), code, pc), 31)
  +nu = U32.add(u, cost(op))
  step.chk(U32.is_gt(nu, T.Env.fuel(env)), U32.to_nat(op), code, env, pc, stk, u, nu, o, dem)

# The run loop: at most n iterations (n = F; every instruction costs >= 1).
def run(n: Nat, +code: T.Vec, +env: T.Env, st: St) -> St:
  match n st:
    case 0n St{h, pc, stk, u, o, dem}:
      St{h, pc, stk, u, o, dem}
    case 1n+p St{True{}, pc, stk, u, o, dem}:
      St{True{}, pc, stk, u, o, dem}
    case 1n+p St{False{}, pc, stk, u, o, dem}:
      run(p, code, env, step(code, env, pc, stk, u, o, dem))

def St.init() -> St:
  St{False{}, 0, Stk{0, Nil{}}, 0, 0, Nil{}}

# Run one trader program to halt or fuel exhaustion.
def run_one(+code: T.Vec, +env: T.Env) -> St:
  run(U32.to_nat(T.Env.fuel(env)), code, env, St.init())

def St.used(s: St) -> U32:
  match s:
    case St{h, pc, stk, u, o, dem}:
      u

def St.ops(s: St) -> U32:
  match s:
    case St{h, pc, stk, u, o, dem}:
      o

def St.dem(s: St) -> List<&2, T.Demand>:
  match s:
    case St{h, pc, stk, u, o, dem}:
      dem

# Opcode names, for hand-written traders and tests.
def NOP() -> U32:
  0

def PUSH() -> U32:
  1

def PUSH16() -> U32:
  2

def ADD() -> U32:
  3

def SUB() -> U32:
  4

def MUL() -> U32:
  5

def AND() -> U32:
  6

def OR() -> U32:
  7

def XOR() -> U32:
  8

def SHL() -> U32:
  9

def SHR() -> U32:
  10

def SAR() -> U32:
  11

def LT() -> U32:
  12

def SLT() -> U32:
  13

def EQ() -> U32:
  14

def NEG() -> U32:
  15

def INC() -> U32:
  16

def DEC() -> U32:
  17

def NOT() -> U32:
  18

def DUP() -> U32:
  19

def DROP() -> U32:
  20

def SWAP() -> U32:
  21

def OVER() -> U32:
  22

def ROT() -> U32:
  23

def JMP() -> U32:
  24

def JZ() -> U32:
  25

def RPRICE() -> U32:
  26

def RHIST() -> U32:
  27

def RHISTW() -> U32:
  28

def RWEALTH() -> U32:
  29

def EMIT() -> U32:
  30

def HALT() -> U32:
  31


# Population run: one balanced fork per tree node (the cube contract).
type RPop is Data:
  RL{t: T.Trader, dem: List<&2, T.Demand>, ops: U32}
  RN{l: RPop, r: RPop}

def run_trader.fin(code: T.Vec, w: U32, a: U32, st: St) -> RPop:
  match st:
    case St{h, pc, stk, u, o, dem}:
      RL{T.Trader{code, w, u, a}, dem, o}

# home, rad: visible-price radius (rad = 2^32-1: everything visible).
def run_trader.at(t: T.Trader, +pr: T.Vec, +hs: T.Vec, +dq: Nat, +ds: Nat, +f: U32, +home: U32, +rad: U32) -> RPop:
  match t:
    case T.Trader{+code, +w, u, a}:
      run_trader.fin(code, w, a, run_one(code, T.Env{dq, ds, pr, hs, w, f, home, rad}))

def run_trader(t: T.Trader, +pr: T.Vec, +hs: T.Vec, +dq: Nat, +ds: Nat, +f: U32) -> RPop:
  run_trader.at(t, pr, hs, dq, ds, f, 0, 4294967295)

def run_pop(p: T.Pop, +pr: T.Vec, +hs: T.Vec, +dq: Nat, +ds: Nat, +f: U32) -> RPop:
  match p:
    case T.PL{t}:
      run_trader(t, pr, hs, dq, ds, f)
    case T.PN{l, r}:
      a b = run_pop(l, pr, hs, dq, ds, f) run_pop(r, pr, hs, dq, ds, f)
      RN{a, b}
