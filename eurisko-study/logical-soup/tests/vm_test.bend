import Base
import ../src/types.bend as T
import ../src/vm.bend as V
# VM unit tests with hand-written traders. Questions: q0 = A, q1 = B,
# q2 = A and B, q3 = spare (dq = 2). Code: 64 bytes (ds = 6).

def prog(xs: List<&2, U32>) -> T.Vec:
  T.Vec.of_list(6n, xs)

def env(pr: List<&2, U32>, hs: List<&2, U32>, w: U32, f: U32) -> T.Env:
  T.Env{2n, 6n, T.Vec.of_list(2n, pr), T.Vec.of_list(2n, hs), w, f, 0, 4294967295}

def run(xs: List<&2, U32>, e: T.Env) -> V.St:
  V.run_one(prog(xs), e)

# demands flattened to [q, neg, amt, ...]
def flat(xs: List<&2, T.Demand>) -> List<&2, U32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{T.Demand{q, n, a}, t}:
      q <> (T.b2u(n) <> (a <> flat(t)))

def leq(xs: List<&2, U32>, ys: List<&2, U32>) -> Bool:
  match xs ys:
    case Nil{} Nil{}:
      True{}
    case Nil{} Con{y, yt}:
      False{}
    case Con{x, xt} Nil{}:
      False{}
    case Con{x, xt} Con{y, yt}:
      U32.is_eq(x, y) && leq(xt, yt)

def len(xs: List<&2, T.Demand>) -> U32:
  match xs:
    case Nil{}:
      0
    case Con{h, t}:
      U32.inc(len(t))

def dem_is(s: V.St, want: List<&2, U32>) -> Bool:
  leq(flat(V.St.dem(s)), want)

def res(name: String, ok: Bool) -> String:
  match ok:
    case True{}:
      "ok   " ++ name
    case False{}:
      "FAIL " ++ name

def stk_n(s: V.St) -> U32:
  match s:
    case V.St{h, pc, V.Stk{n, xs}, u, o, dem}:
      n

def top.go(xs: List<&2, U32>) -> U32:
  match xs:
    case Nil{}:
      0
    case Con{h, t}:
      h

def top(s: V.St) -> U32:
  match s:
    case V.St{h, pc, V.Stk{n, xs}, u, o, dem}:
      top.go(xs)

def prices() -> List<&2, U32>:
  [32768, 16384, 40000, 0]

def hists() -> List<&2, U32>:
  [5, 2, 0, 0]

# constant: buy 5 of q0, sell 3 of q1, halt. 7 instructions, fuel 7.
def t_constant() -> Bool:
  +s = run([V.PUSH(), 0, V.PUSH(), 5, V.EMIT(), V.PUSH(), 1, V.PUSH(), 3, V.NEG(), V.EMIT(), V.HALT()], env(prices(), hists(), 100, 64))
  dem_is(s, [0, 0, 5, 1, 1, 3]) && U32.is_eq(V.St.used(s), 8) && U32.is_eq(V.St.ops(s), 8)

# momentum on q: last outcome 1 -> buy 10, else sell 10. Layout:
# 0 PUSH q 2 PUSH 0 4 RHIST 5 JZ 12 7 PUSH q 9 PUSH 10 11 EMIT... (see below)
def momentum(q: U32) -> List<&2, U32>:
  +qq = q
  [V.PUSH(), qq, V.PUSH(), 0, V.RHIST(), V.JZ(), 14, V.PUSH(), qq, V.PUSH(), 10, V.EMIT(), V.HALT(), V.NOP(), V.PUSH(), qq, V.PUSH(), 10, V.NEG(), V.EMIT(), V.HALT()]

def t_momentum() -> Bool:
  +a = run(momentum(0), env(prices(), hists(), 100, 64))
  +b = run(momentum(1), env(prices(), hists(), 100, 64))
  dem_is(a, [0, 0, 10]) && dem_is(b, [1, 1, 10])

# pattern-k detector: if the last 3 outcomes of q are 1,0,1 (hist & 7 == 5)
# buy 20 of q; else emit nothing.
def pattern(q: U32) -> List<&2, U32>:
  +qq = q
  [V.PUSH(), qq, V.RHISTW(), V.PUSH(), 7, V.AND(), V.PUSH(), 5, V.EQ(), V.JZ(), 16, V.PUSH(), qq, V.PUSH(), 20, V.EMIT(), V.HALT()]

def t_pattern() -> Bool:
  +a = run(pattern(0), env(prices(), hists(), 100, 64))
  +b = run(pattern(1), env(prices(), hists(), 100, 64))
  dem_is(a, [0, 0, 20]) && dem_is(b, Nil{})

# arbitrageur over (A=q0, B=q1, C=A and B=q2): if P(C) > P(A) sell 8 C, buy 8 A;
# then if P(C) > P(B) sell 8 C, buy 8 B. Prices 0.5, 0.25, 0.61: both fire.
def arb() -> List<&2, U32>:
  [V.PUSH(), 0, V.RPRICE(), V.PUSH(), 2, V.RPRICE(), V.LT(), V.JZ(), 20,
   V.PUSH(), 2, V.PUSH(), 8, V.NEG(), V.EMIT(), V.PUSH(), 0, V.PUSH(), 8, V.EMIT(),
   V.PUSH(), 1, V.RPRICE(), V.PUSH(), 2, V.RPRICE(), V.LT(), V.JZ(), 42,
   V.PUSH(), 2, V.PUSH(), 8, V.NEG(), V.EMIT(), V.PUSH(), 1, V.PUSH(), 8, V.EMIT(), V.NOP(), V.NOP(),
   V.HALT()]

def t_arb() -> Bool:
  +a = run(arb(), env(prices(), hists(), 100, 256))
  +b = run(arb(), env([32768, 32768, 16384, 0], hists(), 100, 256))
  dem_is(a, [2, 1, 8, 0, 0, 8, 1, 0, 8]) && dem_is(b, Nil{})

# hard stop: PUSH 0, RPRICE, DROP, JMP 0 costs 1+4+1+1 = 7 per loop. With
# F = 100: 14 loops (98) + PUSH (99), then RPRICE (4) does not fit.
def t_fuel() -> Bool:
  +s = run([V.PUSH(), 0, V.RPRICE(), V.DROP(), V.JMP(), 0], env(prices(), hists(), 100, 100))
  U32.is_eq(V.St.used(s), 99) && U32.is_eq(V.St.ops(s), 57)

# a program with no HALT runs until the fuel is gone, pc wrapping at S.
def t_wrap() -> Bool:
  +s = run([V.NOP()], env(prices(), hists(), 100, 1000))
  U32.is_eq(V.St.used(s), 1000) && U32.is_eq(V.St.ops(s), 1000)

# later emits for the same question overwrite; at most EMAX questions.
def t_overwrite() -> Bool:
  +s = run([V.PUSH(), 0, V.PUSH(), 5, V.EMIT(), V.PUSH(), 1, V.PUSH(), 6, V.EMIT(), V.PUSH(), 0, V.PUSH(), 7, V.EMIT(), V.HALT()], env(prices(), hists(), 100, 64))
  dem_is(s, [0, 0, 7, 1, 0, 6])

# q is masked to the question count (dq = 2 -> q mod 4)
def t_qmask() -> Bool:
  +s = run([V.PUSH(), 6, V.PUSH(), 1, V.EMIT(), V.HALT()], env(prices(), hists(), 100, 64))
  dem_is(s, [2, 0, 1])

# EMAX = 8 distinct questions (dq = 4 here), further new ones dropped.
def env16() -> T.Env:
  T.Env{4n, 6n, T.Vec.new(4n, 0), T.Vec.new(4n, 0), 100, 400, 0, 4294967295}

def t_emax() -> Bool:
  +s = V.run_one(prog([V.DUP(), V.PUSH(), 1, V.EMIT(), V.INC(), V.JMP(), 0]), env16())
  U32.is_eq(len(V.St.dem(s)), 8)

# stack is capped at 16; pops from an empty stack read 0.
def t_stack() -> Bool:
  +s = run([V.PUSH(), 1, V.JMP(), 0], env(prices(), hists(), 100, 40))
  +z = run([V.ADD(), V.ADD(), V.HALT()], env(prices(), hists(), 100, 40))
  U32.is_eq(stk_n(s), 16) && U32.is_eq(stk_n(z), 1) && U32.is_eq(top(z), 0)

# wealth read, arithmetic, compare, PUSH16, signed ops
def t_alu() -> Bool:
  +a = run([V.RWEALTH(), V.PUSH(), 3, V.MUL(), V.HALT()], env(prices(), hists(), 100, 64))
  +b = run([V.PUSH16(), 52, 18, V.HALT()], env(prices(), hists(), 100, 64))
  +c = run([V.PUSH(), 5, V.NEG(), V.PUSH(), 1, V.SAR(), V.HALT()], env(prices(), hists(), 100, 64))
  +d = run([V.PUSH(), 5, V.NEG(), V.PUSH(), 1, V.SLT(), V.HALT()], env(prices(), hists(), 100, 64))
  +e = run([V.PUSH(), 1, V.PUSH(), 2, V.PUSH(), 3, V.ROT(), V.HALT()], env(prices(), hists(), 100, 64))
  U32.is_eq(top(a), 300) && U32.is_eq(top(b), 4660) && U32.is_eq(top(c), 4294967293) && U32.is_eq(top(d), 1) && U32.is_eq(top(e), 1)

# population: 4 traders in a balanced tree, total ops is the sum.
def rsum(r: V.RPop) -> U32:
  match r:
    case V.RL{t, d, o}:
      o
    case V.RN{a, b}:
      U32.add(rsum(a), rsum(b))

def pop4(+c: T.Trader) -> T.Pop:
  T.PN{T.PN{T.PL{c}, T.PL{c}}, T.PN{T.PL{c}, T.PL{c}}}

def t_pop() -> Bool:
  U32.is_eq(rsum(V.run_pop(pop4(T.Trader{prog([V.NOP()]), 10, 0, 1}), T.Vec.new(2n, 0), T.Vec.new(2n, 0), 2n, 6n, 50)), 200)

# visible-price radius: home 0, rad 1 over 4 questions sees q3, q0, q1 only
def renv(+home: U32) -> T.Env:
  T.Env{2n, 6n, T.Vec.of_list(2n, [100, 200, 300, 400]), T.Vec.of_list(2n, [1, 2, 3, 4]), 100, 64, home, 1}

def t_radius() -> Bool:
  +e = renv(0)
  U32.is_eq(V.rd.price(e, 1), 200) && U32.is_eq(V.rd.price(e, 3), 400) && U32.is_eq(V.rd.price(e, 2), 32768) && U32.is_eq(V.rd.hist(e, 2), 0) && U32.is_eq(V.rd.hist(e, 3), 4) && U32.is_eq(V.rd.price(renv(2), 0), 32768) && U32.is_eq(V.rd.price(renv(2), 3), 400)

def summary(ok: Bool) -> String:
  match ok:
    case True{}:
      "vm: PASS (13 tests: constant, momentum, pattern-k, arbitrageur, fuel stop, wrap, overwrite, qmask, emax, stack, alu, population, radius)"
    case False{}:
      "vm: FAIL"

def main() -> IO(Unit):
  do IO<Unit>:
    IO.print(res("constant", t_constant()))
    IO.print(res("momentum", t_momentum()))
    IO.print(res("pattern-k", t_pattern()))
    IO.print(res("arbitrageur", t_arb()))
    IO.print(res("fuel hard stop", t_fuel()))
    IO.print(res("pc wrap", t_wrap()))
    IO.print(res("overwrite", t_overwrite()))
    IO.print(res("q mask", t_qmask()))
    IO.print(res("emax", t_emax()))
    IO.print(res("stack cap", t_stack()))
    IO.print(res("alu", t_alu()))
    IO.print(res("population", t_pop()))
    IO.print(res("radius", t_radius()))
    IO.print(summary(t_constant() && t_momentum() && t_pattern() && t_arb() && t_fuel() && t_wrap() && t_overwrite() && t_qmask() && t_emax() && t_stack() && t_alu() && t_pop() && t_radius()))
