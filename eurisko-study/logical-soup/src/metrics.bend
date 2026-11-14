import Base
import ./types.bend as T
import ./shadow.bend as S
# Per-epoch metrics (P2.2). Inputs are the epoch's question vectors (2^dq
# slots, blocks of four: A, B, A and B, A or B): qv = the market's forecast
# (post-trade marginal quote), os = outcomes, hs = packed histories the
# traders saw, fv = family codes (0..5 base, 6 AND, 7 OR), lv = 1 when the
# question is live (not held out). Docs: docs/metrics.md.

# fam: 8 families x (Brier sum, log-score sum, count); coherence sums (Q16)
# over fully live blocks and their count; shadow ledger for this epoch.
type MA is Data:
  MA{fam: List<&2, Nat>, ca: Nat, co: Nat, nb: Nat, sh: S.Sh}

def zeros(n: Nat) -> List<&2, Nat>:
  match n:
    case 0n:
      Nil{}
    case 1n+k:
      0n <> zeros(k)

def bump(xs: List<&2, Nat>, i: Nat, x: Nat) -> List<&2, Nat>:
  match xs i:
    case Nil{} _:
      Nil{}
    case Con{h, t} 0n:
      Nat.add(h, x) <> t
    case Con{h, t} 1n+k:
      h <> bump(t, k, x)

def MA.zero() -> MA:
  MA{zeros(24n), 0n, 0n, 0n, S.Sh.zero()}

# (p - o)^2 in Q16^2 and -ln(p of the outcome) in micro-nats (p >= 2^-16).
def sq.brier(+p: U32, +o: U32) -> Nat:
  +d = U32.to_nat(T.pick(U32.is_eq(o, 1), U32.sub(65536, U32.min(p, 65536)), U32.min(p, 65536)))
  Nat.mul(d, d)

def sq.log(+p: U32, +o: U32) -> Nat:
  +x = U32.max(1, T.pick(U32.is_eq(o, 1), U32.min(p, 65536), U32.sub(65536, U32.min(p, 65536))))
  F32.to_nat(F32.max(0.0, F32.mul(F32.sub(11.090354888959125, F32.log(F32.from_nat(U32.to_nat(x)))), 1000000.0)))

def fam.add(live: Bool, fam: List<&2, Nat>, +f: U32, +p: U32, +o: U32) -> List<&2, Nat>:
  match live:
    case False{}:
      fam
    case True{}:
      +i = Nat.mul(3n, U32.to_nat(U32.min(f, 7)))
      bump(bump(bump(fam, i, sq.brier(p, o)), Nat.add(i, 1n), sq.log(p, o)), Nat.add(i, 2n), 1n)

type V4 is Data:
  V4{a: U32, b: U32, c: U32, d: U32}

def v4(+dq: Nat, +v: T.Vec, +i: U32) -> V4:
  V4{T.Vec.get(dq, v, i), T.Vec.get(dq, v, U32.add(i, 1)), T.Vec.get(dq, v, U32.add(i, 2)), T.Vec.get(dq, v, U32.add(i, 3))}

def isone(+x: U32) -> Bool:
  U32.is_eq(x, 1)

def pos(+x: U32, +y: U32) -> Nat:
  U32.to_nat(U32.sub(x, U32.min(x, y)))

def MA.coh(all: Bool, ca: Nat, co: Nat, nb: Nat, +pa: U32, +pb: U32, +pc: U32, +pd: U32) -> MA:
  match all:
    case False{}:
      MA{Nil{}, ca, co, nb, S.Sh.zero()}
    case True{}:
      MA{Nil{}, Nat.add(ca, pos(pc, U32.min(pa, pb))), Nat.add(co, pos(U32.max(pa, pb), pd)), Nat.add(nb, 1n), S.Sh.zero()}

def MA.withfam(m: MA, fam: List<&2, Nat>, sh: S.Sh) -> MA:
  match m:
    case MA{f0, ca, co, nb, s0}:
      MA{fam, ca, co, nb, sh}

# One block: per-question family scores and shadow trades, coherence and
# arbitrage when all four questions are live.
def blk.go(m: MA, p: V4, o: V4, h: V4, f: V4, l: V4) -> MA:
  match m p o h f l:
    case MA{fam, ca, co, nb, sh} V4{+pa, +pb, +pc, +pd} V4{+oa, +ob, +oc, +od} V4{ha, hb, hc, hd} V4{fa, fb, fc, fd} V4{+la, +lb, +lc, +ld}:
      +all = isone(la) && isone(lb) && isone(lc) && isone(ld)
      +f1 = fam.add(isone(la), fam, fa, pa, oa)
      +f2 = fam.add(isone(lb), f1, fb, pb, ob)
      +f3 = fam.add(isone(lc), f2, fc, pc, oc)
      +f4 = fam.add(isone(ld), f3, fd, pd, od)
      +s1 = S.Sh.add(S.Sh.gate(isone(la), S.Sh.question(pa, ha, oa)), S.Sh.gate(isone(lb), S.Sh.question(pb, hb, ob)))
      +s2 = S.Sh.add(S.Sh.gate(isone(lc), S.Sh.question(pc, hc, oc)), S.Sh.gate(isone(ld), S.Sh.question(pd, hd, od)))
      +s3 = S.Sh.add(S.Sh.add(s1, s2), S.Sh.arb(all, S.arb(pa, pb, pc, pd, oa, ob, oc, od)))
      MA.withfam(MA.coh(all, ca, co, nb, pa, pb, pc, pd), f4, S.Sh.add(sh, s3))

def blocks(n: Nat, +j: U32, +dq: Nat, +qv: T.Vec, +os: T.Vec, +hs: T.Vec, +fv: T.Vec, +lv: T.Vec, m: MA) -> MA:
  match n:
    case 0n:
      m
    case 1n+k:
      +i = U32.shln(j, 2n)
      blocks(k, U32.inc(j), dq, qv, os, hs, fv, lv, blk.go(m, v4(dq, qv, i), v4(dq, os, i), v4(dq, hs, i), v4(dq, fv, i), v4(dq, lv, i)))

def questions(+dq: Nat, +qv: T.Vec, +os: T.Vec, +hs: T.Vec, +fv: T.Vec, +lv: T.Vec) -> MA:
  blocks(U32.to_nat(U32.shln(1, Nat.sub(dq, 2n))), 0, dq, qv, os, hs, fv, lv, MA.zero())

def MA.sh(m: MA) -> S.Sh:
  match m:
    case MA{fam, ca, co, nb, sh}:
      sh

# ---- population: wealth Gini, lineages, genotypes, active fraction ----

def ONE() -> U32:
  65536

def pop.ws(p: T.Pop, acc: List<&2, U32>) -> List<&2, U32>:
  match p:
    case T.PL{t}:
      T.Trader.wealth(t) <> acc
    case T.PN{l, r}:
      pop.ws(l, pop.ws(r, acc))

def code.hash(v: T.Vec, +h: U32) -> U32:
  match v:
    case T.VL{x}:
      T.hash2(h, x)
    case T.VN{l, r}:
      code.hash(r, code.hash(l, h))

def keep.if(ok: Bool, +x: U32, acc: List<&2, U32>) -> List<&2, U32>:
  match ok:
    case True{}:
      x <> acc
    case False{}:
      acc

# ancestry hashes (kind True) or genome hashes (False) of active cells
def pop.ids(+k: Bool, p: T.Pop, acc: List<&2, U32>) -> List<&2, U32>:
  match p:
    case T.PL{T.Trader{code, +w, u, a}}:
      keep.if(U32.is_ge(w, ONE()), T.pick(k, a, code.hash(code, 0)), acc)
    case T.PN{l, r}:
      pop.ids(k, l, pop.ids(k, r, acc))

def distinct.go(xs: List<&2, U32>, +prev: U32, +n: Nat) -> Nat:
  match xs:
    case Nil{}:
      n
    case Con{+h, t}:
      distinct.go(t, h, T.pick_nat(U32.is_eq(h, prev), n, Nat.add(n, 1n)))

def distinct(xs: List<&2, U32>) -> Nat:
  match xs:
    case Nil{}:
      0n
    case Con{+h, t}:
      distinct.go(t, h, 1n)

def count(xs: List<&2, U32>) -> Nat:
  match xs:
    case Nil{}:
      0n
    case Con{h, t}:
      1n+count(t)

# Kahan-compensated F32 sums over the ascending wealths (in shares):
# s = sum w, a = sum (2i - n - 1) w for i = 1..n; Gini = a / (n s).
type KS is Data:
  KS{s: F32, cs: F32, a: F32, ca: F32}

def kahan(+x: F32, +sum: F32, +c: F32) -> F32 & F32:
  +y = F32.sub(x, c)
  +t = F32.add(sum, y)
  (t, F32.sub(F32.sub(t, sum), y))

def KS.s2(s2: F32, cs2: F32, r: F32 & F32) -> KS:
  (a2, ca2) = r
  KS{s2, cs2, a2, ca2}

def KS.s1(r: F32 & F32, a: F32, ca: F32, +x: F32) -> KS:
  (s2, cs2) = r
  KS.s2(s2, cs2, kahan(x, a, ca))

def KS.step(k: KS, +w: F32, +coef: F32) -> KS:
  match k:
    case KS{s, cs, a, ca}:
      KS.s1(kahan(w, s, cs), a, ca, F32.mul(coef, w))

def gini.go(xs: List<&2, U32>, +i: F32, +n: F32, k: KS) -> KS:
  match xs:
    case Nil{}:
      k
    case Con{+w, t}:
      +coef = F32.sub(F32.sub(F32.mul(2.0, i), n), 1.0)
      gini.go(t, F32.add(i, 1.0), n, KS.step(k, F32.div(F32.from_nat(U32.to_nat(w)), 65536.0), coef))

def gini.fin(+n: F32, k: KS) -> Nat:
  match k:
    case KS{+s, cs, +a, ca}:
      +g = F32.div(a, F32.max(F32.mul(n, s), 0.000001))
      F32.to_nat(F32.round(F32.mul(F32.max(0.0, F32.min(g, 1.0)), 1000000.0)))

# Gini of all cells' wealth, in parts per million.
def gini(+xs: List<&2, U32>) -> Nat:
  +n = F32.from_nat(count(xs))
  gini.fin(n, gini.go(List.sort(~U32, ~U32.is_le, xs), 1.0, n, KS{0.0, 0.0, 0.0, 0.0}))

# gini_ppm, lineages, genotypes, active_ppm
def pop.cols(+logk: Nat, +p: T.Pop) -> List<&2, String>:
  +ws = pop.ws(p, Nil{})
  +an = List.sort(~U32, ~U32.is_le, pop.ids(True{}, p, Nil{}))
  +gs = List.sort(~U32, ~U32.is_le, pop.ids(False{}, p, Nil{}))
  +act = count(an)
  [Nat.show(gini(ws)), Nat.show(distinct(an)), Nat.show(distinct(gs)), Nat.show(Nat.div(Nat.mul(act, 1000000n), U32.to_nat(U32.shln(1, logk))))]

# ---- CSV ----

def fam.names() -> List<&2, String>:
  ["periodic", "lfsr", "ca30", "ca110", "prime", "pi", "and", "or"]

def fam.hdr(xs: List<&2, String>, +pre: String) -> List<&2, String>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{h, t}:
      String.join([pre, h], "") <> fam.hdr(t, pre)

# mean Brier (ppm) and mean log score (micro-nats) per family; nan if none
def fam.mean(+b: Nat, +l: Nat, +n: Nat) -> String & String:
  match n:
    case 0n:
      ("nan", "nan")
    case 1n+k:
      (Nat.show(Nat.div(Nat.div(b, n), 4295n)), Nat.show(Nat.div(l, n)))

def fam.push(r: String & String, bs: List<&2, String>, ls: List<&2, String>) -> List<&2, String> & List<&2, String>:
  (sb, sl) = r
  (sb <> bs, sl <> ls)

def fam.push2(a: String & String, r: List<&2, String> & List<&2, String>) -> List<&2, String> & List<&2, String>:
  (bs, ls) = r
  fam.push(a, bs, ls)

def fam.fin(r: List<&2, String> & List<&2, String>) -> List<&2, String>:
  (bs, ls) = r
  List.concat(&2, String, [List.reverse(&2, String, bs), List.reverse(&2, String, ls)])

def fam.go(xs: List<&2, Nat>, r: List<&2, String> & List<&2, String>) -> List<&2, String> & List<&2, String>:
  match xs:
    case Con{+b, Con{+l, Con{+n, t}}}:
      fam.go(t, fam.push2(fam.mean(b, l, n), r))
    case _:
      r

def fam.cols(xs: List<&2, Nat>, bs: List<&2, String>, ls: List<&2, String>) -> List<&2, String>:
  fam.fin(fam.go(xs, (bs, ls)))

def coh.ppm(+s: Nat, +n: Nat) -> String:
  match n:
    case 0n:
      "nan"
    case 1n+k:
      Nat.show(Nat.div(Nat.mul(s, 15625n), Nat.mul(n, 1024n)))

# Columns this module adds, after soup's own.
def header() -> List<&2, String>:
  List.concat(&2, String, [fam.hdr(fam.names(), "brier_"), fam.hdr(fam.names(), "log_"), ["coh_and_ppm", "coh_or_ppm"], S.names(), ["gini_ppm", "lineages", "genotypes", "active_ppm", "cum_subsidy", "cum_ops"]])

# m: this epoch's question metrics; cum: cumulative shadow ledger;
# pc: population columns; sub/ops: cumulative subsidy (signed) and ops.
def cols(m: MA, cum: S.Sh, pc: List<&2, String>, sub: S.GL, +ops: Nat) -> List<&2, String>:
  match m:
    case MA{fam, +ca, +co, +nb, sh}:
      List.concat(&2, String, [fam.cols(fam, Nil{}, Nil{}), [coh.ppm(ca, nb), coh.ppm(co, nb)], S.Sh.cols(cum), pc, [S.GL.show(sub), Nat.show(ops)]])
