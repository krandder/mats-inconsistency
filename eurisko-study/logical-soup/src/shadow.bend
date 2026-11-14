import Base
import ./types.bend as T
# Shadow reference traders (P2.1). Hand-written forecasters that read the
# same prices and packed histories as the soup's traders and emit one-share
# demands, but are settled in a separate ledger: they never replicate, never
# die and never move prices. Each trades one whole share per question toward
# its belief at the market's forecast for the event (the post-trade marginal
# quote), paid demand x (outcome - price) in Q16 ticks. Docs: docs/shadow.md.
#
# Traders (ledger slots): c50 constant 0.5; mom momentum (the last outcome
# repeats); p2, p4, p8 pattern-k context detectors (find the most recent
# earlier occurrence of the last k outcomes in the 32-outcome history and
# predict the outcome that followed it; abstain when there is none); arb, a
# coherence arbitrageur over each (A, B, A and B, A or B) block.

# Gains and losses, both >= 0, in Q16 ticks.
type GL is Data:
  GL{g: Nat, l: Nat}

type Sh is Data:
  Sh{c50: GL, mom: GL, p2: GL, p4: GL, p8: GL, arb: GL}

def GL.zero() -> GL:
  GL{0n, 0n}

def GL.add(a: GL, b: GL) -> GL:
  match a b:
    case GL{g1, l1} GL{g2, l2}:
      GL{Nat.add(g1, g2), Nat.add(l1, l2)}

# same net value with min(g, l) removed from both, so long-run cumulative
# sums stay far below the 2^48 Nat limit
def GL.norm.go(lt: Bool, +g: Nat, +l: Nat) -> GL:
  match lt:
    case True{}:
      GL{0n, Nat.sub(l, g)}
    case False{}:
      GL{Nat.sub(g, l), 0n}

def GL.norm(x: GL) -> GL:
  match x:
    case GL{+g, +l}:
      GL.norm.go(Nat.is_lt(g, l), g, l)

def Sh.norm(x: Sh) -> Sh:
  match x:
    case Sh{a, b, c, d, e, f}:
      Sh{GL.norm(a), GL.norm(b), GL.norm(c), GL.norm(d), GL.norm(e), GL.norm(f)}

def Sh.zero() -> Sh:
  Sh{GL.zero(), GL.zero(), GL.zero(), GL.zero(), GL.zero(), GL.zero()}

def Sh.add(a: Sh, b: Sh) -> Sh:
  match a b:
    case Sh{a0, a1, a2, a3, a4, a5} Sh{b0, b1, b2, b3, b4, b5}:
      Sh{GL.add(a0, b0), GL.add(a1, b1), GL.add(a2, b2), GL.add(a3, b3), GL.add(a4, b4), GL.add(a5, b5)}

# Signals: 0 abstain, 1 buy one share, 2 sell one share.
def NOSIG() -> U32:
  4294967295

# One share at price p (Q16) on an event with outcome o (0 or 1).
def pnl(+sig: U32, +p: U32, +o: U32) -> GL:
  +one = U32.is_eq(o, 1)
  +act = Bool.not(U32.is_zero(sig))
  +win = (U32.is_eq(sig, 1) && one) || (U32.is_eq(sig, 2) && Bool.not(one))
  +amt = U32.to_nat(T.pick(one, U32.sub(65536, U32.min(p, 65536)), U32.min(p, 65536)))
  GL{T.pick_nat(act && win, amt, 0n), T.pick_nat(act && Bool.not(win), amt, 0n)}

# Trade toward a predicted bit: buy unless the price is already 1, sell
# unless it is already 0.
def toward(+bit: U32, +p: U32) -> U32:
  T.pick(U32.is_eq(bit, 1), T.pick(U32.is_lt(p, 65536), 1, 0), T.pick(U32.is_gt(p, 0), 2, 0))

def sig.c50(+p: U32) -> U32:
  T.pick(U32.is_lt(p, 32768), 1, T.pick(U32.is_gt(p, 32768), 2, 0))

def sig.mom(+h: U32, +p: U32) -> U32:
  toward(U32.and(h, 1), p)

# Most recent j in 1..32-k with ages j..j+k-1 equal to ages 0..k-1; the
# prediction is the outcome at age j-1 (what followed that occurrence).
def pat.hit(n: Nat, hit: Bool, +h: U32, +ctx: U32, +mask: U32, +j: U32) -> U32:
  match n hit:
    case _ True{}:
      U32.and(U32.shrn(h, U32.to_nat(U32.sub(j, 1))), 1)
    case 0n False{}:
      NOSIG()
    case 1n+r False{}:
      +j2 = U32.inc(j)
      pat.hit(r, U32.is_eq(U32.and(U32.shrn(h, U32.to_nat(j2)), mask), ctx), h, ctx, mask, j2)

def pat.predict(+k: Nat, +h: U32) -> U32:
  +mask = U32.sub(U32.shln(1, k), 1)
  +ctx = U32.and(h, mask)
  pat.hit(Nat.sub(31n, k), U32.is_eq(U32.and(U32.shrn(h, 1n), mask), ctx), h, ctx, mask, 1)

def sig.pat(+k: Nat, +h: U32, +p: U32) -> U32:
  +b = pat.predict(k, h)
  T.pick(U32.is_eq(b, NOSIG()), 0, toward(b, p))

# One live question, every single-question trader.
def Sh.question(+p: U32, +h: U32, +o: U32) -> Sh:
  Sh{pnl(sig.c50(p), p, o), pnl(sig.mom(h, p), p, o), pnl(sig.pat(2n, h, p), p, o), pnl(sig.pat(4n, h, p), p, o), pnl(sig.pat(8n, h, p), p, o), GL.zero()}

def Sh.gate(live: Bool, s: Sh) -> Sh:
  match live:
    case True{}:
      s
    case False{}:
      Sh.zero()

# Coherence arbitrage on one block. If P(A and B) > min(P(A), P(B)): sell
# A and B, buy the cheaper parent (payoff >= the violation whatever happens).
# If max(P(A), P(B)) > P(A or B): buy A or B, sell the dearer parent.
def arb(+pa: U32, +pb: U32, +pc: U32, +pd: U32, +oa: U32, +ob: U32, +oc: U32, +od: U32) -> GL:
  +alow = U32.is_le(pa, pb)
  +pm = T.pick(alow, pa, pb)
  +om = T.pick(alow, oa, ob)
  +pM = T.pick(alow, pb, pa)
  +oM = T.pick(alow, ob, oa)
  +va = U32.is_gt(pc, pm)
  +vo = U32.is_gt(pM, pd)
  +g1 = GL.add(pnl(T.pick(va, 2, 0), pc, oc), pnl(T.pick(va, 1, 0), pm, om))
  GL.add(g1, GL.add(pnl(T.pick(vo, 1, 0), pd, od), pnl(T.pick(vo, 2, 0), pM, oM)))

def Sh.arb(live: Bool, g: GL) -> Sh:
  match live:
    case True{}:
      Sh{GL.zero(), GL.zero(), GL.zero(), GL.zero(), GL.zero(), g}
    case False{}:
      Sh.zero()

def names() -> List<&2, String>:
  ["sh_c50", "sh_mom", "sh_pat2", "sh_pat4", "sh_pat8", "sh_arb"]

# Signed decimal of g - l.
def signed.show.go(neg: Bool, +g: Nat, +l: Nat) -> String:
  match neg:
    case True{}:
      String.join(["-", Nat.show(Nat.sub(l, g))], "")
    case False{}:
      Nat.show(Nat.sub(g, l))

def GL.show(x: GL) -> String:
  match x:
    case GL{+g, +l}:
      signed.show.go(Nat.is_lt(g, l), g, l)

def Sh.cols(s: Sh) -> List<&2, String>:
  match s:
    case Sh{a, b, c, d, e, f}:
      [GL.show(a), GL.show(b), GL.show(c), GL.show(d), GL.show(e), GL.show(f)]
