import Base
import ../src/types.bend as T
import ../src/vm.bend as V
# VM throughput bench: 2^LOGN traders run once (balanced fork), fuel F each.
# usage: bun vm_bench.js LOGN F KIND   KIND 0 = random code bytes (seeded),
# 1 = arbitrageur looping forever (every run spends all F), 2 = momentum loop.
# Prints: total instructions executed, total fuel used, demands emitted.

def arb_loop() -> List<&2, U32>:
  [V.PUSH(), 0, V.RPRICE(), V.PUSH(), 2, V.RPRICE(), V.LT(), V.JZ(), 20,
   V.PUSH(), 2, V.PUSH(), 8, V.NEG(), V.EMIT(), V.PUSH(), 0, V.PUSH(), 8, V.EMIT(),
   V.PUSH(), 1, V.RPRICE(), V.PUSH(), 2, V.RPRICE(), V.LT(), V.JZ(), 0,
   V.PUSH(), 2, V.PUSH(), 8, V.NEG(), V.EMIT(), V.PUSH(), 1, V.PUSH(), 8, V.EMIT(), V.JMP(), 0]

def mom_loop() -> List<&2, U32>:
  [V.PUSH(), 0, V.PUSH(), 0, V.RHIST(), V.JZ(), 14, V.PUSH(), 0, V.PUSH(), 10, V.EMIT(), V.JMP(), 0, V.PUSH(), 0, V.PUSH(), 10, V.NEG(), V.EMIT(), V.JMP(), 0]

def rnd(+d: Nat, +h: U32) -> T.Vec:
  match d:
    case 0n:
      T.VL{U32.and(U32.shrn(h, 8n), 255)}
    case 1n+p:
      T.VN{rnd(p, T.hash2(h, 1)), rnd(p, T.hash2(h, 2))}

def code(k: Nat, +h: U32) -> T.Vec:
  match k:
    case 0n:
      rnd(6n, h)
    case 1n:
      T.Vec.of_list(6n, arb_loop())
    case 2n+p:
      T.Vec.of_list(6n, mom_loop())

def pop(+d: Nat, +k: Nat, +h: U32) -> T.Pop:
  match d:
    case 0n:
      T.PL{T.Trader{code(k, h), 1048576, 0, h}}
    case 1n+p:
      a b = pop(p, k, T.hash2(h, 1)) pop(p, k, T.hash2(h, 2))
      T.PN{a, b}

type Tot is Data:
  Tot{ops: Nat, used: Nat, nd: Nat}

def len(xs: List<&2, T.Demand>) -> Nat:
  match xs:
    case Nil{}:
      0n
    case Con{h, t}:
      1n+len(t)

def tot.add(a: Tot, b: Tot) -> Tot:
  match a b:
    case Tot{a1, a2, a3} Tot{b1, b2, b3}:
      Tot{Nat.add(a1, b1), Nat.add(a2, b2), Nat.add(a3, b3)}

def tot(r: V.RPop) -> Tot:
  match r:
    case V.RL{T.Trader{c, w, u, a}, d, o}:
      Tot{U32.to_nat(o), U32.to_nat(u), len(d)}
    case V.RN{a, b}:
      tot.add(tot(a), tot(b))

def show(t: Tot) -> String:
  match t:
    case Tot{o, u, n}:
      "ops=" ++ Nat.show(o) ++ " used=" ++ Nat.show(u) ++ " demands=" ++ Nat.show(n)

def prices() -> T.Vec:
  T.Vec.of_list(2n, [32768, 16384, 40000, 0])

def hists() -> T.Vec:
  T.Vec.of_list(2n, [5, 2, 0, 0])

def main() -> IO(Unit):
  do IO<Unit>:
    ss : List<String> <- IO.args()
    +xs : List<&2, U32> = T.args.u32(ss)
    +p : T.Pop = pop(U32.to_nat(T.arg(xs, 0n, 8)), U32.to_nat(T.arg(xs, 2n, 0)), 12345)
    IO.print(show(tot(V.run_pop(p, prices(), hists(), 2n, 6n, T.arg(xs, 1n, 4096)))))
