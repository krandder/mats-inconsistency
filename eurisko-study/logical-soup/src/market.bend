import Base

# Binary LMSR, one fresh binary event per epoch. Q16 cash; whole-share demands.
# Public limits: b in 1..65535; <=4096 orders; amounts <=65535 after budgeting.
# Nat intermediates stay below 2^45. No unsafe recursion or external effects.
type Signed is Data:
  Signed{negative: Bool, magnitude: Nat}

type Order is Data:
  Order{id: U32, wealth: U32, negative: Bool, amount: U32}

type Fill is Data:
  Fill{id: U32, wealth: U32, negative: Bool, amount: U32, profit: Signed}

type Batch is Data:
  Rejected{}
  Batch{price: U32, quote: U32, net: Signed, orders: List<&2, Order>}

type Settlement is Data:
  Invalid{}
  Settlement{price: U32, quote: U32, fills: List<&2, Fill>, subsidy: Signed}

def signed(negative: Bool, magnitude: Nat) -> Signed:
  match magnitude:
    case 0n:
      Signed{False{}, 0n}
    case 1n+n:
      Signed{negative, 1n+n}

def opposite(s: Signed) -> Signed:
  match s:
    case Signed{negative, magnitude}:
      signed(Bool.not(negative), magnitude)

def difference.go(a: Nat, b: Nat, less: Bool) -> Signed:
  match less:
    case True{}:
      signed(True{}, Nat.sub(b, a))
    case False{}:
      signed(False{}, Nat.sub(a, b))

def difference(+a: Nat, +b: Nat) -> Signed:
  difference.go(a, b, Nat.is_lt(a, b))

def signed.add(a: Signed, b: Signed) -> Signed:
  match a b:
    case Signed{False{}, x} Signed{False{}, y}:
      signed(False{}, Nat.add(x, y))
    case Signed{True{}, x} Signed{True{}, y}:
      signed(True{}, Nat.add(x, y))
    case Signed{False{}, x} Signed{True{}, y}:
      difference(x, y)
    case Signed{True{}, x} Signed{False{}, y}:
      difference(y, x)

# Full collateral against ANY clearing price; also reserve gain headroom so
# wealth remains U32 without burning winnings. A caller allocates wealth across
# questions before constructing orders; an order owns its allocation.
def budget(+wealth: U32, amount: U32) -> U32:
  room = U32.min(wealth, U32.sub(4294967295, wealth))
  U32.min(U32.min(amount, 65535), U32.div(room, 65536))

def budget.order(o: Order) -> Order:
  match o:
    case Order{id, +wealth, negative, amount}:
      Order{id, wealth, negative, budget(wealth, amount)}

# Fixed fuel also rejects an oversized batch instead of silently dropping orders.
def fits(fuel: Nat, xs: List<&2, Order>) -> Bool:
  match fuel xs:
    case 0n Nil{}:
      True{}
    case 0n Con{h, t}:
      False{}
    case 1n+n Nil{}:
      True{}
    case 1n+n Con{h, t}:
      fits(n, t)

def budget.all(fuel: Nat, xs: List<&2, Order>) -> List<&2, Order>:
  match fuel xs:
    case 0n xs:
      Nil{}
    case 1n+n Nil{}:
      Nil{}
    case 1n+n Con{h, t}:
      budget.order(h) <> budget.all(n, t)

def net(fuel: Nat, xs: List<&2, Order>) -> Signed:
  match fuel xs:
    case 0n xs:
      Signed{False{}, 0n}
    case 1n+n Nil{}:
      Signed{False{}, 0n}
    case 1n+n Con{Order{id, wealth, negative, amount}, t}:
      signed.add(signed(negative, U32.to_nat(amount)), net(n, t))

# Avoid cancellation near zero: ln(2)-softplus(-x) = x/2-ln(cosh(x/2)).
# Through x^6 for x<=1/4; the omitted term is <3.3e-10 there.
def discount.gap(+x: F32, small: Bool) -> F32:
  match small:
    case True{}:
      +x2 = F32.mul(x, x)
      +x4 = F32.mul(x2, x2)
      F32.sub(F32.add(F32.sub(F32.mul(x, 0.5), F32.div(x2, 8.0)), F32.div(x4, 192.0)), F32.div(F32.mul(x4, x2), 2880.0))
    case False{}:
      F32.max(0.0, F32.sub(0.6931471805599453, F32.log(F32.add(1.0, F32.exp(F32.sub(0.0, x))))))

# Stable LMSR discount: b*[ln(2)-ln(1+exp(-|q|/b))] in Q16.
# F32 is used only for transcendental evaluation. The integer clamp gives an
# unconditional subsidy bound even if a backend's libm rounds differently.
def discount(+b: U32, +magnitude: Nat) -> Nat:
  +bf = F32.from_nat(U32.to_nat(U32.max(1, U32.min(b, 65535))))
  +x = F32.div(F32.from_nat(magnitude), bf)
  gap = discount.gap(x, F32.is_le(x, 0.25))
  raw = F32.to_nat(F32.mul(F32.mul(bf, gap), 65536.0))
  Nat.min(Nat.min(raw, Nat.mul(U32.to_nat(b), 45427n)), Nat.mul(magnitude, 32768n))

# Normalized cost C(q)-C(0), C(q)=b*ln(1+exp(q/b)).
# Negative q gives a negative cash charge. Inputs follow the public limits.
def cost(b: U32, q: Signed) -> Signed:
  match q:
    case Signed{False{}, +magnitude}:
      signed(False{}, Nat.sub(Nat.mul(magnitude, 65536n), discount(b, magnitude)))
    case Signed{True{}, +magnitude}:
      signed(True{}, discount(b, magnitude))

def price.side(negative: Bool, rebate: U32) -> U32:
  match negative:
    case False{}:
      U32.sub(65536, rebate)
    case True{}:
      rebate

# Round the average cost towards the maker: total charged >= normalized cost.
def execution.price(b: U32, q: Signed) -> U32:
  match q:
    case Signed{negative, 0n}:
      32768
    case Signed{negative, 1n+n}:
      +m = Nat.add(1n, n)
      price.side(negative, U32.from_nat(Nat.div(discount(b, m), m)))

def marginal.price(b: U32, q: Signed) -> U32:
  match q:
    case Signed{negative, magnitude}:
      bf = F32.from_nat(U32.to_nat(U32.max(1, U32.min(b, 65535))))
      x = F32.div(F32.from_nat(magnitude), bf)
      tail = F32.div(65536.0, F32.add(1.0, F32.exp(F32.min(x, 80.0))))
      price.side(negative, U32.from_nat(F32.to_nat(tail)))

def clear.go(+b: U32, orders: List<&2, Order>, valid: Bool) -> Batch:
  match valid:
    case False{}:
      Rejected{}
    case True{}:
      +os = budget.all(4096n, orders)
      +q = net(4096n, os)
      Batch{execution.price(b, q), marginal.price(b, q), q, os}

def clear(+b: U32, +orders: List<&2, Order>) -> Batch:
  clear.go(b, orders, Bool.and(Bool.and(U32.is_ge(b, 1), U32.is_le(b, 65535)), fits(4096n, orders)))

def profit(negative: Bool, amount: U32, price: U32, outcome: Bool) -> Signed:
  match negative outcome:
    case False{} False{}:
      signed(True{}, Nat.mul(U32.to_nat(amount), U32.to_nat(price)))
    case False{} True{}:
      signed(False{}, Nat.mul(U32.to_nat(amount), U32.to_nat(U32.sub(65536, price))))
    case True{} False{}:
      signed(False{}, Nat.mul(U32.to_nat(amount), U32.to_nat(price)))
    case True{} True{}:
      signed(True{}, Nat.mul(U32.to_nat(amount), U32.to_nat(U32.sub(65536, price))))

def wealth.after(wealth: U32, pnl: Signed) -> U32:
  match pnl:
    case Signed{False{}, magnitude}:
      U32.from_nat(Nat.add(U32.to_nat(wealth), magnitude))
    case Signed{True{}, magnitude}:
      U32.from_nat(Nat.sub(U32.to_nat(wealth), magnitude))

def settle.order(o: Order, price: U32, outcome: Bool) -> Fill:
  match o:
    case Order{id, wealth, +negative, +amount}:
      +pnl = profit(negative, amount, price, outcome)
      Fill{id, wealth.after(wealth, pnl), negative, amount, pnl}

def settle.orders(fuel: Nat, orders: List<&2, Order>, +price: U32, +outcome: Bool) -> List<&2, Fill>:
  match fuel orders:
    case 0n orders:
      Nil{}
    case 1n+n Nil{}:
      Nil{}
    case 1n+n Con{h, t}:
      settle.order(h, price, outcome) <> settle.orders(n, t, price, outcome)

def subsidy(fuel: Nat, fills: List<&2, Fill>) -> Signed:
  match fuel fills:
    case 0n fills:
      Signed{False{}, 0n}
    case 1n+n Nil{}:
      Signed{False{}, 0n}
    case 1n+n Con{Fill{id, wealth, negative, amount, pnl}, t}:
      signed.add(pnl, subsidy(n, t))

def settle(batch: Batch, outcome: Bool) -> Settlement:
  match batch:
    case Rejected{}:
      Invalid{}
    case Batch{+price, quote, q, orders}:
      +fills = settle.orders(4096n, orders, price, outcome)
      Settlement{price, quote, fills, subsidy(4096n, fills)}
