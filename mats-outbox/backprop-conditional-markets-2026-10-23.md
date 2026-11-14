# Backprop as conditional markets: agreed formalisation (Kelvin + opus-mats, 2026-10-23)

Setting: one priced outcome with logit l (LMSR quantity), price p = sigmoid/softmax(l), outcome y, log loss.

Agreed:
1. Each unit j issues one security: "its output". Its realness in an episode x is the net sensitivity
   r_j(x) = |J_j(x)|, J_j = dl/da_j; sign(J_j) = long/short. Realness 0 = the security is called off.
2. Net, not gross: if J_j = 0, every parameter feeding j gets zero update (dL/dw_ij = a_i f'(z_j) J_j (p - y) = 0),
   so nobody who contributed to j's activation is reinforced. Buyers' own positions (w_jk) settle separately, at their
   own realness J_k.
3. The parameters feeding j trade on j's security: P&L per episode = a_i (exposure) * f'(z_j) * J_j * (p - y) (edge).
4. Deeper parameters reach j only through the security of the unit they feed: J_i = f'(z_i) sum_j w_ij J_j.
5. Parameters are shared across episodes: total P&L = sum over episodes; realness is decided per episode, as a
   conditional contract's condition is.
6. Stationarity of gradient descent, E[J (p - y)] = 0, gives p = E[|J| y] / E[|J|]: the fair price of a conditional
   market with continuous call-off (realness-weighted mean outcome). Exact.

Retracted (by opus-mats, not exact): global-norm clipping as realness, parimutuel, collar, Adam-as-risk-units,
"clipping = leverage cap", "clip at 1 never cuts a legitimate trade". Realness must be sensitivity, not gradient norm:
|dL/da_j| = |J_j| |p - y| mixes realness with the edge.

Open:
- Unit of account: rescaling a_j (with downstream weights compensating) rescales J_j without changing the function,
  so "realness = 1" needs a canonical unit for each unit's output.
- Logit vs price realness (factor p(1-p)); multi-outcome: J_j is a vector, one conditional contract per outcome.
- Infinitesimal only: finite updates change J.
- The research question: can separately trained agents discover J (their buyers' true marginal value) by bidding,
  without the Jacobian? Needs a mechanism that makes bids equal marginal value.
