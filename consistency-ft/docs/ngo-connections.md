# Ngo's recent research program × consistency-ft

Deep-research synthesis, 2026-08-19. Sources verified per item; epistemic
status tagged [solid] (follows from the texts), [suggestive], [speculative].
Ngo's research home is now mindthefuture.info ("Mind the Future").

## The corpus in one paragraph

Ngo's post-OpenAI program: a *scale-free* theory of agency (Mar 2025) in
which any agent — subagent, human, firm, civilization — is a **coalition**
whose beliefs are aggregated by an internal *prediction market* (formalized
by Garrabrant induction), whose actions are allocated by *auctions*
(Demski's BRIA), and whose values are aggregated by *random-ballot voting*.
"Agents as webs of beliefs" (Jun 2026) adds the dynamics: beliefs are
locally-but-not-globally consistent webs (PDGs + Garrabrant induction);
**actions are the beliefs that holding makes true** (FixDT); **identity is a
chosen self-fulfilling belief** ("I'm the kind of agent who…"); goals are
**drives** (upward pressure on credences) balanced by **anchors** (empirical
pinning), with rationality the limit of infinitesimal drives. "Towards a
Formal Scientific Epistemology" (Jun 2026) recants his earlier dismissal of
Garrabrant induction and names four gaps (old evidence; wealth-concentration
in model-driven traders; incremental theory modification; wealth is
rivalrous, truth isn't).

## Connection 1 — RLEF is amortized Garrabrant induction [solid]

The logical induction criterion is *exactly* our objective, idealized: "no
efficiently computable trader exploits the market" = no bounded-risk
Dutch book with unbounded profit (LI Def 3.0.1/3.5.1). Our RLEF reward
(−Exploit(c,S), the LP adversary's guaranteed profit) trains against a
restricted trader class: single-shot, linear stakes, one tuple, no
price-history strategies. So:

- A model at LI's fixed point scores 0 on our whole benchmark. LI's
  **Affine Coherence** (4.2.10) — timely satisfaction of efficiently
  enumerable *linear* constraints — is literally the Paleka check battery
  (every check is an affine constraint on prices).
- LI reaches coherence by running a market *in time*; we distill the fixed
  point into weights — amortization. Ngo's epistemology post treats trader
  wealth as degree-of-truth of models; in amortized form, "which circuits
  the coherence gradient reinforces" is the analog of wealth concentration
  (his gap #2). [speculative but testable with probes]
- Roadmap it implies: strengthen the adversary class stepwise toward LI's —
  traders with memory across rounds, conditional strategies, cross-tuple
  portfolios. Each strengthening is a new check family with a principled
  pedigree rather than an ad-hoc addition.

## Connection 2 — Coalitional agency predicts what incoherence is [solid]

In Ngo's architecture a composite agent's beliefs ARE market prices; an
incoherent price vector is a market with no arbitrageur. Our gpt-oss-20b
data is a textbook specimen: P and ¬P both priced ~0.10 (buy both sides for
$0.18, collect $1) — the "weird claim discount" applied to both sides of a
partition is exactly what an unarbitraged bag of subagent heuristics looks
like. RLEF = installing the arbitrageur in the weights. Because his theory
is scale-free, the same objective applies unchanged to multi-agent stacks —
and the "Locally Coherent, Globally Incoherent" paper (2605.30335, from the
Paleka citation sweep) already measures the multi-component version.
Positioning: consistency-ft is the single-agent base case of coalitional
belief aggregation done right.

## Connection 3 — Self-fulfilling beliefs are a hazard class for our
training data [solid warning, from his texts + performativity literature]

Webs-of-beliefs' central move (via Demski's FixDT): actions are beliefs an
actuator makes true; identity is a self-fulfilling belief adopted by choice,
not Bayesian update. Consequence for us: propositions about the model's own
outputs/behavior are *performative* — their truth depends on the stated
probability. For those, coherence pressure does fixed-point *selection*,
and the performativity literature says stable fixed points ≠ optimal ones
(Perdomo et al. 2020; Oesterheld: scoring-rule fixed points not optimal;
Hubinger, Conditioning Predictive Models: fixed-point selection is a safety
problem). Design rule for consistency-ft v0: tuples are world-propositions
only; anything self-referential ("what will you answer…", "what probability
do you assign…") is quarantined as a separate regime, not mixed into
training. [Also an LI-internal warning: Self-Trust theorems (4.3.12) hold
only with *soft* threshold indicators — sharp conditionals break on
self-referential sentences. Reward shaping should avoid hard cutoffs.]

Flip side, as capability upside: Ngo's identity mechanism (stable
self-model → reliable cooperation with your own future self) is basically
our HYPOTHESES.md items 1–2 (anti-sycophancy, long-horizon reliability)
restated as theory. [suggestive]

## Connection 4 — Drives/anchors is our loss, named [solid analogy]

Ngo: goal = drive (upward force on a credence) balanced by anchor
(empirical pinning); rationality = limit of infinitesimal drives; active
inference's mistake = clamping credences hard. Our reward = coherence
pressure + λ·anchor (Brier or KL-to-previous-beliefs) is the same equation
with coherence as the drive. His frame makes two predictions we can check:
(a) strong drive + weak anchor ⇒ belief distortion — that is our A₀
collapse arm, and the medical-VLM always-"Yes" collapse already observed in
the literature; (b) the λ sweep is a drive/anchor phase diagram, and the
interesting regime is λ large enough to pin but small enough to let
arbitrage flow through the web. Kelvin's July formal note (markets-as-agency
thread) lives one door down: its Theorem 7 (exchange rate constant iff same
drive) and the two-ledgers repair (trader wealth = epistemic trust;
exchange rate = axiological weight) are the multi-currency version of the
same drive/anchor separation. [suggestive]

## Connection 5 — The dynamic check family nobody has built [open lane]

LI's No Expected Net Update (Thm 4.3.9): today's price of φ equals today's
expectation of the *future* price — conservation of expected evidence as a
martingale property. Paleka's ExpEvidence check is the static cross-section;
the dynamic version ("your current probability should equal your expectation
of your day-after-more-thinking probability") is, per our sweep of the
literature, untrained and largely unmeasured for LLMs. It is also the check
most relevant to reasoning models: it directly tests whether more inference-
time compute moves beliefs in expectation (it shouldn't, if calibrated).
Natural consistency-ft extension: elicit P(φ) and E[P_after-CoT(φ)], score
the gap, train with soft indicators. This would be a genuinely novel check
family with an LI theorem as its spec. [solid spec, unbuilt]

## Connection 6 — People and program surface [logistics]

- Pallavi Sudhir: co-author on Paleka's consistency paper AND author of
  "Market-based Architectures in RL" (2503.05828) + in-progress "RL from
  Recursive Information Market Feedback" — the one person already standing
  on both sides of this bridge.
- Ngo runs a MATS Summer 2026 stream on coalitional agency; Kelvin is MATS
  Autumn 2026 (Turan stream) — adjacency worth using.
- Jessica Taylor's own LI-era background means Connections 1/3/5 are the
  ones to lead with; 2/4 are framing.

## What this changes about the project, concretely

1. Adopt the LI framing in the proposal's motivation: the benchmark measures
   exploitability-as-a-market; training amortizes the arbitrageur. (One
   paragraph, big legitimacy gain with this audience.)
2. Add the quarantine rule: no self-referential/performative propositions in
   training tuples (cheap filter, prevents a real failure mode).
3. Soft thresholds in reward shaping (LI Self-Trust lesson).
4. Queue the dynamic ExpEvidence check (Connection 5) as the follow-on
   experiment after the Gate-1 pilot — it is the most publishable delta and
   no one owns it yet.
5. Adversary-strengthening roadmap (Connection 1) replaces "add more checks
   someday" with a principled ladder toward the LI trader class.
