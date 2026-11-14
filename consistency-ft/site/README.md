# site/ — deployed pages without a generated source

Byte-exact copies of pages live on consistency.futarchy.ai whose content is
not produced by a tracked generator. Deploy = scp to
deploy-host:fleet/apps/consistency-dash/.

- proposal.html — standalone wrapper of proposal/rlef-forecasting.html
- bridge.html — consistency-ft <-> simulations bridge page
- open-agency.html — illustrated Open Agency research connection; original
  inline SVG diagrams and an interactive complement-probability example.
  The dashboard builder copies this source into `dash/` for deployment.
- compiled-rewards.html — interactive explainer of how coherence constraints
  (negation, conjunction bound, partition, conditional chain) compile into
  memoryless RL environments (Jessica Taylor's imperfect-recall framing);
  sliders, reward landscapes, LP-vs-quadratic comparison. Also copied by
  the dashboard builder.
- belief-training.html — companion design page: explicit belief structures,
  accuracy + overlap-coherence losses, martingale updating simulator,
  self-distillation effort ladder, predicted-movement curves, prediction-
  market tuple mining, reward table. Also copied by the dashboard builder.

Generated elsewhere (do not copy here): index.html (scripts/build_dash.py),
plan.html (scripts/build_plan.py), ngo.html (proposal/ngo.html),
environments.html (repo root, pending its owner's commit).
