## 8. Back to the original question: can dense nets be untangled, and is their tangledness needed?

- Needed to represent? For the aggregation tasks we study, no: SFU solutions exist (sections 5.1-5.3), and for
  one-level routing a transformer can implement them in O(rounds) layers (A2). Tangledness is necessary for
  similarity-based routing (T6a, T7), hidden structure near the detection threshold (T4a, T7), multi-hop lookups
  without depth (T8), and genuinely global couplings (T10).
- Used by trained nets? Partly measurable without opening the net (M1) and with attention maps (M2). In the tiny-PFN
  nets, the computation decomposes as the theory says (label routing, then precision weighting), and every net's
  error is partly cross-source; nets that cannot apply the per-source nonlinearity before mixing carry the expected
  signature. Whether the nets' internal messages are as small as the task allows (M4) is not yet measured.
- Learnable? Separate question. World FA (binding) and World 3 (parity) are cheap to represent and were not learned.
