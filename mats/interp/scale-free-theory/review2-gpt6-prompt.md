You reviewed a first draft of outbox/scale-free-theory-2026-10-29.md (your report: work/scale-free-theory/review-gpt6-2026-10-29.md;
Gemini's: work/scale-free-theory/review-gemini-2026-10-29.md). The note has been revised (first draft kept as
work/scale-free-theory/draft-v1-before-review.md; section 10 lists the changes). Working directory: /home/agent/workspaces/mats.

Check only these, briefly:
1. Section 5.1, third bullet (inverse-Wishart, Sylvester's identity): is the claim that the query predictive depends
   on each side of a trader cut only through a (T + 2) x (T + 2) weighted Gram matrix of [E_S, o_S, 1_S], additive
   across cuts, correct for a matrix-normal / inverse-Wishart model with block-diagonal prior scale? Is "diagonal
   plus rank k" handled correctly?
2. Section 5.3: the stated KL rate N^{1-p} and the hypotheses list. Section 5.7: the corrected capacity bound
   m >= (log2(1 + N I) - 2 delta log2 e) / log2(1 + R^2/sigma^2) with forward KL.
3. Section 6.4: do the reported numbers match work/scale-free-theory/n5_interaction_v2.json,
   n6_interaction_noisy_v2.json and n7_routing_classes.json, and is the interpretation (raw interaction share mostly
   tracks error size; deviation shares 0.28-0.48; within-source deficit for one-layer and stuck nets; layer 1 label
   routing, layer 2 precision weighting) supported and not overstated?
4. Any of your first-round high-severity findings that the revision did not address.
Output: numbered findings with severity and the fix. Do not edit files. Be concise.
