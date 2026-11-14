## 4. Verdict: the open-latent profile, with routing and rounds

Kelvin's intuition has three parts: complexity may grow with scale; it is fine for larger structures to have more
dimensions; what stays simple is how information combines. The definition that matches all three, and that the
theorems below support, is:

**Definition (scale-free untangled, SFU).** A task family admits an SFU solution with routing cost b, open-latent
profile o(.) and R rounds if
1. (routing) each item carries a label of at most b bits, and a restricted rule (group-by, sort, contiguous chunks)
   builds the hierarchy from the labels; labels may be recomputed between rounds from what was broadcast;
2. (content) the message at a node is the likelihood of the latents still open there (attached inside and outside
   it), with its size counted in latent states or in the epsilon-rank of the message family at a stated precision:
   a global function of the open latents in one round, a local expansion around a broadcast estimate with more
   rounds;
3. (combination) every merge, at every scale, is the same two operations: add log-likelihoods, and sum out the
   latents that close at that node.

Untangled means b = O(log N) and, at every merge, the open latents of the two children taken together (which include
those that close there) stay small in states or epsilon-rank: bounded, or growing with the number of groups rather
than of items. Bounding only the parent's open latents is not enough: summing out a closing block whose internal
structure is an expander is #P-hard even when nothing stays open.

This is junction-tree message passing with the items as leaves. Open latents are the separators (Lauritzen and
Spiegelhalter 1988); the size of what is merged along an elimination order is the induced width (Dechter 1999);
routing by labels is a form of context-specific independence (Boutilier, Friedman, Goldszmidt, Koller 1996). What is
added here is the reading of Kelvin's three phrases in these terms, the role of rounds, and the measurements.

The intrinsic quantity is the task's minimal message (the Nerode class of T1, or its internal epsilon-cover at a
stated precision); the open latents explain it. Counting latent *variables* is not invariant (two can be merged into
one), so sizes are counted in states or epsilon-rank. For a single cut, the smallest number of latent states making
the two sides conditionally independent is the nonnegative rank of the joint law across that cut (**PROVEN**); this
bounds the separator at that cut, not the profile of one latent model consistent across all cuts.

How the three parts of the intuition map:
- "zooming out, things get more complicated": at larger scales the open latents are those of larger structures
  (groups, groups of groups); they may be richer, so the message dimension may grow with scale;
- "it is fine for larger structures to have more dimensions": the message size is set by the open latents at that
  scale, not by how many items lie below; under noise it must in fact grow slowly (5.7);
- "how information combines stays simple": every merge is addition of log-likelihoods plus summing out what closes,
  at every scale; what changes with scale is which latents are open.

This note is about cuts across items. Cuts across depth (how many directions of layer l still matter for the output)
are the Lyapunov lane's subject (pool-yw7kwn.1-.8); in a computation with R rounds, R messages cross each item cut, so
the two pictures meet in the (items x depth) grid of a transformer.

In this language D1 is SFU with a fixed hierarchy, one round, counting bits; D3 is the message dimension; D4 at the
item level is what remains after the latents are summed out (a shared latent turns its items into a clique), which is
why it overstates tangledness; D5 bounds how much the input law itself must carry across a cut; D6 is the linear
special case where the "latents" are multipole moments. Status: the definition is a choice, not a claim; T2-T4 below
are what make it the natural one.
