### 6.2 The decomposition

For layer l and head h, write the logits as L_ij = R(lambda_i, lambda_j) + w_j + e_ij: a **routing** term from
discrete labels (same source, same question, token type), a **weighting** term per key (a log-precision-like scalar
from the key's content), and a residual. Values are the **content** (natural-parameter-like messages). Layers are
**rounds**. "Dense attention mostly pays for input-dependent routing" becomes four testable claims:
(a) rows cluster into few routing classes C, explained by labels; (b) given routing and weighting, the content is
low-dimensional and merges additively; (c) a static (input-independent) routing of the same sparsity fails while
label routing does not; (d) the union of supports over inputs is dense while each input's pattern is sparse.

Two consequences.
(i) Expressivity (**PROVEN UNDER** a flat, one-level routing hierarchy and bounded group size; **HEURISTIC**
otherwise): a transformer whose labels are linearly readable can implement a one-level R-round SFU computation with
message size k in O(R) layers and width O(k + label dimension): per round, one head gathers each group's natural
parameters (normaliser kept by a sink), and an MLP turns sums into the next round's local statistics. A multi-level
hierarchy of depth H needs about H layers per round, and undoing a normaliser of order 1/N needs MLP precision growing
with N. So density is not needed to *represent* SFU computations of this kind; it is needed when routing is by
similarity rather than labels (T6a, T7). Multi-step in-context estimation by transformers is established in other
settings (von Oswald et al., arXiv:2212.07677; Ahn et al., arXiv:2306.00297; Bai et al., arXiv:2306.04637).
(ii) Learning (**HEURISTIC**): at uniform attention (World FA's measured own/other ratio 1.00) the gradient on the
query-key form along the binding direction is a covariance between "same ID" and how useful the gathered content is,
which is near zero until the content path is useful, and the content path is useless until binding exists: a saddle
neither part can leave alone. Consistent with World FA's 0 of 15 flat runs and with success once binding was given;
plateaus of this routing-content kind are analysed for induction heads and in-context classification (Bietti et al.,
arXiv:2306.00802; Reddy, arXiv:2312.03002).

A caution about our worlds. In the tiny-PFN world the grouping (3 sources x 5 questions) is the same in every episode;
the net must find it by matching random per-episode ID embeddings (no positional encoding, `work/tiny-pfn/pfn.py`),
but an architecture designer can hard-wire it, and with a fixed token order a "static" pattern already encodes the
grouping. So the PFN prior tests content and combination; routing is tested by World FA (flat tokens, random order,
binding by trader ID), or by permuting token order per episode.
