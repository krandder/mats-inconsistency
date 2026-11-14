# Literature check: attention aggregation motifs, and transformers trained on math/synthetic data without human language

Checked at 2026-10-26. Method: each paper's PDF fetched from arXiv (Nature HTML for AlphaGeometry, goodfire.com HTML for the Goodfire post), text extracted with pypdf, and every quote string-matched against the extracted text. Quotes are exact except for whitespace and line-break hyphenation. "p.N" is the PDF page. Anything I could not match in the source text is listed under "Dropped / unverified" at the end, not used above it.

Relevance lines refer to the question: **how to get transformers to aggregate information well in domains like forecasting or small synthetic aggregation worlds.**

---

## Topic A: how attention aggregates information in practice

### A1. Tigges, Hollinsworth, Geiger, Nanda (2023). Linear Representations of Sentiment in Large Language Models. https://arxiv.org/abs/2310.15154
- **What is aggregated:** sentiment, a single linear direction. **Where:** commas, periods, and some nouns/names (the "SUM" positions), written there by mid-layer heads (GPT-2 small: heads 7.1 and 7.5 write to SUM; "summary readers" read it at END). **Read later:** END-position heads read the summary token instead of, or in addition to, the original valenced words.
- **Quote (Abstract, p.1):** "sentiment is not solely represented on emotionally charged words, but is additionally summarised at in- termediate positions without inherent sentiment, such as punctuation and names. We show that in Stanford Sentiment Treebank zero-shot classification, 76% of above-chance classification accuracy is lost when ablating the sentiment direction, nearly half of which (36%) is due to ablating the summarized sentiment direction exclusively at comma positions."
- **Quote (§4.2, p.8):** "this summarization results in a partial information bottleneck, in which the summarization points become as important (or sometimes more important) than the phrases that precede them".
- **Why it is learned:** not explained mechanistically. The authors only frame it as "a naturally emerging analogue to the explicit classification token in BERT-like models" (Intro, p.1–2) and as an "information bottleneck" (p.2).
- **Relevance:** this is the clearest causal evidence that a pretrained LM stores clause-level running summaries at delimiter positions. A forecasting or aggregation world that has delimiters between evidence items gives the model a natural place to put per-item or running summaries, and those places can be probed and ablated.

### A2. Goodfire: Bushnaq, Braun, Clive-Griffin et al. (2026-05-05). Interpreting Language Model Parameters. https://www.goodfire.com/research/interpreting-lm-parameters
- I found no Goodfire post specifically about comma or punctuation *summarization* (I checked the 40+ /research pages listed on goodfire.com/research). The closest is this parameter-decomposition post. Its decomposition finds a query-key subcomponent pair in layer 1 that drives attention to earlier punctuation, newlines and similar tokens across most heads.
- **Quote (web page):** "this interaction contributes significantly to the attention patterns of most heads on previous periods, commas, and newline characters." and "This suggests that this pair of subcomponents orchestrates a syntax boundary detector with a variety of short- or long-offset ranges."
- **Why:** not addressed. This is an early-layer boundary detector, not evidence of summary storage.
- **Relevance:** weak. It shows that attending to boundary tokens is a dedicated, low-level circuit component, which you could look for in a small model trained on an aggregation world.

### A3. Razzhigaev et al. (2025). LLM-Microscope: Uncovering the Hidden Role of Punctuation in Context Memory of Transformers. https://arxiv.org/abs/2502.15007
- **What/where:** how much of the prefix can be reconstructed from a single token's hidden state ("contextualization"). Punctuation, stopwords and articles score highest. **Read later:** not localized to heads, but removing these tokens hurts downstream tasks.
- **Quote (Abstract, p.1):** "removing these tokens — especially stopwords, articles, and commas — consistently degrades performance on MMLU and BABILong-4k, even if removing only irrelevant tokens."
- **Quote (§1, p.2):** "certain “filler” tokens, such as punctuation marks, stopwords, and articles, are highly contextualized and act as key aggregators in language understanding."
- **Why:** not explained.
- **Relevance:** this is a probe-based way to find aggregation tokens without a circuit analysis: train a decoder that reconstructs the prefix from one position. It ports directly to a synthetic world, where you can ask which positions carry the evidence-so-far.

### A4. Wang et al. (2023). Label Words are Anchors: An Information Flow Perspective for Understanding In-Context Learning. https://arxiv.org/abs/2305.14160
- **What:** demonstration content. **Where:** the label-word tokens of the demonstrations, in shallow layers (GPT2-XL, GPT-J). **Read later:** in deep layers, the final position's attention to label words correlates with the prediction.
- **Quote (Abstract, p.1):** "(1) semantic information aggregates into label word representations during the shallow compu- tation layers’ processing; (2) the consolidated information in label words serves as a reference for LLMs’ final predictions."
- **Causal test (§2, p.2):** "By blocking the information aggregation path to label words in certain layers, we find that such isolation in shal- low layers significantly impairs model performance."
- **Use:** a "Hidden_anchor" compression that keeps only the label-word hidden states works, and the authors attribute this "to the aggregation of information in label" words (p.7).
- **Why:** no stated explanation for why the model learns this.
- **Relevance:** aggregation points are the structurally predictable positions that the final answer later depends on. In an aggregation world, the answer or label slots of earlier episodes are candidate anchors.

### A5. Hendel, Geva, Globerson (2023). In-Context Learning Creates Task Vectors. https://arxiv.org/abs/2310.15916
- **What:** the task (the mapping implied by the demonstrations). **Where:** the hidden state of the separator token "→" after the query, at an intermediate layer L. **Read later:** the layers after L apply θ to x. Patching θ into a demonstration-free forward pass recovers ICL accuracy.
- **Quote (Abstract, p.1):** "ICL can be seen as compressing S into a single task vector θ(S) and then using this task vector to modu- late the transformer to produce the output."
- **Quote (Fig. 2 caption, p.3):** "we use a dummy query ( x′ = Plum) and use the representation of→ at the Lth layer as θ."
- **Quote (§3.3, p.3):** "all models exhibit a performance peak at a similar intermediate layer, irrespective of their parameters and layer count differences."
- **Why:** not addressed.
- **Relevance:** the compressed summary lives at the delimiter just before the answer, in middle layers. This gives a concrete, testable target: patch the separator state in a small model trained on an aggregation world.

### A6. Todd et al. (2023). Function Vectors in Large Language Models. https://arxiv.org/abs/2310.15213
- **What:** a function vector, the sum of the outputs of a few causally important heads. **Where:** heads clustered in early-middle layers, read at the final token. These heads attend to the output tokens of the demonstrations. **Read later:** adding the vector in middle layers triggers the task zero-shot.
- **Quote (Abstract, p.1):** "a small number attention heads transport a compact representation of the demonstrated task, which we call a function vector (FV)."
- **Quote (§3, p.4):** "These heads primarily attend to token positions corresponding to example outputs; this observation is consistent with the high salience of ICL label tokens observed by Wang et al. (2023a)".
- **Why:** not addressed.
- **Relevance:** aggregation is done by a sparse, identifiable set of mover heads reading example-output positions. Causal mediation over heads is a ready-made method for locating the aggregator in a small model.

### A7. Xiao et al. (2023). Efficient Streaming Language Models with Attention Sinks. https://arxiv.org/abs/2309.17453
- **What:** surplus attention mass, not content. **Where:** the initial tokens, in all layers beyond the first two. **Why (stated):** softmax must sum to 1, and early tokens are visible to every later token.
- **Quote (§1, p.2):** "We attribute the reason to the Softmax operation, which requires attention scores to sum up to one for all contextual tokens. Thus, even when the current query does not have a strong match in many previous tokens, the model still needs to allocate these unneeded attention values somewhere".
- **Quote (§1, p.2):** "initial tokens are visible to almost all subsequent tokens because of the autoregressive language modeling nature, making them more readily trained to serve as attention sinks."
- **Trained fix (Abstract):** "adding a placeholder token as a dedicated attention sink during pre-training can further improve streaming deployment."
- **Relevance:** a sink is a dump for attention mass, not a summary. Separate the two roles before reading a high-attention token as an aggregation point.

### A8. Sun, Chen, Kolter, Liu (2024). Massive Activations in Large Language Models. https://arxiv.org/abs/2402.17762
- **What:** a few huge, input-agnostic activation values. **Where:** the starting token, delimiter tokens, and low-semantic words such as "and" and "of". **Read later:** they concentrate attention on those tokens and act as implicit biases.
- **Quote (§1, p.2):** "Many of these activations are found within the starting word token and delimiter tokens."
- **Quote (Abstract, p.1):** "their values largely stay constant regardless of the input, and they function as indispensable bias terms in LLMs. Third, these massive activations lead to the concentration of attention probabilities to their corresponding tokens, and further, implicit bias terms in the self- attention output."
- **Why (stated):** "LLMs try to learn implicit bias components in self-attention via massive activations, during their pretraining phase" (§1, p.2).
- **Relevance:** delimiters host both input-agnostic bias machinery and, per A1/A3, input-dependent summaries. Any claim that a delimiter aggregates evidence has to show input dependence.

### A9. Gu et al. (2024). When Attention Sink Emerges in Language Models: An Empirical View. https://arxiv.org/abs/2410.10781
- **Why (stated):** sinks emerge with sufficient training and are tied to softmax normalization. They can be moved to other positions, and they vanish with sigmoid attention.
- **Quote (Abstract, p.1):** "attention sink acts more like key biases, storing extra attention scores, which could be non-informative and not contribute to the value computation."
- **Quote (§1, p.2):** "The sink position is highly related to the loss function and data distribution and can be shifted to other positions rather than the first token."
- **Quote (Abstract):** "replacing softmax attention with other attention operations, such as sigmoid attention without normalization, attention sinks do not emerge in LMs up to 1B parameters."
- **Relevance:** in a synthetic world you control the data distribution, so the sink position, and possibly the summary position, is a design variable. Normalization choice is a lever too.

### A10. Barbero et al. (2025). Why do LLMs attend to the first token? https://arxiv.org/abs/2504.02732
- **Why (stated):** sinks let deep or long-context models avoid over-mixing, that is, collapse of token representations.
- **Quote (Abstract, p.1):** "we argue theoretically and empirically that this mechanism provides a method for LLMs to avoid over-mixing".
- **Relevance:** aggregation has a cost. With many evidence items and deep stacks, the model needs a way not to mix, so good aggregation requires a no-op path as well as a mixing path.

### A11. Bondarenko, Nagel, Blankevoort (2023). Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing. https://arxiv.org/abs/2306.12929
- **What/where:** outliers in BERT sit at delimiter positions. Heads park probability mass on [SEP], "." and "," whose values are small, which gives a (soft) no-op.
- **Quote (§3, p.3):** "the majority of outliers (> 97%) correlate with the position of delimiter tokens – [SEP], “.”, and “,”." and "the attention head assigns almost all of its probability mass to [SEP] tokens, and other less informative tokens like dots/commas, while these tokens also have small values inV associated with those tokens."
- **Why (stated, Abstract):** "To achieve the exact zeros needed in the attention matrix for a no-update, the input to the softmax is pushed to be larger and larger during training".
- **Relevance:** this is direct counter-evidence to reading every heavy attention to a comma as aggregation. Value-weighted attention, which Tigges et al. also used (A1, p.8), is the right check.

### A12. Clark, Khandelwal, Levy, Manning (2019). What Does BERT Look At? https://arxiv.org/abs/1906.04341
- **Quote (Abstract, p.1):** "A surprisingly large amount of BERT’s attention focuses on the deliminator to- ken [SEP], which we argue is used by the model as a sort of no-op."
- **Evidence (§3.2, p.4):** gradients of the loss with respect to [SEP] attention become very small once that attention is high.
- **Relevance:** the original no-op interpretation of delimiter attention. Same caution as A11.

### A13. Darcet, Oquab, Mairal, Bojanowski (2023). Vision Transformers Need Registers. https://arxiv.org/abs/2309.16588
- **What:** global image information. **Where:** high-norm tokens on redundant, low-information background patches. **Why (stated):** the model recycles tokens whose local content is redundant.
- **Quote (§2.1, p.3):** "the model learns to recognize patches containing little useful information, and recycle the corresponding tokens to aggregate global image information while discarding spatial information."
- **Quote (Abstract):** "We propose a simple yet effective solution based on providing additional tokens to the input se- quence of the Vision Transformer to fill that role."
- **Relevance:** this is the most explicit stated mechanism for *why* summaries end up on low-information tokens: those slots are cheap to overwrite. It predicts that giving explicit register or scratch tokens in an aggregation world will relocate the summaries, which is a testable prediction.

### A14. Mu, Li, Goodman (2023). Learning to Compress Prompts with Gist Tokens. https://arxiv.org/abs/2304.08467
- **Trained version of the motif.** Gist tokens are inserted after the prompt, and an attention mask blocks later tokens from seeing the prompt, so everything has to pass through the gist tokens' activations.
- **Quote (§1, p.2):** "doing instruction tuning [38] with gist tokens inserted after the prompt, and a modified attention mask preventing tokens after the gist tokens from attending to tokens before the gist tokens."
- **Quote (Abstract):** "gisting enables up to 26x compression of prompts".
- **Relevance:** a mask-enforced bottleneck is a direct way to force an aggregation location in a synthetic world and then measure what survives it.

### A15. Pang et al. (2024). Anchor-based Large Language Models. https://arxiv.org/abs/2402.07616
- **Trained motif:** an anchor token (the last token of a sequence) is trained, via anchor-based attention masks, to hold the sequence information. Only the anchors' KV cache is kept.
- **Quote (§1, p.2):** "the final token in a sequence may be better suited to serve as a natural information com- pression point compared to other tokens, as they cannot observe future tokens."
- **Quote (Abstract):** "up to 99% keys/values cache reduc- tion and up to 3.5 times faster inference."
- **Relevance:** in this work the end-of-segment delimiter is engineered as the summary slot, which matches where pretrained models put it naturally (A1, A5).

### A16. Goyal et al. (2023). Think before you speak: Training Language Models With Pause Tokens. https://arxiv.org/abs/2310.02226 · Pfau, Merrill, Bowman (2024). Let's Think Dot by Dot. https://arxiv.org/abs/2404.15758
- **Pause tokens (Abstract):** "inference-time delays show gains on our tasks when the model is both pretrained and finetuned with delays."
- **Filler tokens (Abstract):** "learning to use filler tokens is difficult and requires specific, dense supervision to converge."
- **Relevance:** extra content-free positions add parallel compute, but a model only learns to use them with suitable training data. Adding aggregation slots to a world is not free.

### A17. Agarwal, Dalal, Misra (2025, v5 2026). The Bayesian Geometry of Transformer Attention. https://arxiv.org/abs/2512.22471
- **Mechanistic, on synthetic Bayesian tasks** with known posteriors (bijection elimination, HMM filtering). Small transformers match the posterior. A layer-0 "hypothesis-frame" head is critical. Value vectors lie on a low-dimensional manifold parameterized by posterior entropy.
- **Quote (Abstract):** "small transformers reproduce exact Bayesian posteriors for filtering and hypothesis elimination with 10−3–10−4 bit accuracy, while capacity-matched MLPs fail by orders of magnitude."
- **Quote (Abstract):** "(i)belief accumulation—integrating evidence into a running posterior; (ii)belief transport—propagating beliefs forward through stochastic dynamics; and (iii)random-access binding—retrieving stored hypotheses by content rather than position."
- **Quote (§5.1, p.15):** "Layer 0 attention constructs thehypothesis spacein which all subsequent inference takes place. Keys at this layer form an approximately orthogonal basis over input tokens".
- **Relevance:** this is the closest existing "small synthetic aggregation world" with a verified mechanism. Its three primitives are a usable checklist for what a forecasting-style world must require.

### A18. Shai et al. (2024). Transformers represent belief state geometry in their residual stream. https://arxiv.org/abs/2405.15943
- **Trained on HMM-generated token sequences (no language).** Belief states, i.e. posteriors over hidden states, are linearly decodable from the residual stream, sometimes spread across layers.
- **Quote (Abstract):** "belief states are linearly represented in the residual stream of transformers, even in cases where the predicted belief state geometry has highly nontrivial fractal structure."
- **Relevance:** this gives a ground-truth target for what "aggregated well" means (the Bayes belief state) and a linear-probe test for it.

### A19. Bigelow, Wurgaft, … Lubana (2025; Goodfire AI is affiliation 1). Belief Dynamics Reveal the Dual Nature of In-Context Learning and Activation Steering. https://arxiv.org/abs/2511.00617
- This is a behavioral Bayesian model of an LLM, not a circuit study. ICL accumulates evidence, and steering shifts the prior.
- **Quote (Abstract):** "steering operates by changing concept priors, while in-context learning leads to an ac- cumulation of evidence."
- **Quote (§1, p.2):** "a sigmoidal growth of posterior belief as a function of in-context exemplars".
- **Relevance:** it gives a behavioral test, a log-odds-additive evidence curve, that a well-aggregating model in a forecasting world should pass.

---

## Topic B: transformers trained on math or synthetic data without human language

### B1. Ye, Xu, Li, Allen-Zhu (2024). Physics of Language Models Part 2.1: Grade-School Math and the Hidden Reasoning Process. https://arxiv.org/abs/2407.20311
- **Plans and computes only necessary quantities.** Probes show the model has marked which parameters are necessary before it starts generating. It also learns all-pair dependencies that are never needed in training.
- **Quote (§1, Result 3, p.3):** "the model can learn to generate shortest solutions, almost always avoid- ing unnecessary computations. This suggests that the model formulates a plan before it generates, in order to avoid computing any quantities that are not needed".
- **Quote (§1, Result 4, p.3):** "the model (mentally!) preprocesses the full set of necessary parameters before it starts any generation."
- **Quote (§4, p.11):** "language models can pre-compute the all-pair dependency graph dep(A,B ) mentally even before a question is asked. We consider this a “level-2” reasoning skill".
- **Quote (Result 7, p.13):** "Language model depth is crucial for mathematical reasoning."
- **Caveat:** iGSM is synthetic but uses English-like templated entity names (e.g. "Riverview High's Film Studio", Fig. 5), so it is not language-free in the strict sense.
- **Relevance:** this is the strongest evidence that from-scratch training on a dependency-graph world produces planning and relevance tracking. The V-probing tasks (nece(A), dep(A,B), can_next(A)) carry over directly as probes for evidence relevance in an aggregation world.

### B2. Allen-Zhu & Li (2023). Physics of Language Models Part 1: Learning Hierarchical Language Structures. https://arxiv.org/abs/2305.13673
- **Synthetic CFGs (no language).** NT ancestor/boundary information is linearly encoded *at subtree-end tokens*, and attention links NT-end tokens to adjacent NT-ends, which looks like dynamic programming.
- **Quote (§1 Corollary, p.3):** "Boundary-based attention allows a token to attend to its closest NT symbols in CFG tree, even when separated by hundreds of tokens. This resembles DP".
- **Quote (Result 5, §4.2, p.10):** "the information of position i’s NT ancestor/boundary is locally encoded around position i± 1 when i is on the NT boundary."
- **Relevance:** this is the synthetic-data analogue of the summarization motif. Subtree summaries live at boundary tokens and later attention reads boundary to boundary. It suggests hierarchical evidence structure with explicit segment ends as a world design.

### B3. Allen-Zhu (2025). Physics of Language Models Part 4.1: Architecture Design and the Magic of Canon Layers. https://arxiv.org/abs/2512.17351
- Uses five synthetic pretraining tasks, including Depo (reasoning depth) and Brevo (reasoning breadth). Canon layers are local weighted sums over neighbouring tokens.
- **Quote (Abstract):** "Canon layers: lightweight architectural components—named after the musical term “canon”—that promote horizontal information flow across neighboring tokens."
- **Quote (Abstract):** "Canon layers enhance reasoning depth (e.g., by 2×), reasoning breadth".
- **Quote (§1 Result 1, p.3):** "Transformers lack horizontal information flow within layers, leading to inefficiencies even on simple tasks like associative recall."
- **Relevance:** this is an architectural lever specifically for aggregation and multi-hop, and the paper evaluates it on synthetic worlds of the kind proposed here.

### B4. Lee et al. (2023). Teaching Arithmetic to Small Transformers. https://arxiv.org/abs/2307.03381
- **Quote (Abstract):** "we then train on chain-of-thought style data that includes intermediate step results. Even in the complete absence of pretraining, this approach significantly and simultaneously improves accuracy, sample complexity, and convergence speed."
- **Quote (§1, p.3):** "By training on samples with reversed results, i.e., ‘A3A2A1 + B3B1B1 = C1C2C3’, we enable the model to learn a simpler function, significantly improving sample complexity."
- **Length generalization (§1, p.4):** "We find length generalization beyond trained digit lengths difficult."
- **Relevance:** output order and format decide whether each output token depends locally or globally on the inputs. The same applies to the order in which an aggregate is emitted.

### B5. McLeish et al. (2024). Transformers Can Do Arithmetic with the Right Embeddings. https://arxiv.org/abs/2405.17399
- **Quote (Abstract):** "their inability to keep track of the exact position of each digit inside of a large span of digits. We mend this problem by adding an embedding to each digit that encodes its position relative to the start of the number."
- **Quote (Abstract):** "achieving up to 99% accuracy on 100 digit addition problems."
- **Relevance:** positional indexing of items is a precondition for aggregation. Giving items explicit within-group indices can unlock length or extrapolation generalization.

### B6. Lample & Charton (2019). Deep Learning for Symbolic Mathematics. https://arxiv.org/abs/1912.01412
- **Quote (Abstract):** "We achieve results that outperform commercial Computer Algebra Systems such as Matlab or Mathematica."
- **Caveat (§4.5, p.9):** the comparison uses "Mathematica with a timeout delay of 30 seconds."
- **Relevance:** seq2seq models trained purely on generated math data can be strong, and the generator distribution is the main design object.

### B7. Charton (2021). Linear algebra with transformers. https://arxiv.org/abs/2112.01898 · Charton (2023). Learning the greatest common divisor: explaining transformer predictions. https://arxiv.org/abs/2308.15594
- **Linear algebra (Abstract):** "models trained to predict Laplace-distributed eigenvalues generalize to diﬀerent classes of matrices: Wigner matrices or matrices with positive eigenvalues. The reverse is not true."
- **GCD (Abstract):** "Models trained from uniform op erands only learn a handful of GCD (up to 38 GCD ≤ 100). Log-uniform operands boost perfor- mance to 73 GCD ≤ 100, and a log-uniform distribution of outcomes (i.e. GCD) to 91."
- **Relevance:** in synthetic worlds the training distribution decides which rules are learned and how far they generalize out of distribution. Evidence or outcome distributions for an aggregation world should be chosen deliberately, e.g. heavy-tailed or balanced.

### B8. Power et al. (2022). Grokking. https://arxiv.org/abs/2201.02177 · Nanda et al. (2023). Progress measures for grokking via mechanistic interpretability. https://arxiv.org/abs/2301.05217
- **Power (Abstract):** "this improvement in general- ization can happen well past the point of overﬁtting"; **(§1, p.2):** "weight decay is particularly effective at improving generalization on the tasks we study."
- **Nanda (Abstract):** "uses discrete Fourier transforms and trigonometric identities to convert addition to rotation about a circle" and "split training into three continuous phases: memorization, circuit formation, and cleanup."
- **Relevance:** on small synthetic worlds a general aggregation circuit can form well after the training loss is near zero. Track progress measures, not just loss, and treat weight decay as a lever.

### B9. Liu et al. (2022). Transformers Learn Shortcuts to Automata. https://arxiv.org/abs/2210.10749
- **Quote (Abstract):** "Empirically, we ﬁnd that Transformers converge to shortcut solutions with standard training, across a wide variety of automata. We further investigate the brittleness of these solutions".
- **Relevance:** sequential evidence-accumulation worlds (state tracking) invite parallel shortcut solutions that break out of distribution. Test generalization in length and distribution, not in-distribution accuracy.

### B10. Trinh, Wu, Le, He, Luong (2024). Solving olympiad geometry without human demonstrations (AlphaGeometry). Nature 625:476–482. https://www.nature.com/articles/s41586-023-06747-5
- **Quote:** "we extracted 100 million synthetic theorems and their proofs, many with more than 200 proof steps".
- **Quote:** "a neural language model, trained from scratch on our large-scale synthetic data, to guide a symbolic deduction engine".
- **Relevance:** a symbolic engine can generate unlimited verified training data, and the network only has to learn the hard, branching part. The analogous design for forecasting worlds uses an exact Bayes or generator oracle.

### B11. Synthetic vs. natural-language pretraining for transfer
- **Wu, Li, Liang (2022). Insights into Pre-training via Simpler Synthetic Tasks.** https://arxiv.org/abs/2206.10139
  - Abstract: "the best synthetic pre-training method, LIME, attains an average of 67% of the beneﬁts of natural pre-training" and "pre-training on a simple and generic synthetic task deﬁned by the Set function achieves 65% of the beneﬁts".
- **Wu et al. (2021). LIME.** https://arxiv.org/abs/2101.06223
  - Abstract: "we design three synthetic tasks that are intended to require the model to have these three abilities [deduction, induction, abduction]. We speciﬁcally design these tasks to be synthetic and devoid of mathematical knowledge".
- **Hu et al. (2025). Between Circuits and Chomsky.** https://arxiv.org/abs/2502.19249
  - Abstract: "pre-pretraining achieves the same loss and better linguistic generalization with a 33% smaller token budget" and "attention heads acquired during pre-pretraining remain crucial for the model’s performance on syntactic evaluations."
- **Shinnick et al. (2025). Transformers Pretrained on Procedural Data Contain Modular Structures for Algorithmic Reasoning.** https://arxiv.org/abs/2505.22308
  - Abstract: "Attention layers often carry the most transferable information, but some pretraining rules impart useful structure to MLP blocks instead."
- **Zhang et al. (2024). Intelligence at the Edge of Chaos.** https://arxiv.org/abs/2410.02536
  - Abstract: "models trained on more complex data exhibit greater predictive ability" and "Both uniform and periodic systems, and often also highly chaotic systems, resulted in poorer downstream performance".
- **Lee, Han, Kumar, Agrawal (2026). Training Language Models via Neural Cellular Automata.** https://arxiv.org/abs/2603.10055
  - Abstract: "pre-pre-training on only 164M NCA tokens improves downstream language modeling by up to 6%" and "this even outperforms pre-pre-training on 1.6B tokens of natural language from Common Crawl with more compute" and "attention layers are the most transferable".
- **Relevance:** language-free synthetic pretraining transfers. The transferable part is mostly attention, i.e. routing and aggregation machinery, and the data must have intermediate complexity. This supports training aggregation-specific attention on synthetic worlds first.

---

## Cross-cutting points, each tied to the entries above
1. **Where summaries go.** Pretrained LMs store summaries at delimiter or segment-end positions in mid layers: commas and periods (A1), "→" (A5), label words (A4). A synthetic model trained on CFGs does the same at subtree ends (B2). The motif appears in both natural-language and language-free training.
2. **Sink versus summary.** Heavy attention to a delimiter is often a no-op or bias sink with small values (A7–A12), not aggregation. Separating the two requires value-weighted attention or ablation of an input-dependent direction (A1).
3. **Stated reasons for the location.** Only these are stated: the softmax sum-to-one constraint plus visibility under the causal mask (A7, A9); avoiding over-mixing (A10); recycling redundant or low-information slots (A13, a ViT); the last token being a natural compression point under causal attention (A15, a design motivation). No source here gives a learning-dynamics account of why *content* summaries (as opposed to sinks) go to punctuation.
4. **Ways to impose aggregation by design.** Attention masks (gist tokens, AnLLM; A14, A15); explicit registers or sink tokens (A7, A13); Canon layers (B3); output order and format (B4); positional indexing (B5).

## Dropped / unverified
- **Goodfire post on comma or punctuation summarization:** none found on goodfire.com/research. Tigges is now at Goodfire, but A1 predates the company. A2 and A19 are the only Goodfire items used.
- **An "AlphaGeometry arXiv 2401.09319" ID** returned by a search tool is wrong (that ID is a sub-Finsler geometry paper). Only the Nature URL is used.
- A search-tool claim that the NCA paper used "1.6B parameters" was not checked against the PDF and is not used.
- Physics of LMs Part 4.2, and Agarwal et al. Papers II and III (2512.22473, 2512.23752), were not read. They are not cited above.
