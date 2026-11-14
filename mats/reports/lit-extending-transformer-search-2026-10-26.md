# Extending what transformers already do: in-context search, aggregation, order choice

Literature search, 2026-10-26. The framing is Kelvin's: don't look for new architectures; learn and extend what transformers already do. They aggregate information and run search algorithms over their context, using fixed-order next-token prediction, and they choose what to think about first *inside* that framework.

**How this was checked.** Every paper below was fetched from arXiv on 2026-10-26: metadata from the arXiv API, full text from arxiv.org/html (one paper via PDF and pypdf). Every quote is copied verbatim from that text. "Abstract" means the arXiv abstract; § means the section of the arXiv HTML version. Venues are given only when the arXiv comment field, or a proceedings or conference page found in search, states them; otherwise the paper is listed as arXiv. Papers I could not open are left out. Equations stripped from the HTML show up as gaps, so some quotes skip numbers.

---

## Thread 1. Learning search from linearised traces, and what limits it

**1.1 Saparov et al., "Transformers Struggle to Learn to Search"** (2024; ICLR 2025). https://arxiv.org/abs/2412.04703
- **What it does:** trains small transformers on graph connectivity with effectively unlimited data, then extracts the learned algorithm mechanistically.
- **Quotes:**
  - Abstract: "when given the right training distribution, the transformer is able to learn to search"
  - Abstract: "transformers perform search at every vertex in parallel: For each vertex in the input graph, transformers compute the set of vertices reachable from that vertex."
  - Abstract: "as the input graph size increases, the transformer has greater difficulty in learning the task. This difficulty is not resolved even as the number of parameters is increased"
  - Abstract: "performing search in-context (i.e., chain-of-thought) does not resolve this inability to learn to search on larger graphs."
  - §1: "The training distribution must be carefully constructed so as to preclude the usefulness of shortcuts or heuristics"
  - Future work, §7: "Alternate training procedures may help transformers to more easily learn to search, such as curriculum learning. Alternate architectures may help as well, such as looped transformers."
- **For extending the mechanism:** the learned in-weights search is parallel path-merging, bounded by depth. Two things decide whether it generalises: whether heuristics are removed from the data (a "balanced", lookahead-uniform distribution), and graph size. Scale does not, and a linearised DFS trace does not rescue it.

**1.2 Gandhi et al., "Stream of Search (SoS): Learning to Search in Language"** (2024, arXiv). https://arxiv.org/abs/2404.03683
- **What it does:** pretrains a 250M LM on flattened search traces from mixed symbolic strategies on Countdown, then improves it with STaR and APA.
- **Quotes:**
  - Abstract: "SoS pretraining increases search accuracy by 25% over models trained to predict only the optimal search trajectory"
  - Abstract: "The finetuned SoS models solve 36% of previously unsolved problems, including problems that cannot be solved by any of the heuristic solvers."
  - §4: context length forced a size cap: "search trajectories for games with 5 input numbers can be 60,000 tokens long"
  - Future work, §7: "introducing other formalizable operations such as limits, summarization, cycle checks, and subgoal setting could enhance the SoS framework."
- **For extending the mechanism:** trace format matters. Messy traces with backtracking beat optimal paths. The binding limit is trace length against context, and the authors name in-trace summarisation as the next operation (this links to Thread 5).

**1.3 Lehnert et al., "Beyond A*: Better Planning with Transformers via Search Dynamics Bootstrapping" (Searchformer)** (2024, arXiv). https://arxiv.org/abs/2402.14083
- **What it does:** an encoder-decoder imitates A* execution traces, then is fine-tuned on its own shorter successful traces.
- **Quotes:**
  - Abstract: "optimally solves previously unseen Sokoban puzzles 93.7% of the time, while using up to 26.8% fewer search steps than the A* implementation"
  - Abstract: it beats solution-only baselines "with a 5-10× smaller model size and a 10× smaller training dataset"
  - §5.1 Limitations: "the trace length may grow exponentially in the length of an optimal plan"
  - Future work, §5.2: "use curriculum learning: starting from simple tasks with reasonably long execution traces, train and fine-tune the Searchformer to reduce the trace length, and then adapt the improved model to more complex tasks." Also from §5.2: "Integrating hierarchical planning methods and temporal abstractions ... are another avenue."
- **For extending the mechanism:** imitation followed by self-bootstrapping can beat the teacher algorithm on trace length. Exponential trace growth is the stated wall.

**1.4 Ye, Fu, Jia, Sharan, "Transformers Provably Learn Algorithmic Solutions for Graph Connectivity, But Only with the Right Data"** (2025, arXiv). https://arxiv.org/abs/2510.19753
- **What it does:** theory plus experiments on which data makes a transformer learn matrix-powering connectivity rather than a degree heuristic.
- **Quotes:**
  - Abstract: "Within-capacity graphs (diameter ≤ 3^L) drive the learning of the algorithmic solution while beyond-capacity graphs drive the learning of a simple heuristic based on node degrees."
  - Abstract: "restricting training data to stay within a model's capacity makes both standard and Disentangled Transformers learn the exact algorithm."
- **For extending the mechanism:** this is a precise account of why held-out generalisation of learned search fails. Training examples beyond the model's depth capacity reward heuristics. It suggests a curriculum keyed to capacity, which Saparov et al. only proposed.

**1.5 Schultz et al., "Mastering Board Games by External and Internal Planning with Language Models"** (2024; ICML 2025, PMLR v267). https://arxiv.org/abs/2412.12119
- **What it does:** "internal search" distils MCTS-like trees into the LM as a linearised in-context tree, and compares this with external MCTS.
- **Quotes:**
  - Abstract: "in internal search, the model is trained to generate in-context a linearized tree of search and a resulting final choice."
  - §4: "prompts were composed with diverse search parameters, ranging depth 1–3 and breadth 2–5"
- **For extending the mechanism:** the SoS idea scales to chess with search budget as an explicit prompt parameter. Breadth and depth are still fixed by the prompt, not chosen by the model.

**Also verified (anchors):**
- **Fan et al., "Looped Transformers for Length Generalization"** (ICLR 2025). https://arxiv.org/abs/2409.15647
  - Abstract: "looped Transformers with an adaptive number of steps significantly improve length generalization"
- **Zhou et al., "Transformers Can Achieve Length Generalization But Not Robustly"** (2024). https://arxiv.org/abs/2402.09371
  - Abstract: success is "intricately linked to the data format and the type of position encoding"
  - Abstract: success is "significantly influenced by factors like random weight initialization and training data order"
- **Jenner et al., "Evidence of Learned Look-Ahead in a Chess-Playing Neural Network"** (NeurIPS 2024). https://arxiv.org/abs/2406.00877
  - Abstract: "attention heads that move important information 'forward and backward in time'"
  - Abstract: a probe "can predict the optimal move 2 turns ahead with 92% accuracy"

---

## Thread 2. Parallel and branching reasoning with learned aggregation

Axes to compare: how branches are spawned, what crosses the merge (text summary, full KV, or a vector), and whether the merge is learned.

**2.1 Pan et al., "Learning Adaptive Parallel Reasoning with Language Models" (APR)** (2025; COLM 2025). https://arxiv.org/abs/2504.15466
- **What it does:** the model emits `spawn()` and `join()` itself. Children run in parallel and return messages. Parent and children are trained end-to-end with RL on Countdown, from scratch.
- **Quotes:**
  - Spawn and merge: Fig. 1 caption: "Child threads have access only to a limited context passed to them by the parent thread and return a summary of their execution to the parent thread." §3.2: the parent "continues, conditioned only on its previous context before spawning the child threads and messages returned by the child threads."
  - RL result, §4: "the primary benefit of reinforcement learning comes from scaling test-time compute rather than improving decision quality within a fixed budget." With 10 threads forced, accuracy went "from 83.2 to 83.3%".
  - Future work, §5: "Better orchestration and communication protocols ... any-to-any messaging, all-to-all communication patterns, or subscription-based methods."
- **For extending the mechanism:** branching and merging are learned, but the merge message is model-written text. RL learned to *spend more*, not to branch or merge *better*.

**2.2 Yang, An, Liu, Chen, Chen, "Multiverse: Your Language Models Secretly Decide How to Parallelize and Merge Generation"** (2025, arXiv). https://arxiv.org/abs/2506.09991
- **What it does:** Map, Process and Reduce stages generated natively, with an attention mask that isolates branches. Trained by SFT from an AR model on 1K curated examples.
- **Quotes:**
  - §1: in s1K long CoTs, "over 98% exhibit parallelizable branches"
  - §4.2: the Reduce stage is "conditioned on preceding tokens from all branches", which ensures "every branch remains fully accessible in the Reduce stage and beyond."
  - Future work, §9: "a key direction in future research is to integrate Reinforcement Learning (RL) into training to explore and encourage more parallelism"
- **For extending the mechanism:** this is the "lossless merge" end of the spectrum. The merge attends to all branch KVs, so it is rich but unbounded. Branching was learned only by imitation.

**2.3 Venkatraman et al., "Recursive Self-Aggregation Unlocks Deep Thinking in Large Language Models" (RSA)** (2025, arXiv). https://arxiv.org/abs/2509.26626
- **What it does:** keeps a population of chains. Each step aggregates random subsets into a new population, and an "aggregation-aware" RL step trains the combine operation.
- **Quotes:**
  - Abstract: "Each step of RSA refines a population of candidate reasoning chains through aggregation of subsets"
  - §1: "RSA maintains a candidate population larger than the aggregation set size and can therefore jointly consider significantly more proposals than single-step aggregation, which is constrained by the model's effective context length."
  - Future work, §6: "use multi-step reinforcement learning to train the policy for the end-to-end RSA procedure, moving beyond the greedy single-step aggregation explored in this work."
- **For extending the mechanism:** this is bounded-context consensus done iteratively, with aggregation inside the context window. The authors name end-to-end multi-step training of the recursion as open.

**2.4 Zhao et al., "The Majority is not always right: RL training for solution aggregation" (AggLM)** (2025, arXiv). https://arxiv.org/abs/2509.06870
- **What it does:** RLVR trains an aggregator that reads k candidate solutions in context and writes a final answer.
- **Quotes:**
  - Abstract: "we propose to learn aggregation as an explicit reasoning skill"
  - Abstract: "careful balancing of easy and hard training examples, allowing the model to learn both to recover minority-but-correct answers as well as easy majority-correct answers."
  - Body: "learned generative aggregation outperforms frequency- and reward-model–based selection at surfacing correct minority solutions."
- **For extending the mechanism:** a learned in-context aggregator beats voting precisely where voting fails, on minority-correct answers. The generator stays fixed.

**2.5 Wen et al., "ParaThinker: Native Parallel Thinking as a New Paradigm to Scale LLM Test-time Compute"** (2025, arXiv). https://arxiv.org/abs/2509.04475
- **What it does:** trains `<think i>` control tokens, per-path learnable positional embeddings, and a merge that reuses the branch KV caches.
- **Quotes:**
  - Abstract: names the failure it targets, "Tunnel Vision", "where a model's imperfect initial steps lock it into a suboptimal reasoning path."
  - §1: "a two-phase attention mask design that enforces independence during reasoning and controlled integration during summarization."
  - §3: "merges the reasoning paths by reusing their KV-caches to generate the final answer."
- **For extending the mechanism:** the merge reads high-dimensional KV state, not text summaries. The number of branches is fixed, not chosen.

**Also verified:**
- **Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in Language Models"** (ICLR 2023). https://arxiv.org/abs/2203.11171
  - Abstract: "selects the most consistent answer by marginalizing out the sampled reasoning paths". The aggregation is not learned.
- **Yao et al., "Tree of Thoughts"** (NeurIPS 2023). https://arxiv.org/abs/2305.10601
  - Abstract: "considering multiple different reasoning paths and self-evaluating choices". The search is external scaffolding.
- **Rodionov et al., "Hogwild! Inference"** (NeurIPS 2025). https://arxiv.org/abs/2504.06261
  - Abstract: "modern reasoning-capable LLMs can perform inference with shared Key-Value cache out of the box, without additional fine-tuning."
- **Aggarwal et al. (incl. ParaGator), "Reasoning over mathematical objects: on-policy reward modeling and test time aggregation"** (2026). https://arxiv.org/abs/2603.18886
  - §3: "the self-aggregation method is upper bounded by the pass@k performance and that the model is unable to synthesize correct solutions based on incorrect ones"
  - §3: generation and aggregation are trained jointly, pass@k for generation and pass@1 for aggregation, "online and end-to-end".
- **Wu et al., "Native Parallel Reasoner"** (2025). https://arxiv.org/abs/2512.07461
  - Abstract: PAPO "optimizes branching policies directly within the execution graph"
  - Abstract: "100% genuine parallel execution"
- **Lian et al., "ThreadWeaver"** (ICML 2026 oral). https://arxiv.org/abs/2512.07843
  - Abstract: RL "trains the model to balance reasoning accuracy with effective parallelization."
- **Qi et al., "Learning to Reason Across Parallel Samples" (SSA)** (2025). https://arxiv.org/abs/2506.09014
  - Abstract: a separate compact aggregator "takes a concatenated sequence of multiple samples and output the final answer"
- **Zheng et al., "Parallel-R1"** (2025). https://arxiv.org/abs/2509.07980
  - Abstract: parallel thinking acts as a "mid-training exploration scaffold"
- **Hao et al., "Coconut"** (COLM 2025). https://arxiv.org/abs/2412.06769
  - Abstract: "continuous thoughts can encode multiple alternative next steps, allowing the model to perform a breadth-first search"
- **Zhu et al., "Reasoning by Superposition"** (NeurIPS 2025). https://arxiv.org/abs/2505.12514
  - Abstract: "encoding of multiple search frontiers as a superposition state automatically emerges in training continuous CoTs"
- **Tang et al., "Multiplex Thinking"** (2026). https://arxiv.org/abs/2601.08808
  - Abstract: "samples K candidate tokens and aggregates their embeddings into a single continuous multiplex token"

---

## Thread 3. Choosing what to think about first: learned and adaptive generation order

**3.1 Kim, Shah, Kontonis, Kakade, Chen, "Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions"** (2025, arXiv). https://arxiv.org/abs/2502.06768
- **What it does:** shows that MDMs train on intractable infilling subproblems, and that adaptive decoding order (top-probability margin) sidesteps them at inference.
- **Quotes:**
  - Abstract: "adaptive inference can boost solving accuracy in pretrained MDMs from <7% to ≈90%, even outperforming ARMs with 7× as many parameters and that were explicitly trained via teacher forcing to learn the right order of decoding."
  - §4: "a pretrained MDM, even if it performs poorly on certain hard subproblems, still contains sufficient information to avoid them when paired with an effective oracle"
  - Future work, §5: "An important direction for future work is to go beyond the relatively simple adaptive strategies to find a better generation order like top probability and top probability margin considered here."
- **For extending the mechanism:** this is the cleanest evidence that order choice matters for reasoning. The "what first" decision can be read off the model's own uncertainty. The order rule itself is a heuristic, and the authors name learning it as open.

**3.2 Xu, Liu, Agrawal, Chen, Huang, "Scheduling Thoughts: Learning the Order of Thought in Diffusion Language Models" (SAS)** (2026, arXiv). https://arxiv.org/abs/2606.23567
- **What it does:** trains a lightweight GRPO order policy over a frozen denoiser, using a reward derived from a KL bound on decoding mismatch.
- **Quotes:**
  - Abstract: "the unmasking order defines an 'order of thought' that strongly influences generation quality yet is typically chosen heuristically."
  - Abstract: "On Sudoku with 1B MDM, SAS improves puzzle accuracy from 82.0% (best heuristic schedule) to 91.8%, and reaches 97.5% with second-stage fine-tuning along learned trajectories."
  - Abstract: "on GSM8K from 64% to 76%"
- **For extending the mechanism:** this is a directly learned "what to think about first" policy that beats Kim et al.'s heuristic. The denoiser is frozen during order learning.

**3.3 Jazbec et al., "Learning Unmasking Policies for Diffusion Language Models"** (2025; ICML 2026 oral spotlight). https://arxiv.org/abs/2512.09106
- **What it does:** frames unmasking as an MDP with the dLLM as environment, and trains a single-layer transformer policy on token confidences with RL.
- **Quotes:**
  - §1 Introduction: "What is the optimal order in which to unmask the sequence of tokens, in order to strike a good balance between correctnes [sic] and efficiency?"
  - Abstract: trained policies "match the performance of state-of-the-art heuristics when combined with semi-autoregressive (block) generation, while outperforming them in the full-diffusion setting."
  - Future work, §6: "learning to remask previously unmasked tokens"
- **For extending the mechanism:** the policy sees only confidences, not hidden state, and its gains appear where order freedom is largest. Revisability (remasking) is still open.

**3.4 Ma, Zhong, Wang, "Reinforced Context Order Recovery for Adaptive Reasoning and Planning" (ReCOR)** (2025, arXiv). https://arxiv.org/abs/2508.13070
- **What it does:** RL recovers data-dependent token orders without order annotations. The order policy and the token predictor are trained jointly.
- **Quotes:**
  - Abstract: "current causal and diffusion models encounter difficulties in problems that require adaptive token generation orders to solve tractably"
  - Abstract: "sometimes outperforming oracle models supervised with the ground-truth order."
  - §1: "ReCOR jointly optimizes the token prediction model and the order prediction policy"
- **For extending the mechanism:** this is the closest to an autoregressive transformer that learns its own order end-to-end. It has been tested on arithmetic and puzzles only.

**3.5 Bachmann and Nagarajan, "The pitfalls of next-token prediction"** (ICML 2024). https://arxiv.org/abs/2403.06963
- **What it does:** a minimal path-star planning task on which teacher-forced next-token training fails to learn the predictor at all. Reversing the target order or teacherless training fixes it.
- **Quotes:**
  - Abstract: "in certain classes of tasks, teacher-forcing can simply fail to learn an accurate next-token predictor in the first place."
  - The section list names the fixes: "Removing the Clever Hans cheat via teacherless training" and "Removing the Indecipherable Token failure via path reversal."
- **For extending the mechanism:** the fixed left-to-right order can make a learnable task unlearnable. Order is a training-time variable, not only an inference-time one.

**Also verified:**
- **Pannatier et al., σ-GPT** (ECML/PKDD 2024). https://arxiv.org/abs/2404.09562
  - Abstract: "this order can be modulated on-the-fly per-sample"
  - Note: σ-GPT trains on random or fixed orders and does *not* learn an order policy. §2.5: "Using a randomized order during training allows conditional density estimation, infilling, and burst-sampling."
- **Wang et al., LO-ARM** (ICML 2025, PMLR v267). https://arxiv.org/abs/2503.05979
  - Abstract: "a trainable probability distribution, referred to as an order-policy, that dynamically decides the autoregressive order in a state-dependent manner". Tested on images and molecules, not reasoning.
- **Garg, Kohli, Sarawagi, "Masked Diffusion Models are Secretly Learned-Order Autoregressive Models"** (2025). https://arxiv.org/abs/2511.19152
  - Abstract: the MDM objective "decomposes precisely into a weighted auto-regressive losses over these orders"
- **Yu et al., "Thinking Out of Order"** (2026). https://arxiv.org/abs/2601.22035
  - Abstract: answer-before-reasoning prompts cause "up to 67% relative drop" for AR models, against "≤14%" for MDLMs.
- **Sato et al., "Discovering Learning-Friendly Generation Orders"** (2025). https://arxiv.org/abs/2506.23875
  - Abstract: "the generation order of intermediate states strongly affects whether training succeeds"
- **Patel et al., "Insertion Language Models"** (2025). https://arxiv.org/abs/2505.05755
  - Abstract: "ILMs outperform both ARMs and MDMs on common planning tasks"
- **Nie et al., LLaDA** (2025). https://arxiv.org/abs/2502.09992
  - Abstract: "LLaDA addresses the reversal curse"
- **Zelikman et al., Quiet-STaR** (2024). https://arxiv.org/abs/2403.09629
  - Abstract: "LMs learn to generate rationales at each token to explain future text"
- **Goyal et al., pause tokens** (ICLR 2024). https://arxiv.org/abs/2310.02226
  - Abstract: "inference-time delays show gains when the model is both pre-trained and finetuned with delays"
- **Wang et al., planning tokens** (COLM 2024). https://arxiv.org/abs/2310.05707
  - Abstract: "a planning token at the start of each reasoning step"

---

## Thread 4. Transformers as in-context aggregators and Bayesian reasoners

**4.1 Müller, Hollmann, Pineda Arango, Grabocka, Hutter, "Transformers Can Do Bayesian Inference" (PFNs)** (ICLR 2022). https://arxiv.org/abs/2112.10510
- **What it does:** trains a transformer on prior samples so that its in-context prediction approximates the posterior predictive. TabPFN is the tabular instance: https://arxiv.org/abs/2207.01848
- **Quotes:**
  - Abstract: "PFNs make probabilistic predictions for arbitrary other data points in a single forward propagation, having learned to approximate Bayesian inference."
  - Abstract: "PFNs can near-perfectly mimic Gaussian processes"
- **For extending the mechanism:** amortised aggregation is learned by choosing the training prior, the mirror image of Thread 1's "right training distribution". The inputs are exchangeable data points, not agent messages.

**4.2 Agarwal, Dalal, Misra, "The Bayesian Geometry of Transformer Attention"** (2025, arXiv). https://arxiv.org/abs/2512.22471
- **What it does:** "Bayesian wind tunnels" with closed-form posteriors in which memorisation is impossible, plus a mechanistic reading of how small transformers implement the update.
- **Quotes:**
  - Abstract: "small transformers reproduce Bayesian posteriors with 10^-3-10^-4 bit accuracy, while capacity-matched MLPs fail by orders of magnitude"
  - Abstract: "residual streams serve as the belief substrate, feed-forward networks perform the posterior update, and attention provides content-addressable routing."
  - Future work, §8: "Future wind tunnels could incorporate richer latent-variable structures, including Kalman filtering, hierarchical Bayesian models, or causal graphical models"
- **For extending the mechanism:** this gives a verifiable testbed and a mechanism split: attention routes, MLP updates. Their tasks are not multi-source aggregation.

**4.3 Bakalova, Veitsman, Huang, Hahn, "Contextualize-then-Aggregate: Circuits for In-Context Learning in Gemma-2 2B"** (2025, arXiv). https://arxiv.org/abs/2504.00132
- **What it does:** causal interventions on information flow during few-shot ICL in a real 2B model.
- **Quotes:**
  - Abstract: "In the lower layers, the model builds up representations of individual fewshot examples, which are contextualized by preceding examples ... In the higher layers, these representations are aggregated to identify the task"
- **For extending the mechanism:** this is mechanistic evidence that in-context aggregation is a two-stage, layer-separated process in a pretrained LM.

**4.4 Falck, Wang, Holmes, "Is In-Context Learning in Large Language Models Bayesian? A Martingale Perspective"** (ICML 2024). https://arxiv.org/abs/2406.00793
- **What it does:** martingale checks that any Bayesian learner on exchangeable data must pass, applied to LLM ICL.
- **Quotes:**
  - Abstract: "we provide evidence for violations of the martingale property, and deviations from a Bayesian scaling behaviour of uncertainty, falsifying the hypothesis that ICL is Bayesian."
- **For extending the mechanism:** a pretrained LLM is not automatically a coherent aggregator. Coherence has to be trained (as in PFNs) or tested for.

**4.5 Kawada and Kellis, "Evidence Integration in Large Language Models"** (2026, arXiv). https://arxiv.org/abs/2609.04290
- **What it does:** a distributional theory of how a receiving LLM updates on external candidate answers, tested over 10M trials and 12 models, with causal localisation.
- **Quotes:**
  - Abstract: "receivers more readily integrate characteristic errors of their own than foreign errors from different sources"
  - Abstract: "LLMs also integrate candidates even after internally verifying their invalidity"
  - Introduction: "sequential evidence produces the predicted sub-additivity and order effects."
- **For extending the mechanism:** this measures what goes wrong when a transformer merges others' outputs: correlated errors and order effects. It bears directly on learned merges in Thread 2.

**Also verified:**
- **Xie et al., "An Explanation of In-context Learning as Implicit Bayesian Inference"** (ICLR 2022). https://arxiv.org/abs/2111.02080
  - Abstract: "in-context learning occurs when the LM also infers a shared latent concept between examples in a prompt."
- **Panwar, Ahuja, Goyal, "In-Context Learning through the Bayesian Prism"** (ICLR 2024). https://arxiv.org/abs/2306.04891
  - Abstract: "Where Bayesian inference is tractable, we find evidence that high-capacity transformers mimic the Bayesian predictor."
- **Schoenegger et al., "Wisdom of the Silicon Crowd"** (2024; Science Advances). https://arxiv.org/abs/2402.19379
  - Abstract: giving GPT-4 or Claude 2 the human median in context improved accuracy, "though this leads to less accurate predictions than simply averaging human and machine forecasts." Untrained in-context aggregation lost to a fixed average.

---

## Thread 5. Bounded context with continued search

**5.1 Aghajohari et al., "The Markovian Thinker" (Delethink)** (2025, arXiv). https://arxiv.org/abs/2510.06557
- **What it does:** the RL environment resets context every C tokens and keeps only a short carryover. The policy learns what state to write.
- **Quotes:**
  - Abstract: "Through RL, the policy learns to write a textual state near the end of each chunk sufficient for seamless continuation of reasoning after reset."
  - Abstract: "an R1-Distill 1.5B model reasons in 8K-token chunks yet thinks up to 24K tokens, matching or surpassing LongCoT-RL trained with a 24K budget."
  - What is lost, §7: "tasks that require writing to a memory like crossword might be challenging as solved cases may be lost if not carried by the model"
  - §7: "As the markovian state carried between chunks shrinks too much, the model may lose the thread of reasoning"
- **For extending the mechanism:** the compression is learned by outcome RL, not supervised. The loss shows up on tasks with a large live state.

**5.2 Yang, Srebro, McAllester, Li, "PENCIL: Long Thoughts with Short Memory"** (ICML 2025). https://arxiv.org/abs/2503.14337
- **What it does:** a `[CALL] ... [SEP] ... [RETURN]` reduction rule erases finished sub-computations. It is trained on traces annotated with the rule.
- **Quotes:**
  - §1 Introduction: "Once a computation completes (marked by [RETURN]), all intermediate reasoning steps (those between [CALL]and [SEP]) will be removed, merging the answer back into the context."
  - Abstract: "a small 25M-parameter transformer and 2048 context length solves Einstein's puzzle"
  - What is lost, §3.1: "once a branch returns, its intermediate reasoning steps are discarded, therefore search paths will be discarded, preserving only the final answer."
- **For extending the mechanism:** space-optimal search inside next-token prediction (there is a Turing-machine simulation result). When to reduce is learned by imitation, not discovered.

**5.3 Yan et al., "InftyThink"** (2025; ICLR 2026). https://arxiv.org/abs/2503.06692
- **What it does:** SFT on long CoTs rewritten into reason-then-summarise segments.
- **Quotes:**
  - Abstract: "interleaving short reasoning segments with concise progress summaries"
  - What is lost, §4.3: larger segments "may commit the model to reasoning paths that become difficult to revise, whereas smaller segments allow more flexible exploration over multiple iterations."
  - What is lost, §4.4, without training: "the model may lose important information across multiple reasoning iterations"
- **For extending the mechanism:** summary granularity trades commitment against exploration, which ties back to ParaThinker's "Tunnel Vision".

**5.4 Sun et al., "Scaling Long-Horizon LLM Agent via Context-Folding"** (2025, arXiv). https://arxiv.org/abs/2510.11967
- **What it does:** the agent calls `branch` and `return` actions. Returned branches are folded to a summary. Trained with FoldGRPO.
- **Quotes:**
  - Abstract: "matches or outperforms the ReAct baselines while using an active context 10× smaller and significantly outperforms models that rely on summarization-based context management."
  - §4.5.3: "Whether the folding agent can benefit from parallel branching ... remains an open question." Their parallel version performed "similarly to the single-branch version."
- **For extending the mechanism:** the same spawn/return-summary shape as APR, applied to context management. Parallel fan-out did not add accuracy here.

**5.5 Zhou et al., "MEM1"** (2025, arXiv). https://arxiv.org/abs/2506.15841
- **What it does:** RL trains a constant-size internal state that merges memory and reasoning at every turn.
- **Quotes:**
  - Abstract: "updates a compact shared internal state that jointly supports memory consolidation and reasoning"
  - Abstract: the agent "generalizes beyond the training horizon"
- **For extending the mechanism:** consolidation is learned and bounded, but tested on multi-hop QA and web tasks, not on search with backtracking.

**Also verified:**
- **Zhang et al., "LightThinker"** (EMNLP 2025). https://arxiv.org/abs/2502.15589
  - Abstract: "compresses verbose thought steps into compact representations and discards the original reasoning chains". The compressed state is hidden-state gist tokens, not text.

---

## Thread 6. RL and in-context search: reweighting or creating?

**6.1 Yue et al., "Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?"** (NeurIPS 2025 oral). https://arxiv.org/abs/2504.13837
- **Quotes:**
  - Abstract: "the current training setup does not elicit fundamentally new reasoning patterns"
  - Abstract: "the base models achieve a higher pass@k score when k is large"
  - Abstract: "distillation can introduce new reasoning patterns from the teacher"
- **For extending the mechanism:** under standard RLVR, the search strategies are the base model's. New ones come from data (SFT or distillation).

**6.2 Liu et al., "ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models"** (2025, arXiv). https://arxiv.org/abs/2505.24864
- **Quotes:**
  - Abstract: "prolonged RL (ProRL) training can uncover novel reasoning strategies that are inaccessible to base models, even under extensive sampling."
  - Abstract: "reasoning boundary improvements correlates strongly with task competence of base model and training duration"
- **For extending the mechanism:** the counter-evidence depends on long training, KL control, resets and diverse tasks, and the gains are largest where the base model is weak.

**6.3 Zhang, Neubig, Yue, "On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models"** (2025; ICML 2026 spotlight per icml.cc). https://arxiv.org/abs/2512.07783
- **Quotes:**
  - Abstract: "RL produces true capability gains (pass@128) only when pre-training leaves sufficient headroom and when RL data target the model's edge of competence"
  - Abstract: "Process-level rewards reduce reward hacking and improve reasoning fidelity."
- **For extending the mechanism:** a controlled synthetic setup that reconciles 6.1 and 6.2. Whether RL creates a strategy depends on where the training tasks sit relative to competence.

**6.4 Yuan et al., "From f(x) and g(x) to f(g(x)): LLMs Learn New Skills in RL by Composing Old Ones"** (2025, arXiv). https://arxiv.org/abs/2509.25123
- **Quotes:**
  - Abstract: "When an LLM has already learned f and g prior to RL, our experiments reveal that RL enables it to learn unseen compositions"
  - Abstract: "next-token training with the same data yields none of these findings."
- **For extending the mechanism:** RL can create *compositions* of existing in-context skills, which matters if search is composed from expand, evaluate and merge primitives.

**6.5 RL on the branch and merge decisions themselves:** APR (2.1) and ParaGator (2.x).
- **APR, §4:** RL's gain is "scaling test-time compute rather than improving decision quality within a fixed budget".
- **ParaGator, §3:** "standard outcome-based RL encourages the model to converge toward a single high-probability solution mode", and "This diversity collapse leaves the aggregator with redundant candidates".
- **Also:** RSA (2.3) and AggLM (2.4) train the aggregation step with RLVR.
- **For extending the mechanism:** where RL acts directly on branching, the measured effect is more compute or better merging of an existing pool. No verified result shows RL learning a qualitatively better *branching policy* at fixed compute.

**Also verified:**
- **Wu et al., "The Invisible Leash"** (2025). https://arxiv.org/abs/2507.14843
  - Abstract: "the shrinkage of observed support outweighs its expansion"
- **Zhao et al., "Echo Chamber"** (COLM 2025). https://arxiv.org/abs/2504.07912
  - Abstract: "RL algorithms consistently converge towards a dominant output distribution, amplifying patterns in the pretraining data"
- **Gandhi et al., "Cognitive Behaviors that Enable Self-Improving Reasoners"** (2025). https://arxiv.org/abs/2503.01307
  - Abstract: "the presence of reasoning behaviors, rather than correctness of answers, proves to be the critical factor"
- **Zhai et al., "Does RL Expand the Capability Boundary of LLM Agents? A PASS@(k,T) Analysis"** (2026). https://arxiv.org/abs/2604.14877
  - Abstract: "tool-use RL genuinely enlarges the capability boundary"
  - Abstract: "RL reweights the base strategy distribution toward the subset whose downstream reasoning more often yields a correct answer, with the improvement concentrated on how the agent integrates retrieved information."
- **Nguyen et al., "The Reasoning Boundary Paradox"** (2025). https://arxiv.org/abs/2510.02230
  - Abstract: a "winner-take-all phenomenon" in RLVR.

---

## The 5 papers to read first

1. **Saparov et al. 2024/ICLR 2025 (1.1).** What in-weights search a transformer actually learns (parallel path-merging), and the exact conditions under which held-out generalisation fails. Read with Ye et al. (1.4) for the capacity-keyed data account.
2. **Kim et al. 2025 (3.1).** The strongest evidence that "what to think about first" matters, and that the model's own uncertainty already carries the signal. Its future-work line asks for learned orders, which SAS (3.2) then partly supplies.
3. **Pan et al., APR, 2025 (2.1).** Learned spawn and join inside next-token generation, trained end-to-end with RL. It contains the sobering finding that RL bought compute, not decision quality, and lists richer communication protocols as future work.
4. **Venkatraman et al., RSA, 2025 (2.3).** Bounded-context, iterated in-context aggregation over a population, with a learned combine step. Its future-work line asks for multi-step RL of the whole recursion.
5. **Aghajohari et al., Delethink, 2025 (5.1).** Learned constant-size state for continued reasoning under a hard context bound, with a concrete account of where it breaks (large live state).

---

## Three open gaps (none of the papers verified here fills them)

These are stated as open questions, not as the direction to take. "No paper" means none of the papers I found and read.

**Gap 1. Learned order of *search expansions*, not of output tokens.**
- **What exists:** every learned-order result above (SAS, Jazbec et al., ReCOR, LO-ARM, Kim et al.'s heuristics) chooses which *output position* to fill in a fixed-length target (Sudoku cells, GSM8K answer tokens, molecules).
- **What is missing:** none lets a next-token transformer learn which *open subproblem or frontier node* to expand next inside a variable-length search trace, and then tests held-out larger instances (the generalisation axis of Thread 1).
- **Closest:** ReCOR (3.4), which jointly learns order and tokens, and Kim et al. (3.1), whose future work asks for exactly the learned order. On the search side, SoS and Searchformer fix the expansion order by imitating a symbolic solver.

**Gap 2. Training the merge or compress step so the merged state supports *continued search*, under a fixed bound, end-to-end over many rounds.**
- **What exists:**
  - Learned merges are trained on final-answer reward after one round: AggLM; ParaGator; RSA's "greedy single-step aggregation" (its own words).
  - Bounded-state methods (Delethink, Context-Folding, PENCIL) compress one serial thread.
  - APR's join is learned, but RL did not improve decision quality at fixed compute.
- **What is missing:** none trains a multi-round branch → merge → re-branch loop end-to-end. None measures whether the merged state is *sufficient* for the next round of search, rather than only whether the final answer is right.
- **Closest:** RSA (2.3), whose future work names multi-step RL of the recursion; APR (2.1); Delethink (5.1).
- **Open design axis:** what crosses the merge. Text summaries (APR, Context-Folding, InftyThink), full branch KV (Multiverse, ParaThinker, Hogwild), or a vector (Coconut, Multiplex Thinking). No paper compares these at a fixed bound.

**Gap 3. Whether a learned in-context aggregator combines *dependent* branch evidence correctly.**
- **What exists:**
  - Bayesian-ICL results (PFNs, wind tunnels, Panwar et al.) verify posteriors over *exchangeable* data with a known prior.
  - Parallel-sample aggregators (AggLM, SSA, RSA, ParaGator) are trained only on answer correctness. Their candidates all come from one model, so errors are correlated.
  - Kawada and Kellis show LLM receivers over-adopt errors congruent with their own and show sequential sub-additivity and order effects.
  - ParaGator states the aggregator "is unable to synthesize correct solutions based on incorrect ones."
- **What is missing:** none tests a learned branch aggregator against a known-posterior benchmark in which branches share errors, so it is unknown whether learned aggregation discounts correlated evidence or merely votes better.
- **Closest:** Kawada and Kellis (4.5) for the measurement; Agarwal et al. (4.2) for the wind-tunnel method; AggLM (2.4) for the trained aggregator.
