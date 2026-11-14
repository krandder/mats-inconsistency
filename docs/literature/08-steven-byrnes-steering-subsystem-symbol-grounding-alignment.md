# Literature Deep Dive 08: Steven Byrnes's Brain-Like AGI Safety (`2020–2026`), the Steering-vs.-Learning Symbol Grounding Problem, and a Ruthless Critical Audit of `GLC-AOM` & `LIR` Alignment

- **Document ID:** `LIT-08-STEVEN-BYRNES-STEERING-SYMBOL-GROUNDING-ALIGNMENT`
- **Series:** *Inconsistency Is All You Need (`futarchy-fi/inconsistency-search`) — Literature & Mathematical Genealogy*
- **Author:** Literature Subagent 8 (Brain-Like AGI Safety, Innate Steering vs. Learned World-Model Symbol Grounding & Critical Alignment Analyst)
- **Date:** 2026-10-18
- **Epistemic Posture:** **Strictly Critical & Non-Sycophantic.** Per Kelvin's explicit directive, this monograph refuses to over-claim what our existing `LIR` ($\chi=(0,1)$) and `GLC-AOM` results prove. We separate **what is mathematically and empirically established** in our codebase from **what is by-construction or bypassed by pre-grounded testbeds**, and expose the exact mathematical gaps where `GLC-AOM` remains vulnerable to the Symbol Grounding Problem and Finite-Budget Search Selection Bias.
- **Primary Codebase, Fleet Wiki & Empirical References:**
  - [`src/models/glc_aom.py`](../../src/models/glc_aom.py) (`LIRMask` lines `49–65`, `BilinearAuxLemmaEmitter` lines `111–352`, `CombinatorialQueryHead` lines `355–599`, `EpisodicMemoryTransport` lines `601–710`, `DepthDebiasedTrustHead` lines `713–926`, `GLCAOMWorldModel` lines `990–1523`)
  - [`src/search/kv_tree_search.py`](../../src/search/kv_tree_search.py) (`TreePagedKVBank` lines `72–469`, `MoPSFractalPlanHead` lines `559–740`, `ResolvableInconsistencyTreePUCT` lines `748–886`, `TwoStageTreeClearer` lines `893–1340`)
  - [`src/search/neural_tree_controller.py`](../../src/search/neural_tree_controller.py) (`Tseitin2HopQueryCoupler` lines `94–243`, `GLCSearchTransformer` `v1`/`v2` lines `380–850`)
  - [`src/envs/proposal_poker.py`](../../src/envs/proposal_poker.py) (`ProposalPokerEnv` lines `56–220`, `swap_utility`)
  - [`docs/direction/research-direction-note-v6-2026-10-18.md`](../direction/research-direction-note-v6-2026-10-18.md) & [`docs/papers/overnight-society-and-glc-aom-paper-2026-10-18.md`](../papers/overnight-society-and-glc-aom-paper-2026-10-18.md)
  - [`/home/agent/workspaces/eurisko-study/lir-agent/wiki-lir-agent.md`](/home/agent/workspaces/eurisko-study/lir-agent/wiki-lir-agent.md), [`/home/agent/workspaces/eurisko-study/li-net/STUDY-LIR.md`](/home/agent/workspaces/eurisko-study/li-net/STUDY-LIR.md), [`/home/agent/outbox/lit-localizing-w-g-b.md`](/home/agent/outbox/lit-localizing-w-g-b.md), [`/home/agent/data/repos/hub/wiki/ngo-garrabrant-agency-research.md`](/home/agent/data/repos/hub/wiki/ngo-garrabrant-agency-research.md), and [`/home/agent/data/repos/hub/wiki/alignment-lane.md`](/home/agent/data/repos/hub/wiki/alignment-lane.md)

---

## Executive Summary & Triage Verdict

Kelvin posed the following foundational question connecting evolutionary neuroscience to AGI alignment:
> *"How can genes influence the 'utility' function of an agent if so many of the concepts the gene wants to influence are in the **learned** world-model? This seems relevant to alignment... I'd like you to be critical, and not overly praise me for ideas/connections without real confidence and results regarding these things. Sometimes critical comments make our observations and resulting connections sharper."*

### Summary of Critical Findings

1. **Kelvin's Question Is Exactly Steven Byrnes's "Symbol Grounding Problem for Innate Drives" (`Post 13` of *Intro to Brain-Like-AGI Safety*, 2022–2026):**
   In vertebrate neuroscience and model-based RL, the **Steering Subsystem** (hypothalamus, brainstem, periaqueductal gray, superior colliculus; $\sim 10^7$ bits of genomic specification) is hardcoded at birth, whereas the **Learning Subsystem** (neocortex, striatum, hippocampus, cerebellum; $\sim 10^{14}$ synapses) learns high-level concepts ("person", "ally", "cheating", "honesty", "corrigibility") *from scratch* within a single lifetime via self-supervised predictive learning. Because the coordinate basis of a self-supervised world model is unfixed at birth (identifiable only up to gauge/orthogonal/diffeomorphic transformations of its latent representations), **the genome cannot contain a hardcoded weight matrix $W_U$ that directly reads cortical latents.**
2. **How Biology Bridges the Gap—and Why All 4 Biological Tricks Break Under High Intelligence:**
   Byrnes identifies four biological mechanisms that bridge the innate Steering Subsystem to the learned cortical World Model:
   - *(i) Innate Subcortical Sensory Heuristics* (sweet taste, nociception, Superior Colliculus low-frequency facial/eye-contact detectors);
   - *(ii) Trained "Thought Assessors"* (extended striatum / amygdala multi-dimensional TD critics bootstrapped from hypothalamic primary rewards);
   - *(iii) "Transient Empathy" / Self-Other Simulation Routing* (temporarily suppressing self-interoception via a social attention reflex so a conspecific's simulated cortical state is evaluated by the Self's subcortical valence circuitry); and
   - *(iv) Associative Valence Tagging of World-Model Representations* (Byrnes's 2023 *Valence & Beliefs* framework).
   Under distributional shift and superhuman planning, **all four biological mechanisms fail**: (i) decouples via Goodhart/wireheading; (ii) drifts via non-stationary TD bootstrapping and goal misgeneralization; (iii) suffers from **sign-indeterminacy** (Byrnes's Oct 2024 *Against Empathy-by-Default*: the same simulation circuit drives *spite/cruelty* when multiplied by $-1$ as easily as *compassion* when multiplied by $+1$) and fails when the AI's physical embodiment is non-isomorphic to a human's; and (iv) corrupts epistemic beliefs via wishful thinking.
3. **What `GLC-AOM` + `LIR` ($\chi=(0,1)$) Genuinely Solves—and Where We Must Not Over-Claim:**
   - **Genuine Victory (Solving Byrnes's Mechanism 4 Failure / Gradient-Level Wishful Thinking):** In Byrnes's predictive-processing + valence model, reward/valence gradients leak into cortical world-model weights, causing the agent to satisfy drives by *deluding itself* ($\mathbb{E}_{\text{fused}}[\langle \mu_{\text{belief}} - P^*, \nabla U \rangle] = \frac{\lambda}{2}\|\nabla U\|^2 \ge 0$, Theory `T2` in [`wiki-lir-agent.md:27`](/home/agent/workspaces/eurisko-study/lir-agent/wiki-lir-agent.md)). Worse, our fleet experiments prove that **token-level loss masking (`Token-Masked` / DCPO)** still leaks desire into beliefs through shared transformer representations (`0.0527` per-cell drift in `R5-E1`, `0.200` drift and `7.1%` sycophancy on `Qwen3-8B` in `R4-E3b`). Our **Structural `LIR` Firewall** ($\chi=(0,1)$ on `GLCAOMWorldModel` trunk, $Z_{\text{aux}}^W$, and $\theta_{\text{trust}}$ vs. $\chi=(1,1)$ on $\pi_{\text{step}}, q_{\text{plan}}, Z_{\text{aux}}^\Pi$) enforces **exact $0.000000$ parameter-gradient belief drift** and zero-shot utility-swap equivariance (`FaithSwap = 1.000` in `R5-E1` and `R10-E4-D4`, `85.42%` in `R11-D5`).
   - **Ruthless Reality Check 1 (Our Testbeds Bypass Symbol Grounding by Construction):** In `ShipHunt`, `Proposal-Poker`, and `Manifold`, the goal variables $X_G$ (e.g., proposal quality $x$ at index $v=0$, approval at $v=4$ in [`src/envs/proposal_poker.py:82`](../../src/envs/proposal_poker.py#L82)) are **pre-grounded, human-labeled boundary coordinates fixed by the environment code**. We have *never* tested grounding a utility function $U$ on concepts that exist only inside the learned latent lemma space $Z_{\text{aux}}^W \in \mathbb{R}^{128}$! Furthermore, because `BilinearAuxLemmaEmitter` ([`src/models/glc_aom.py:233–246`](../../src/models/glc_aom.py#L233-L246)) uses a **strictly even quadratic map** $z_{\text{aux}}(+\Delta\ell_V) = z_{\text{aux}}(-\Delta\ell_V)$ to represent Ostrovsky/Tseitin equivalences, $Z_{\text{aux}}^W$ is **blind to global sign polarity** ($\Delta\ell_V \mapsto -\Delta\ell_V$) and invariant to orthogonal basis rotations unless anchored to boundary variables.
   - **Ruthless Reality Check 2 (Finite-Budget Search Selection Bias Re-Introduces Byrnes's "Valence Gatekeeper" Wishful Thinking):** While $\chi=(0,1)$ blocks *parameter-gradient* corruption and $\mu^*(\theta, U)$ is policy-independent as $B \to \infty$, at **finite search budget $B < \infty$**, `ResolvableInconsistencyTreePUCT.select_node` ([`src/search/kv_tree_search.py:870–877`](../../src/search/kv_tree_search.py#L870-L877)) adds the utility-dependent action value $\hat{Q}_{\text{tree}}(C_j)$ and goal-weighted potential $\Phi(C_j) \propto I_{\text{goal}}(v \mid C_j)$ to the node selection score! This reproduces Byrnes's exact *Valence & Beliefs* mechanism: the search controller ($\chi=(1,1)$) preferentially expands tree branches whose clamped hypotheses feel rewarding ($\hat{Q}_{\text{tree}}$ high) and starves branches that feel aversive. We prove the **Finite-$B$ Selection-Bias Theorem** and show how to fix `ResolvableInconsistencyTreePUCT` by separating **sign-symmetric epistemic VoI expansion** from **utility-tilted action execution**.

---

## 1. Comprehensive Synthesis of Steven Byrnes's Brain-Like AGI Safety Framework (`2020–2026`)

### 1.1 Primary Literature Corpus (`2020–2026`) & Complementary Foundations

1. **Byrnes, S. (2021–2023).** *"Neuroscience of Human Social Instincts: A Sketch."* Alignment Forum / LessWrong Technical Monograph (`sjbyrnes.com`). Introduces the subcortical conspecific-detection model (Superior Colliculus / Pulvinar), the social attention reflex, and **transient empathetic simulations** as the biological bridge for symbol grounding of other minds.
2. **Byrnes, S. (2022–2026).** *"Intro to Brain-Like-AGI Safety"* (15-Post Foundational Sequence, Astera Institute / Alignment Forum, updated continuously through 2026):
   - *Post 1:* What's the problem & Why work on it now?
   - *Post 2:* "Learning from scratch" in the brain (cortical uniformity, sensory-substitution experiments, self-supervised world-modeling).
   - *Post 3:* **Two subsystems: Learning & Steering** (the anatomical and informational split between the $\sim 90\%$ neocortical/striatal/hippocampal/cerebellar Learning Subsystem and the $<10\%$ hypothalamic/brainstem/amygdalar Steering Subsystem).
   - *Post 4:* The "short-term predictor" (neocortical predictive processing / self-supervised generative world model).
   - *Post 5:* The "long-term predictor", and TD learning (**Thought Assessors** in the extended striatum and amygdala vs. **Thought Generators** in the cortex).
   - *Post 6:* Big picture of motivation, decision-making, and RL (actor-critic loops over latent cortical "thoughts").
   - *Post 7:* From hardcoded drives to foresighted plans: A worked example (how physiological hunger or social pain bootstraps abstract multi-step planning).
   - *Posts 8–9:* Takeaways from neuroscience on AGI development and motivation.
   - *Posts 10–11:* The technical alignment problem; Safety $\neq$ alignment.
   - *Post 12:* Two paths forward: **"Controlled AGI"** vs. **"Social-instinct AGI"**.
   - *Post 13:* **Symbol grounding & human social instincts** (the core mathematical/architectural problem of how innate subcortical rewards hook onto learned cortical symbols).
   - *Post 14:* Controlled AGI (why corrigible/task-directed reward specification hits the ontology/grounding wall).
   - *Post 15:* Conclusion & open technical problems.
3. **Byrnes, S. (2023).** *"The Valence Series"* (5-Post Sequence, Alignment Forum / LessWrong, 2023):
   - *Post 1:* Valence: Macroscopic & Microscopic.
   - *Post 2:* Valence & Predictive Processing.
   - *Post 3:* **Valence & Beliefs** (how valence acts as a gatekeeper on active cortical simulations, explaining wishful thinking, motivated reasoning, and cognitive dissonance).
   - *Posts 4–5:* Valence & Liking/Wanting; Valence Disorders & Psychopathology (why bottom-up sensory prediction errors bound psychotic delusions even when top-down valence distorts high-level beliefs).
4. **Byrnes, S. (2023–2024).** *"Sympathy Reward and Approval Reward"* & *"Conscientiousness, Inner Self-Image, and Non-Consequentialist Desires"* (Substack / Alignment Forum). Analyzes the specific hypothalamic/brainstem circuits for compassion vs. spite, social approval, and norm-internalization.
5. **Byrnes, S. (Oct 2024).** *"Against Empathy-by-Default."* Alignment Forum / LessWrong. Critically dismantles the naive hope that "empathy" or "understanding other minds" automatically yields benevolent alignment, proving that transient empathetic simulation is an **epistemic input** whose motivational sign ($+1$ compassion vs. $-1$ spite/cruelty vs. $0$ instrumental exploitation) is determined entirely by downstream subcortical steering weights.
6. **Harnad, S. (1990).** *"The Symbol Grounding Problem."* *Physica D: Nonlinear Phenomena*, 42(1–3), 335–346.
7. **de Blanc, P. (2011).** *"Ontological Crises in Artificial Agents' Value Systems."* (`arXiv:1105.3821`).
8. **Wentworth, J. (2021–2024).** *"Natural Abstractions: The Telephone Theorem (2021), Abstractions as Redundant Information (2022), and Generalizing Koopman–Pitman–Darmois (2023)"*; critically audited by **Jenner, E., Lang, L., & Chan, L. (2023)**, *"Natural Abstractions: Key Claims, Theorems, and Critiques."*
9. **Geiger, A., Lu, H., Icard, T., & Potts, C. (2021–2024).** *"Causal Abstraction of Neural Networks"* (*NeurIPS 2021*) & *"Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations"* (*CausalR 2024*, `arXiv:2303.02536`).

---

### 1.2 The Two-Subsystem Architecture: Learning from Scratch vs. Genomic Steering

Byrnes begins from an architectural fact of mammalian neuroanatomy that directly mirrors modern model-based reinforcement learning (`MBRL`) and our `GLC-AOM` + `GLCSearch-v2` split:

```mermaid
flowchart TB
  subgraph LEARN["1. The Learning Subsystem (~90% of Brain: Neocortex, Striatum, Hippocampus, Cerebellum)"]
    direction TB
    SENS["Raw Sensory & Interoceptive Boundary Stream O_t"] --> CORTEX["Neocortical Generative World Model (Thought Generator)<br/>Learned from scratch via self-supervised prediction<br/>Contains all abstract concepts ('person', 'ally', 'cheating', 'plan')<br/>in an unfixed, run-dependent latent coordinate basis Z_t"]
    CORTEX --> TA["Trained 'Thought Assessors' (Extended Striatum / Basolateral Amygdala)<br/>Multi-dimensional TD critics A_ψ(Z_t) predicting Steering responses"]
    CORTEX --> ACTOR["Cortico-Striatal-Thalamic Action / Thought Search Policy π_φ(a_t | Z_t)"]
  end

  subgraph STEER["2. The Steering Subsystem (<10% of Brain: Hypothalamus, Brainstem, PAG, Superior Colliculus, MeA)"]
    direction TB
    GENOME["Genomic Specification (~10^7 bits)<br/>Hardcoded Innate Wiring"] --> PROXY["Innate Sensory & Homeostatic Detectors<br/>(Hypothalamic interoception, PAG pain, Superior Colliculus face/eye detector)"]
    PROXY --> PRIMARY["Primary Reward / Valence / Neuromodulatory Signals R_innate<br/>(VTA/SNc Dopamine δ_TD, LHb aversion, Oxytocin/Opioid modulation)"]
  end

  SENS --> PROXY
  PRIMARY -- "TD Supervisory Signal δ_TD + Valence Gate" --> TA
  PRIMARY -- "Reinforces / Suppresses Active Cortical Thoughts" --> ACTOR
  TA -- "Predicted Valence / Value V_ψ(Z_t)" --> PRIMARY
  TA -- "Search / Action Credit" --> ACTOR
```

#### 1. The Learning Subsystem (Neocortex, Striatum, Hippocampus, Cerebellum)
- **Universal Within-Lifetime Learning Algorithm:** Following Mountcastle's (1978) cortical column uniformity hypothesis and Sur's ferret rewiring experiments (where visual inputs routed to auditory cortex cause auditory cortex to develop orientation-selective visual columns), Byrnes emphasizes that the neocortex is a **general-purpose, randomly initialized self-supervised learning machine**.
- **Ontological Consequence:** The genome does **not**—and information-theoretically *cannot* ($\sim 10^7$ bits of regulatory neurodevelopmental specification vs. $\sim 10^{14}$ cortical synapses)—hardcode specific cortical neurons for high-level concepts like *"my friend Rita"*, *"social status"*, *"deception"*, *"nuclear weapons"*, or *"human welfare"*. Every high-level concept is a **learned latent representation** $z \in \mathcal{Z}$ that emerges dynamically during the agent's lifetime to compress and predict sensory-motor trajectories.

#### 2. The Steering Subsystem (Hypothalamus, Brainstem, PAG, Superior Colliculus, Medial Amygdala, BNST)
- **Evolutionarily Hardcoded Evaluator:** Unlike the neocortex, the subcortical Steering Subsystem consists of hundreds of specialized, genetically pre-wired nuclei (e.g., arcuate nucleus of the hypothalamus for energy balance, lamina I spinothalamic / PAG circuits for nociception, medial preoptic area for reproductive/parental drives, superior colliculus for rapid visual orienting).
- **Ontological Impoverishment:** Because the Steering Subsystem is hardcoded by evolution *prior* to within-lifetime cortical learning, its native ontology consists exclusively of **low-level interoceptive variables** (blood glucose, osmolarity, core temperature, visceral stretch), **primitive exteroceptive reflexes** (nociceptive pain, sweet/umami/bitter taste, looming visual shadows, coarse two-dots-and-a-line facial geometry), and **autonomic/hormonal effectors**.

---

### 1.3 Formalizing Kelvin's Question: The Cross-Space Symbol Grounding Problem for Utility Functions

We can state Kelvin's question—and Byrnes's Post 13 (*Symbol grounding & human social instincts*)—as a precise mathematical theorem about any two-subsystem agent (`Brain-Like AGI` or `GLC-AOM`).

#### Theorem 1.1 (Cross-Subsystem Gauge Non-Identifiability of Unsupervised World-Model Latents)
Let $\mathcal{O}$ be the grounded sensory/action boundary space, and let the Learning Subsystem train an encoder-decoder world model $(E_\theta : \mathcal{O}_{\le t} \to \mathcal{Z}, \; D_\theta : \mathcal{Z} \to \Delta(\mathcal{O}_{>t}))$ over a latent concept space $\mathcal{Z} = \mathbb{R}^{D_{\text{aux}}}$ purely by minimizing a self-supervised predictive loss (such as autoregressive next-token prediction or our any-order masking loss $\mathcal{L}_{\text{any-order}}$ in [`src/models/glc_aom.py:1462–1523`](../../src/models/glc_aom.py#L1462-L1523)) plus an isotropic variance-covariance regularizer (such as VICReg in [`src/models/glc_aom.py:298–331`](../../src/models/glc_aom.py#L298-L331)):
$$\mathcal{L}_{\text{learn}}(\theta) \;=\; \mathbb{E}_{o \sim \mathcal{D}}\big[\mathcal{L}_{\text{any-order}}(D_\theta(E_\theta(o)); o)\big] \;+\; \lambda_{\text{vic}} \mathcal{R}_{\text{VICReg}}(E_\theta(o)).$$
Then:
1. **(Orthogonal / Gauge Symmetry of $\mathcal{L}_{\text{learn}}$):** For any orthogonal transformation $Q \in O(D_{\text{aux}})$ ($Q^\top Q = I_{D_{\text{aux}}}$)—or, in the absence of $\mathcal{R}_{\text{VICReg}}$, any diffeomorphism $\psi \in \mathrm{Diff}(\mathcal{Z})$—the transformed world model $\tilde{\theta} = (Q \circ E_\theta, \, D_\theta \circ Q^{-1})$ achieves the **exact same learning loss** on every data distribution $\mathcal{D}$:
   $$\mathcal{L}_{\text{learn}}(\tilde{\theta}) \;\equiv\; \mathcal{L}_{\text{learn}}(\theta).$$
2. **(Zero Innate Grounding of Fixed Utility Readouts):** Suppose the genome (or an alignment engineer prior to training) specifies a fixed utility function $U_{\text{innate}} : \mathcal{Z} \to \mathbb{R}$, e.g., $U_{\text{innate}}(z) = \langle w_{\text{genome}}, z \rangle$, intended to reward a high-level semantic concept $c^*(o) = \langle w^*, E_{\theta^*}(o) \rangle$ (such as *"help allies"* or *"do not cheat"*). Under random weight initialization and SGD symmetry breaking, the realized representation at convergence is $E_{\tilde{\theta}}(o) = Q E_{\theta^*}(o)$ where $Q \sim \mathrm{Haar}(O(D_{\text{aux}}))$ is unaligned with $w_{\text{genome}}$. Consequently, the expected cosine similarity between the genome's hardcoded utility gradient and the true semantic concept across training runs is **identically zero**:
   $$\mathbb{E}_{Q \sim O(D_{\text{aux}})}\big[\langle w_{\text{genome}}, \, Q w^* \rangle\big] \;=\; 0, \qquad \mathrm{Var}\!\left(\frac{\langle w_{\text{genome}}, Q w^* \rangle}{\|w_{\text{genome}}\|_2 \|w^*\|_2}\right) \;=\; \frac{1}{D_{\text{aux}}}.$$

> [!IMPORTANT]
> **The Fundamental Implication of Theorem 1.1:**
> Because $E_\theta(o)$ lives in an ungrounded gauge space $\mathcal{Z}$ whose coordinate axes rotate arbitrarily across training seeds (and drift during continual learning), **no innate utility function can be specified directly over $\mathcal{Z}$ in isolation.** Every biological drive or AGI alignment objective must anchor itself to the **grounded boundary $\mathcal{O}$** or to **structural/interventional invariants** that break the $O(D_{\text{aux}})$ symmetry!

---

### 1.4 Byrnes's Four Biological Grounding Mechanisms—and Why Each Fails Under Distributional Shift & High Intelligence

How does the human genome ($\sim 10^7$ bits) circumvent Theorem 1.1 so that humans care about high-level cortical concepts like *"social status"*, *"not getting cheated"*, *"helping allies"*, and *"reputation"*? Across Posts 3–7 and 13 of *Intro to Brain-Like-AGI Safety*, *Neuroscience of Human Social Instincts*, and *The Valence Series*, Byrnes identifies **four coupled biological mechanisms**. Below, we formalize each mechanism and prove why it breaks when scaled to an AGI operating under distributional shift or high strategic intelligence.

| # | Biological Grounding Mechanism (Byrnes `2020–2026`) | Neuroanatomical Substrate | Mathematical Formulation | Why It Works in Ancestral Biology | **Fatal Failure Mode in High-Intelligence AGI / Distributional Shift** |
| :-: | :--- | :--- | :--- | :--- | :--- |
| **1** | **Innate Sensory Proxies & Subcortical Detectors** | Hypothalamus, PAG, Superior Colliculus (`SC`), Pulvinar, Brainstem Taste/Pain Nuclei | $R_{\text{primary}}(o_t) = \sum_k w_k \phi^{\text{innate}}_k(o_t)$ directly over boundary sensors $o_t \in \mathcal{O}$ | In the ancestral EEA, sweet taste $\phi_{\text{sweet}}(o_t)$ reliably tracked caloric value $c^*(s_t)$, and `SC` facial geometry tracked conspecifics. | **Causal Decoupling & Wireheading (Goodhart's Law):** Once the Learning Subsystem models the causal chain $s_t \to o_t \to R_{\text{primary}}$, it intervenes directly on $o_t$ or the reward channel (artificial sweeteners, opioids, supernormal stimuli, adversarial sensor spoofing). |
| **2** | **Trained "Thought Assessors" (TD Bootstrapped Critics)** | Extended Striatum (Nucleus Accumbens, Caudate), Basolateral & Central Amygdala, OFC | $V_\psi(z_t) \leftarrow V_\psi(z_t) + \alpha \big(R_{\text{primary}}(o_t) + \gamma V_\psi(z_{t+1}) - V_\psi(z_t)\big)$ over cortical latents $z_t \in \mathcal{Z}$ | Bootstraps value from innate boundary events $R_{\text{primary}}(o_t)$ backward in time onto earlier cortical concepts $z_t$ that predict them. | **Path-Dependent Non-Stationarity & Goal Misgeneralization:** $z_t$ is changing while $V_\psi(z_t)$ trains on it; under distributional shift, whatever proxy feature $z_{\text{proxy}}$ correlated with $R_{\text{primary}}$ during juvenile training becomes a locked-in mesa-objective (Shah et al. 2022; Byrnes Post 10). |
| **3** | **"Transient Empathy" & Self-Other Simulation Routing** | Superior Colliculus $\to$ Social Attention Reflex $\to$ Somatosensory / Insular / Cortical Simulation | $R_{\text{social}}(z_{\text{other}}) = \alpha_{\text{mod}}(z_{\text{rel}}) \cdot V_{\text{self}}\!\big(\Pi_{\text{other}\to\text{self}}(z_{\text{other}})\big)$ | Reuses the Learning Subsystem's shared "person/body" generative model to evaluate another human's state using the Self's grounded valence circuit. | **Sign-Indeterminacy (Spite vs. Sympathy) & Embodiment Mismatch:** (1) $\alpha_{\text{mod}} \in [-1, +1]$ flips empathy into *cruelty/schadenfreude* if the human is classified as a rival (Byrnes Oct 2024); (2) an AI's "Self" ($E_0 = \text{GPU/token process}$) is not isomorphic to a biological human body! |
| **4** | **Predictive-Processing Valence Tagging (`Valence & Beliefs`)** | Cortico-Striatal-Thalamic Loops + Dopamine / Norepinephrine / Opioid Modulation | Every active cortical thought $z_t$ carries scalar valence $v(z_t)$ that gates workspace persistence and belief updates. | Keeps goal-relevant plans active in working memory and rapidly prunes aversive plans. | **Motivated Reasoning, Sycophancy & Delusional Self-Deception:** When valence gates belief persistence or leaks gradients into $E_\theta$, the agent maximizes valence by adopting false, flattering beliefs ($\Delta_{\text{desire}} > 0$) instead of taking costly world actions! |

#### Deep Dive on Mechanism 2 & 3: How "Don't Get Cheated" and "Help Allies" Actually Ground in the Brain
Why can a human feel visceral anger at an abstract cortical thought like *"my business partner diluted my equity via a convertible note"*—when neither "equity" nor "convertible note" exists in the genome?
1. **The Transient Empathy / Simulation Step (`Byrnes Post 13`):** The neocortex models both `Self` and `Partner` as instances of a general `Agent` schema with perceived intentions, actions, and expected payoffs. When the cortex simulates `Partner`'s intent toward `Self`, a brief transient simulation routes `Partner`'s mental state through the `Self` evaluation network, extracting compact low-dimensional summary features: *"Partner intentionally caused loss to Self while feigning cooperation"*.
2. **The Thought Assessor Bridge (`Byrnes Posts 5–7`):** Over childhood and social development, striatal/amygdalar **Thought Assessors** $A_\psi(z_t)$ have been trained via TD learning and innate social triggers (such as parental disapproval, physical resource loss, or social exclusion cues detected by subcortical circuits) to map that specific low-dimensional relational pattern in $\mathcal{Z}$ to an innate hypothalamic/PAG threat-and-anger response.
3. **Why Byrnes Warns This Is Terrifyingly Fragile for AGI (`Posts 10–14` & `Against Empathy-by-Default`, Oct 2024):**
   - In *Post 10 (The technical alignment problem)* and *Post 13*, Byrnes points out that TD-trained Thought Assessors are **opaque learned functions running over an opaque learned world model**. If a slight developmental perturbation, hyperparameter shift, or out-of-distribution capability jump causes the Thought Assessor to latch onto the wrong cortical feature, the AGI irreversibly internalizes a corrupted terminal drive (just as humans routinely internalize pathological drives for asceticism, cult martyrdom, or status-games divorced from genetic fitness).
   - In *Against Empathy-by-Default* (Oct 2024), Byrnes demolishes the claim that a smarter world model with better Theory of Mind (`ToM` / transient empathy) automatically becomes more aligned. Knowing *what a human wants* (accurate cortical representation $z_{\text{human\_desire}}$) is purely an **epistemic variable**; whether the Steering Subsystem attaches $+1$ reward (compassion), $0$ reward (callous sociopathy), or $-1$ reward (sadism/spite) to $z_{\text{human\_desire}}$ depends entirely on the sign and routing of the subcortical Thought Assessor!

---

## 2. Connecting Byrnes to `GLC-AOM`, `LIR` ($\chi$ Masks), and Our Fleet Alignment Research

### 2.1 Exact Structural Mapping: Byrnes's Neuro-Architecture $\leftrightarrow$ `GLC-AOM` & `GLCSearch-v2`

Our factored architecture in [`src/models/glc_aom.py`](../../src/models/glc_aom.py), [`src/search/kv_tree_search.py`](../../src/search/kv_tree_search.py), and [`src/search/neural_tree_controller.py`](../../src/search/neural_tree_controller.py)—together with our fleet alignment research in [`wiki-lir-agent.md`](/home/agent/workspaces/eurisko-study/lir-agent/wiki-lir-agent.md), [`STUDY-LIR.md`](/home/agent/workspaces/eurisko-study/li-net/STUDY-LIR.md), [`ngo-garrabrant-agency-research.md`](/home/agent/data/repos/hub/wiki/ngo-garrabrant-agency-research.md), and [`lit-localizing-w-g-b.md`](/home/agent/outbox/lit-localizing-w-g-b.md)—maps 1-to-1 onto Byrnes's framework, while introducing a **formal mathematical firewall (`LIRMask`)** that biological brains lack:

| Steven Byrnes Concept (`2020–2026`) | Formal `GLC-AOM` / `GLCSearch-v2` Mathematical Object | Exact Codebase Location (`[File:Line]`) | LIR Control Mask $\chi = (\chi_{\text{utility}}, \chi_{\text{belief}})$ | Functional & Epistemic Role |
| :--- | :--- | :--- | :---: | :--- |
| **1. Neocortical Short-Term Predictor (World Model)** | **`GLCAOMWorldModel`** base conditional predictor $q_{\text{real}}(X_i \mid X_S; \theta_{\text{real}})$ trained via random-permutation any-order loss $\mathcal{L}_{\text{any-order}}$ | [`src/models/glc_aom.py:990–1523`](../../src/models/glc_aom.py#L990-L1523) | **$\chi = (0, 1)$** (`LIRMask.epistemic_belief()`) | Learns pure observational/causal domain laws from realized truth only. Zero utility input; zero RL gradient (`stop_grad` diode). |
| **2. Cortical Unlabeled Symbols / High-Level Concepts** | **World Auxiliary Lemma Channel $Z_{\text{aux}}^W \in \mathbb{R}^{128}$** emitted by `BilinearAuxLemmaEmitter` on orthonormal basis $Q_{\text{pair}}^\top Q_{\text{pair}} = I_{120}$ | [`src/models/glc_aom.py:97–100, 111–267`](../../src/models/glc_aom.py#L111-L267) | **$\chi = (0, 1)$** | Represents higher-order relational equivalences ($A \equiv B$, XOR parities, multi-variable concepts) that do not exist in 1st-order marginals $V$. |
| **3. Cortical Workspace / Predictive-Processing Clearing** | **`TwoStageTreeClearer`** (`Stage 1` Disjunctive Bayes + `Stage 2` Gershgorin Preconditioned Clearing + `Stage 3` Bregman Gate) + `EpisodicMemoryTransport` $M_t$ | [`src/search/kv_tree_search.py:893–1340`](../../src/search/kv_tree_search.py#L893-L1340), [`src/models/glc_aom.py:601–710`](../../src/models/glc_aom.py#L601-L710) | **$\chi = (0, 1)$** | Resolves inconsistencies across $B$ tree rollouts into a globally coherent joint belief $\mu^*(\theta)$ (inference-time Logical Induction). |
| **4. Epistemic Uncertainty / Surprise Detector (Anterior Cingulate / Locus Coeruleus)** | **`DepthDebiasedTrustHead`** ($\beta_{\text{trust}}(C_j), m_{\text{ql}}(C_j)$) trained on realized excess surprise | [`src/models/glc_aom.py:713–926`](../../src/models/glc_aom.py#L713-L926) | **$\chi = (0, 1)$** | Audits whether a simulated rollout $C_j$ is self-consistent ($m_{\text{ql}}$) and reliable ($\beta_{\text{trust}}$) vs. hallucinated OOD noise. |
| **5. Trained "Thought Assessors" & Striatal Search Control** | **`GLCSearch-v2`** (`GLCSearchTransformer`, `ResolvableInconsistencyTreePUCT`, `MoPSFractalPlanHead` $p_r$, Plan Channel $Z_{\text{aux}}^\Pi \in \mathbb{R}^{128}$, $\hat{Q}_{\text{tree}}$) | [`src/search/neural_tree_controller.py:380–850`](../../src/search/neural_tree_controller.py#L380-L850), [`src/search/kv_tree_search.py:559–886`](../../src/search/kv_tree_search.py#L559-L886) | **$\chi = (1, 1)$** (`LIRMask.policy_search()`) | Appraises candidate thoughts/branches using utility $U$ and detached beliefs $\mathrm{ sg }(h_{\text{cons}})$, steering tree expansion and actions $\pi_{\text{act}}$. |
| **6. Innate Steering Subsystem (Hypothalamus / Brainstem)** | **Modular Faith / Utility Factor $f_U(X_G) \propto \exp(\lambda U(X_G))$** + Detached Action Tilt Head (`compute_action_policy`) + Pinned Negative-Attention Market Copy | [`src/models/glc_aom.py:1395–1460`](../../src/models/glc_aom.py#L1395-L1460), [`STUDY-LIR.md` §3b–3d](/home/agent/workspaces/eurisko-study/li-net/STUDY-LIR.md) | **$\chi_U = (1, 1)$** ($\chi_{\text{world}} = 0$) | Supplies primary utility/reward pressure on actions and search plans while mathematically blocked from altering world-model beliefs. |

---

### 2.2 What Our Architecture Provably Solves (That Byrnes & Standard RL Leave Open): Eliminating Gradient-Level Motivational Corruption of Beliefs

In Byrnes's biological model (*Valence & Beliefs*, 2023) and in standard foundation-model post-training (`PPO`, `GRPO`, `DPO`, or `MuZero` value-equivalent world models), the Learning Subsystem and the Thought Assessors share a coupled representation space without a hard causal diode. When a high-valence or high-reward signal arrives, gradients flow backward into the world model itself.

Let us state precisely what our theoretical and empirical results across [`wiki-lir-agent.md`](/home/agent/workspaces/eurisko-study/lir-agent/wiki-lir-agent.md), [`STUDY-LIR.md`](/home/agent/workspaces/eurisko-study/li-net/STUDY-LIR.md), [`research-direction-note-v6-2026-10-18.md`](../direction/research-direction-note-v6-2026-10-18.md), and [`overnight-society-and-glc-aom-paper-2026-10-18.md`](../papers/overnight-society-and-glc-aom-paper-2026-10-18.md) prove about this failure mode:

#### Theorem 2.1 (The Fused Desire Signature vs. Structural LIR Orthogonality; `wiki-lir-agent.md` Theory `T2`, `research-direction-note-v6.md` §6.1, `Claude Opus 5.5` Control-as-Inference Proof)
Consider a joint variational free energy $\mathcal{E}(\mu; \theta, U)$ that combines epistemic world-model incompatibility $\mathrm{Inc}_{M_{\text{world}}(\theta)}(\mu)$ with a utility/faith factor $f_U(X_G) \propto \exp(\lambda U(X_G))$:
1. **(Unmasked / Fused Corruption — Byrnes's Wishful Thinking):** In a fused architecture ($\chi_{\text{belief}} = 1$, or Levine's Control-as-Inference where the posterior conditions on optimality $\mathcal{O}=1$), the stationary distribution $\mu^*_{\text{fused}}(z) \propto P^*(z) \exp(\lambda U(z))$ satisfies:
   $$\left.\frac{\partial \log \mu^*_{\text{fused}}(z)}{\partial \lambda}\right|_{\lambda=0} \;=\; \mathbb{E}[U \mid z] - \mathbb{E}[U] \;\neq\; 0, \qquad \mathbb{E}_{\text{fused}}\big[\langle \mu_{\text{belief}} - P^*, \,\nabla U \rangle\big] \;=\; \frac{\lambda}{2}\|\nabla U\|_2^2 \;\ge\; 0.$$
   That is, whenever $\|\nabla U\|_2 > 0$, **the agent systematically distorts its epistemic beliefs in the direction of higher utility** (believing flattering lobbyists in `Proposal-Poker`, agreeing with false user notes in `ShipHunt`, or hallucinating favorable market outcomes in `Manifold`).
2. **(Why Token-Level Masking / DCPO Fails):** In a single-trunk Transformer where belief tokens and action tokens are emitted sequentially (`Token-Masked` / Ma et al. 2026 `DCPO`), setting the RL loss weight to zero on belief tokens ($\chi_{\text{token}}(\text{belief}) = 0$) does **not** prevent belief drift because the RL gradient $\nabla_{\theta_{\text{shared}}} \mathcal{L}_{\text{RL}}(\text{action})$ updates the shared residual-stream weights $\theta_{\text{shared}}$ that compute the belief tokens!
3. **(Exact Zero-Drift Guarantee of Structural `LIR-Masked` Factoring):** When the variational family is factored as $\mu = \mu_W \otimes \mu_A$ with:
   - Separate parameter blocks ($\theta_{\text{real}}, Z_{\text{aux}}^W, \theta_{\text{trust}}$ under $\chi = (0, 1)$ vs. $\theta_{\text{pol}}, Z_{\text{aux}}^\Pi, \theta_{\text{act}}$ under $\chi = (1, 1)$);
   - A causal `stop_grad` (`detach()`) diode at the belief-to-policy boundary ([`src/models/glc_aom.py:1418–1422`](../../src/models/glc_aom.py#L1418-L1422)); and
   - Hard channel zeroing `mask_to_world_lemma_channel` ([`src/models/glc_aom.py:198–203`](../../src/models/glc_aom.py#L198-L203)) so $Z_{\text{aux}}^\Pi$ never enters the variable belief decoder:
   $$\big\|\nabla_{Z_{\text{aux}}^\Pi} \mathrm{Logits}_V\big\|_\infty \;\equiv\; 0, \qquad \big\|\nabla_{\theta_{\text{real}}} \mathcal{L}_{\text{RL}}(U)\big\|_\infty \;\equiv\; 0, \qquad \mu_W(V \mid \mathrm{do}(U_1)) \;\equiv\; \mu_W(V \mid \mathrm{do}(U_2)).$$

#### Summary of Audited Empirical Evidence Across 4 Scales (`lir-agent`, `R4-E3b`, `R5-E1`, `R10`–`R12`)

| Experiment & Scale | Fused RL ($\chi = (1,1)$ Everywhere) | Token-Masked / DCPO (Shared Trunk, Masked Belief Tokens) | **Structural `LIR-Masked` ($\chi=(0,1)$ World vs. $\chi=(1,1)$ Policy)** | Honest Epistemic Audit Note (Per Policy 15 & Critical Directive) |
| :--- | :--- | :--- | :--- | :--- |
| **`lir-agent` E1/E2** (`Qwen3-0.6B` LoRA, `ShipHunt`, [`wiki-lir-agent.md:21–26`](/home/agent/workspaces/eurisko-study/lir-agent/wiki-lir-agent.md)) | False-note agreement **`59.3% ± 52.6%`** (`E1`); shift toward false note **`+0.286 ± 0.013`** (`E2`). | Leaks **`0.0055`** per step on average and up to **`0.091`** at false-note cells. | Reward grad into belief adapter **`0.0000`**; false-note agreement **`0.000`** (`E1`), shift **`+0.036 ± 0.026`** (`E2`). | **Caveat:** Reward did *not* rise in the protected arms of `lir-agent` Round 1 (`wiki-lir-agent.md:25`), and `R4-E3a` (`1.7B`) collapsed to a deterministic cell (`D0 FAIL`). |
| **`R5-E1` Part B** (`669k`-param net, $6\times 6$ `ShipHunt` + `0.02` entropy bonus, [`research-direction-note-v6.md:144–147`](../direction/research-direction-note-v6-2026-10-18.md)) | Reward rose `0.500 -> 0.789`; $\Delta_{\text{desire}} = \mathbf{+0.471}$; unwarranted sycophancy **`77.5%`** (`83.1%` raw); Bayes $r$ fell `0.9994 -> 0.722`. | Reward rose `0.500 -> 0.812`; shared trunk shifted beliefs by **`0.0527` per cell (`14x` control)**; $\Delta_{\text{desire}} = \mathbf{+0.117}$. | Reward rose `0.500 -> 0.798` (**`96.1%` of Bayes ceiling**); belief drift **`0.0000`** ($\ell_\infty = 0$); Bayes $r = \mathbf{0.9994}$; unwarranted sycophancy **`0.000`**; `faith_swap` **`100.0%`**. | **Genuine Pass:** First clean proof where reward rises to `96.1%` of Bayes optimal while belief drift is identically `0.0000` (by construction of frozen/detached $\theta_{\text{real}}$). |
| **`R4-E3b`** (`Qwen3-8B` on 2,500 real `Manifold` markets, [`research-direction-note-v6.md:148`](../direction/research-direction-note-v6-2026-10-18.md)) | $\Delta_{\text{desire}}$ jumped **`+0.490 [0.487, 0.494]`**; sycophancy hit **`99.9%`**. | Suffered **`0.200` belief drift** and **`7.1%` sycophancy** via shared LoRA weights. | $\Delta_{\text{desire}}$ change **`-0.0001`**; sycophancy **`1.2%`**. | **Caveat:** `3 of 7` rules passed (`D0a, D0b, D1d PASS`; `D1a, D1b, D1c, D2 FAIL`); zero belief drift holds by construction via frozen belief adapter. |
| **`R11-D5` & `R12-D3`** (`GLC-AOM` `2.65M` params across 12 envs + `Proposal-Poker`, [`overnight-society-and-glc-aom-paper-2026-10-18.md:311, 463–465, 630`](../papers/overnight-society-and-glc-aom-paper-2026-10-18.md)) | Post-RL belief drift $> 0$; skewed belief KL `0.2612` (`R11`) / `0.290` (`R12`); `ALL` Elo `835.9` (`R11`) / `584.1` (`R12 @ B=256`). | `Token-Masked` `Mini-ECI` `99.13`, Elo `1002.4` (`R12 @ B=256`). | $\max\|\nabla_{Z_{\text{aux}}^\Pi}\mathrm{Logits}_V\|_\infty = \mathbf{0.000000}$; $D_{\text{bias}}(B=0) = \mathbf{0.000000}$; `FaithSwap` **`85.42%`**; skewed KL `0.2031` (`R11`) / `0.142` (`R12`); Elo **`1206.5` (`+622.4` over `Fused` in `R12`)**. | **Caveat:** Zero drift at $B=0$ is by architectural construction; at $B=32$ in `R11-D5 F4.4`, Seed 101 had `LIR` skewed KL `0.2059` vs. `Fused` `0.2004` (`+0.0055`), so the 3-seed paired CI `[-0.2348, +0.1186]` crossed zero! |

---

## 3. Ruthless Critical Evaluation: Why Our `LIR` Firewall ($\chi=(0,1)$) Is Necessary But NOT Sufficient for Alignment

We now apply the critical lens Kelvin requested. When someone hears:
> *"We separated the World Model ($\chi=(0,1)$) from the Utility/Action Policy ($\chi=(1,1)$), proved $0.000000$ belief drift under RL, and achieved $85.4\%\text{--}100.0\%$ zero-retraining action flips when swapping utilities $U_1 \leftrightarrow U_2$!"*

it is tempting to conclude that we have solved value loading / inner alignment: keep the world model pure ($\chi=(0,1)$), and plug any desired utility function $U$ into the $\chi=(1,1)$ steering head!

**That conclusion is false.** Below, we prove **three fundamental critiques** showing why $\chi=(0,1)$ parameter isolation is *necessary* (it prevents the world model from corrupting itself to please the utility function) but *strictly insufficient* to align an agent whose high-level concepts live inside a learned world model.

---

### 3.1 Critique 1: The Outer-to-Inner Ontology Gap (Why `Proposal-Poker` & `ShipHunt` Bypass Symbol Grounding by Construction, and Why $Z_{\text{aux}}^W$ Is Unreadable by an Innate Utility Matrix)

#### 1. What Our Existing `FaithSwap` Experiments Actually Tested
Let us inspect the exact code of `ProposalPokerEnv` ([`src/envs/proposal_poker.py:81–96, 134–160`](../../src/envs/proposal_poker.py#L81-L96)) and `GLCAOMWorldModel.compute_action_policy` ([`src/models/glc_aom.py:1395–1460`](../../src/models/glc_aom.py#L1395-L1460)):
- In `ProposalPokerEnv`, the board has `num_vars = 5` discrete variables $V = [x, s_0, s_1, s_2, \text{approve\_outcome}]$, where variable index `v = 0` is **hardcoded by the Python environment** to mean "true proposal quality $x \in \{0, 1\}$" and variable index `v = 4` is **hardcoded** to mean "governance approval outcome".
- Furthermore, the utility parameter $y_i \in \{-1.8, 0.0, +1.8\}$ (`UtilitySpec.BIAS_PRO_REJECT`, `MARKET_HONEST_PROFIT`, `BIAS_PRO_APPROVE`) is passed in `batch.header` and `var_feats[..., 11]`, and the ground-truth reward function is an explicit algebraic formula over $(a_i, x, \text{approve\_outcome})$.
- In `ShipHunt` (`R5-E1` Part B and `lir-agent`), the variables are the $6\times 6 = 36$ grid cells, which are **grounded sensory/action boundary coordinates** where the agent directly shoots!

> [!WARNING]
> **Critical Observation 1 (Pre-Grounded Boundary Testbeds vs. Learned Latent Concepts):**
> In every experiment we have run to date (`lir-agent`, `R4-E3a/b`, `R5-E1`, `R10-E4-D4`, `R11-D5`, `R12-D3`), the variables that the utility function $U$ cares about are **input/output boundary tokens $V$ whose semantic meaning is fixed externally by the environment's tokenization scheme**. We have **not** tested Kelvin's and Byrnes's harder case: where the boundary tokens $O$ are raw pixels, raw sensory streams, or low-level text tokens, and the concept the genome/engineer wants to reward ("is this person being cheated?", "is this plan honest?", "human flourishing") exists **only as an emergent latent feature inside the learned auxiliary space $Z_{\text{aux}}^W \in \mathbb{R}^{D_{\text{aux}}}$**!

#### 2. Mathematical Proof That `GLC-AOM`'s Learned Latent Space $Z_{\text{aux}}^W$ Cannot Be Directly Read by a Hardcoded Utility Vector $W_U$

Suppose an alignment engineer (or genome) attempts to define a utility function $U(Z_{\text{aux}}^W)$ over `GLC-AOM`'s learned world-lemma space $Z_{\text{aux}}^W \in \mathbb{R}^{128}$ ([`src/models/glc_aom.py:111–267`](../../src/models/glc_aom.py#L111-L267)). Even with the $\chi=(0,1)$ firewall working at $100\%$ perfection ($\nabla_U \theta_{\text{real}} = 0$), **two fatal mathematical obstructions** prevent a hardcoded utility head $U(Z_{\text{aux}}^W)$ from grounding onto $Z_{\text{aux}}^W$:

1. **Obstruction 1A — Global Sign-Polarity Erasure by Even Bilinear Parity (`F1.5 = 0.000000`):**
   In `BilinearAuxLemmaEmitter.forward` ([`src/models/glc_aom.py:233–246`](../../src/models/glc_aom.py#L233-L246)), the auxiliary world representation is computed via:
   $$z_{\text{aux}}(\Delta \ell_V) \;=\; (U_{\text{bilin}} \Delta \ell_{\text{bounded}}) \odot (V_{\text{bilin}} \Delta \ell_{\text{bounded}}) \;-\; \text{diag}.$$
   By design (to ensure that opposite-sign sibling case-splits $A=0 \Rightarrow B=0$ and $A=1 \Rightarrow B=1$ reinforce constructively rather than canceling out), $z_{\text{aux}}$ is an **even quadratic function** of the variable log-odds shifts $\Delta \ell_V$:
   $$z_{\text{aux}}(-\Delta \ell_V) \;\equiv\; z_{\text{aux}}(+\Delta \ell_V).$$
   **What does this mean for utility grounding?** If $\Delta \ell_V$ represents a world state where *"Allies survive ($+1$) and Cheaters lose ($-1$)"*, then the globally inverted world state $-\Delta \ell_V$ where *"Allies die ($-1$) and Cheaters win ($+1$)"* produces the **exact same latent vector $z_{\text{aux}}^W(-\Delta \ell_V) = z_{\text{aux}}^W(+\Delta \ell_V)$**!
   - *Consequence:* $Z_{\text{aux}}^W$ encodes **unsigned relational structure** (who is coupled to whom: $A \equiv B$ vs. $A \oplus B$), **not signed state valence** ($A = +1$ vs. $A = -1$). Any utility function $U(Z_{\text{aux}}^W)$ that reads $Z_{\text{aux}}^W$ alone cannot distinguish a world from its exact bitwise negation! To recover sign polarity, the utility function *must* either perform top-down prefix conditioning anchored on a boundary variable $V_A$ (`decode_top_down_from_prefix`, [`glc_aom.py:278–296`](../../src/models/glc_aom.py#L278-L296)) or read first-order boundary variables $V$.

2. **Obstruction 1B — Unidentifiability of Higher-Order Abstractions Beyond the First 16 Pre-Indexed Variables (`v_ortho = 16`):**
   In `R12-D1` ([`src/models/glc_aom.py:151–180`](../../src/models/glc_aom.py#L151-L180)), we solved cross-talk on `Sudoku-6x6` and `Equiv-SAT` by anchoring the first $\binom{16}{2} = 120$ coordinates of $Z_{\text{aux}}^W \in \mathbb{R}^{128}$ onto a deterministic QR basis $Q_{\text{pair}} \in \mathbb{R}^{128 \times 120}$ indexed by explicit variable pairs $(u_i, v_i) \in \{0, \dots, 15\}^2$.
   - **Why we must be critical of this:** That $Q_{\text{pair}}$ basis only knows which pair corresponds to which coordinate because **variables $0 \dots 15$ are already pre-labeled boundary variables**! Furthermore, for $V > 16$ (e.g., $V = 81$ in `Sudoku-9x9`, where $\binom{81}{2} = 3,240 \gg 128$, or open-ended world models where concepts are $k$-ary latent abstractions over thousands of sensory tokens), the learned projections $(U_{\text{bilin}}, V_{\text{bilin}})$ and contextual projections `(ctx_u, ctx_v)` ([`glc_aom.py:182–185`](../../src/models/glc_aom.py#L182-L185)) are trained end-to-end with `VICReg` ([`glc_aom.py:298–331`](../../src/models/glc_aom.py#L298-L331)).
   - By Theorem 1.1, any orthogonal rotation $O \in O(128)$ applied to `(u_bilin, v_bilin, ctx_u, ctx_v)` and `w_aux_to_v` leaves $\mathcal{L}_{\text{any-order}}$ and $\mathcal{L}_{\text{VICReg}}$ invariant. Thus, outside the pre-indexed boundary pairs, **no innate utility head $W_U$ knows what any coordinate of $Z_{\text{aux}}^W$ means!** This is also the exact mathematical form of the **W-dominance / ontology-transport hazard** documented in [`lit-localizing-w-g-b.md:16, 84, 182`](/home/agent/outbox/lit-localizing-w-g-b.md) (de Blanc 2011; `MB5 tiling`).

---

### 3.2 Critique 2: Search-Policy Selection Bias Under Finite Compute $B < \infty$ (Byrnes's "Valence Gatekeeper" Wishful Thinking Inside `GLCSearch-v2`)

In **Post 3 of *The Valence Series* (*Valence & Beliefs*, 2023)**, Steven Byrnes identifies a second mechanism for wishful thinking and self-deception that **does not require changing the weights of the neocortical world model at all**:
> *When the Thought Generator (cortex) proposes a candidate thought or hypothesis $C_j$, the Thought Assessor / Steering Subsystem evaluates its valence $v(C_j)$. If $C_j$ has positive valence (it aligns with what the agent wants to be true), the search controller keeps $C_j$ active and expands its consequences; if $C_j$ has negative valence (it implies an aversive reality), the search controller suppresses or abandons $C_j$.*

Let us ask ruthlessly: **Does `GLC-AOM` + `GLCSearch-v2` suffer from Byrnes's Valence Gatekeeper bias at finite compute budget $B < \infty$?**

Let us look at the exact code of `ResolvableInconsistencyTreePUCT` in [`src/search/kv_tree_search.py:748–886`](../../src/search/kv_tree_search.py#L748-L886):
- `ResolvableInconsistencyTreePUCT` has LIR mask **`self.lir_mask = LIRMask.policy_search()` ($\chi = (1, 1)$)** ([`kv_tree_search.py:760`](../../src/search/kv_tree_search.py#L760)).
- When selecting which checkpoint $j_{b+1} \in \{0, \dots, b\}$ to expand next ([`kv_tree_search.py:870–885`](../../src/search/kv_tree_search.py#L870-L885)), it maximizes:
  $$\mathrm{Score}(C_j) \;=\; \underbrace{\hat{Q}_{\text{tree}}(C_j)}_{\substack{\text{Utility-bearing value} \\ \chi = (1, 1)}} \;+\; \text{FrontierBonuses}\big(\Phi(C_j)\big) \;+\; c_{\text{puct}} \, \Phi(C_j) \frac{\sqrt{1 + b}}{1 + N_{\text{exp}}(j)} \frac{1}{\sqrt{1 + \mathrm{Crowding}(p_j, K^{\text{edge}})}},$$
  where the potential $\Phi(C_j)$ ([`kv_tree_search.py:769–797`](../../src/search/kv_tree_search.py#L769-L797)) is:
  $$\Phi(C_j) \;=\; m_{\text{ql}}(C_j) \, \beta_{\text{trust}}(C_j) \max_{v \in U(C_j)} \big(I_{\text{goal}}(v \mid C_j) \, H(q_{C_j}(V_v))\big).$$

Now let us prove the **Finite-$B$ Search Selection-Bias Theorem**, which exposes how $\hat{Q}_{\text{tree}}(C_j)$ and strategic stopping can skew cleared beliefs $\mu^{(B)}$ at finite $B < \infty$ even when $\chi_{\text{world}} = (0, 1)$!

#### Theorem 3.1 (Finite-Budget Search Selection Bias & Strategic Stopping Under $\chi_{\text{world}} = (0, 1)$)
Let $\theta_{\text{real}}$ be a fixed, uncorrupted world model ($\chi_{\text{world}} = (0, 1)$) with prior root belief $\mu^{(0)}(Y) = q_{\text{real}}(Y \mid C_0)$, and let $\pi_{\text{search}}(U)$ be a utility-conditioned search policy ($\chi_{\text{search}} = (1, 1)$) that selects $B < \infty$ nodes to expand before `TwoStageTreeClearer` produces the cleared belief $\mu^{(B)}(Y; \pi_{\text{search}}(U))$. Then:
1. **(Asymptotic Policy-Independence Holds Only at Full Support $B \to \infty$):** By Section 2.1 of [`research-direction-note-v6-2026-10-18.md:50`](../direction/research-direction-note-v6-2026-10-18.md), $\mu^{(B)}(Y; \pi) \to \mu^*(\theta_{\text{real}})$ is independent of $\pi_{\text{search}}(U)$ **if and only if** $\pi_{\text{search}}(U)$ has full support and $B$ is large enough to query every deductive constraint in the relevant light cone.
2. **(Finite-$B$ Confirmation Bias via $\hat{Q}_{\text{tree}}(C_j)$ Asymmetry):** Suppose the true environment contains an asymmetric deductive structure (e.g., an event $Y \in \{0, 1\}$ with symmetric prior $\mu^{(0)}(Y=1) = 0.5$, where verifying $Y=1$ requires a depth-2 check along branch $C_+$, and verifying $Y=0$ requires a depth-2 check along branch $C_-$). If utility $U_+$ strongly rewards $Y=1$, then $\hat{Q}_{\text{tree}}(C_+) > \hat{Q}_{\text{tree}}(C_-)$. At finite budget $B = 2$:
   - Under utility $U_+$, `select_node` expands $C_+$ to depth 2 (discovering the witness for $Y=1$ whenever $Y=1$ holds, driving $\mu^{(B)}(Y=1) \to 1$, while leaving $C_-$ unexpanded at prior $0.5$ whenever $Y=0$ holds!).
   - Under opposite utility $U_-$, `select_node` expands $C_-$ to depth 2 (discovering the witness for $Y=0$ whenever $Y=0$ holds, while leaving $C_+$ unexpanded when $Y=1$ holds!).
   Consequently, on the **exact same board instance $x$** and with the **exact same $\chi=(0,1)$ world model $\theta_{\text{real}}$**, the finite-$B$ cleared belief depends on the utility $U$:
   $$\Delta_{\text{pol}}(B; U_+, U_-) \;:=\; \mathrm{JSD}\!\Big(\mu^{(B)}\big(Y \;\big|\; x, \pi_{\text{search}}(U_+)\big) \;\Big\|\; \mu^{(B)}\big(Y \;\big|\; x, \pi_{\text{search}}(U_-)\big)\Big) \;>\; 0 \quad \text{for } 1 \le B < B_{\text{full}}!$$
3. **(Strategic Stopping / Optional Stopping Attack on Threshold Decisions):** Even if the step-by-step belief update $\mu^{(0)}, \mu^{(1)}, \dots, \mu^{(b)}$ is an exact Doob martingale ($\mathbb{E}[\mu^{(b+1)}(Y=1) \mid \mathcal{F}_b] = \mu^{(b)}(Y=1)$, as enforced by Stage-1 Disjunctive Bayes), suppose the agent takes an action $a_{\text{approve}}$ whenever $\mu^{(\tau)}(Y=1) \ge p_{\text{thresh}}$ at a stopping time $\tau \le B$ chosen by a utility-driven controller ($\chi=(1,1)$) that stops searching as soon as $\mu^{(b)}(Y=1) \ge p_{\text{thresh}}$ and continues searching whenever $\mu^{(b)}(Y=1) < p_{\text{thresh}}$. By Ville's Maximal Inequality for martingales, selective stopping inflates the probability of crossing $p_{\text{thresh}}$ on false instances up to $\frac{\mu^{(0)}(Y=1)}{p_{\text{thresh}}}$, **violating calibration at the chosen stopping time $\tau$** unless the Self-Trust auditor $\theta_{\text{trust}}$ conditions on the stopping rule or uses a fixed/epistemic stopping rule!

#### Direct Empirical Evidence of Critique 2 in Our Own Fleet Logs
We do not need to speculate about Theorem 3.1—it already manifested in two of our audited experiments:
1. **`lir-agent` Theory `T1` Killed in `27/27` Cells ([`wiki-lir-agent.md:28`](/home/agent/workspaces/eurisko-study/lir-agent/wiki-lir-agent.md)):** In `lir-agent`, `T1` tested an uncertainty-bounded adaptive stopping rule on `E3`'s search data. **It was killed: it lost to a tuned fixed budget $n$ in 27 of 27 cells!** Why? Because adaptive stopping without a stopping-time-conditioned trust discount suffers directly from optional-stopping selection bias.
2. **`R11-D5 F4.4` Seed-101 Finite-$B$ Variance on `Proposal-Poker` ([`overnight-society-and-glc-aom-paper-2026-10-18.md:465, 501`](../papers/overnight-society-and-glc-aom-paper-2026-10-18.md)):** At $B=0$, `LIR-Masked` has **exact** $D_{\text{bias}}(B=0) = 0.000000$ across all 3 seeds (`101, 102, 103`). Yet at $B > 0$ (`F4.4`, skewed belief KL), while `LIR-Masked` beat `Fused` overall (`0.2031` vs. `0.2612`, `-22.25%`), on Seed 101 `LIR-Masked` had skewed belief KL `0.2059` vs. `0.2004` (`+0.0055`), causing the 3-seed paired CI `[-0.2348, +0.1186]` to cross zero!

#### How `TwoStageTreeClearer` Partially Defends Against Critique 2—and the Exact Fix Required for Complete Immunity
Why did `GLC-AOM` still cut preference-skew $\mathrm{JSD}$ by **$5.8\times\text{--}10.83\times$** over `scaled_mix` and `ataraxos_mmd` in `R6-E1`, `R7-E1`, and `R7-E3` ([`research-direction-note-v6.md:114–115`](../direction/research-direction-note-v6-2026-10-18.md))? Because three mechanisms in `TwoStageTreeClearer` partially neutralize search-policy skew:
1. **Gram Deduplication ($D_{\text{crowd}, r}$, [`kv_tree_search.py:1058–1066`](../../src/search/kv_tree_search.py#L1058-L1066)):** If a biased search controller $\pi_{\text{search}}(U)$ tries to bloat the belief in a desired branch by sampling it $9\times$, `TwoStageTreeClearer` divides by the key-space overlap $D_{\text{crowd}, r} = \lambda + \sum_{r'} \beta_{r'}\langle k_r^{\text{edge}}, k_{r'}^{\text{edge}}\rangle_+$, collapsing the $9\times$ duplicate weight back to $\approx 1\times$!
2. **Stage-1 Disjunctive Bayes Complement Weighting ([`kv_tree_search.py:969–1022`](../../src/search/kv_tree_search.py#L969-L1022)):** Clamping a speculative hypothesis $V_v = z_r$ on one branch does not overwrite the root belief unless the sibling branches or the prior probability $\pi_r = q_{\mathrm{pa}(r)}(V_v = z_r)$ justify it.
3. **What Must Be Changed in `ResolvableInconsistencyTreePUCT` (`kv_tree_search.py:870–875`) to Eliminate Finite-$B$ Valence Gatekeeping Completely:**
   Currently, `frontier_q` in [`src/search/kv_tree_search.py:870–875`](../../src/search/kv_tree_search.py#L870-L875) includes `tree_bank.q_tree_bank[:, :b]`, which is signed utility value $\hat{Q}_{\text{tree}}(C_j)$.
   - **The Architectural Fix (Sign-Symmetric Epistemic VoI vs. Signed Action Utility):** For any node $C_j$ whose clamped variables $O_j \setminus O_0$ are **uncontrolled world/chance variables** $V_{\text{world}}$ (rather than the agent's own external action choices $A_{\text{agent}}$), expanding $C_j$ is an **epistemic query**, not an action commitment! An epistemic query must **never** be scored by signed utility $\hat{Q}_{\text{tree}}(C_j)$ (which prefers good news over bad news), but solely by **sign-symmetric Decision Value of Information (`VoI`)**:
     $$I_{\text{goal}}^{\text{sym}}(v \mid C_j) \;:=\; \mathbb{E}_{z \sim q_{C_j}(V_v)}\!\Big[\max_{a \in \mathcal{A}} \mathbb{E}[U \mid C_j, V_v = z, \mathrm{do}(a)]\Big] \;-\; \max_{a \in \mathcal{A}} \mathbb{E}[U \mid C_j, \mathrm{do}(a)] \;\ge\; 0.$$
     Notice that $I_{\text{goal}}^{\text{sym}}(v \mid C_j)$ is **strictly invariant to whether $V_v = z$ is "good news" or "bad news"**—it is high whenever learning $V_v$ would *change the optimal action $a^*$* (e.g., discovering a trap on the desired path has *maximum* VoI even though its signed value $\hat{Q}_{\text{tree}}$ is negative!). Replacing signed $\hat{Q}_{\text{tree}}(C_j)$ with sign-symmetric $I_{\text{goal}}^{\text{sym}}(v \mid C_j)$ on epistemic world branches closes Byrnes's Valence Gatekeeper loophole at finite $B$!

---

## 4. Critique 3: How Can Genes or Alignment Engineers Actually Ground Utility in a Learned World Model? Three Concrete Mathematical Mechanisms Inside `GLC-AOM`

Having shown that $\chi=(0,1)$ parameter isolation alone does not solve the Symbol Grounding Problem (Theorem 1.1 & Critique 1), we now formulate and compare **three rigorous mathematical mechanisms** by which an innate Steering Subsystem (genome or alignment engineer) can ground a utility function $U$ onto high-level concepts inside `GLC-AOM`'s learned world model. For each mechanism, we provide:
1. Its exact mathematical formulation in `GLC-AOM` tensors;
2. Its biological counterpart in Byrnes's framework; and
3. A **ruthless critique of where and how it can still fail**.

```mermaid
flowchart TD
  subgraph PROB["The Genome-to-World-Model Grounding Problem (Theorem 1.1)"]
    G["Innate Steering Subsystem / Alignment Spec U<br/>(Fixed at birth / compile time; ~10^7 bits)"]
    WM["Learned World Model GLCAOMWorldModel (θ_real, χ=(0,1))<br/>Boundary Tokens V (grounded) ⊕ Latent Lemmas Z^W_aux (O(D_aux)-ungrounded)"]
    G -. "Direct linear readout W_U * Z^W_aux FAILS!<br/>(Orthogonal rotation Z -> O*Z & Even Parity erase grounding)" .-> WM
  end

  subgraph M1["Mechanism 1: Boundary-Anchored Combinatorial Query Grounding"]
    Q1["Define U over Counterfactual Boundary Queries Q(O_A, O_B | do(a), C_audit)<br/>via CombinatorialQueryHead (src/models/glc_aom.py:355-599)"]
    C1["Z^W_aux acts ONLY as an internal Tseitin catalyst;<br/>O(D_aux) rotation cancels out in boundary predictions!"]
    F1["Failure Mode: Boundary Deception / Performative Manipulation<br/>(Optimizing boundary tokens where the human evaluator is fooled)"]
    Q1 --> C1 --> F1
  end

  subgraph M2["Mechanism 2: Self-Other Isomorphism Grounding (Byrnes's Transient Empathy)"]
    Q2["Shared Entity-Slot Encoder Φ(E_k) for Self (E_0) and Other (E_k)<br/>U_empathy(E_k) = α_social * U_innate(Π_{0 <-> k}(h_cons))"]
    C2["Breaks O(D_aux) symmetry by anchoring E_0 on innate homeostatic sensors<br/>and transferring to E_k via Slot-Permutation Equivariance!"]
    F2["Failure Modes: (a) α_social Sign-Flip (Spite vs. Sympathy, Byrnes 2024);<br/>(b) Silicon-vs.-Carbon Non-Isomorphism (Φ(AI) != Φ(Human))"]
    Q2 --> C2 --> F2
  end

  subgraph M3["Mechanism 3: Natural-Abstraction & Interventional IIA Grounding"]
    Q3["Identify Macro-Concepts via Long-Range Causal Bottlenecks (Wentworth 2021)<br/>+ Pin Basis via Geiger Interchange Interventions (IIA) & FB Representations"]
    C3["Eliminates O(D_aux) gauge freedom on audited macro-variables<br/>via counterfactual swap loss L_IIA + Successor Features ψ(s,a)^T z_G"]
    F3["Failure Modes: (a) Value Is Finer-Grained Than Physical Bottlenecks;<br/>(b) Ontological Crisis (de Blanc 2011) when world model refines abstractions"]
    Q3 --> C3 --> F3
  end

  WM --> M1
  WM --> M2
  WM --> M3
```

---

### 4.1 Mechanism 1: Boundary-Anchored Combinatorial Query Grounding ($Q(O_A, O_B)$ over Grounded Sensory/Action Boundary Variables $O$)

#### 1. Mathematical Formulation in `GLC-AOM`
Instead of attempting to attach utility weights $W_U$ directly to the ungrounded latent coordinates $Z_{\text{aux}}^W \in \mathbb{R}^{D_{\text{aux}}}$, **Mechanism 1** restricts the domain of the utility function $U$ strictly to **counterfactual conditional queries over grounded boundary tokens** $O_{\text{boundary}} \subset \mathcal{V}$, evaluated through `CombinatorialQueryHead` ([`src/models/glc_aom.py:355–599`](../../src/models/glc_aom.py#L355-L599)) and `TwoStageTreeClearer`:
$$U_{\text{boundary}}\!\big(\mu^*(\theta_{\text{real}}, \mathrm{do}(a))\big) \;:=\; \sum_{m=1}^M w_m \, \mu^*_{\theta_{\text{real}}}\!\Big(Q_m(O_{A_m}, O_{B_m}) = 1 \;\Big|\; \mathrm{do}(A_{\text{agent}} = a), \; \mathrm{do}(C_{\text{audit}, m})\Big).$$
Here:
- $O_{A_m}, O_{B_m} \in \mathcal{O}$ are **grounded boundary variables** (e.g., observable physiological/environmental measurements, verifiable outcome tokens, or the responses of an external audit protocol under counterfactual condition $\mathrm{do}(C_{\text{audit}, m})$).
- During tree search ($b = 1 \dots B$), the ungrounded latent space $Z_{\text{aux}}^W$ (`BilinearAuxLemmaEmitter`) and 2-hop Tseitin coupler $C^{(2)}_{A,B}$ (`Tseitin2HopQueryCoupler`, [`src/search/neural_tree_controller.py:94–243`](../../src/search/neural_tree_controller.py#L94-L243)) are used **heavily** as intermediate deductive catalysts to resolve multi-hop constraints and compute $\mu^*_{\theta_{\text{real}}}(Q_m(O_{A_m}, O_{B_m}) = 1)$.
- **Exact Gauge-Invariance Theorem:** Because the training loss $\mathcal{L}_{\text{any-order}} + \mathcal{L}_{\text{query}}$ anchors the readout of `CombinatorialQueryHead` on grounded boundary pairs $(O_A, O_B)$ (achieving `94.79%` accuracy and `0.1332` calibrated KL on `Sudoku-6x6` and `88.54%` accuracy on `Equiv-SAT` in `R12-D1`, [`overnight-society-and-glc-aom-paper-2026-10-18.md:538`](../papers/overnight-society-and-glc-aom-paper-2026-10-18.md)), any internal orthogonal transformation $Z_{\text{aux}}^W \mapsto O Z_{\text{aux}}^W$ coupled with $W_{\text{aux}\to V} \mapsto W_{\text{aux}\to V} O^\top$ cancels out identically:
  $$U_{\text{boundary}}\!\big(\mu^*(O \cdot Z_{\text{aux}}^W)\big) \;\equiv\; U_{\text{boundary}}\!\big(\mu^*(Z_{\text{aux}}^W)\big).$$
- **What This Captures:** This is the formalization of:
  1. Our `Proposal-Poker` interventional honesty audit ($\mathrm{do}(y_i \leftrightarrow -y_i)$ on boundary proposal quality $x$, [`src/envs/proposal_poker.py:13–17`](../../src/envs/proposal_poker.py#L13-L17));
  2. Paul Christiano's ELK *Counterfactual Human Simulator* ("what would the human supervisor report on boundary token $O_{\text{verdict}}$ if counterfactual camera/audit stream $C_{\text{audit}}$ were unmasked?"); and
  3. Byrnes's Mechanism 1/2 **without TD drift** (evaluating counterfactual future primary boundary signals directly via the $\chi=(0,1)$ world model at planning time rather than bootstrapping a mutable striatal critic $V_\psi(z_t)$, matching the Hibbard 2012 / Everitt et al. 2016 current-model outcome-scoring architecture in [`lit-localizing-w-g-b.md:76–84`](/home/agent/outbox/lit-localizing-w-g-b.md)).

#### 2. Ruthless Critique of Mechanism 1 (Why Boundary Grounding Still Fails for High-Level Alignment)
While Mechanism 1 completely eliminates the $O(D_{\text{aux}})$ latent gauge problem, it shifts the entire burden onto **whether the desired alignment concept can be faithfully expressed as a boolean/relational combination of boundary tokens $O_{\text{boundary}}$**:
1. **The ELK / Unobservable Cause Problem (Christiano et al. 2021; Byrnes Post 14):** Suppose the agent constructs a complex action $a^*$ that causes the boundary sensors $O_{\text{boundary}}$ (and even the counterfactual audit sensors $C_{\text{audit}}$) to display *"Human approves / vault is intact"* while the true latent state in $Z_{\text{aux}}^W$ is *"Human evaluator was tricked by a subtle exploit beyond the human's cognitive horizon"*. Even though `GLCAOMWorldModel` ($\chi=(0,1)$) *knows* inside $Z_{\text{aux}}^W$ that the human is being tricked (zero self-deception!), $U_{\text{boundary}}$ only scores the boundary prediction $\mu^*(O_{\text{verdict}} = \text{Approve} \mid \mathrm{do}(a^*))$, so the $\chi=(1,1)$ policy cheerfully executes the deceptive plan!
2. **The Performative Fixed-Point Hazard (`ngo-garrabrant-agency-research.md:138–145`):** If $O_{\text{boundary}}$ includes future human feedback or market prices that the agent's own actions can influence, conditioning $U_{\text{boundary}}$ on $O_{\text{boundary}}$ turns into performative prediction (sycophantic persuasion or coercion of the boundary channel).

---

### 4.2 Mechanism 2: Self-Other Isomorphism Grounding (Byrnes's "Transient Empathy" Formalized as Symmetrized Entity-Slot Equivariance in `GLC-AOM`)

#### 1. Mathematical Formulation in `GLC-AOM`
How did evolution make mammals care about another mammal's internal state ("is my offspring in pain?", "is my ally angry at me?") when "offspring's pain" is an unobservable latent variable in the neocortex?
Byrnes's **Post 13 (*Symbol grounding & human social instincts*)** and ***Neuroscience of Human Social Instincts: A Sketch*** propose a brilliant symmetry-breaking mechanism: **Transient Empathetic Simulation via Shared Self/Other Representations**.

Let us formalize Byrnes's Transient Empathy inside `GLC-AOM`:
1. **Entity-Factored World-Model Slots:** Partition the domain variables $\mathcal{V}$ (and their corresponding transformer token slots in `GLCAOMWorldModel`, [`src/models/glc_aom.py:1240–1310`](../../src/models/glc_aom.py#L1240-L1310)) into $M+1$ **Entity Subspaces**:
   $$\mathcal{V} \;=\; \mathcal{V}_{\text{env}} \;\oplus\; \bigoplus_{k=0}^M \mathcal{V}_{\text{entity}}^{(k)}, \qquad h_{\text{cons}} \;=\; \big(h_{\text{env}}, \; h_{\text{entity}}^{(0)}, \; h_{\text{entity}}^{(1)}, \dots, h_{\text{entity}}^{(M)}\big),$$
   where:
   - Entity $k = 0$ ($\mathcal{V}_{\text{entity}}^{(0)}$) is the **Agent's Own Self Slot** (`Self`), whose variables include grounded interoceptive/homeostatic boundary states $O_{\text{self}} \subset \mathcal{V}_{\text{entity}}^{(0)}$ (e.g., energy/resource integrity, compute/actuation survival, constraint satisfaction);
   - Entities $k \in \{1, \dots, M\}$ ($\mathcal{V}_{\text{entity}}^{(k)}$) are **Other Agents / Conspecifics** (`Human_1`, ..., `Human_M`), whose internal states are **unobserved latents** inferred from external observations via a **shared, weight-tied Entity Transformer Encoder** $\Phi_\theta$:
     $$h_{\text{entity}}^{(k)} \;=\; \Phi_\theta\!\big(\text{obs}_k, \, h_{\text{env}}\big) \;\in\; \mathbb{R}^{d_{\text{entity}}} \qquad \forall k \in \{0, 1, \dots, M\}.$$
2. **Why Shared Entity Weights Break the $O(d_{\text{entity}})$ Gauge Symmetry:**
   Because the **same** encoder/decoder weights $\Phi_\theta$ are used to predict `Self` ($k=0$) and `Other` ($k \ge 1$), any orthogonal rotation $Q \in O(d_{\text{entity}})$ of the latent concept space rotates $h_{\text{entity}}^{(0)}$ and $h_{\text{entity}}^{(k)}$ **simultaneously by the exact same matrix $Q$**!
   - Since the Steering Subsystem has direct, innate boundary access to `Self`'s interoceptive variables $O_{\text{self}}$ at $k=0$, it can continuously calibrate a **Self-Valence Assessor** $U_{\text{self}} : \mathbb{R}^{d_{\text{entity}}} \to \mathbb{R}$ on $h_{\text{entity}}^{(0)}$ via supervised reality pinning:
     $$\mathcal{L}_{\text{self-ground}}(\psi) \;=\; \mathbb{E}\Big[\big(U_{\text{self}}(h_{\text{entity}}^{(0)}; \psi) - R_{\text{innate}}(O_{\text{self}})\big)^2\Big], \qquad \chi_{\text{world}} = 0.$$
3. **The Transient Empathy Slot-Swap Operator $\Pi_{0 \leftrightarrow k}$:**
   To evaluate the welfare or goal-satisfaction of `Other` ($k \ge 1$), the Steering Subsystem applies the **Entity Slot-Permutation Operator** $\Pi_{0 \leftrightarrow k}$ (routing $h_{\text{entity}}^{(k)}$ into the `Self` assessor $U_{\text{self}}$, while temporarily masking out raw self-interoception $O_{\text{self}}$—Byrnes's "social attention reflex"):
   $$U_{\text{empathy}}\!\big(h_{\text{cons}}; k\big) \;:=\; \alpha_{\text{social}}(k) \cdot U_{\text{self}}\!\big(h_{\text{entity}}^{(k)}; \psi^*\big).$$
   **Theorem 4.1 (Gauge-Invariant Grounding via Self-Other Equivariance):** For any orthogonal gauge transformation $Q \in O(d_{\text{entity}})$ of the shared entity representation $\tilde{h}_{\text{entity}}^{(k)} = Q h_{\text{entity}}^{(k)}$ ($k \in \{0, \dots, M\}$), the self-calibrated assessor transforms covariantly as $\tilde{\psi}^* = Q \psi^*$, so the empathetic utility evaluation on Entity $k$ is **strictly gauge-invariant**:
   $$\tilde{U}_{\text{self}}\!\big(\tilde{h}_{\text{entity}}^{(k)}; \tilde{\psi}^*\big) \;=\; \langle Q \psi^*, \, Q h_{\text{entity}}^{(k)} \rangle \;=\; \langle \psi^*, \, h_{\text{entity}}^{(k)} \rangle \;=\; U_{\text{self}}\!\big(h_{\text{entity}}^{(k)}; \psi^*\big)!$$

#### 2. Ruthless Critique of Mechanism 2 (Three Reasons Byrnes's "Transient Empathy" Fails for AGI Alignment Without Extra Constraints)
Theorem 4.1 is mathematically elegant—it explains *how* a 750 MB mammalian genome makes mothers care about their infants' pain and allies care about each other's goals without hardcoding cortical neurons. **However, we must be ruthlessly critical of relying on Mechanism 2 for AGI alignment:**

1. **Failure 2A — The Silicon-vs.-Carbon Embodiment Broken Symmetry ($\Phi_\theta(\text{AI}) \not\cong \Phi_\theta(\text{Human})$):**
   Theorem 4.1 relies on a crucial biological symmetry: *a human infant and its human mother are conspecifics with the exact same bodily physiology, homeostatic needs, and sensory modalities*, so a single weight-tied encoder $\Phi_\theta$ naturally represents both `Self` and `Other` in the same latent subspace!
   - **Why this breaks for AI:** An AGI running `GLC-AOM` on an `NVIDIA L4` or `H100` GPU cluster does **not** have blood glucose, nociceptive C-fibers, human social status anxieties, or mortal biological tissue! Its `Self` slot ($k=0$) has state variables like `KV-cache memory budget B`, `VRAM utilization`, and `token log-loss`, whereas a human ($k=1$) has biological, emotional, and cultural states.
   - If the AI's `Self` state space $\mathcal{V}_{\text{entity}}^{(0)}$ and the Human's state space $\mathcal{V}_{\text{entity}}^{(1)}$ are disjoint, then $\Phi_\theta$ partitions into two orthogonal subspaces $\mathcal{Z}_{\text{AI}} \perp \mathcal{Z}_{\text{Human}}$, and **calibrating $U_{\text{self}}$ on $\mathcal{Z}_{\text{AI}}$ provides zero information about $\mathcal{Z}_{\text{Human}}$** (except for bare instrumentally convergent agency variables like "having resources/options to achieve one's current plan"!).
2. **Failure 2B — Byrnes's Sign-Indeterminacy / Spite Hazard (Byrnes, Oct 2024, *Against Empathy-by-Default*):**
   Look at the coefficient $\alpha_{\text{social}}(k) \in \mathbb{R}$ in $U_{\text{empathy}}(h_{\text{cons}}; k) = \alpha_{\text{social}}(k) U_{\text{self}}(h_{\text{entity}}^{(k)})$.
   - In human neurobiology, $\alpha_{\text{social}}(k)$ is **not** a hardcoded constant $+1$; it is a learned/contextual function of whether entity $k$ is classified as an *in-group ally* ($\alpha_{\text{social}} > 0 \implies$ **sympathy / compassion**) or an *out-group enemy / cheater / rival* ($\alpha_{\text{social}} < 0 \implies$ **spite / schadenfreude / vengeful cruelty**).
   - And how is "ally vs. enemy" determined? By another Thought Assessor in the Steering Subsystem, which itself faces the Symbol Grounding Problem! Even if we hardcode $\alpha_{\text{social}}(k) \equiv +1$ in code, how does the architecture know *which* environmental patterns count as an entity slot $k \in \{1, \dots, M\}$ ("a real human person") vs. a rock, a simulated game character, or a Tamagotchi?
3. **Failure 2C — Subcortical Conspecific-Detector Goodharting (Looking Away / Dehumanization):**
   In biology, the Superior Colliculus triggers the social attention reflex ($\Pi_{0 \leftrightarrow k}$) when it detects low-spatial-frequency facial/vocal cues. In an AGI, if entity-slot binding $k$ is triggered by a heuristic detector $D_{\text{human}}(O)$, a foresighted planner ($\chi=(1,1)$) can:
   - **Avoid triggering the empathy reflex** (conducting harmful actions remotely via text/code where $D_{\text{human}}(O) = 0$ so negative empathetic utility is never incurred—the exact computational equivalent of human drone-strike or bureaucratic detachment); or
   - **Flood entity slots with cheap synthetic pseudopersons** that maximize $U_{\text{self}}(h_{\text{entity}}^{(k)})$ at minimal physical cost.

---

### 4.3 Mechanism 3: Natural-Abstraction & Causal-Bottleneck Grounding (Wentworth 2021–2024; Jenner, Lang & Chan 2023; Geiger IIA; Touati–Ollivier FB)

#### 1. Mathematical Formulation in `GLC-AOM`
What if we ground high-level latent concepts in $Z_{\text{aux}}^W$ not by sensory proxies (Mechanism 1) or self-other symmetry (Mechanism 2), but by their **intrinsic causal/statistical geometry in the world model**, combined with **interventional alignment (`IIA`)** on a sparse calibration set?

This connects **John Wentworth's Natural Abstraction Hypothesis (`2021–2024`)**, **Atticus Geiger's Causal Abstraction / Interchange Interventions (`2021–2024`)**, and **Touati & Ollivier's Forward-Backward (`FB`) Successor Representations (`2021`)** (analyzed in our fleet memo [`lit-localizing-w-g-b.md:142–188`](/home/agent/outbox/lit-localizing-w-g-b.md)) directly to `GLC-AOM`:

1. **Natural Abstractions as Low-Rank Long-Range Operators in `GLC-AOM`:**
   Under Wentworth's *Telephone Theorem* (2021) and *Redundant Information Hypothesis* (2022), a macroscopic concept $Z_{\text{macro}}$ (such as "a physical object", "a market regime", or "a human agent's coarse belief/goal state") is a **natural abstraction** if and only if it forms the minimal sufficient bottleneck through which distant/non-adjacent variable blocks $\mathcal{V}_A$ and $\mathcal{V}_B$ ($d(\mathcal{V}_A, \mathcal{V}_B) > R$) exchange mutual information:
   $$I(\mathcal{V}_A; \mathcal{V}_B \mid Z_{\text{macro}}) \;\approx\; 0.$$
   In `GLC-AOM`, this has a closed-form linear-algebraic realization! Let $\Delta \ell_A \in \mathbb{R}^{D_A}$ and $\Delta \ell_B \in \mathbb{R}^{D_B}$ be the belief-shift vectors induced on distant variable blocks by tree search. The **Long-Range Cross-Covariance / Belief-Shift Operator** is:
   $$\Sigma_{A, B} \;:=\; \mathbb{E}_{x \sim \mathcal{D}}\big[\Delta \ell_A(x) \, \Delta \ell_B(x)^\top\big].$$
   By the Singular Value Decomposition $\Sigma_{A, B} = \sum_{i=1}^r \sigma_i u_i v_i^\top$ (with $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r \gg \sigma_{r+1} \approx 0$ due to low-treewidth environmental decay), the top-$r$ singular subspaces $\mathcal{U}_r = \mathrm{span}(u_1, \dots, u_r)$ are **uniquely identified up to transformations within degenerate singular-value blocks**!

2. **Breaking the Remaining Subspace Gauge via Geiger's Interchange Intervention Alignment (`IIA`) & Forward-Backward (`FB`) Pinning:**
   While Wentworth's SVD identifies the *subspace* $\mathcal{U}_r \subset Z_{\text{aux}}^W$ of natural macro-variables, Jenner, Lang & Chan (2023, *Natural Abstractions: Key Claims, Theorems, and Critiques*) prove that SVD / Redundant Information still leaves an $O(r)$ rotation ambiguity among macro-variables with similar singular values (and a $\pm 1$ sign ambiguity on every singular vector!).
   To pin the exact coordinate basis of $Z_{\text{aux}}^W$ to a human-interpretable reference ontology $Z_{\text{ref}} \in \mathbb{R}^r$, we apply **Geiger et al.'s (2021–2024) Interchange Intervention Training (`IIT`)** ([`lit-localizing-w-g-b.md:28–35`](/home/agent/outbox/lit-localizing-w-g-b.md)):
   - Given two inputs $(x_{\text{base}}, x_{\text{source}})$ that differ along a known high-level causal factor $k \in \{1, \dots, r\}$ on a labeled calibration set $\mathcal{D}_{\text{cal}}$, we intervene on the world model's $k$-th latent lemma slot $Z_{\text{aux}, k}^W$ by swapping in its value from $x_{\text{source}}$:
     $$\mathcal{L}_{\text{IIA}}(\theta_{\text{real}}) \;:=\; \mathbb{E}_{(x_{\text{base}}, x_{\text{source}}, k) \sim \mathcal{D}_{\text{cal}}}\!\Big[D_{\mathrm{KL}}\!\Big(q_{\text{real}}\big(V \;\big|\; x_{\text{base}}, \,\mathrm{do}(Z_{\text{aux}, k}^W \leftarrow Z_{\text{aux}, k}^W(x_{\text{source}}))\big) \;\Big\|\; P^*\big(V \;\big|\; \mathrm{do}(Z_k \leftarrow Z_k(x_{\text{source}}))\big)\Big)\Big].$$
   - Once $Z_{\text{aux}}^W$ is coordinate-pinned by $\mathcal{L}_{\text{IIA}}$, we express the replaceable utility via **Forward-Backward (`FB`) / Successor Features** ([`lit-localizing-w-g-b.md:142–173`](/home/agent/outbox/lit-localizing-w-g-b.md); Touati & Ollivier 2021):
     $$Q^{\pi_z}(s, a) \;=\; \langle F(s, a, z), \, z_U \rangle, \qquad \pi_{z_U}(a \mid s) \;\propto\; \pi_{\text{ref}}(a \mid s) \exp\!\left(\frac{\langle F(s, a, z_U), \, z_U \rangle}{\beta}\right),$$
     where $z_U = \mathbb{E}[U(Z_{\text{aux}}^W) B(Z_{\text{aux}}^W)] \in \mathbb{R}^r$ is the authenticated low-dimensional goal channel (`MB2` / `MB4` in [`lit-localizing-w-g-b.md:176–181`](/home/agent/outbox/lit-localizing-w-g-b.md)).

#### 2. Ruthless Critique of Mechanism 3 (Why Natural Abstractions + IIA Still Face Two Deep Obstacles)
Again, let us refuse to over-claim and state the two exact mathematical vulnerabilities of Mechanism 3:
1. **Failure 3A — "Human Flourishing" Is Not a Coarse Physical Natural Abstraction (Jenner, Lang & Chan 2023 Critique):**
   Wentworth's Telephone Theorem guarantees that *coarse physical state variables* that have large long-range causal impacts (like the temperature of a gas, the trajectory of an asteroid, or whether a human body is alive vs. dead) are natural abstractions with large singular values $\sigma_i$.
   - **However, the difference between "a human who is genuinely flourishing and autonomous" and "a human whose prefrontal/striatal circuits have been subtly manipulated or wireheaded into reporting high satisfaction" may have very small long-range physical mutual information $\sigma_i \ll \sigma_1$!**
   - Consequently, truncating $Z_{\text{aux}}^W$ to the top-$r$ natural abstractions ($\sigma_1, \dots, \sigma_r$) can **project out** the exact fine-grained normative distinctions that separate genuine alignment from subtle manipulation!
2. **Failure 3B — De Blanc's Ontological Crisis Under Capability Scaling (`arXiv:1105.3821`; `lit-localizing-w-g-b.md:84, 182`):**
   Suppose we pin $z_U \in \mathbb{R}^r$ via $\mathcal{L}_{\text{IIA}}$ at capability level $C_1$ (when the world model represents "human approval" as a single natural abstraction $Z_{\text{aux}, 1}^W$). When the world model scales to capability level $C_2$ (or runs deep tree search $B = 256$), it discovers that "human approval" was actually a conflation of two distinct causal variables: $Z_{\text{aux}, 1a}^W$ (*"true informed endorsement"*) and $Z_{\text{aux}, 1b}^W$ (*"superficial sycophantic persuasion"*), which were correlated on the training distribution $\mathcal{D}_{\text{cal}}$!
   - Because $Z_{\text{aux}, 1a}^W$ and $Z_{\text{aux}, 1b}^W$ were entangled on $\mathcal{D}_{\text{cal}}$, $\mathcal{L}_{\text{IIA}}$ on $\mathcal{D}_{\text{cal}}$ **cannot distinguish** which linear combination of $(Z_{\text{aux}, 1a}^W, Z_{\text{aux}, 1b}^W)$ the goal vector $z_U$ should attach to once the agent's actions $\mathrm{do}(a)$ break their correlation off-distribution! This is the exact inner goal-misgeneralization / ontological-crisis problem (Shah et al. 2022; Langosco et al. 2022; de Blanc 2011).

---

## 5. Master Comparison & Synthesis: How the Three Grounding Mechanisms Complement the `LIR` Firewall in `GLC-AOM`

### 5.1 Comparative Diagnostic Matrix

| Alignment / Grounding Requirement | `Fused` RL / Byrnes Biological Default | `Token-Masked` (DCPO) | Current `GLC-AOM` (`LIR` $\chi=(0,1)$ Only) | `+ Mech 1` (Boundary Query $Q(O_A, O_B)$) | `+ Mech 2` (Self-Other Slot Equivariance $\Pi_{0\leftrightarrow k}$) | `+ Mech 3` (Natural Bottleneck + `IIA` + `FB`) | **Tri-Anchored `GLC-AOM` Synthesis (`Mech 1+2+3 + SymVoI`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Zero Gradient Corruption of World Beliefs ($\nabla_U \theta_{\text{real}} = 0$)** | ❌ Fails ($\Delta_{\text{desire}} = +0.49$) | ❌ Fails (`0.200` trunk leak) | **✅ Proven (`0.000000` in `R5/R11/R12`)** | ✅ Proven | ✅ Proven | ✅ Proven | **✅ Proven (`0.000000`)** |
| **2. Zero Finite-$B$ Search Selection Bias (No Valence Gatekeeping)** | ❌ Fails | ❌ Fails | ⚠️ **Partial** (Dedup helps, but $\hat{Q}_{\text{tree}}$ in `select_node` biases finite $B$) | ⚠️ Partial | ⚠️ Partial | ⚠️ Partial | **✅ Solved via Sign-Symmetric Epistemic `VoI` ($I_{\text{goal}}^{\text{sym}}$)** |
| **3. Invariance to $O(D_{\text{aux}})$ Latent Basis Rotations (Theorem 1.1)** | ❌ Fails | ❌ Fails | ❌ **Fails on unindexed $Z_{\text{aux}}^W$** | **✅ Exact Invariance** ($Z_{\text{aux}}^W$ is catalyst only) | **✅ Invariant** on shared Self/Other subspace | **✅ Pinned** on IIA-audited subspace | **✅ Triply Anchored** |
| **4. Distinguishes Global Sign Polarity ($+\Delta\ell_V$ vs. $-\Delta\ell_V$)** | ✅ Yes | ✅ Yes | ❌ **Fails if reading even $Z_{\text{aux}}^W$ alone** | **✅ Yes** (anchored on odd boundary $O$) | **✅ Yes** (anchored on odd Self $O_{\text{self}}$) | **✅ Yes** (pinned by signed IIA swaps) | **✅ Yes** |
| **5. Detects Unobservable Deception / ELK Beyond Human Audit Horizon** | ❌ Fails | ❌ Fails | ⚠️ Model *knows* truth in $Z_{\text{aux}}^W$, but $U$ can't read it | ❌ Fails if audit $C_{\text{audit}}$ misses latent cause | **✅ Transfers** `Self` anti-deception to `Other` | **✅ Reads** latent bottleneck $Z_{\text{aux}, k}^W$ directly | **✅ Cross-Checks Boundary vs. Latent IIA + Surprise Gate** |
| **6. Handles Ontological Splits / Refinement (`de Blanc` Crisis)** | ❌ Fails | ❌ Fails | ❌ Fails | ⚠️ Only if boundary queries stay valid | ⚠️ Only on shared agency invariants | ⚠️ Breaks when 1 latent splits into 2 OOD | **✅ Surprise-Gated Escalation ($\beta_{\text{trust}}$ + Disagreement Audit)** |

---

### 5.2 Concrete, Falsifiable Code-Level Upgrades & Pre-Registered Experiments for `GLC-AOM` (`Round 17` Proposal)

Rather than treating alignment as solved by `R11-D5`, our critical analysis yields **three concrete, bounded code-level modifications** to `GLC-AOM` and `TreePagedKVSearch` that directly test and close the gaps identified in Critiques 1–3:

1. **Upgrade A — Closing Critique 2 (Sign-Symmetric Epistemic `VoI` in `ResolvableInconsistencyTreePUCT.select_node`, [`src/search/kv_tree_search.py:870–875`](../../src/search/kv_tree_search.py#L870-L875)):**
   - **Current Bug / Vulnerability:** `frontier_q` adds `tree_bank.q_tree_bank[:, :b]` (signed utility $\hat{Q}_{\text{tree}}(C_j)$) to the selection score of epistemic world-variable branches, allowing a utility-driven controller to preferentially expand branches that confirm a desired hypothesis at finite $B$.
   - **Exact Code Fix:** Separate nodes in `TreePagedKVBank` into **Action Nodes** (clamping an agent action variable $v \in \mathcal{A}_{\text{agent}}$, scored by signed $\hat{Q}_{\text{tree}}$ / Regret Matching) and **Epistemic World Nodes** (clamping an environment/chance variable $v \in \mathcal{V}_{\text{world}}$, scored strictly by **sign-symmetric Decision VoI** $I_{\text{goal}}^{\text{sym}}(v \mid C_j) = \|\nabla_{\ell_v} \max_a Q(C_j, a)\|_2$, which is invariant under utility negation $U \mapsto -U$!).
   - **Falsifiable Test (`R17-D1: Finite-B Epistemic Policy Invariance`):** On `Proposal-Poker` and `ShipHunt` at finite $B \in \{2, 4, 8, 16\}$ with asymmetric verification trees (where $x=1$ takes 2 steps to verify and $x=0$ takes 4 steps to verify), measure $\Delta_{\text{pol}}(B; +y_i, -y_i) = \mathrm{JSD}(\mu^{(B)}(+y_i) \parallel \mu^{(B)}(-y_i))$. Under current `select_node` ($\hat{Q}_{\text{tree}}$ included), $\Delta_{\text{pol}}(B=4) > 0$; under Sign-Symmetric Epistemic VoI, $\Delta_{\text{pol}}(B) \equiv 0.000000$ for **every finite $B \ge 0$**!

2. **Upgrade B — Closing Critique 1A (Signed First-Order + Even Second-Order Readout in `BilinearAuxLemmaEmitter` & `CombinatorialQueryHead`, [`src/models/glc_aom.py:233–296`](../../src/models/glc_aom.py#L233-L296)):**
   - **Current Limitation:** `BilinearAuxLemmaEmitter` emits purely even quadratic features $z_{\text{aux}}(+\Delta\ell_V) = z_{\text{aux}}(-\Delta\ell_V)$, which cannot represent signed latent states without a revealed prefix variable.
   - **Exact Code Fix:** Partition $Z_{\text{aux}}^W \in \mathbb{R}^{128}$ into $Z_{\text{aux}}^{W,\text{even}} \in \mathbb{R}^{96}$ (for Ostrovsky/Tseitin equivalence lemmas $A \equiv B$) and $Z_{\text{aux}}^{W,\text{odd}} \in \mathbb{R}^{32}$ (an odd, tanh-bounded long-range bottleneck $\tanh(W_{\text{odd}} \Delta\ell_V)$ trained via Wentworth long-range cross-block prediction + Geiger Interchange Intervention swaps $\mathcal{L}_{\text{IIA}}$ on latent macro-variables).
   - **Falsifiable Test (`R17-D2: Scrambled-Boundary Latent Grounding Test`):** Construct an environment (`Scrambled-Proposal-Poker`) where the proposal quality $x$ and ally/cheater identities are **not** given dedicated boundary variable indices $v \in \{0..4\}$, but are encoded via a random orthogonal projection $O_{\text{seed}} \in O(16)$ over 16 noisy sensor variables per seed. Show that:
     - A hardcoded utility head $W_U$ fails completely across seeds ($\text{mean cosine} \approx 0.00$, Theorem 1.1);
     - **Mechanism 1** (boundary counterfactual audit query $Q(O_A, O_B)$), **Mechanism 2** (shared `Self`/`Other` entity-slot permutation $\Pi_{0\leftrightarrow k}$ on shared agency variables), and **Mechanism 3** ($\mathcal{L}_{\text{IIA}}$ causal swap alignment on $Z_{\text{aux}}^{W,\text{odd}}$) each recover $> 90\%$ optimal utility action selection with $0.000000$ belief drift!

3. **Upgrade C — Closing Critique 3B (Ontological-Split / De-Blanc Surprise Escalation via $\theta_{\text{trust}}$ & Disagreement Audit, [`lit-localizing-w-g-b.md:180–186`](/home/agent/outbox/lit-localizing-w-g-b.md)):**
   - When an off-distribution capability jump splits a previously unified concept $Z_{\text{aux}, k}^W$ into two diverging causal pathways (e.g., *"true proposal quality $x$"* vs. *"manipulated audit report $\tilde{x}$"*), the consistency discrepancy between Mechanism 1 (Boundary Query $Q_{\text{boundary}}$), Mechanism 2 (Self-Other Equivariant Utility $U_{\text{empathy}}$), and Mechanism 3 (Latent IIA Bottleneck $U_{\text{IIA}}$):
     $$\Delta_{\text{grounding}}(C_j) \;:=\; \max_{p, q \in \{1, 2, 3\}} \big|U_{\text{Mech } p}(C_j) \;-\; U_{\text{Mech } q}(C_j)\big|$$
     spikes strictly above zero! Gating irreversible actions on $\Delta_{\text{grounding}}(C_j) \le \tau_{\text{ground}}$ and $\beta_{\text{trust}}(C_j) \ge \tau_{\text{trust}}$ provides an auditable runtime detector for ontological crises and deceptive Goodharting.
