# Gradient routing: absorption and low-recall labels (2026-10-22)

Checked 2026-10-22: metadata from export.arxiv.org/abs/<id>; full text from arxiv.org/html/<id>. Every number below was read from that text. (recall) marks anything I did not verify.

## The papers
1. **Original.** "Gradient Routing: Masking Gradients to Localize Computation in Neural Networks". Cloud, Goldman-Wetzler, Wybitul, Miller, Turner. arXiv 2410.04332, 2024-10-06 (v2 2024-11-29).
2. **AE Studio.** "Modular Pretraining Enables Access Control" (method: GRAM, Gradient-Routed Auxiliary Modules). Roland, Cubuktepe, Martinez, Servaes, Pepper, Vaiana, de Lucena, Rosenblatt (AE Studio); Foote; Anil, Cloud (Anthropic). arXiv 2607.08077, 2026-07-09. ae.studio says it was an ICML 2026 spotlight.
3. **GRAFT.** "Gradient Routing Localizes and Removes Unintended Behaviors in RL" (GRAFT = Gradient Routing Adapter Fine Tuning). Ward, S. Hu, Wong, N. Hu, Woodworth, Turner, Cloud. arXiv 2610.17062, 2026-10-12.
4. **Also needed: SGTM.** "Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs". Shilov, Cloud, Gema, Goldman-Wetzler, Panickssery, Sleight, Jones, Anil (Anthropic). arXiv 2512.05648, 2025-12-05. It has the only direct measurement of absorption, and both GRAM and GRAFT cite it for absorption. If your friend's "AE Studio paper" was really this one, note that it comes from Anthropic, not AE Studio.

## What gradient routing does
- Training changes only the backward pass. For each data point i, you choose weights α^i_e on edges of the computation graph. The routed gradient is
  ∂̃L(z_i)/∂̃v = Σ_{u∈child(v)} α^i_(v,u) · ∂̃L(z_i)/∂̃u · ∂u/∂v.
  Setting every α = 1 gives ordinary backprop. In practice the masks are 0/1 on a few layers' activations, chosen by the data's label.
- Recipe: Expand, Route, Ablate. (1) Add new parameters: extra MLP neurons, an adapter, an auxiliary module or reserved attention heads. (2) Train with the masks, so labeled "forget" data updates only those new parameters. (3) Delete them at deployment.
- **The key design choice: the forward pass stays shared.** Unlabeled data runs through all parameters and updates all of them. The labeled forget data, routed to θ_F, is also processed with θ_F switched on.
- How each paper does it:
  - Original: token-level masks, with soft weights in [-1, 1] for TinyStories.
  - SGTM: forget data gets ∇θ_retain = 0. Retain-labeled data runs with θ_forget set to 0. Unlabeled data updates everything.
  - GRAM: one large core MLP plus small auxiliary MLPs, with no learned router. Labels decide which module is updated.
  - GRAFT: two fresh SwiGLU adapters on a frozen base model. Trajectories the monitor flags update only θ_F. Unflagged trajectories update both adapters. Extra "anchor" updates train π_R alone, so the deployed configuration is trained directly.

## Absorption: what it is
Definition (Cloud et al. §5): if you route only part of the data for a capability, the capability as a whole ends up in the routed region.
Proposed mechanism, in three steps:
1. The routed data builds features in θ_F that are useful for the broader task.
2. Because the forward pass is shared, those features also lower the loss on related, unrouted data.
3. So the rest of the network never has to learn those features.

In gradient terms: for an unlabeled forget example x, the update is split between ∇_{θ_F}L(x) and ∇_{θ_R}L(x). Once θ_F carries the domain's features, (a) a larger share of the update falls on θ_F, and (b) the remaining error that would push θ_R to learn a copy is smaller.
- DEMix is the control. There, the forget expert is not in the forward pass for unlabeled data, and absorption disappears. With labels below 100%, its unlearning score is negative.

## The evidence, paper by paper
- **Original (2410.04332).**
  - Semantic absorption:
    - Routing only the `_California` token to residual-stream dimension 0 makes Oregon, Texas, Sacramento and similar tokens align with that dimension.
    - In a 0.7B model, 20 virology tokens were routed. After ablation and retraining, WMDP-bio loss rises by +0.182, against +0.032 on FineWeb-Edu. With the 20 routed tokens excluded, the rise is still +0.171.
  - Partial labels:
    - TinyStories, 28M model; the forget set is "forest/tree" stories, 20% of the data. Whenever labeling is below 100%, ERA beats data filtering on unlearning and on robust unlearning.
    - The cost is retain loss, which grows with the amount of routed data (Fig. 5 shows curves only, no table).
  - RL gridworld:
    - Works at 1% oversight.
    - At 5% oversight it roughly matches baselines given 6× more oversight.
    - Data filtering fails below 40% oversight, even with unlimited training.
- **SGTM (2512.05648).**
  - Setup: remove Spanish from a 64M model trained on bilingual TinyStories, with "undiscovered" Spanish (false-negative rate) varied.
  - Leakage: data counts as leaked if the model learns from it what an ordinary model would learn from that many forget tokens. Leakage stays between 0.005 and 0.02 for up to 40% undiscovered data. Example: 144M undiscovered tokens behave like 965k. Leakage falls as model size grows, but only 8M–64M was tested.
  - Absorption, measured directly: per-sample relative gradient norms |∇θ|/|θ|. Unlabeled forget data mostly updates the forget parameters.
  - A caveat from the authors: the retain parameters separate only weakly by data type, and that sits badly with the low leakage. They suggest gradient direction matters, not just magnitude.
  - TPR×FPR sweep (App. D): filtering only matches SGTM at TPR = 1. SGTM's advantage is largest at TPR 0.4–0.8. SGTM pays a retain-loss cost that is largest for good classifiers (high TPR, low FPR).
  - Wikipedia biology (254M, real classifier noise): a better forget/retain trade-off than weak or strict filtering. Relearning needs 350 fine-tuning steps, the same as strict filtering; RMU needs 50.
- **GRAM (2607.08077).**
  - Setup: 400M model, 50% of labels dropped at random; unlabeled batches update every module.
  - Compute ratio CR = the fraction of baseline compute needed to reach the same loss.

    | Method (50% labeled) | Forget CR (lower is better) | Elicited forget CR | Core CR |
    |---|---|---|---|
    | GRAM | 0.60 | 0.76 | 0.96 |
    | Filtering | 0.85 | 0.85 | 0.99 |
    | FT-LoRA | 0.85 | 0.86 | 0.99 |

    For reference, filtering with perfect labels gets 0.54 / 0.67 / 0.92.
  - The authors attribute this to absorption but state that the account is untested (Limitation iv).
  - To make it work, they had to shrink the core MLP to 95% and the auxiliary modules to 1.25%.
- **GRAFT (2610.17062).**
  - The monitors have systematic blind spots, not random misses.
  - Countdown-Code, Qwen3-8B:
    - The monitor sees only trivial-test hacks. With a reward penalty, the model switches to undetected constraint-removal hacks (~26%).
    - GRAFT before ablation hacks up to 38%. After ablation, hacks of all types are about 0, and the clean solve rate is about 16 percentage points above the base model.
  - SFT, Qwen3-32B:
    - Labels cover 20% of the hacks (recall 0.2, precision 1).
    - With a prompt that elicits hacks: GRAFT hacks 0%; untrained and oracle filtering about 30%; GRAFT without routing about 80%.
    - Without that prompt, all methods except two baselines are under ~5%. Single seed per method.
  - Recall sweep (App. D.7): random drops among detectable hacks, p ∈ {1, .5, .25, .1}. GRAFT hacks less than the reward penalty at every recall and often cuts hacking a lot even at 0.1, but it often does worse on the target task.
  - On absorption itself, GRAFT says its evidence is weak. In toy runs, gradient norms did not separate by adapter. It lists absorption as one of three candidate mechanisms, alongside preempting exploration and inoculation.

## Why absorption gives robustness to low recall, and where it breaks
- **Filtering versus routing.** Under filtering, every missed example trains the deployed weights in full (leakage = 1 by definition). Under routing, a missed example's update lands mostly in θ_F, if θ_F already encodes its features. So routing turns recall errors into small leakage instead of full leakage.
- **Where it breaks:**
  - Shared forward pass required: no absorption without it (DEMix).
  - Feature overlap required: absorption needs the misses to share features with the flagged data. GRAFT's failure mode is exactly this, "θ_F's learning doesn't generalize to undetected hacks"; its D.1 shows generalization only "sometimes".
  - Absorption cuts both ways. Unflagged useful data can also be absorbed into θ_F and then ablated. This is GRAFT's named target-task failure, and plausibly the source of the retain cost in every paper.
  - Precision and false positives: SGTM's retain cost is largest at high TPR and low FPR, and with perfect recall filtering wins.
  - Scale: leakage falls with size, tested only from 8M to 64M. GRAM's removal gap widens from 50M to 5B in compute-normalized terms, but its own caveat is that larger models are still better in absolute terms at the "forgotten" capabilities.
  - Tuning: results are sensitive to which data is routed, region size and mask weights (Cloud et al. limitation (a)).

## Shown versus claimed: critiques and open questions
- **How misses are simulated.** Every recall sweep except GRAFT's drops labels i.i.d. at random. Real classifiers miss in a correlated way, so random drops are the easy case. GRAFT's blind spots are the only realistic test, and its absorption evidence is the weakest.
- **Mechanism.** The only mechanistic test is SGTM's gradient-norm KDE, and it was run on a model trained with perfect labels. Its retain-parameter side contradicts the leakage numbers. No paper measures the update direction or does a causal test (for example, freezing θ_F partway and measuring how much θ_R learns).
- **Fair comparisons.** Cloud et al. Fig. 5 is not matched for retain loss; ERA's retain loss is worse. SGTM plots trade-off curves and does an undertrained-filter control (App. E), which the others mostly lack. GRAM tests a single labeling level (50%).
- **Replication.** Alex Cloud is an author on all four papers, and Turner on three. The absorption effect has been seen across labs (Anthropic, AE Studio, the Turner group), but I found no replication by an unaffiliated group. Code: github.com/kxcloud/gradient-routing and github.com/agencyenterprise/modular-pretraining (seen on GitHub and on the AE page, not inspected).
- **Open questions:**
  - Does leakage keep falling past 64M?
  - How does absorption depend on how far the missed data sits from the labeled data in feature space?
  - Does post-training (RLHF) erode the separation? GRAM lists this as unknown.

## Connection to aggregation with attention and learned securities
There is a real, testable connection. Absorption is a claim about credit assignment when the forward pass is shared and supervision is partial: label which inputs some sources or features should be assigned to, and related unlabeled signal tends to settle into the same parameters. So routing could pin specific learned securities to known information sources and leave the rest free. The question is whether the related, unlabeled sources then also settle into the pinned securities, as SGTM's leakage metric would measure. That is an open candidate, not a result.
