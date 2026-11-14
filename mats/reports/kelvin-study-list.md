# Kelvin's study list (MATS)

Every arXiv entry was checked against the arXiv API on 2026-10-29.

## Already in hand
- Bach, *Learning Theory from First Principles* (MIT Press 2024).
- Radhakrishnan, Beaglehole, Pandit, Belkin, "Mechanism for feature learning in neural networks and backpropagation-free machine learning models", Science 2024 (bookmarked).
- H-Net: Hwang, Wang, Gu 2025, "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling", https://arxiv.org/abs/2507.07955 (registered for the info-aggregation track).

## Lottery Ticket Hypothesis (added 2026-10-29)
Suggested order:
1. Frankle & Carbin 2019, "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks", https://arxiv.org/abs/1803.03635 — the original: dense nets contain sparse subnetworks that train to similar accuracy from their original initialization.
2. Frankle, Dziugaite, Roy, Carbin 2020, "Linear Mode Connectivity and the Lottery Ticket Hypothesis", https://arxiv.org/abs/1912.05671 — tickets exist at scale only after early training ("rewinding"), tied to stability to SGD noise.
3. Frankle, Dziugaite, Roy, Carbin 2019, "Stabilizing the Lottery Ticket Hypothesis", https://arxiv.org/abs/1903.01611 — late rewinding.
4. Zhou, Lan, Liu, Yosinski 2019, "Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask", https://arxiv.org/abs/1905.01067 — what matters in a ticket; masks alone perform well.
5. Ramanujan et al. 2020, "What's Hidden in a Randomly Weighted Neural Network?", https://arxiv.org/abs/1911.13299 — good subnetworks inside random, untrained nets.
6. Malach, Yehudai, Shalev-Shwartz, Shamir 2020, "Proving the Lottery Ticket Hypothesis: Pruning is All You Need", https://arxiv.org/abs/2002.00585 — the strong version, proved: a large enough random net contains a subnetwork approximating any target net.
7. Pensia et al. 2020, "Optimal Lottery Tickets via SubsetSum: Logarithmic Over-Parameterization is Sufficient", https://arxiv.org/abs/2006.07990.
Why for us: density as a search aid versus a representational need (the tangledness question).

## Tensor networks: correlations across a cut, trees and MERA (added 2026-10-29)
Primers:
1. Bridgeman & Chubb 2017, "Hand-waving and Interpretive Dance: An Introductory Course on Tensor Networks", https://arxiv.org/abs/1603.03039 — the gentlest start.
2. Orús 2014, "A Practical Introduction to Tensor Networks: Matrix Product States and Projected Entangled Pair States", https://arxiv.org/abs/1306.2164.
Deep learning ↔ tensor networks (Cohen, Sharir, Levine, Shashua):
3. Cohen, Sharir, Shashua 2016, "On the Expressive Power of Deep Learning: A Tensor Analysis", https://arxiv.org/abs/1509.05009 — depth efficiency via hierarchical tensor decompositions.
4. Cohen & Shashua 2017, "Inductive Bias of Deep Convolutional Networks through Pooling Geometry", https://arxiv.org/abs/1605.06743 — which correlations across a partition a net can model (separation rank) is set by its pooling geometry.
5. Levine, Yakira, Cohen, Shashua 2018, "Deep Learning and Quantum Entanglement: Fundamental Connections with Implications to Network Design", https://arxiv.org/abs/1704.01552 — the min-cut of the network's tensor-network graph bounds the correlations across a cut. This is the key paper for the bond-dimension statement.
6. Levine, Sharir, Cohen, Shashua 2019, "Quantum Entanglement in Deep Learning Architectures", PRL, https://arxiv.org/abs/1803.09780.
7. Levine, Sharir, Ziv, Shashua 2017, "On the Long-Term Memory of Deep Recurrent Networks", https://arxiv.org/abs/1710.09431.
8. Cohen, Sharir, Levine, Tamari et al. 2017, "Analysis and Design of Convolutional Networks via Hierarchical Tensor Decompositions", https://arxiv.org/abs/1705.02302 — an overview of the line.
Tree tensor networks and MERA (physics):
9. Shi, Duan, Vidal 2006, "Classical simulation of quantum many-body systems with a tree tensor network", https://arxiv.org/abs/quant-ph/0511070.
10. Vidal 2007, "Entanglement renormalization", https://arxiv.org/abs/cond-mat/0512165 — MERA.
11. Vidal 2008, "A class of quantum many-body states that can be efficiently simulated", https://arxiv.org/abs/quant-ph/0610099.
Tensor networks for ML and probabilistic modelling:
12. Stoudenmire & Schwab 2016, "Supervised Learning with Quantum-Inspired Tensor Networks", https://arxiv.org/abs/1605.05775.
13. Stoudenmire 2018, "Learning Relevant Features of Data with Multi-scale Tensor Networks", https://arxiv.org/abs/1801.00315 — tree / coarse-graining for learning.
14. Glasser, Sweke, Pancotti, Eisert, Cirac 2019, "Expressive power of tensor-network factorizations for probabilistic modeling", https://arxiv.org/abs/1907.03741 — closest to our probabilistic families (relations to HMMs and Born machines).
Renormalization group ↔ deep learning:
15. Mehta & Schwab 2014, "An exact mapping between the Variational Renormalization Group and Deep Learning", https://arxiv.org/abs/1410.3831.
Why for us: a precise notion of "not tangled at any scale" (bounded bond dimension, or min-cut, per scale), and architectures where exact inference is tractable by construction.

## Residual networks with small per-layer changes (added 2026-10-29, Kelvin: "beyond the ARC setup, we can also study resnets, where we can cap each layer to cause a small change")
1. Li, Nica, Roy 2021, "The Future is Log-Gaussian: ResNets and Their Infinite-Depth-and-Width Limit at Initialization", https://arxiv.org/abs/2106.04013 — in the joint limit, the output norm is log-Gaussian, governed by depth/width. This is the "size" mode in our nets.
2. Hayou & Yang 2023, "Width and Depth Limits Commute in Residual Networks", https://arxiv.org/abs/2302.00453 — with 1/sqrt(depth) branch scaling, the infinite-depth limit is well defined.
3. Yang, Yu, Zhu, Hayou 2023, "Tensor Programs VI: Feature Learning in Infinite-Depth Neural Networks", https://arxiv.org/abs/2310.02244.
4. Bordelon, Noci, Li, Hanin, Pehlevan 2023, "Depthwise Hyperparameter Transfer in Residual Networks: Dynamics and Scaling Limit", https://arxiv.org/abs/2309.16620.
Background for the Lyapunov count: Newman 1986, "The distribution of Lyapunov exponents: exact results for random matrices", Comm. Math. Phys. 103 (not on arXiv).

## Attention flow and Shapley values (added 2026-11-06)
Checked against the arXiv API on 2026-11-06. Kelvin's note: the link between attention flows and Shapley values looks significant for our earlier market-inspired ideas (paying the parts of a network, credit assignment as in markets).
1. Abnar & Zuidema 2020, "Quantifying Attention Flow in Transformers", https://arxiv.org/abs/2005.00928 — attention rollout and attention flow (max-flow over the layered attention graph) as measures of how much each input token contributes.
2. Ethayarajh & Jurafsky 2021, "Attention Flows are Shapley Value Explanations", https://arxiv.org/abs/2105.14652 — attention flows are Shapley values (at the layer level), unlike raw attention weights or leave-one-out.
Why for us: Shapley values are a fair-division rule for credit; if attention flow computes them, a transformer's information routing already has a market-like credit assignment, which bears on the "markets inside networks" track (G2) and on how aggregation credit is split across sources (G4, G5).
