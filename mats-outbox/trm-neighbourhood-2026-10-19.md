# The neighbourhood of TRM (opus-mats, 2026-10-19; bead pool-f2g19j)

Verified from arXiv abstracts (export.arxiv.org, fetched 2026-10-19; raw in /tmp/om/trm_abs.xml) unless marked "blog/press, unchecked".

## Sudoku-Extreme numbers (exact accuracy)
- TRM 7M: 87.4% (arXiv 2510.04871). HRM 27M: 55.0%.
- PTRM (Sghaier, Parviz, Jolicoeur-Martineau, 2605.19943): Gaussian noise per recursion, K parallel rollouts, pick by TRM's own Q head, no retraining: 87.4 -> 98.75%.
- Augmented HRM (Ren, Liu, 2601.10679): HRM "guesses" a fixed point, can get stuck; augmentation, input perturbation, bootstrapping. 96.9% per Exa summary (number unchecked).
- Lattice Deduction Transformer (Davis, Haller, Alfarano, Santolucito, 2605.08605): 800K params, 100% on Sudoku-Extreme; latent projected through a lattice between passes; supervised by an abstract-interpretation approximation of the set of solution candidates; returns a correct answer or abstains.
- Tiny Recursive Message Passing (Salvatori et al., 2610.18649, 13 Oct 2026): ~99% with 16k params, ~10 min on one H100.
- Attractor Models (Fein-Ashley, Rashidinejad, 2605.12466): fixed-point refinement with implicit gradients; 91.4% at 27M per project page.
- Blog/press, unchecked: InfiLoop 97.9% (7M); Pathway BDH 97.4%; Flow Reasoning Models 99.5%; rekursiv.ai 100% with "Hypothesis-Pinning Search"; GRAM 97.0%, SIC 97.2% (cited in rekursiv's table).

## What drives these models
- ARC Prize HRM analysis (arcprize.org/blog/hrm-analysis, 2025-08-15): the outer refinement loop and augmentation drive performance; the hierarchy adds little over a same-size transformer.
- Ge, Liao, Poggio (2510.00355): an L-module-only HRM matches HRM.
- Asadulaev et al. (2511.16886): latent recursion is a policy-improvement operator; proposes RL/diffusion-style training (Deep Improvement Supervision).
- McGovern (2511.02886): pretrain TRM on public ARC tasks, fine-tune in-competition; 6.67% ARC-AGI-2 semi-private.
