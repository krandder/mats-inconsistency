# Citations checked for scale-free-theory (2026-10-29, leaf pool-yw7kwn.10)

Checked against the arXiv API (abstracts saved in cites/batch_*.xml) or a web search of the publisher / author page.
"Claim used" is what the deliverable relies on. These are existence-and-abstract checks, not theorem-level checks:
the hypotheses of the cited theorems were read from abstracts or secondary descriptions, and the reviews
(review-gpt6-2026-10-29.md, review-gemini-2026-10-29.md) flagged where the note's first draft over-extended them.
Nisan-Wigderson constants were taken from later papers' descriptions; Dyer 2003's #P-completeness statement from a
seminar abstract and citing papers.

| Reference | Where checked | Claim used |
|---|---|---|
| Alman & Song 2023, Fast Attention Requires Bounded Entries, arXiv:2302.13214 | arXiv abstract | Q,K,V in [-B,B]^{n x d}, A = exp(QK^T/d), d = O(log n): B = o(sqrt(log n)) -> n^{1+o(1)} approx to 1/poly(n); B = Theta(sqrt(log n)) -> no n^{2-Omega(1)} algorithm under SETH |
| Keles, Wijewardena, Hegde 2022, arXiv:2209.04881 | arXiv abstract | self-attention time quadratic unless SETH false, also approximately; Taylor-series linear-time approximation with cost exponential in the order |
| Chakrabarti & Regev 2010/2012, arXiv:1009.3460 | arXiv abstract | randomized CC of gap-Hamming (gap sqrt(n)) is Omega(n) |
| Sanford, Hsu, Telgarsky 2023, arXiv:2306.02896 | arXiv abstract | sparse averaging: transformers log in input size; triple detection linear; CC proof techniques |
| Sanford, Hsu, Telgarsky 2024, arXiv:2402.09268 | arXiv abstract | constant attention layers <-> constant MPC rounds; log depth suffices for basic tasks |
| Sanford, Hsu, Telgarsky 2024, arXiv:2408.14332 | arXiv abstract | one-layer transformers need exponentially larger size than two-layer for induction heads (CC argument) |
| Amarilli, Capelli, Monet, Senellart 2018/2020, arXiv:1811.02944 | arXiv abstract | bounded-treewidth circuits -> d-SDNNF in time linear in size, singly exponential in treewidth |
| Abo Khamis, Ngo, Rudra 2016 (FAQ), arXiv:1504.04044 | arXiv abstract | one framework for joins, CSP, graphical-model inference; InsideOut; width governed by variable order / fractional covers |
| Markov & Shi 2008, arXiv:quant-ph/0511069 | arXiv abstract | contraction of a network of treewidth d in T^{O(1)} exp(O(d)) |
| Nguyen et al. 2021 FMMformer, arXiv:2108.02347 | arXiv abstract | attention = banded near field + low-rank far field |
| Zhu & Soricut 2021 H-Transformer-1D, arXiv:2107.11906 | arXiv abstract | H-matrix-structured attention, linear cost |
| Levine, Wies, Sharir, Bata, Shashua 2020, arXiv:2006.12467 | arXiv abstract | depth-to-width interplay in self-attention (separation-rank analysis) |
| Levine, Yakira, Cohen, Shashua 2017/2018, arXiv:1704.01552 | arXiv abstract | ConvAC <-> many-body wave function; entanglement measures quantify correlations; min-cut statements |
| Wu & Yang 2018/2020, arXiv:1807.07237 | arXiv abstract | k-component Gaussian location mixture estimated from moments (denoised method of moments) |
| Hahn 2020, arXiv:1906.06755 | arXiv abstract | self-attention cannot model periodic finite-state languages / hierarchy unless layers or heads grow |
| Rubinstein 2018, arXiv:1803.00904 | arXiv abstract | (1+eps)-approx bichromatic closest pair needs n^{2-delta} under SETH |
| Zaheer et al. 2017 Deep Sets, arXiv:1703.06114 | arXiv abstract | permutation-invariant functions as rho(sum phi) |
| Xiao et al. 2023 attention sinks, arXiv:2309.17453 | arXiv abstract | "attention sink" phenomenon (initial tokens absorb attention) |
| Lee, Lee, Oum 2012, arXiv:1001.0461 | arXiv abstract | rw(G(n,p)) = ceil(n/3) - O(1) for constant p; > r n for p = c/n, c > 1; G(n,c/n) linear treewidth for c > 1 |
| Lin & Tegmark 2017, arXiv:1606.06737 | arXiv abstract | MI decays exponentially for regular grammars, can decay as a power law for context-free |
| Lerner & Parr 2001, UAI, arXiv:1301.2288 | web (arXiv PDF, JAIR review) | CLG inference NP-hard even on polytrees; approximate too unless P = NP |
| Oum 2008, J. Graph Theory 57(3):239-244 | web (ACM DL, Oum survey arXiv:1601.03800) | rank-width <= branch-width; rank-width <= treewidth + 1 |
| Dyer 2003, STOC, Approximate counting by dynamic programming | web (ACM DL + seminar abstract) | counting 0-1 knapsack solutions is #P-complete; FPRAS by DP + rejection |
| Forster 2002, JCSS 65(4):612-625 | web (publisher abstract) | sign-rank(M) >= n / ||M||; Hadamard -> >= sqrt(n) |
| Nisan & Wigderson 1993, SIAM J. Comput. 22(1):211-219 | web (citing papers) | k-step pointer chasing: (k-1)-round lower bounds (det. Omega(n - k log n); randomized Omega(n/k^2 - k log n)); k rounds O(k log n) |
| Bova, Capelli, Mengel, Slivovsky 2016, IJCAI | web (TU Wien TR) | DNNF size lower bounds via multi-partition communication complexity (rectangle covers) |
| Evans, Kenyon, Peres, Schulman 2000, Ann. Appl. Probab. 10(2):410-433 | web (Project Euclid) | broadcasting a bit on a tree; reconstruction iff br(T)(1-2eps)^2 > 1 (Kesten-Stigum) |
| Chandrasekaran, Srebro, Harsha 2008, arXiv:1206.3240 | lit file (verified 10-29) | for arbitrary potentials, low treewidth is the only graph property giving tractable inference (grid-minor hypothesis) |
| Wagstaff et al. 2019, arXiv:1901.09006 | lit file (verified 10-29) | continuous sum-decomposition of arbitrary set functions needs latent dim >= N |

Classical, not re-fetched (standard textbook statements): Koopman 1936 (Trans. AMS 39) / Pitman 1936 / Darmois 1935
(fixed-dimensional sufficient statistics under regularity -> exponential family); Clarke & Barron 1990 (IEEE Trans.
IT 36(3)): I(theta; X^n) = (d/2) log(n / 2 pi e) + O(1); van der Vaart 1998, Asymptotic Statistics (one-step
estimators 5.7, Bernstein-von Mises ch. 10); Razborov 1992 (TCS 106) and Kalyanasundaram & Schnitger 1992 (SIAM J.
Discrete Math. 5) randomized disjointness Omega(n); Lauritzen 1992 (JASA 87) and Lauritzen & Jensen 2001 (Stat.
Comput. 11) conditional-Gaussian propagation; Valiant 2015 (JACM 62(2)) subquadratic light-bulb / outlier
correlations; Greengard & Rokhlin 1987 (J. Comput. Phys. 73) FMM; Grasedyck 2010 (SIMAX 31(4)) hierarchical SVD;
Dickson (quadratic forms over GF(2): zero count fixed by rank and Arf invariant); Kasteleyn 1961 / Fisher 1961
(planar zero-field Ising / dimers by Pfaffians); Paturi & Simon 1986 (unbounded-error CC = log sign-rank, up to 1).

## Added after review (2026-10-29, verified on the arXiv API)
| Reference | Claim used |
|---|---|
| von Oswald et al. 2022, Transformers learn in-context by gradient descent, arXiv:2212.07677 | transformers implement multi-step in-context estimation |
| Ahn, Cheng, Daneshmand, Sra 2023, arXiv:2306.00297 | transformers implement preconditioned gradient descent in context |
| Bai, Chen, Wang, Xiong, Mei 2023, Transformers as Statisticians, arXiv:2306.04637 | provable in-context algorithms and algorithm selection |
| Bietti et al. 2023, Birth of a Transformer: A Memory Viewpoint, arXiv:2306.00802 | routing-content (induction head) learning dynamics |
| Reddy 2023, arXiv:2312.03002 | abrupt learning and data dependence in an in-context classification task |
| Liu & Ihler 2014, arXiv:1410.2653 | one-shot combination of local MLEs: information loss tied to deviation from full exponential families (asymptotic, second order) |
| Jordan, Lee, Yang 2016, arXiv:1605.07689 | communication-efficient surrogate likelihood; quasi-posterior |
| Huang & Huo 2015, arXiv:1511.01443 | distributed one-step estimator: one extra round reaches centralised asymptotics |
| Duchi, Jordan, Wainwright, Zhang 2014, arXiv:1405.0782 | lower bounds for distributed estimation, independent vs interactive (broadcast) protocols |
| Abbe, Kamath, Malach, Sandon, Srebro 2021, arXiv:2108.04190 | SGD/GD simulate SQ; with fine precision they go beyond SQ |
| Shalev-Shwartz, Shamir, Shammah 2017, arXiv:1703.07950 | gradient-based failures, including parity-type problems |
Classical, cited from memory: Thatcher & Wright 1968 (tree automata); Lauritzen & Spiegelhalter 1988 (junction trees);
Dechter 1999 (bucket elimination, induced width); Boutilier, Friedman, Goldszmidt, Koller 1996 (context-specific
independence); Chor & Goldreich 1988 (inner product mod 2, distributional lower bound under the uniform law);
Grasedyck 2010 (hierarchical SVD); Daniels 1954 (saddlepoint).

