# Proposition market: literature for (a) a proximity space and (b) proposition keys, 2026-10-22

Design (Kelvin, addendum 2026-10-22): fixed base bits y, some settled; an evolving population of propositions z with rule-based links to y; wealth grows with accuracy gains and decays when unused; broke propositions are replaced.
Checks: every arXiv id was checked today on export.arxiv.org (title, date) and every DOI on Crossref. "Cites" = Semantic Scholar count today; CR = Crossref, OA = OpenAlex where S2 had none.
Toy = bits y1..y6, random propositions of 2-3 literals joined by AND/OR/XOR. Effort is for that toy.

## The fact that frames everything
With 6 bits there are 64 worlds, so any z = f(y) is fully described by its 64-entry truth table. Market-making theory already treats a security this way, as a payoff vector over outcomes (Abernethy et al. 2013, below), and its price under a belief p is the inner product (truth table · p).
So in the toy both (a) and (b) have exact, rule-based answers. The literature matters for two things: (i) what to use when worlds can no longer be enumerated, and (ii) which mutation, niche and deletion rules keep a rule population healthy.

## 1. Semantic distance and semantic search operators (a)
- **Moraglio, Krawiec & Johnson 2012, Geometric Semantic GP**, PPSN, DOI 10.1007/978-3-642-32937-1_3; 413 cites. A program's "semantics" is its output vector on the inputs. Mutation and crossover are built so the offspring move a bounded distance in that space. Boolean mutation is T∨M or T∧¬M, where M is a random minterm, so it flips exactly one truth-table entry. Crossover (T1∧R)∨(¬R∧T2) takes each entry from one of the two parents.
  For (a): Hamming distance on truth tables is a proximity space, and these operators are exactly local in it. The catch is that formulas grow every generation. Effort: about 1 h if propositions are stored as truth tables and re-minimised (trivial at 6 variables).
- **Beadle & Johnson 2008**, semantically driven crossover, IEEE CEC, DOI 10.1109/CEC.2008.4630784; 132 cites, on Boolean problems. **Uy et al. 2011**, GPEM, DOI 10.1007/s10710-010-9121-2; 355 cites. Beadle rejects offspring whose meaning equals a parent's (checked with BDDs). Uy keeps a subtree swap only when the semantic distance falls inside a band.
  For (a): the rule "mutate the syntax, then accept the child only if its truth table changed, but not by too much". Effort: hours.
- **Pawlak, Wieloch & Krawiec 2015**, semantic backpropagation, IEEE TEVC, DOI 10.1109/TEVC.2014.2321259; 108 cites. Works out the output a subtree should have, then picks a replacement subtree from a library.
  For (a): directed mutation. If the parent is wrong on these settled worlds, change the literal that makes it right there. Effort: about 1 day.
- **Vanneschi, Castelli & Silva 2014**, survey of semantic GP, GPEM, DOI 10.1007/s10710-013-9210-0; 160 (CR). A map of the area.
- **Evans et al. 2018, "Can Neural Networks Understand Logical Entailment?"**, arXiv:1802.08535; 139 cites. PossibleWorldNet represents a formula by its values in many randomly sampled worlds, and it was the best model on their entailment benchmark.
  For (a) and (b): semantic vectors over sampled worlds, the scalable version of truth tables. Effort: minutes for the vectors.

## 2. Walsh–Fourier (parity) spectrum (a, b)
- **O'Donnell 2014, Analysis of Boolean Functions**, CUP, DOI 10.1017/CBO9781139814782 (452 CR); free copy arXiv:2105.10386. Any f on {0,1}^n can be written f = Σ_S f̂(S) χ_S, where χ_S is the ±1 parity of the bits in S. Three consequences for us:
  (1) Parseval: for 0/1-valued f and g, Pr[f≠g] = Σ_S (f̂(S)−ĝ(S))². Spectrum distance therefore equals truth-table Hamming distance, so the spectrum gives no new proximity, only a rotated basis.
  (2) Price: P_p(z) = Σ_S f̂(S)·E_p[χ_S]. The spectrum is a fixed key, the belief's parity moments are the query, and the price is their exact inner product.
  (3) Shape: XOR(y3,y5) = ½(1−χ_{35}) is two coefficients, one of them non-constant; AND(y3,y5) = ¼(1−χ_3−χ_5+χ_{35}) is four. Any proposition of at most 3 literals has degree ≤3, so its key lives in 1+6+15+20 = 42 dimensions.
- **Linial, Mansour & Nisan 1993**, JACM, DOI 10.1145/174130.174138; 836 cites. **Kushilevitz & Mansour 1993**, SIAM J Comput, DOI 10.1137/0222080; 328 (OA). Functions whose spectrum sits on low degree or on few coefficients can be learned by estimating those coefficients from samples.
  For (b): a predictor that estimates the moments E[χ_S] for |S|≤3 can price every 2-3 literal proposition, including ones it has never seen.
- **Abbe, Boix-Adserà & Misiakiewicz 2022**, arXiv:2202.08658; 161 cites. **Barak et al. 2022**, arXiv:2207.08799; 223 cites. SGD learns sparse functions fast when each higher-degree coefficient extends a lower one ("staircase"). AND has this property; XOR has no degree-1 part. Parities of k bits take n^{O(k)} SGD steps.
  Prediction for the toy: learned predictors will master AND/OR propositions before XOR ones, so report results by Fourier degree. Effort: none.

## 3. Logical-query embeddings from knowledge graphs (b, partly a)
- **GQE**, Hamilton et al. 2018, arXiv:1806.01445 (345). **Query2Box**, Ren, Hu & Leskovec 2020, arXiv:2002.05969 (390). **BetaE**, Ren & Leskovec 2020, arXiv:2010.11465 (274). **ConE**, Zhang et al. 2021, arXiv:2110.13715 (164). **CQD**, Arakelyan et al. 2021, arXiv:2011.03459 (179).
  What they do: a logical query becomes a point (GQE), a box (Q2B), a Beta distribution (BetaE, which adds NOT) or a cone (ConE, closed under complement). AND is a learned intersection and OR goes through DNF. CQD trains only on single atoms and composes them with t-norms at test time, so it handles query shapes it never saw.
  For (a): overlap between regions approximates overlap in meaning. For (b): keys are composed from fixed atom embeddings plus operators, so "y3 AND y6" is assembled from parts already seen in "y3 AND y5". In the toy, the region they approximate is just the set of satisfying worlds, which we can store exactly, so they add nothing at n=6. Effort: 1-2 days with the snap-stanford KGReasoning code; worth it only at large n.
- **Vilnis et al. 2018**, box lattice measures, arXiv:1805.06627; 162 cites. Boxes whose volumes are probabilities, so P(A∧B) is the volume of the intersection. This is a compact market state with consistent conjunction prices, not a proposition key.

## 4. Syntax and execution encoders (b)
- **Tree-LSTM**, Tai et al. 2015, arXiv:1503.00075 (3301). **FormulaNet**, Wang et al. 2017, arXiv:1709.09994 (153): a graph embedding of formulas that is invariant to variable renaming. **NeuroSAT**, Selsam et al. 2018, arXiv:1802.03685 (523): message passing on the literal–clause graph, trained only on the sat/unsat bit. **code2vec**, Alon et al. 2018, arXiv:1803.09473 (1401): a bag of syntax-tree paths.
  For (b): keys from syntax that scale to any number of variables. The catch: syntactic closeness is not closeness in meaning (x∧y and x⊕y differ by one token).
- **EqNet**, Allamanis et al. 2017, arXiv:1611.01423 (111): trains expression embeddings so that equivalent Boolean expressions land together; standard tree encoders did poorly at this. **Wang, Singh & Su 2017**, arXiv:1711.07163 (145): embeddings of execution traces beat syntax embeddings. Lesson: embed behaviour. Effort for a Tree-LSTM key: about 1 day; worth it only as a baseline in the toy.

## 5. Differentiable logic and tractable circuits
- **NLM**, Dong et al. 2019, arXiv:1904.11694 (311). **∂ILP**, Evans & Grefenstette 2018, arXiv:1711.04574 (580). **LNN**, Riegel et al. 2020, arXiv:2006.13155 (225). **difflogic**, Petersen et al. 2022, arXiv:2210.08277 (119).
  What they do: learn rules by gradient. ∂ILP puts soft weights on candidate clauses, NLM runs MLPs over predicate tensors, LNN uses weighted real-valued connectives with truth bounds, and difflogic makes each gate a softmax over the 16 two-input Boolean ops.
  For (a): difflogic's relaxed choice of operator can propose mutations. LNN's downward inference pushes bounds from y to its parts, which is the y = z1⊕z2 case. All four learn the connections, which Kelvin wants to keep out of the market, so use them at most as proposal generators. Effort: days.
- **Semantic loss**, Xu et al. 2018, arXiv:1711.11157 (588): loss = −log P(constraint holds) under the model, computed through a compiled circuit. This is the mechanism behind Kelvin's "gradient flows to unsettled propositions because some mass is contradicted". Effort: hours at n=6, by enumeration.
- **SDD**, Darwiche 2011, IJCAI pp. 819-826 (PDF checked on ijcai.org; 388 cites). **PSDD**, Kisa et al. 2014, KR (104 OA). **Knowledge compilation map**, Darwiche & Marquis 2002, JAIR, DOI 10.1613/jair.989 (1067).
  What they do: an SDD is a canonical compiled form of a Boolean function. A PSDD is a tractable distribution over that form: the exact marginal of any compiled proposition, in time linear in circuit size.
  For us: canonical forms let us remove duplicate propositions, and PSDDs give exact prices past enumeration size. Effort: hours with PySDD.

## 6. Rule populations with strength or wealth (a)
- **XCS**, Wilson 1995, Evol Comput, DOI 10.1162/evco.1995.3.2.149 (1367). Butz & Wilson 2002, the reference algorithm, DOI 10.1007/s005000100111 (599). ZCS, Wilson 1994, DOI 10.1162/evco.1994.2.1.1 (570). Review: Urbanowicz & Moore 2009, DOI 10.1155/2009/736398 (320).
  How XCS works: a rule is a condition over {0,1,#} plus an action, carrying a payoff prediction, a prediction error and a fitness. Fitness is accuracy (low error) relative to the other rules in the niche, not the size of the payoff. Wilson argues that payoff-based "strength" (Holland, ZCS) breeds over-general rules and loses niches.
  The GA runs only inside the action set (the rules that just matched and advocated the same action). Deletion acts on the whole population, with probability proportional to the size of the rule's niche, which keeps the niches roughly equal. Covering: when no rule matches, create one that matches the current input, with random #s. Subsumption: an accurate, more general rule absorbs the more specific ones.
  Why the ternary code is a proximity space: the condition 1#0# matches the subcube "bit1=1 and bit3=0". One mutation changes one symbol:
  - 0↔# halves or doubles the matched set; parent and child are nested, so this specialises or generalises.
  - 0↔1 moves to the adjacent, disjoint subcube.
  Hamming distance between two conditions therefore bounds how much their match sets differ. Generalisation enters through #: rules that are accurate and carry more #s match more inputs, so they sit in more action sets, reproduce more often and take over (Wilson's "generalisation hypothesis"). The limit: conditions express only conjunctions, so XOR of two bits takes two rules.
  For (a): local reproduction in the niche of the y that just settled, global deletion weighted by crowding, fitness by accuracy rather than raw wealth, and covering for outcomes nothing explains. Effort: half a day.
- **Hayek**, Baum & Durdanovic 2000, Neural Comput, DOI 10.1162/089976600300014700 (39). Baum 1999, Mach Learn, DOI 10.1023/A:1007593124513 (49). Agents bid in auctions for control of the world, pay the previous owner and collect reward. Broke agents are removed, and wealthy agents fund mutated children. Baum argues that conservation of money and property rights are what made credit assignment work where Holland's bucket brigade failed.
  For (a): "divide" means a wealthy proposition endows its offspring out of its own wealth, so money stays conserved. Effort: half a day.
- **Logical Induction**, Garrabrant et al. 2016, arXiv:1609.03543 (43). A market over logical sentences: traders' strategies are continuous functions of prices, and the prices are made consistent across logically linked sentences. It is the closest prior design to unsettled propositions getting prices through logical links. It is computable but impractical, so it is a conceptual reference only.

## 7. One predictor for a changing set of securities (b), and the market side
- **HyperNetworks**, Ha et al. 2016, arXiv:1609.09106 (2093). **FiLM**, Perez et al. 2017, arXiv:1709.07871 (5135). **Conditional Neural Processes**, Garnelo et al. 2018, arXiv:1807.01613 (917). **Deep Sets**, Zaheer et al. 2017, arXiv:1703.06114 (3342). **Oh et al. 2017**, arXiv:1706.05064 (291).
  What they do: condition one network on a task code, either by generating its weights (hypernetwork) or by a per-feature scale and shift (FiLM). CNPs condition on a context set; Deep Sets pools over a changing set in any order. Oh et al. add an analogy loss so that unseen combinations of instruction parts generalise.
  For (b): one predictor, trade = g(context, key(z), price), serves every security. FiLM on the key takes about 1 h.
- **LMSR**, Hanson 2003, DOI 10.1023/A:1022058209073 (545). **Chen et al. 2008**, arXiv:0802.1362 (74): pricing Boolean combinations under LMSR is #P-hard in general. **Abernethy, Chen & Wortman Vaughan 2013**, DOI 10.1145/2465769.2465777 (110): a security is a payoff vector, and a cost-function market maker is an online learner.

## Recommended toy design (proposed, not built)
(a) Proximity and reproduction
1. Store each z as its formula plus its truth table t(z) ∈ {0,1}^64. Distance d(z,z') = Pr_w[z(w)≠z'(w)], with w drawn from the current market distribution (uniform at the start). By Parseval this equals the Fourier distance.
2. Mutate the syntax and accept by semantics (XCS and semantic-GP operators): swap a literal (y5→y6), flip a sign, change one operator (AND/OR/XOR), or add or drop a literal. Reject the child if its truth table is unchanged, duplicates an existing proposition, or has d > 1/4.
3. Divide by a Shannon split, z → (z∧y_k, z∧¬y_k), on the y_k that best separates z's recent errors. The two children partition z, so the parent's wealth splits by their prices and no money is created (Hayek).
4. Selection follows XCS. Reproduce only among propositions linked to the y that just settled. Delete across the whole population, with probability rising with crowding (the number of propositions within d < 1/8) and falling with wealth. Decay is a per-step tax, as in Holland's classifier systems.
(b) Keys
5. key(z) = its 42 Fourier coefficients of degree ≤3 (exact and rule-based). A predictor outputs a query: its moment estimates m_S = E[χ_S]. Its price for any z is key·m, so moving from y3∧y5 to y3∧y6 needs only m_6 and m_36.
6. For trading behaviour beyond pricing, put a FiLM network on key(z). Compare truth-table, Fourier and Tree-LSTM keys on held-out propositions.
7. For y = z1⊕z2 with latent z: extend each world to (y, z) bits and compute keys on that joint space; only the y entries ever settle.

## Risks
1. Over-general propositions: one that is right on average (e.g. y3∨y5 when y3 is usually 1) gains wealth while adding little. Pay wealth for improvement over the market price, not for hit rate (the XCS lesson).
2. Scale: truth tables and full spectra have size 2^n. Above about 20 bits, switch to sampled worlds and low-degree coefficients; dense, high-degree propositions then get poor keys.
3. The exact key makes the toy too easy: a predictor that only outputs moments never learns anything proposition-specific, so the toy does not test learned generalisation. Keep a held-out-proposition test with a learned key, and expect XOR to lag AND (the staircase results).

## Open
- I found no paper that combines wealth-based rule populations with semantic (truth-table) mutation: XCS mutates syntax, and semantic GP has no wealth.
- How to split wealth on mutation without creating money, and how fast decay should be relative to the settlement rate.
- Latent z (y = f(z)): proximity and keys over joint spaces where z never settles, and identifiability (z1⊕z2 is unchanged when both are negated).
- Whether rollout-based predictors can output parity moments, or only prices for the listed securities.
