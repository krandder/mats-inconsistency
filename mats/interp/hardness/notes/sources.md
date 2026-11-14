# Sources checked for the hardness note (29 Oct 2026)

Checked = I read the cited statement in the paper text (local copy under ../papers/ unless noted) or the arXiv API record.

| Source | What I use | Check |
|---|---|---|
| Impagliazzo, Kabanets, Wigderson, "In search of an easy witness", JCSS 65(4) 2002 (CCC 2001); papers/ikw/ikw02.pdf | Thm 32: NEXP not in P/poly <=> CAPP "nontrivially approximable" (nondet. 2^{n^eps} time, n^eps advice, i.o., additive 1/6), also by black-box non-adaptive algorithms. Cor 34: promise-BPP in promise-SUBEXP => NEXP not in P/poly. Remark 33: deterministic poly-time CAPP within 1/6 <=> promise-BPP = promise-P | read Sec 5.1 text |
| Chen, Tell, "Simple and fast derandomization from very hard functions", STOC 2021, ECCC TR20-148; papers/chen-tell/ct.pdf | Thm 6.4: under #NSETH, for all k >= 1, eps > 0, BPTIME[n^k] not in DTIME[n^{k+1-eps}] (hard problem: checking a univariate polynomial identity H = Q by one random evaluation, from (k+1)-OV). Thm 1.3: under OWF + strong hardness, prBPTIME[T] in prDTIME[n^{1+eps} T]; average-case derandomization with n^eps overhead over samplable distributions | read Sec 1.2, Sec 6 |
| Neyman, Lecomte, Wu, Winer, Hilton, Robinson, "Competing with sampling", ARC blog 18 Nov 2025 | MSP statement: G_M(theta, pi, c, eps); pi depends on theta only, |pi| <= O(|theta|); time O((1/eps^2) Time(M)); E_c error <= eps^2 E_c Var_x; "mechanistic" informal ("avoids any random or pseudorandom sampling"); train-and-explain variant (Time(E) <= O(Time(T))); random-MLP instance has theta empty, c = W, fixed depth k, n -> infinity | WebFetch of the post |
| Wu, Lecomte, Winer, Robinson, Hilton, Christiano, arXiv 2605.05179v2; papers/wu/wu.pdf | conjecture MSE <= c_K (L/n)^K (Sec 5, App D); runtimes c'_K L n^{K+1}, c''_K L^2 n^K; Sec 7.3: open whether one can match sampling at large depth or for narrow nets | read Sec 5, 7.3, App D |
| Hanin, Nica, arXiv 1812.05994 (CMP 2020) | log of the norm of a product of random matrices applied to a vector is asymptotically Gaussian when depth and width grow together | abstract |
| Hanin, arXiv 1801.03744 (NeurIPS 2018) | variance of squared Jacobian entries exponential in beta = sum 1/n_l | abstract |
| Goldwasser, Kim, Vaikuntanathan, Zamir, arXiv 2204.06974 (FOCS 2022) | undetectable backdoors; white-box undetectable for RFF / random ReLU nets | abstract |
| Naor, Reingold, J. ACM 51(2) 2004 | PRFs in TC^0 (DDH or factoring) | abstract via search |
| Dunbar, Aaronson, arXiv 2510.06527 | wide random nets have nearly independent outputs iff the activation has zero Gaussian mean; not plain ReLU | abstract |
| He, O'Donnell, arXiv 2404.14648 | random reversible circuits of depth n*O~(k^2) give almost k-wise independent permutations | abstract |
| Kunisky, Wein, Bandeira, arXiv 1907.11636 | the low-degree method is a heuristic for hypothesis testing problems | abstract |
| Christiano, Neyman, Xu, arXiv 2211.06738 | heuristic estimators, presumption of independence | abstract |
| Rolnick, Kording arXiv 1910.00744; Carlini, Jagielski, Mironov arXiv 2003.04884 | black-box extraction of ReLU networks | titles/abstracts via arXiv API |
| Bartlett, Harvey, Liaw, Mehrabian arXiv 1703.02930 | VC dimension of piecewise-linear nets | title |
| Montanaro arXiv 1504.06987 | quantum quadratic speedup of Monte Carlo | title |
| Polyanskiy, Wu arXiv 1508.06025 | SDPI | title |
| Poole et al. arXiv 1606.05340; Schoenholz et al. arXiv 1611.01232 | order/chaos phases, correlation maps | titles |
| Daniely, Vardi arXiv 2006.03177; Chen, Gollakota, Klivans, Meka arXiv 2202.05258 | hardness of learning random-weight networks | titles |

Not verified, so not cited: Williams 2010/2013 ("Improving exhaustive search ...") exact CAPP statement (PDF not found); Novak's IBC book (I prove the black-box bound directly instead).
