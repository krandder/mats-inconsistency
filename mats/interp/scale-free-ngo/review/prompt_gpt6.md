You are an adversarial reviewer. Read-only: do not edit or create any files; print your review to stdout.

TASK. Check a short research note that compares Richard Ngo's "scale-free theory of intelligent agency" (coalitional agency) with our formal definition of "scale-free information combination". Your main job is to find MISREADINGS OF NGO: places where the note attributes to Ngo (or to Lee, Kondor and Ngo 2025/2026) a claim, quote, definition, motivation or position he does not hold, quotes him inaccurately or out of context, or describes his framework in a way he would reject. Second job: check the other citations and the math.

FILES (paths relative to the working directory /home/agent/workspaces/mats):
- The note under review: work/scale-free-ngo/review/draft-v1.md
- Ngo's original post (plain text, from the LessWrong API): work/scale-free-ngo/sources/ngo-2025-03-21-scale-free.txt
- Its 55 LessWrong comments, threaded: work/scale-free-ngo/sources/ngo-2025-03-21-comments-lesswrong.txt
- Ngo's later posts: work/scale-free-ngo/sources/ngo-*.txt (well-foundedness Apr 2025; On Goal-Models Feb 2026; Distributed vs centralized agents Feb 2026; Towards a Formal Scientific Epistemology Jun 2026; Agents as Webs of Beliefs Jun 2026; Explaining Knightianism on one foot Sep 2026; Social Theory and Agent Foundations Oct 2026)
- Lee, Kondor, Ngo, arXiv:2509.06701 v2 (PDF text): work/scale-free-ngo/sources/lee-kondor-ngo-2509.06701-v2.txt
- Ostrovsky 2012 (PDF text): work/scale-free-ngo/sources/ostrovsky-2012-aggregation.txt
- arXiv abstracts (Atom XML) for Beygelzimer-Langford-Pennock 2012, Storkey 2011, Storkey-Millin-Geras 2012, Abernethy et al. 2014, Richardson LIR 2026, Pallavi Sudhir 2025/2026: work/scale-free-ngo/sources/arxiv-abstracts-*.xml
- Our definition and theorems (read "Plain summary", sections 4, 5.1-5.4, 5.7 T11d, 6.1, 6.4): outbox/scale-free-theory-2026-10-29.md
- Our experiment draft (summary section): outbox/scale-free-exp-2026-10-29.md

CHECK IN PARTICULAR
1. Every sentence in section 1 and every attribution to Ngo in sections 2-3 against the source texts (quotes must be verbatim; paraphrases must be fair). Is "Ngo's framework has no mechanism for the second part" fair? Is "His current programme is explicitly non-Bayesian" fair? Is the reading of well-foundedness (Holistan/Fractistan) and of "demarcate regions" fair? Is it fair to say precision-weighted averaging is what he rejects and why?
2. The note's claims about Lee-Kondor-Ngo (weights summing to one; geometric/log pool of beliefs; unanimity results; ICML 2026).
3. The LMSR identity in the table (a myopic risk-neutral trader who takes the price as prior, evidence conditionally independent of earlier traders' given outcome k, buys b log L(x|k) + c and is paid b log[L(x|k*)/sum_k p_k L(x|k)]). Is it correct? Are the stated conditions sufficient?
4. "No weights summing to one make the log pool of N >= 2 posteriors equal Bayes for all evidence" and the Dietrich-List "individualwise Bayesian" / multiplicative pooling attribution (Theorem 3 of "Probabilistic opinion pooling").
5. Chen et al. 2010 (Algorithmica 58:930-969) as stated; Ostrovsky's Example 1 as stated (XNOR, "motivating non-separable example"); Beygelzimer et al. 2012 as stated; Storkey 2011 ("markets can run message passing"); Laskey et al. 2018 junction-tree market maker; Wainwright and Jordan 2008 sec. 4.1 (local vs global consistency on trees/junction trees); Lee 1993 as summarised by Bikhchandani-Hirshleifer-Welch.
6. Predictive coding: is "score and curvature at the broadcast estimate = precision-weighted prediction error tau(x - mu) and precision" exact for Gaussian sources?
7. Tags: flag any claim tagged [exact] that is really an interpretation, and any number that does not match our two outbox files.
8. Anything important in Ngo's post, comments or follow-ups that the note misses and that changes the comparison (at most 3 items).

OUTPUT. Numbered findings, most severe first, at most 20. For each: severity (MISREADING / WRONG CITATION / MATH ERROR / OVERCLAIM / MINOR), the exact text from the note, the evidence (quote the source text with file name), and a concrete fix. End with a one-paragraph overall verdict. If you can use web search to check a citation, do; say which ones you checked that way.
