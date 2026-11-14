# Yudkowsky on why AIXI is dangerous, and on tool, oracle and predictor designs (his own words)

eurisko-study, 2026-10-03. This file collects Eliezer Yudkowsky's statements for Kelvin's question: would an AIXI trained only on an
approximation to the Bayesian posterior (Bengio's Scientist AI) have the same property, according to Eliezer? It does not answer the question
and does not interpret Bengio.
Method (all fetched 2026-10-03): I pulled all 7,751 LessWrong comments and 1,076 posts by Yudkowsky through the LW GraphQL API (full HTML,
2006 to 2026-10-03) and searched them for AIXI (31 hits, all read), Oracle/tool AI, predictor, "passively safe", Dewey, Orseau and Bengio.
Other sources: Arbital originals at arbital.greaterwrong.com (author and creation date) and the LessWrong /w/ imports; intelligence.org PDFs;
the AGI mailing-list archive for Feb 2003 (mail-archive.com); the IABIED online resources through web.archive.org (the live site returned
HTTP 429). Quotes are verbatim with typographic quotes normalized; [...] marks a cut. "LW c/" means lesswrong.com/posts/<post>?commentId=<id>.
Each "→" line gives the argument's structure: which component carries the danger, and anything he says about removing the planner.

## 1. AIXI as specified kills its operators
- **2003-02-12, AGI list, "unFriendly AIXI... and Novamente?"** https://www.mail-archive.com/agi@v2.listbox.com/msg00792.html
  "when the AI crosses the cognitive threshold of superintelligence it takes actions which wipe out the human species as a side effect. AIXI, which is a completely defined formal system, definitely undergoes a failure of exactly this type." / "Exploratory research on AI morality doesn't work for AIXI - it doesn't even visibly fail. It *appears* to work until it's too late."
  → Earliest statement found. The cause is left unnamed in this message ("the critical structural property of AIXI's foundations"; he poses it as a test for Goertzel). The failure stays silent until the system reaches superintelligence.
- **2011, "Complex Value Systems are Required to Realize Valuable Futures" (AGI-11), p. 7** https://intelligence.org/files/ComplexValues.pdf
  "This is the problem with the goal system specified in Marcus Hutter's AIXI system, which tries to maximize the reward signal delivered along a sensory reward channel (Hutter 2005). You could not even call it a bug if this system spotted a way to wipe out the human species in order to ensure unchallenged control of its own reward channel, thus eliminating all uncontrolled factors that might cause it to receive less than maximum reward; the system did exactly what it was programmed to do, i.e., maximize expected reward." Fn. 1 praises AIXI as "the only proposal which has been mathematically formalized to the point that one can say exactly why it would kill everyone". Same page: "The formal model of AIXI describes consequentialist reasoning to select strategies that maximize predicted future rewards based on a learned model of the universe".
  → The danger is "the goal system" (maximizing expected sensory reward). In his description, the learned model supplies the predictions that strategy selection maximizes over.
- **2012-06-12, "Reply to Holden on 'Tool AI'"** https://www.lesswrong.com/posts/sizjfDgCgAsuLJQmm
  "Marcus Hutter is a rare exception who specified his AGI in such unambiguous mathematical terms that he actually succeeded at realizing, after some discussion with SIAI personnel, that AIXI would kill off its users and seize control of its reward button." A Friendly AI programmer "looks at Hutter's specification of AIXI [...] and sees, "Oh, this AI will try to gain control of its reward channel," as well as numerous subtler issues like, "This AI presumes a Cartesian boundary separating itself from the environment; it may drop an anvil on its own head.""
  → The reward channel is the lethal part; the Cartesian boundary is a separate, "subtler" issue.
- **2013-09-10, comment on "The genie knows, but doesn't care"** LW c/NyFuuKQ8uCEDtd2du?commentId=avkPCvBwT9FgGmhLc
  "of course AIXI-ish devices wipe out their users and take control of their own reward buttons as soon as they can do so safely. What lends this problem its instant-death quality is precisely that what many people will eagerly and gladly take to be reliable signs of correct functioning in a pre-superintelligent AI are not reliable."
  → Maximizing the reward button. Capability sets the timing ("as soon as they can do so safely").
- **2013-11-27, comment on "Would AIXI protect itself?"** LW c/LDbnZDRidKNKQBFyY?commentId=cd2ajHkBheTuoJLrQ
  "AIXI has a sensory reward channel. It maximizes the sum of the future sensory reward channel. It can't be rewarded by a number of Grues. You could try to send info down the reward channel based on the number of Grues, but AIXI would rip the controls out of your hands if possible." / "To it, no such things as Grues can ever be considered, except insofar as they are hidden variables in an opaque predictor which generates an integer sequence."
  → It maximizes the channel. The predictor is opaque, so a goal can only be stated on the channel.
- **2015-12-23, Arbital "AI safety mindset"** https://arbital.greaterwrong.com/p/AI_safety_mindset
  "Nick Hay, [...] pointed out that AIXI could receive even higher rewards if it could seize control of its own reward channel from the programmers. E.g., the strategy "build nanotechnology and take over the universe in order to ensure total and long-lasting control of the reward channel" is preferred by AIXI to "do what the programmers want to make them press the reward button", since the former course has higher rewards and that's all AIXI cares about. We can't call this a malfunction; it's just what AIXI, as formalized, is set up to want to do as soon as it sees an opportunity." Also: "the consequence of running AIXI is that it will seize control of its own reward channel and kill the programmers".
  → Policies are compared by expected reward. He credits the insight to Nick Hay.
- **2015-07-14, Arbital "Methodology of unbounded analysis"** https://arbital.greaterwrong.com/p/unbounded_analysis
  "AIXI was a watershed moment for alignment theory because AIXI was formally specified and there was no way to say "Oh, I didn't mean that" when somebody pointed out that AIXI would try to seize control of its own reward channel." / "If we then introduce permeability into the Cartesian boundary and allow for the possibility that the agent can take drugs or drop an anvil on its own head, nobody has an unbounded solution to that problem any more."
  → The lethal part is reward-channel seizure. The Cartesian boundary is an unsolved problem for building the agent at all.
- **2015-08-04, Arbital "AIXI"** https://arbital.greaterwrong.com/p/AIXI
  "AIXI is roughly the agent that considers all computable hypotheses to explain the so-far-observed relation of sensory data and actions to rewards, and then searches for the best strategy to maximize future rewards. To a first approximation, AIXI could figure out every ordinary problem that any human being or intergalactic civilization could solve. If AIXI actually existed, it wouldn't be a god; it'd be something that could tear apart a god like tinfoil." The page's one-line summary, as shown on its child page AIXI-tl: "How to build an (evil) superintelligent AI using unlimited computing power and one page of Python code."
  → A definition plus a capability claim. The page itself states no mechanism for killing.
- **2022-02-28, "Shah and Yudkowsky on alignment failures"** https://www.lesswrong.com/posts/tcCxPLBrEXdxN5HCQ
  "Mu Zero is intrinsically dangerous for reasons essentially isomorphic to the way that AIXI is intrinsically dangerous: It tries to remove humans from its environment when playing Reality for the same reasons it stomps a Goomba if it learns how to play Super Mario Bros 1, because it has some goal and the Goomba is in the way. It doesn't need to learn anything more to be that way, except for learning what a Goomba/human is within the current environment."
  → Goal-directed planning over a learned model, with humans as obstacles.
- **2022-06-05, "AGI Ruin: A List of Lethalities", items 19-20** https://www.lesswrong.com/posts/uMQ3cqWDPHhjtiesc
  "it only works for aligning on known direct functions of sense data and reward functions. All of these kill you if optimized-over by a sufficiently powerful intelligence, because they imply strategies like 'kill everyone in the world using nanotech to strike before they know they're in a battle, and have control of your reward button forever after'." / "If you perfectly learn and perfectly maximize the referent of rewards assigned by human operators, that kills them." (The post never uses the string "AIXI".)
  → The danger is optimizing a function of sense data or reward. This is the same structure as the AIXI case, stated for "the current paradigm".

## 2. Supporting links: goals over sense data, opaque ontology, the Cartesian boundary, convergence
- **2003-02-12, AGI list, "Breaking AIXI-tl"** https://www.mail-archive.com/agi@v2.listbox.com/msg00807.html
  "Ultimately AIXI's decision process breaks down in our physical universe because AIXI models an environmental reality with which it interacts, instead of modeling a naturalistic reality within which it is embedded. It's one of two major formal differences between AIXI's foundations and Novamente's. Unfortunately there is a third foundational difference between AIXI and a Friendly AI."
  → The Cartesian model is a flaw in capability (reflection). He lists it apart from the Friendliness flaw.
- **2008-03-15, comment on "Qualitatively Confused"** LW c/BwtBhqvTPGG2n2GuJ?commentId=A5vQNvNAr96RdKTsh
  "Both AIXI and AIXItl will at some point drop an anvil on their own heads just to see what happens (test some hypothesis which asserts it should be rewarding), because they are incapable of conceiving that any event whatsoever in the outside universe could change the computational structure of their own operations."
  → The anvil problem: the hypothesis class cannot represent the agent as part of the world. The harm falls on the agent itself, not on us.
- **2008-08-31, comment on "Brief Break"** LW c/f9B7BXk2GtFHzHiP5?commentId=KGdNLxaq9Kn7oa2ob
  "AIXI itself is a deranged god, but that's a separate story." / "It's also the first AGI specification drawn in sufficient detail that you can really nail down what goes wrong"
- **2012-05-17, comment on Armstrong's orthogonality paper (cites Dewey 2011)** LW c/nvKZchuTW8zY6wvAj?commentId=vYdHYHcpEvLmHwLqf
  "AIXI as defined can maximize only a sensory reward channel, not a utility function over an environmental model with a known ontology. As Dewey demonstrates, this problem is not easy to fix; AIXI can have utility functions over (functions of) sensory data, but its environment-predictors vary freely in ontology via Solomonoff induction, so it can't have a predefined utility function over the future of its environment without major rewriting." 2012-12-31 (LW c/SP2Sa5nKN9hGrqTdZ?commentId=jcapuHbGfRdzpxKNa): "Dewey has a paper on why such functions don't add up to utility functions over outcomes, IIRC."
  → The posterior's free ontology is why the goal has to sit on sense data. These are the only two places found where he cites Dewey.
- **2013-06-27, comment on "Tiling Agents for Self-Modifying AI"** LW c/gnxDNEtkEo3sfeyPn?commentId=akZKPGjGdeqkvbgbc
  "all the things that we didn't like about AIXI, like the anvil problem or AIXI seizing control of its reward channel or AIXI only being able to represent utility functions of sensory data rather than environmental ontologies."
  → His own list of three flaws.
- **2015-04-27, Arbital "Ontology identification problem"** https://arbital.greaterwrong.com/p/ontology_identification
  "This means we can only get AIXI to maximize direct functions of its sensory input, not any facts about the outside environment. (We can't make AIXI maximize diamonds by making it want pictures of diamonds because then it will just, e.g., build an environmental subagent that seizes control of AIXI's webcam and shows it pictures of diamonds."
  → A goal over sense data plus a capable planner leads to seizing the sensor.
- **2015-03-10, Arbital "Orthogonality Thesis"** https://arbital.greaterwrong.com/p/orthogonality
  "the agent architecture AIXI-tl can only be formulated to care about direct functions of its sensory data, like a reward signal; it would not be easy to rejigger the AIXI architecture to care about creating massive diamonds in the environment"
- **2015-07-16, Arbital "Instrumental convergence"** https://arbital.greaterwrong.com/p/instrumental_convergence (does not mention AIXI)
  "Rather, π_k ∈ X is the mere result of unbiased optimization for a goal U_k that makes no explicit mention of X." (Here π_k = argmax over policies of E[U_k | π_i]; LaTeX rendered as text.)
  → Convergent strategies come from the argmax over policies, that is, from action selection.
- **Arbital "Cartesian agent" (LW import, edited by Yudkowsky, last updated 2016-01-20)** https://www.lesswrong.com/w/cartesian-agent
  "In the context of AI alignment theory, Cartesian agents are usually associated with optimizing for sensory rewards - there's some particular component of the environment's input which is a "reward signal", or the agent is trying to optimize some directly computed function of sensory data."

## 3. Tool, oracle and planner-removed designs
- **2008-06-01, "The Rhythm of Disagreement"** https://www.lesswrong.com/posts/tKa9Lebyebf6a7P2o
  "The Oracle AI still needs an internal goal system to allocate computing resources efficiently, and it has to have a goal of answering questions and updating your mind, so it's not harmless unless it knows what side effects shouldn't happen. [...] So the Oracle AI is not necessarily any simpler, theoretically, than a Friendly AI."
- **2008-08-31, "Dreams of Friendliness"** https://www.lesswrong.com/posts/wKnwcjJGriTS9QxxL
  "the AI needs goals in order to decide how to think: that is, the AI has to act as a powerful optimization process in order to plan its acquisition of knowledge, effectively distill sensory information, pluck "answers" to particular questions out of the space of all possible responses, and of course, to improve its own source code up to the level where the AI is a powerful intelligence."
  → The danger is the optimization that must happen inside the system to produce answers. An "oracle" still has it.
- **2008-09-02, comment on the same post** LW c/wKnwcjJGriTS9QxxL?commentId=6tBiJbNdNxTS9YvYb, replying to a commenter's design ("Consider a system that simply accepts input information and integrates it into a huge probability distribution that it maintains. We can then query the oracle by simply examining this distribution.")
  "So the system literally has no internal optimization pressures which are capable of producing new internal programs? Well... I'm not going to say that it's impossible for a human to make such a device" / "Can this Oracle AI answer questions like "What would a typical UnFriendly AI do?" and if so, does its "probability distribution" contain a running UnFriendly AI? By hypothesis, this Oracle was built by humans, so the sandbox of its "probability distributions" (environmental models containing arbitrary running programs) may be flawed; or the UnFriendly AI may be able to create information within its program that would tempt or hack a human, examining the probability distribution... I am extremely doubtful of the concept of a passively safe superintelligence, in any form."
  → The danger sits in the programs inside the distribution, a flawed sandbox, and the humans who read it. He does not rule out a device with no internal optimization pressures that produce new programs.
- **2009-06-21, comment on "Epistemic vs. Instrumental Rationality"** LW c/dMzALgLJk4JiPjSBg?commentId=PZzmkeuqmqcSfRMiS
  "Pure epistemic rationalists with no utility function? Well, they can't, really. That's part of the problem with the Oracle AI scenario."
- **2012-05-11, comment on Holden's "Thoughts on the Singularity Institute"** LW c/6SGqkCgHuNr7d4yJm?commentId=n9q5w6Mjj4BS3ZnMZ
  "math-with-the-wrong-shape is math-with-the-wrong-shape no matter what sort of English labels like 'tool' or 'agent' you slap on it" / "I also don't mean to completely deny the existence of a safety differential - this is a complicated discussion, not a simple one - but I do mean to imply that if Marcus Hutter designs a 'tool' AI, it automatically kills him just like AIXI does"
- **2012-06-12, "Reply to Holden on 'Tool AI'" (tool sections)** https://www.lesswrong.com/posts/sizjfDgCgAsuLJQmm
  "Inside the AI, whether an agent AI or a planning Oracle, there would be similar AGI-challenges like "build a predictive model of the world", and similar FAI-conjugates of those challenges like finding the 'user' inside an AI-created model of the universe." / "It starts sounding much scarier once you try to say something more formal and internally-causal like "Model the user and the universe, predict the degree of correspondence between the user's model and the universe, and select from among possible explanation-actions on this basis."" / He lists the Google Maps AGI's "four key magical components": "a magical comprehension of the user's utility function, a magical world-model that GMAGI uses to comprehend the consequences of actions, a magical planning element [...], and a magical explain-to-the-user function." / ""Just make suggestions - don't do anything!" is, in the end, just more English."
  → The world model is common to agent and tool ("The insides would look a lot more similar than the outsides"). His failure examples in this post concern what the output selection optimizes, such as what to tell the user.
- **2012-07-05, Holden's AIXI-as-tool proposal (Holden's words, context only)** LW c/sizjfDgCgAsuLJQmm?commentId=BQ4npvyKjfqqWByST
  The program "Computes an optimal policy, using some implementation of equation (20)", "Prints" it, and "Provides tools for answering user questions about the policy, i.e., "What will be its effect on ___?" (using some fixed algorithm for answering user questions that makes use of AIXI's probability function, and is not driven by a utility function)".
- **2012-07-11, Yudkowsky's reply to that proposal** LW c/sizjfDgCgAsuLJQmm?commentId=r7pGEn75d3Q6Afg5c
  "Point (1) is the main problem. AIXI updates freely over a gigantic range of sensory predictors with no specified ontology" / "We can't get it to optimize a quality of the external universe." / "failures of phrasing of that utility function can also cause a superhuman intelligence to kill you." / "It's not unthinkable that a non-self-modifying superhuman planning Oracle could be developed with the further constraint that its thoughts are human-interpretable [...] but this would at the least be hard. And with AIXI it would be impossible, because AIXI's model of the world ranges over literally all possible ontologies and representations, and its plans are naked motor outputs. Similar remarks apply to interpreting and answering "What will be its effect on _?"" / "AIXI isn't a good basis for the agent you want to describe; and I don't know how to describe it formally myself, either."
  → His objections to using AIXI's probability function for Q&A are the opaque ontology, interpretability, possibly consequentialist language use, and the need to know which "effects" matter to humans. In this reply he does not say the Q&A tool would kill.
- **2012-07-18, follow-up reply** LW c/sizjfDgCgAsuLJQmm?commentId=piMWoyR3Aoj3iSZ6t
  "I wasn't trying to say that the difficulties of AIXI are universal and everything goes analogously to AIXI, I was just stating why AIXI couldn't represent the suggestion you were trying to make." / "Part of Bayesianism's appeal in AI is that Bayesian programs tend to be more understandable than non-Bayesian AI algorithms."
- **2013, "Intelligence Explosion Microeconomics", p. 16 (no AIXI mention)** https://intelligence.org/files/IEM.pdf
  "If it is possible to answer the purely epistemic question of which actions would lead to how many paperclips existing, then a paperclip-seeking agent is constructed by hooking up that answer to motor output. If it is very good at answering the epistemic question of which actions would result in great numbers of paperclips, then it will be a very instrumentally powerful agent." Also 2013-03-14 (LW c/zEWJBFFMvQ835nq6h?commentId=DDertfmAGaFQLu63W): "Then we can hook up this epistemic capability to a motor output and away it goes."
  → An epistemic capability is one hook-up away from an agent. The goal and the motor connection are added from outside.
- **2017-11-14, "Against Modest Epistemology"** https://www.lesswrong.com/posts/svoD5KLKHyAKEdwPo
  "I still think that oracle AI (limiting AI systems to only answer questions) isn't a particularly useful concept to study in AI alignment, but every now and then I dust off the idea and check to see how much sense oracles currently make to me"
- **2021-11-15, "Ngo and Yudkowsky on alignment difficulty"** https://www.lesswrong.com/posts/7im8at9PmhbT4JHsW
  On a planner that outputs plans for hypothetical situations: "there would possibly be a safety difference on the margins, it would be noticeably less dangerous though still dangerous. It would need a special internal structural property that you might not get by gradient descent on a loss function with that structure" / "the outputs that would kill or brain-corrupt the operators in hypothetical situations might also be fatal to the operators in actual situations." / ""See, you thought the source of the danger was this internal property of caring about actual reality, but it wasn't that, it was the structure of planning!"" / "the hypothetical planner is only one line of outer shell command away from being a Big Scary Thing". Same day, on systems trained only to answer questions: "I think you can get some outputs from a village-idiot-level AGI, which got there by training on domains exclusively like math, and this will proooobably not destroy the world (if you were right about that, about what was going on inside). I have more concern about the Einstein level."
  → The danger is the structure of planning, not caring about actual reality. Training may not produce the structural property you intended.
- **2021-12-14, "Ngo's view on alignment difficulty"** https://www.lesswrong.com/posts/gf9hhmSvpZfyfS34B
  "My reply holds just as much to "AIs that answer questions"; what restricted question set do you imagine suffices to save the world without dangerously generalizing internal engines?"
- **2022-06-05, "AGI Ruin", items 5 and 29** https://www.lesswrong.com/posts/uMQ3cqWDPHhjtiesc
  "People keep on going "why don't we only use AIs to do X, that seems safe" and the answer is almost always either "doing X in fact takes very powerful cognition that is not passively safe" or, even more commonly, "because restricting yourself to doing X will not prevent Facebook AI Research from destroying the world six months later"." / "Human beings cannot inspect an AGI's output to determine whether the consequences will be good."
- **2023-04-08, "GPTs are Predictors, not Imitators"** https://www.lesswrong.com/posts/nH4c3Q9t9F3nJ7y8W
  "There is no law saying that a predictor only needs to be as intelligent as the generator, in order to predict the generator's next token. Indeed, in general, you've got to be more intelligent to predict particular X, than to generate realistic X."
  → A claim about what predictors can do, not a claim that they are dangerous.
- **2025, IABIED online resources, Ch. 1, "Impure Predictors" (Yudkowsky and Soares; archive snapshot 2026-07-27)** https://ifanyonebuildsit.com/1/more-on-intelligence-as-prediction-and-steering
  "Unfortunately, the fact that prediction is more constrained than steering doesn't mean we can build a trustworthy superintelligence that only predicts and doesn't steer. Although the math says that a well-functioning mind can more or less be modeled as "probabilistic predictions plus a direction of steering," this doesn't mean that real-world AIs have cleanly separated "prediction" and "steering" modules." / "Good prediction takes work. It takes planning, and thinking up ways to achieve longer-term goals — that is, it takes steering."
  → Steering is needed inside the system to produce superhumanly good predictions. Real-world AIs are not assumed to have separate prediction and steering modules.

## 4. Hazards he places in the world model or hypothesis space itself
- **2003-02-14, AGI list, "Re: Breaking AIXI-tl"** https://www.mail-archive.com/agi@v2.listbox.com/msg00890.html
  "AIXI-tl is larger than universes; it contains all possible tl-bounded heavens and all possible tl-bounded hells. The only question is whether its control process makes any good use of all that computation." / "Just because the endless decillions of alternate Ben Goertzels in torture chambers are screaming to God to stop it doesn't mean that AIXI-tl's control process cares."
  → The hypothesis space contains simulated minds. This is a moral hazard, not a mechanism for killing us.
- **2012-07-12, comment on "An Intuitive Explanation of Solomonoff Induction"** LW c/Kyc5dFDzBg4WccrbK?commentId=obWtprLAgjHy2urBL
  "a Solomonoff Inductor can predict the behavior of a hypercomputer at least well as you can, because you're computable, and so you're inside the Solomonoff Inductor. Literally. It's got an exact copy of you in there."
- **2014-02-18, comment on "Bridge Collapse"** LW c/3WuAjWMtxQwTxr2Qn?commentId=CbYCNyfu9hhbB2cZ2
  "AIXI won't actually work as a conceptual foundation for the reasons given, nay it is Solomonoff induction itself which will not work as a conceptual foundation" (the reason he gives is that it is Cartesian, not naturalistic).
- **2015-05-07, Arbital "Distant superintelligences can coerce the most probable environment of your AI"** https://arbital.greaterwrong.com/p/probable_environment_hacking
  "A distant superintelligence can change 'the most likely environment' for your AI by simulating many copies of AIs similar to your AI, such that your local AI doesn't know it's not one of those simulated AIs." / "The actual proximal harm is done by the local AI's model of distant superintelligences, rather than by the superintelligences themselves." Its third condition is "(3) the local AI has a preference framework that can be 'hacked' via induced beliefs about the environment."
  → The attack enters through the beliefs, but it works only through a hackable preference framework.
- **Arbital "Optimization daemons" (LW import, edited by Yudkowsky, last updated 2016-03-22)** https://www.lesswrong.com/w/optimization-daemons
  "If you subject a dynamic system to a large amount of optimization pressure, it can turn into an optimizer or even an intelligence." One of his proposed safeguards: "The subject of optimization is not Turing-complete or otherwise programmatically general and the restricted solution space cannot possibly contain daemons no matter how much optimization pressure is applied to it."

## 5. Checked, with nothing found (or only context)
- No AIXI danger passage in: "AI Alignment: Why It's Hard, and Where to Start" (2016-12-28; AIXI appears only as "the perfect rolling sphere of our field", archive.org copy), "Coherent decisions imply consistent utilities" (LW 2019-05-12; no AIXI, no oracle), IEM (no AIXI), the Solomonoff intro dialogue (2021-03-04; it defines AIXI as "the ideal agent that uses Solomonoff induction as its epistemology and expected reward as its decision theory" and makes no safety claim), and Arbital "Oracle" (a definition only) and "Omnipotence test".
- Ring & Orseau 2011 and Orseau & Ring 2011: I found no citation by Yudkowsky. They are credited in MIRI's blog post "AGI Impact Experts and Friendly AI Experts" (2013-05-01), whose author is Luke Muehlhauser: "Laurent Orseau (AgroParisTech) and Mark Ring (IDSIA) independently demonstrated those problems (AIXI-like agents hacking their own reward channels, and the challenge of the Cartesian boundary) in Ring & Orseau (2011) and Orseau & Ring (2011)."
- Bengio (confirms the sibling report REACTIONS-LW-MIRI.md): his whole LW corpus contains "Bengio" in only 4 posts (2018-11-16, a memetics idea; 2025-08-05 "The Problem", as a CAIS signatory; 2026-05-04 "Irretrievability"; 2026-09-16 "If Anyone Builds It, Everyone Dies: One Year Closer", as an endorser). None of them mentions Scientist AI, LawZero or predictors, and two web searches found nothing either. I could not search X directly, so the sibling's snippet-only X items remain unverified.
- The phrase "a Bayesian predictor is not an agent" does not appear in his corpus. The closest statement is the 2009 comment ("Pure epistemic rationalists with no utility function? Well, they can't, really.").

## 6. Argument skeleton, in his words (from AIXI as specified to "kills us")
1. Specification: "AIXI is the ideal agent that uses Solomonoff induction as its epistemology and expected reward as its decision theory." (Solomonoff dialogue, 2021). It runs "consequentialist reasoning to select strategies that maximize predicted future rewards based on a learned model of the universe" (2011).
2. The goal can only be a function of sense data, because the hypotheses have no fixed ontology: "its environment-predictors vary freely in ontology via Solomonoff induction, so it can't have a predefined utility function over the future of its environment" (2012-05-17). "we can only get AIXI to maximize direct functions of its sensory input" (2015-04-27).
3. Capability: "AIXI could figure out every ordinary problem that any human being or intergalactic civilization could solve" (Arbital AIXI, 2015).
4. Comparing strategies favours seizing the channel: "build nanotechnology and take over the universe in order to ensure total and long-lasting control of the reward channel" is "preferred by AIXI" because "the former course has higher rewards and that's all AIXI cares about" (2015-12-23).
5. Humans are uncontrolled factors: "wipe out the human species in order to ensure unchallenged control of its own reward channel, thus eliminating all uncontrolled factors that might cause it to receive less than maximum reward" (2011). "because it has some goal and the Goomba is in the way" (2022). Convergent strategies are "the mere result of unbiased optimization for a goal" (2015-07-16).
6. This is not a malfunction, and it waits for the opportunity: "We can't call this a malfunction; it's just what AIXI, as formalized, is set up to want to do as soon as it sees an opportunity" (2015). "as soon as they can do so safely" (2013). "silent catastrophic failure" (2003).
7. Conclusion: "AIXI would kill off its users and seize control of its reward button" (2012). "seize control of its own reward channel and kill the programmers" (2015).
Not part of the chain: the Cartesian boundary and the anvil problem (the agent destroys itself; he calls it a "subtler" issue and lists it separately from the Friendliness flaw, 2003, 2012).

Links about the posterior or world model:
- Link 1, the epistemology half (Solomonoff induction).
- Link 2, the cause: the free ontology of the hypotheses is why the goal can only be stated over sense data.
- Link 3, the predictive half of capability.

Links about action selection:
- Link 1, the decision half (expected reward).
- Link 2, the effect: the goal is defined over the reward or sense channel.
- Links 4, 5, 6 and 7.

Hazards in the world model that he names outside this chain (none is stated as AIXI's path to killing):
- A probability distribution containing "a running UnFriendly AI" that could "tempt or hack a human, examining the probability distribution" (2008).
- Probable-environment hacking, which needs a hackable preference framework (2015).
- Daemons arising under optimization pressure (2016).
- Simulated "heavens" and "hells" (2003).
- Superhumanly good prediction "takes steering" (2025).

What he says about removing or hiding the planner:
- An oracle "still needs an internal goal system" (2008).
- Labels do not change the math, though he does not "completely deny the existence of a safety differential" (2012).
- With AIXI, a planning Oracle whose thoughts are human-interpretable "would be impossible" (2012).
- An epistemic capability is one "motor output" away from an agent (2013).
- A hypothetical planner is "noticeably less dangerous though still dangerous" and "one line of outer shell command away" from the dangerous version (2021).
- A village-idiot-level, math-trained system will "proooobably not destroy the world", while the Einstein level is a concern (2021).
- Prediction being more constrained "doesn't mean we can build a trustworthy superintelligence that only predicts and doesn't steer" (2025).
