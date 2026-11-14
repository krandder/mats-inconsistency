**TL;DR:** Audited all 23 numbered statements across Sections 1–6 (no tools used). None are fatally wrong, 6 have concrete fixable gaps/constant bugs (most notably the noise constant in **Theorem 4.1(c)**, the Lévy distance factor of 2 in **Proposition 3.2**, an unstated orthogonality-closure hypothesis in **Corollary 6.9**, and variance normalization $V(W) = \Theta(1)$ in **Corollary 4.3** / **Theorem 2.3**), and 17 are fully correct. No decision needed; full referee report and severity-ordered fix list follow.

---

# Adversarial Referee Report (Sections 1–6)

## Priority 1: Section 4 (Circuit $\to$ ReLU Reduction and Hardness Corollaries)

### **Theorem 4.1 (Boolean circuits reduce to Gaussian means of ReLU networks)**
- **4.1(a) (Biased DAG / MLP):** **CORRECT.**
  - *Verification:* $\beta_i(x) = \text{relu}(K x_i) - \text{relu}(K x_i - 1) \in [0, 1]$ uses 2 units per input. On $[0, 1]^2$, $\text{NOT}(a) = \text{relu}(1-a) = 1-a \in [0, 1]$, $\text{AND}(a,b) = \text{relu}(a+b-1) \in [0, 1]$, and $\text{OR}(a,b) = \text{relu}(a+b) - \text{relu}(a+b-1) = \min(1, a+b) \in [0, 1]$ preserve (I1) and (I2). Substituting difference-of-two-units wires ($a = u_1 - u_2$) into downstream gates keeps all weights in $\{0, \pm 1, \pm K\}$, biases in $\{0, \pm 1\}$, fan-in $\le 4$, units $\le 2m + 2s$, edges $\le 2m + 8s$, depth $d + 1$, and layered width $\le 2\max(w, m)$. Outside $B = \bigcup_{i=1}^m \{0 < x_i < 1/K\}$, $f(x) = C(u(x))$ a.s., and $|f(x) - C(u(x))| \le 1$ on $B$, giving $|\mathbb{E} f - \Pr[C=1]| \le \Pr(B) \le m \phi(0)/K = m/(K\sqrt{2\pi})$.
- **4.1(b) (Bias-free homogeneous):** **CORRECT.**
  - *Verification:* Layer 1 computes $r = |x_0| = \text{relu}(x_0) + \text{relu}(-x_0)$ and $x_i = \text{relu}(x_i) - \text{relu}(-x_i)$; Layer 2 computes $\beta_i = \text{relu}(K x_i) - \text{relu}(K x_i - r)$ and carries $\text{relu}(r) = r$; Layers $3, \dots, d+2$ compute the gates (plus 1 optional output ReLU for depth $d+3$). Since $x_0$ is independent of $(x_1, \dots, x_m)$, $r$ is independent of $u$, $0 \le f^\circ(x) \le r$, and
    $$\left|\mathbb{E} f^\circ - \mathbb{E}[r]\Pr[C=1]\right| \le \mathbb{E}[r \mathbf{1}_{B^\circ}] \le \sum_{i=1}^m \mathbb{E}\left[r \cdot \frac{r}{K}\phi(0)\right] = \frac{m \mathbb{E}[x_0^2]}{K\sqrt{2\pi}} = \frac{m}{K\sqrt{2\pi}}.$$
    Dividing by $\mathbb{E}[r] = \sqrt{2/\pi}$ yields $\frac{m}{2K}$ exactly.
- **4.1(c) (Noisy):** **CORRECT WITH A FIXABLE GAP (factor-of-$3/2$ bug in $\tau$, changing $512\sigma^2 \to 1152\sigma^2$).**
  - *Failing step in proof:* Each restore gadget $R(v) = \text{relu}(2v - 1/2) - \text{relu}(2v - 3/2)$ is the difference of **two** noisy ReLU units:
    $$\tilde{R}(v) = \left(\text{relu}(2v - 1/2) + \xi_1\right) - \left(\text{relu}(2v - 3/2) + \xi_2\right) = \text{clamp}(2v - 1/2, 0, 1) + \xi_1 - \xi_2.$$
    When $|\xi_i| \le \tau$, a restored wire carries noise $|\xi_1 - \xi_2| \le 2\tau$, **not** $\le \tau$. Notice the proof contradicts itself within a single sentence: *"a gate fed by restored wires (each Boolean $\pm \tau$) produces a value within $4\tau = 1/4$ of its Boolean value ... so restoration returns it exactly, up to the restore units' own noise ($\le 2\tau$)"*.
  - *Exact error propagation with $\pm 2\tau$ restored wires:*
    - Feed two restored wires $a = a_{\text{bool}} + e_a$, $b = b_{\text{bool}} + e_b$ ($|e_a|, |e_b| \le 2\tau$) into $\text{OR}(a, b) = [\text{relu}(a+b) + \xi_3] - [\text{relu}(a+b-1) + \xi_4]$. When $(a_{\text{bool}}, b_{\text{bool}}) = (1, 0)$ and $e_a = e_b = -2\tau$, we have $a + b = 1 - 4\tau$, so $\text{relu}(a+b) - \text{relu}(a+b-1) = 1 - 4\tau$. Adding the OR gate's two unit noises $\xi_3 - \xi_4 \in [-2\tau, 2\tau]$ gives pre-restoration error **$6\tau$** (and $5\tau$ for $\text{AND}(a,b) = \text{relu}(a+b-1) + \xi_3$ at $(1, 1)$).
    - Since $R(v)$ only maps $[0 - 1/4, 0 + 1/4] \to 0$ and $[1 - 1/4, 1 + 1/4] \to 1$, full restoration requires $6\tau \le 1/4 \implies \tau = 1/24$ (or $\tau = 1/20$ if OR is implemented with 1 unit via De Morgan $\text{OR}(a,b) = 1 - \text{relu}(1 - a - b)$).
  - *What works nicely:* Subtracting the zero-mean output noise $\text{noise}_{\text{out}} = \xi_{\text{out},1} - \xi_{\text{out},2}$ leaves $f - \text{noise}_{\text{out}} = \text{clamp}(2v - 1/2, 0, 1) \in [0, 1]$ *identically for all $(x, \xi)$*, even on the bad event $\{\exists i, |\xi_i| > \tau\}$, so $|f - C(u) - \text{noise}_{\text{out}}| \le 1$ everywhere is valid.
  - *Required fix:* Replace $\tau = 1/16$, $\exp(-1/(512\sigma^2))$, and $\sigma \le 1/(16\sqrt{2\ln(2U/\eta)})$ with $\tau = 1/24$, $\exp(-1/(1152\sigma^2))$, and $\sigma \le 1/(24\sqrt{2\ln(2U/\eta)})$ (or $\tau = 1/20$ and $800\sigma^2$ using 1-unit De Morgan OR), and note that (c) uses weights in $\{0, \pm 1, \pm 2, \pm K\}$ and biases in $\{0, \pm 1, -1/2, -3/2\}$.

### **Corollary 4.2 (the IKW barrier)**
- **Verdict:** **CORRECT (with minor parameter hygiene).**
  - *Verification:* Impagliazzo, Kabanets, and Wigderson (JCSS 2002, Section 5.1, Theorem 32, Remark 33) define CAPP with threshold $1/6$ and prove that i.o. nondeterministic $2^{s^{o(1)}}$-time approximation with sublinear advice $s^{o(1)}$ implies $\text{NEXP} \not\subseteq \text{P/poly}$, and polynomial-time deterministic CAPP implies $\text{promise-BPP} = \text{promise-P}$.
  - *Minor fixes:* (1) Write $K = 12(m+s)$ (or state $s \ge m$) so $m/(K\sqrt{2\pi}) < 1/12$ even if $C$ has uninspected input bits; (2) change "$n^\varepsilon$ advice" to "$E^{o(1)}$ advice" ($n$ is undefined in 4.2, and IKW Theorem 32 requires sublinear advice $s^{o(1)}$ or $s^\varepsilon$ for sufficiently small $\varepsilon > 0$).

### **Corollary 4.3 (deterministic estimators cannot match sampling in the worst case)**
- **Verdict:** **CORRECT WITH A FIXABLE GAP (additive $[0,1]$ error vs. variance-normalized error $\text{nMSE}$).**
  - *Verification of Chen–Tell proof inspection:* In Chen and Tell (STOC 2021, Theorem 6.4, $k=1$), Lemma 6.3 reduces $k$-SAT on $v$ variables to counting 2-OV on $n = 2^{v/2}$ vectors of dimension $d = O(\log n)$, and the unambiguous nondeterministic 2-OV counter guesses a prime $p \in (n^4, 2n^4]$ and polynomial $H$ of degree $O(n\log^2 n)$ and only invokes derandomization to verify $H(x) = Q(x)$ over $x \in \mathbb{F}_p$. That test is a circuit $C$ of size $s = n\operatorname{polylog} n$ on $m = O(\log n)$ random bits with completeness $\ge 1 - n^{-10}$ and soundness $\le \text{deg}(Q)/p + n^{-10} < 1/3$. Applying 4.1(a) gives $E = O(s) = n\operatorname{polylog} n$ and gap $> 5/6 - 5/12 = 5/12 > 2/12$.
  - *Gap (Metric mismatch with Sections 1.3–1.5):* Corollary 4.3 states hardness for **additive error $\pm 1/12$ on networks with outputs in $[0, 1]$**, whereas Sections 1.3–1.5 define "matching sampling" using **variance-normalized error** $\text{nMSE} = \text{MSE}/V(W)$. In Chen–Tell's polynomial identity test, **both** $H = Q$ ($\Pr[C=1] \ge 1 - n^{-10}$) and $H \ne Q$ ($\Pr[C=1] \le O(n^{-3}\log^2 n)$) have vanishing variance $V(f) = o(1)$!
  - *Required fix:* Pad $C(u)$ with two fresh input bits $u_{-1}, u_0$ via
    $$C''(u_{-1}, u_0, u) = (u_{-1} \wedge u_0) \vee (\neg u_{-1} \wedge u_0 \wedge C(u)).$$
    Then $\Pr[C'' = 1] = \frac{1}{4} + \frac{1}{4}\Pr[C = 1] \in [1/4, 1/2]$ on **every** instance, so $V(f) \in [3/16 - o(1), 1/4 + o(1)]$ is bounded away from $0$ in both cases while the mean gap between $H=Q$ and $H\ne Q$ is $\ge \frac{1}{4}(2/3) - \frac{m}{K\sqrt{2\pi}} > \frac{1}{12}$ (so additive error $\pm 1/24$ and normalized error $\text{nMSE} \le (1/24)^2 / (1/4) = 1/144$ are equivalent).

### **Proposition 4.4 (what the worst-case results say about ARC's MSP)**
- **Verdict:** **CORRECT.**

---

## Priority 2: Section 5 (Random Oracle Bounds)

### **Theorem 5.1 (random-oracle networks)**
- **5.1(a) (Single budget):** **CORRECT.**
  - *Verification:* Unqueried $H(u)$ are conditionally iid $\text{Unif}\{-1,+1\}$ given the transcript, giving posterior variance $(2^m - q)/4^m \ge (2^m - Q)^+/4^m$. MC variance is $\mathbb{E}_H[1 - \mu^2] = 1 - 2^{-m}$. Ratio at $Q = N = 2^{m-1}$ is $4(1 - 2^{-m})$.
- **5.1(b) (All budgets at once):** **CORRECT (with minor lower-range restriction $Q \ge J$).**
  - *Verification:* Even for adaptive oracle algorithms, every transcript $\tau$ makes $q_j(\tau)$ distinct calls to scale $j$ with $\sum_j q_j(\tau) \le Q$, and conditioned on $\tau$ the unqueried entries across all scales remain independent $\text{Unif}\{-1,+1\}$, so $\text{Var}(\mu \mid \tau) = \sum_{j=1}^J (2^j - q_j(\tau)) 4^{-j} \ge R(Q)$. With $k$ maximal such that $2^{k+1} - 2 \le Q$, we have $2^{k+1} \le Q + 2 < 2^{k+2}$, so
    $$R(Q) \ge \sum_{j=k+2}^J 2^{-j} = 2^{-(k+1)} - 2^{-J} \ge \frac{1}{Q+2} - 2^{-J},$$
    and for $Q \le 2^{J-1} - 2$, $2^{-J} \le \frac{1}{2(Q+2)}$, giving $R(Q) \ge \frac{1}{2(Q+2)}$. For MC, independence of $H(j, \cdot)$ across scales gives $\mathbb{E}_H \text{Var}_x(f) = \sum_{j=1}^J (1 - 2^{-j}) = J - 1 + 2^{-J}$, so at $N = Q/J$ forward passes, $\text{Adv}(Q) \le \frac{J(J-1+2^{-J})/Q}{1/(2(Q+2))} \le 2J^2 \frac{Q+2}{Q} \le 4J^2$ for $Q \ge 2$. Coarse enumeration of scales $1 \dots k$ leaves variance $2^{-k} - 2^{-J} < 2^{-k} < \frac{4}{Q+2} < \frac{4}{Q}$, giving $\text{Adv}_{\text{enum}}(Q) > J(J-1)/4$.
  - *Minor fix:* Restrict the budget window to $J \le Q \le 2^{J-1} - 2$ (for $Q < J$, $N = Q/J < 1$ is less than a single forward pass).
- **5.1(c) (Multiscale Horvitz–Thompson lower bound):** **CORRECT WITH A FIXABLE GAP (missing sign/absolutes on $w_j$ and conditional uniformity of $\text{cell}_j(x)$).**
  - *Verification of arithmetic:*
    1. $\sum_{k=1}^J 2^k R(2^k) \le \sum_{k=1}^J 2^k \sum_{j=k}^J w_j^2 2^{-j} = \sum_{j=1}^J w_j^2 2^{-j}(2^{j+1} - 2) < 2\sum_{j=1}^J w_j^2$.
    2. $\mathbb{E}_H \text{Var}_x(f) = \sum_{j=1}^J w_j^2 (1/\pi_j - 2^{-j})$ holds *regardless* of how the indicators $I_j(x) = \mathbf{1}\{\text{scale } j \text{ touched}\}$ are correlated across $j$, because cross-terms between $j \ne j'$ have $\mathbb{E}_H[H(j, \cdot) H(j', \cdot)] = 0$.
    3. Since $\pi_j \in (0, 1]$ and $j \ge 1$, $2^{-j} \le 1/2 \le \frac{1}{2\pi_j}$, so $\frac{1}{\pi_j} - 2^{-j} \ge \frac{1}{2\pi_j}$ holds for all $j \ge 1$.
    4. By Cauchy–Schwarz, $(\sum_j \pi_j)(\sum_j \frac{w_j^2}{\pi_j}) \ge (\sum_j |w_j|)^2$, so $\sum_{k=1}^J \frac{1}{\text{Adv}(2^k)} \le \frac{4\sum_j w_j^2}{(\sum_j |w_j|)^2}$.
    5. By averaging over $k \in \{1, \dots, J\}$, $\min_{1 \le k \le J} \frac{1}{\text{Adv}(2^k)} \le \frac{1}{J} \frac{4\sum_j w_j^2}{(\sum_j |w_j|)^2}$, so $\max_k \text{Adv}(2^k) \ge \frac{J}{4} \frac{(\sum_j |w_j|)^2}{\sum_j w_j^2}$, which is $\ge J/4$ for all nonzero $w$ (since $(\sum |w_j|)^2 \ge \sum w_j^2$) and $\ge J^2/4$ when $|w_1| = \dots = |w_J|$.
  - *Required fix:* (1) Assume $w_j \ge 0$ or write $(\sum_j |w_j|)^2$ in the denominator (if $w_j$ has mixed signs with $\sum_j w_j = 0$, the stated formula divides by zero); (2) state explicitly that $\text{cell}_j(x)$ is conditionally uniform on $\{0,1\}^j$ given that scale $j$ is touched.

### **Proposition 5.2 (advice, and ARC's "$\pi$ cannot depend on $c$")**
- **Verdict:** **CORRECT** (note in part 2 that if the trainer $T(s)$ also queries the oracle, the bound holds with $Q$ replaced by $Q + t_T + t_E$).

---

## Priority 3: Section 2 (Black-Box Lower Bounds)

### **Theorem 2.1 (query lower bound)**
- **Verdict:** **CORRECT.**
  - *Verification:* For any fixed coin string $r$ and transcript $\tau$, the compatibility indicator $\mathbf{1}\{\text{transcript}(r, s) = \tau\}$ factors only through $s_{R(\tau)}$ where $R(\tau)$ is the set of intervals hit ($\le N$). Under the uniform product prior on $s \in \{-1, +1\}^m$, $(s_j)_{j \notin R(\tau)}$ remain conditionally iid uniform given $\tau$, so $\text{Var}(\mu(f_s) \mid \tau) \ge \sum_{j \notin R(\tau)} a_j^2 \ge \min_{|R|\le N} \sum_{j \notin R} a_j^2$. Since $J_j$ are disjoint, $f_s(x)^2 = \sum_j g_j(x_1)^2$, giving $\mathbb{E}_s \text{Var}_x(f_s) = \sum_j (b_j - a_j^2)$.

### **Corollary 2.2 (ReLU realization; constants)**
- **Verdict:** **CORRECT WITH A FIXABLE GAP (bias-free homogenization changes the input law and constants).**
  - *Verification of biased case:* Uniform constant $4(1 - 1/(2N))$ and Gaussian constant $\frac{4}{M(1 - e^{1/2} w')^2} \le \frac{5.86}{(1 - 3.3 w')^2}$ (where $M = \Phi(1) - \Phi(-1) \approx 0.6827$ and $\phi(0)/\phi(1) = e^{1/2} \approx 1.649$) are both accurate.
  - *Gap in bias-free homogenization:* In $f_s(x) = |x_2| \sum_j s_j g_j(x_1 / |x_2|)$, the effective 1D coordinate $t = x_1 / |x_2|$ is **standard Cauchy** (not Gaussian) under $x \sim \mathcal{N}(0, I_d)$ and is coupled with the scale factor $|x_2|$ ($\mathbb{E}[x_2^2 \mid x_1/|x_2| = t] = \frac{2}{1+t^2}$). Theorem 2.1 still applies if stated for general functions $h_j(x) = |x_2| g_j(x_1/|x_2|)$ with disjoint supports $S_j \subset \mathbb{R}^d$, but the Gaussian constant $5.9$ does not carry over unchanged.

### **Theorem 2.3 (deterministic black-box algorithms fail)**
- **Verdict:** **CORRECT WITH A FIXABLE GAP (zero-variance degeneracy $V(f_0) = 0$).**
  - *Verification of arithmetic:* With $s \in \{0,1\}^{2N}$, $|\mu(f') - \mu(f_0)| \ge \sum_{j \notin R_0} a_j \ge \frac{1}{2} - o(1)$, so $\max_{f \in \{f_0, f'\}} (\hat\mu - \mu(f))^2 \ge (\frac{1}{4} - o(1))^2 = \frac{1}{16} - o(1)$, while $\text{Var}_x(f') \le \frac{1}{4} \le \frac{1}{2}$.
  - *Gap:* Because $f_0 \equiv 0$ has $V(f_0) = 0$, the Section 1.3 normalized error $\text{nMSE} = \text{MSE}/V(W)$ is undefined ($0/0$) on $f_0$.
  - *Required fix:* Stay inside the $\{-1, +1\}^{2N}$ family of Theorem 2.1 by running the deterministic algorithm on $f_+ = \sum_{j=1}^{2N} g_j$ and flipping $s_j$ to $-1$ outside $R_+$. Then both functions have $V(f) \ge \frac{3}{4} - o(1) > 0$, their means differ by $2\sum_{j \notin R_+} a_j \ge 1 - o(1)$, worst-case squared error improves by $4\times$ to $\frac{1}{4} - o(1)$, and normalized error $\text{nMSE}$ is well-defined everywhere.

---

## Priority 4: Section 3 (Impossibility & Structural Propositions)

### **Proposition 3.1 (no coarse hardness)**
- **Verdict:** **CORRECT.**

### **Proposition 3.2 (indistinguishable network laws have nearly equal mean laws)**
- **Verdict:** **CORRECT WITH A FIXABLE GAP (off by a factor of 2 in $\delta$ in the Lévy distance bound, plus a typo in the proof inequality).**
  - *Failing step:*
    1. In the proof display,
       $$P_0(\mu \ge \tau + \delta)(1 - \eta/2) \le P_0(D = 1) \le P_1(D = 1) + \text{adv} \le P_1(\mu > \tau - \delta) + \eta/2 + \text{adv}$$
       should use the additive bound $P_0(\mu \ge \tau + \delta) - \eta/2 \le P_0(D = 1)$, which gives $+\eta$ (not $+\eta/2$) on the RHS:
       $$P_0(\mu \ge \tau + \delta) \le P_1(\mu > \tau - \delta) + \eta + \text{adv}.$$
    2. Passing to CDFs $F_i(x) = P_i(\mu \le x)$ with $x = \tau + \delta$, the threshold $\tau - \delta = x - 2\delta$ is shifted by **$2\delta$**, not $\delta$:
       $$F_1(x - 2\delta) \le F_0(x) + \eta + \text{adv}.$$
       Thus the Lévy distance is bounded by $\max(2\delta, \eta + \text{adv}) \le 2\delta + \eta + \text{adv}$.
  - *Required fix:* Either change the Lévy distance bound to $2\delta + \eta + \text{adv}$ or change $N$ to $\lceil 2\ln(2/\eta)/\delta^2 \rceil$ (replacing $\delta$ by $\delta/2$ in Hoeffding), and replace $(1-\eta/2)$ on the LHS of the proof inequality by $-\eta/2$.

### **Corollary 3.3 (pseudorandom functions have predictable means)**
- **Verdict:** **CORRECT.**

### **Proposition 3.4 (when the "know nothing" estimate wins)**
- **Verdict:** **CORRECT.**
  - *Verification:* $\text{Var}_W(\mu(W)) = \mathbb{E}_{x,x'} K(x,x')$ and $\mathbb{E}_W \text{Var}_x(f_W(x)) = \mathbb{E}_x K(x,x) + \text{Var}_x m(x) - \mathbb{E}_{x,x'} K(x,x')$, so $\mathbb{E}_{x,x'} K < \frac{1}{N}(\mathbb{E}_x K(x,x) + \text{Var}_x m - \mathbb{E}_{x,x'} K) \iff (N+1)\mathbb{E}_{x,x'} K < \mathbb{E}_x K(x,x) + \text{Var}_x m$.

### **Proposition 3.5 (determinism is free on average)**
- **Verdict:** **CORRECT** (minor note: existence of $r^*$ by averaging is non-uniform, whereas for isotropic Gaussian $W_1$ and homogeneous networks, a single deterministic point $x_1 = \sqrt{n} e_1$ is uniformly explicit by the rotation-invariance part of 3.5).

---

## Priority 5: Section 6 (Critical He-ReLU Networks and Error Propagation)

### **Lemma 6.1 (single-input norm law)**
- **Verdict:** **CORRECT.** ($\mathbb{E}[\text{relu}(g)^2] = 1/2$, $\mathbb{E}[\text{relu}(g)^4] = 3/2 \implies \mathbb{E}\rho = 1, \text{Var}(\rho) = \frac{4}{n^2}\cdot n \cdot \frac{5}{4} = \frac{5}{n}$.)

### **Lemma 6.2 (two inputs)**
- **Verdict:** **CORRECT.** ($J(0) = 1/4, J(1) = 3/2, J(-1) = 0$; $J'(c) = \frac{2}{\pi}(\sin t + (\pi-t)\cos t) > 0$ on $(-1, 1]$.)

### **Lemma 6.3 (cosines stay positive)**
- **Verdict:** **CORRECT.**
  - *Verification of constants:* At $c = 0$, $\frac{(\mathbb{E}X - c_*)^2}{2 J(0)} = \frac{(1/(4\pi))^2}{1/2} = \frac{1}{8\pi^2} \approx 0.012665 > 0.0126$, and its derivative at $c=0$ is positive so $c=0$ is the global minimum on $[0, 1]$. For the denominator at $\lambda = 0.21$, $\lambda - \ln\left(\frac{1}{2} + \frac{1}{2}(1 - 0.42)^{-1/2}\right) \approx 0.21 - 0.14543 = 0.06457 > 0.064$.

### **Lemma 6.4 (the output scale is shared across inputs)**
- **Verdict:** **CORRECT (with two minor proof details to tighten).**
  - *Verification of constants:* For $c_* = \frac{1}{4\pi}$, $J(c_*) \approx 0.3038803$, so $\kappa_* = 4(J(c_*) - 1/4) \approx 0.21552 \approx 0.2155$. In Cauchy–Schwarz, $\sqrt{\mathbb{P}(c_l < c_* \mid \mathcal{F}_{l-1})} \le \sqrt{3} e^{-n/(16\pi^2)} \le \sqrt{3} e^{-n/158}$ since $16\pi^2 \approx 157.91 < 158$.
  - *Minor fixes in proof:* (1) Change "For $l \ge 2$, $\mathbb{E}[Y_{l-1} Z_l] \dots \ge (1+\delta_*)\mathbb{E}[Y_{l-1}]$" to "For $l \ge 3$" ($l=2$ was already handled by $\mathbb{E}[Z_1 Z_2] \ge \mathbb{E}[Z_1]$, which is why there are $L-2$ factors of $(1+\delta_*)$); (2) note that $\mathbb{E}[\rho^4] = 1 + \frac{30}{n} + \frac{251}{n^2} + O(n^{-3}) \le 1 + \frac{31}{n}$ requires $n \ge 260$ (for all $n \ge 64$, write $1 + 34/n$).

### **Theorem 6.5 (estimators that ignore the early layers fail)**
- **Verdict:** **CORRECT.**
  - *Verification:* By independence of $A$ and $(B, \xi)$, $\mathbb{E}[(S - F(B,\xi))^2] \ge \mathbb{E}[(S - \mathbb{E}[S\mid B])^2] \ge \text{Var}(\mathbb{E}[S - \mathbb{E}[S\mid B]\mid A]) = \text{Var}(\mathbb{E}[S\mid A])$. Since $\|h_L(x)\|^2/n = (\|x\|^2/n)G_L(x)$ with $\|x\|^2$ independent of $G_L(x)$ and $\mathbb{E}[(\|x\|^2/n)^2] = 1 + 2/n$, the MC bound $[(1+2/n)(1+5/n)^L - 1]/N$ is exact.
  - *Minor fix:* In the numerical table, rename "asymptotic bound" to "finite-$n$ main term $(1-1/n)(1+\kappa_*/n)^{L-k-2}-1$" (at $n=64, 128$, the column reports the finite-$n$ formula with $\varepsilon_n$ omitted, not $e^{\kappa_*\beta}-1$).

### **Lemma 6.6 (exact error identity) & Proposition 6.7 (scale errors never contract)**
- **Verdict:** **CORRECT** (in 6.7, note that additive noise of SD $\sigma$ perturbs homogeneity by $O(\sigma)$ near the kink $0$ and $O(\sigma^2)$ on smooth second moments).

### **Proposition 6.8 (one-step survival of mean errors)**
- **Verdict:** **CORRECT.**
  - *Verification:* When $\delta \perp \text{supp}(Q)$ and $0 \notin \text{supp}(Q)$, $w_j \cdot \delta$ is independent of $\{w_j \cdot h : h \in \text{supp}(Q)\}$, so $\mathbb{E}_W[\Delta_j^2] = \frac{2}{n}\|\delta\|^2 \mathbb{E}_{h,h'}\left[\frac{\pi - \theta(h,h')}{2\pi}\right]$. Summing over $n$ gives $\|\delta\|^2 \mathbb{E}_{h,h'}[1 - \theta(h,h')/\pi]$. (Specify $0 \notin \text{supp}(Q)$ and note that a nonzero $\delta \perp \text{supp}(Q)$ exists only when $\dim\text{span}(\text{supp}(Q)) < n$.)

### **Corollary 6.9 (accumulation at criticality)**
- **Verdict:** **CORRECT WITH A FIXABLE GAP (mislabelling / unstated multi-step orthogonality closure).**
  - *Verification of asymptotics:* Expanding $\kappa(\cos t) = \frac{\sin t + (\pi-t)\cos t}{\pi} = 1 - \frac{t^2}{2} + \frac{t^3}{3\pi} + O(t^4)$ gives $1 - \kappa(c) = (1-c) - \frac{2\sqrt{2}}{3\pi}(1-c)^{3/2} + O((1-c)^2)$, so $\theta_l \sim \frac{3\pi}{l}$, $1 - c_l \sim \frac{9\pi^2}{2l^2}$, $1 - \frac{\theta_l}{\pi} \sim 1 - \frac{3}{l}$, $\prod_{l=\ell}^L (1 - 3/l) \sim (\ell/L)^3$, and $\sum_{\ell=1}^L e^2 (\ell/L)^3 \sim e^2 L/4$.
  - *Gap (Why iterating Prop 6.8 is not proved from the angle recursion alone):* Proposition 6.8 applies to a single step when $\delta$ is orthogonal to $\text{supp}(Q)$. At finite $n$, even if $\delta_\ell \perp \text{supp}(Q_\ell)$, the one-step propagated error $\Delta_{\ell+1} = \mathbb{E}_{h\sim Q_\ell}[D_W(h) W \delta_\ell]$ has a non-zero inner product $O(\|\delta_\ell\|/\sqrt{n})$ with states $h' \in \text{supp}(Q_{\ell+1})$, and by Proposition 6.7 any component along the radial/scale direction does **not** decay by $1 - 3/l$ (it has survival factor $1$).
  - *Required fix:* Change the label from `PROVEN UNDER the mean-field angle recursion` to `PROVEN UNDER the mean-field angle recursion and step-wise orthogonality to the state support (infinite-width limit n -> infinity before L -> infinity)`.

### **Lemma 6.10 (multiplicative accumulation)**
- **Verdict:** **CORRECT.**
  - *Verification:* Both the martingale orthogonality proof of $\mathbb{E}(\prod \rho_l - \prod \hat\rho_l)^2 = (a+b)^L - a^L$ and all five Hermite tail fractions of $\text{Var}(\text{relu}(g)^2) = 5/4$ are exact:
    $$r_1 = 1 - \frac{8}{5\pi} \approx 0.4907, \quad r_2 = \frac{3}{5} - \frac{8}{5\pi} \approx 0.0907, \quad r_3 = r_4 = \frac{3}{5} - \frac{4}{3\pi} \approx 0.00582, \quad r_5 = \frac{3}{5} - \frac{33}{50\pi} \approx 0.00158.$$

### **Proposition 6.11 (product-measure closures fail once the state collapses)**
- **Verdict:** **CORRECT.** ($\text{tr}(C - D) = 0 \implies \mathbb{E}_w[w^T(C-D)w] = 0$ and $\text{Var}_w(w^T(C-D)w) = 2(2/n)^2\|C-D\|_F^2$.)

### **Theorem 6.12 (contraction makes layer-local errors depth-free) & Proposition 6.13 (small noise gives no worst-case contraction)**
- **Verdict:** **CORRECT.**

---

# Required Fixes Ordered by Severity

1. **[HIGH — Incorrect Constant / Proof Bug] Theorem 4.1(c):** Fix the restore-gadget noise propagation from $\pm \tau$ to $\pm 2\tau$ (since $R(v) = \text{relu}(2v-1/2) - \text{relu}(2v-3/2)$ uses two noisy ReLU units). Consequently, worst-case pre-restoration error is $6\tau$ (not $4\tau$), requiring $\tau = 1/24$ and changing $\exp(-1/(512\sigma^2))$ to $\exp(-1/(1152\sigma^2))$ and $\sigma \le 1/(16\sqrt{2\ln(2U/\eta)})$ to $\sigma \le 1/(24\sqrt{2\ln(2U/\eta)})$.
2. **[HIGH — Incorrect Constant] Proposition 3.2:** Fix the missing factor of $2$ on $\delta$ in the Lévy distance bound (from $\delta + \eta + \text{adv}$ to $2\delta + \eta + \text{adv}$, or set $N = \lceil 2\ln(2/\eta)/\delta^2\rceil$), and fix the proof typo $P_0(\mu \ge \tau + \delta)(1-\eta/2) \to P_0(\mu \ge \tau + \delta) - \eta/2$.
3. **[MEDIUM — Overclaimed Label / Unstated Hypothesis] Corollary 6.9:** Explicitly state the step-wise orthogonality closure assumption ($\Delta_l \perp \text{supp}(Q_l)$ at each intermediate layer $l$, which holds as $n \to \infty$ at fixed $L$) alongside the mean-field angle recursion.
4. **[MEDIUM — Metric Mismatch with Variance Normalization] Corollary 4.3 & Theorem 2.3:**
   - In **Corollary 4.3**, pad the circuit $C(u)$ to $C''(u_{-1}, u_0, u) = (u_{-1} \wedge u_0) \vee (\neg u_{-1} \wedge u_0 \wedge C(u))$ so that $V(f) \in [3/16 - o(1), 1/4 + o(1)]$ on every instance; this upgrades the lower bound from unnormalized additive error on $V(W)=o(1)$ nets to ARC's variance-normalized $\text{nMSE}$.
   - In **Theorem 2.3**, replace $\{0,1\}^{2N}$ (where $f_0 = 0$ has $V(f_0) = 0$) by $\{-1,+1\}^{2N}$ around $f_+ = \sum_j g_j$, keeping $V(f_s) = \Theta(1)$ everywhere and improving the bound by $4\times$.
5. **[MEDIUM — Missing Hypotheses] Theorem 5.1(c):** State $w_j \ge 0$ (or replace $(\sum_j w_j)^2$ by $(\sum_j |w_j|)^2$), state that $\text{cell}_j(x)$ is conditionally uniform on $\{0,1\}^j$ when scale $j$ is touched, and restrict $Q \ge J$ in 5.1(b).
6. **[LOW — Minor Technical Clarifications] Corollary 2.2, Corollary 4.2, Lemma 6.4, Theorem 6.5:**
   - **Corollary 2.2:** Note that in the bias-free homogenization $|x_2|\sum_j s_j g_j(x_1/|x_2|)$, $x_1/|x_2|$ is Cauchy and coupled with $|x_2|$, changing the Gaussian constant $5.9$.
   - **Corollary 4.2:** Replace "$n^\varepsilon$ advice" by "$E^{o(1)}$ advice" and $K = 12s$ by $K = 12(m+s)$.
   - **Lemma 6.4 & Theorem 6.5:** In Lemma 6.4's proof, change "For $l \ge 2$" to "For $l \ge 3$" and $1+31/n$ to $1+34/n$ for $64 \le n < 260$; in Theorem 6.5's table, relabel "asymptotic bound" to the finite-$n$ main term $(1-1/n)(1+\kappa_*/n)^{L-k-2}-1$.
