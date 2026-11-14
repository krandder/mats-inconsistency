# Check of the Lambert-Ostrovsky-Panov (2018) closed form (NBER WP 20516, Sec. 4.2).
import numpy as np, time
rng = np.random.default_rng(0)

def lop(Om, ks, kM):
    """Om: cov of (v; theta_1..theta_n; theta_M; u). ks: list of signal dims. Returns alpha list, betaD, betaM."""
    K = sum(ks); idx_v = 0; th = slice(1, 1+K); M = slice(1+K, 1+K+kM); iu = 1+K+kM
    Sth = Om[th, th]; Stv = Om[th, 0]; Stu = Om[th, iu]
    Sdiag = np.zeros_like(Sth); o = 0
    for k in ks: Sdiag[o:o+k, o:o+k] = Sth[o:o+k, o:o+k]; o += k
    if kM > 0:
        SMM = Om[M, M]; StM = Om[th, M]; SMv = Om[M, 0]; SMu = Om[M, iu]
        iMM = np.linalg.inv(SMM)
        Lam = Sdiag + Sth - StM @ iMM @ StM.T
        Au = np.linalg.solve(Lam, Stu - StM @ iMM @ SMu)
        Av = np.linalg.solve(Lam, Stv - StM @ iMM @ SMv)
        a = -Av @ Sdiag @ Av
        b = Av @ (2*Sdiag + Lam) @ Au + Om[iu, M] @ iMM @ SMv - Om[iu, 0]
        # c = Var(Au'theta - u | theta_M)
        w = np.concatenate([Au, -np.ones(1)])
        sel = list(range(1, 1+K)) + [iu]
        S = Om[np.ix_(sel, sel)]; C = Om[np.ix_(sel, list(range(1+K, 1+K+kM)))]
        c = w @ (S - C @ iMM @ C.T) @ w
    else:
        Lam = Sdiag + Sth
        Au = np.linalg.solve(Lam, Stu); Av = np.linalg.solve(Lam, Stv)
        a = -Av @ Sdiag @ Av; b = Av @ (2*Sdiag + Lam) @ Au - Om[iu, 0]
        c = Au @ Sth @ Au - 2*Au @ Stu + Om[iu, iu]
    gam = -(b + np.sqrt(b*b - 4*a*c)) / (2*a)
    bD = 1/gam; al = gam*Av - Au
    bM = None
    if kM > 0:
        bM = iMM @ (SMv - StM.T @ Av) - bD * iMM @ (SMu - StM.T @ Au)
    return al, bD, bM

def verify(Om, ks, kM, al, bD, bM):
    K = sum(ks); n = 1 + K + kM + 1
    # price as linear map of mu: P = bM'thM + bD (al'th + u)
    wP = np.zeros(n); wP[1:1+K] = bD*al; wP[1+K+K*0+0:1+K] += 0
    if kM > 0: wP[1+K:1+K+kM] += bM
    wP[-1] += bD
    # market maker consistency: P == E[v | thM, D]
    wD = np.zeros(n); wD[1:1+K] = al; wD[-1] = 1
    obs = [wD] + [np.eye(n)[1+K+j] for j in range(kM)]
    Wo = np.array(obs); Sobs = Wo @ Om @ Wo.T; cvo = Wo @ Om[:, 0]
    coef = np.linalg.solve(Sobs, cvo); wE = coef @ Wo
    err_mm = np.abs((wE - wP) @ Om @ (wE - wP))
    # best responses
    err_br = 0; o = 0
    for i, k in enumerate(ks):
        wO = np.zeros(n); wO[0] = 1; wO -= wP; wO[1+o:1+o+k] += bD*al[o:o+k]  # v - P excluding own demand
        Sii = Om[1+o:1+o+k, 1+o:1+o+k]; cov = Om[1+o:1+o+k, :] @ wO
        ci = np.linalg.solve(Sii, cov) / (2*bD)
        err_br = max(err_br, np.abs(ci - al[o:o+k]).max()); o += k
    return err_mm, err_br

if __name__ == "__main__":
    # 1) Kyle 1985: v~N(0,S0), one insider sees v, u~N(0,su2): lambda = sqrt(S0)/(2 su)
    S0, su = 2.0, 0.7
    Om = np.array([[S0, S0, 0], [S0, S0, 0], [0, 0, su**2]])
    # full-rank requirement on Sigma_ii holds (S0>0); Om itself is singular, allowed
    al, bD, _ = lop(Om, [1], 0)
    print("Kyle: lambda", bD, "expected", np.sqrt(S0)/(2*su), " beta", al[0], "expected", su/np.sqrt(S0))
    # 2) N identical insiders (static Holden-Subrahmanyam): lambda = sqrt(N S0)/((N+1) su)
    N = 5; Om = np.zeros((N+2, N+2)); Om[:N+1, :N+1] = S0; Om[-1, -1] = su**2
    al, bD, _ = lop(Om, [1]*N, 0)
    print("HS N=5: lambda", bD, "expected", np.sqrt(N*S0)/((N+1)*su))
    # 3) LOP Example 2: theta1 = v + xi, theta2 = xi -> alpha ∝ (2, -1)
    Om = np.array([[1,1,0,0],[1,2,1,0],[0,1,1,0],[0,0,0,1.]])
    al, bD, _ = lop(Om, [1,1], 0); print("LOP ex2 alpha ratio", al/al[0]*2)
    # 4) random complex structures with a market-maker signal and correlated u; verify equilibrium conditions
    worst = (0, 0)
    for t in range(200):
        n = rng.integers(2, 9); kM = rng.integers(0, 3); K = n
        F = rng.normal(size=(1+K+kM+1, 1+K+kM+1+3))
        Om = F @ F.T / F.shape[1]
        al, bD, bM = lop(Om, [1]*n, kM)
        e = verify(Om, [1]*n, kM, al, bD, bM); worst = (max(worst[0], e[0]), max(worst[1], e[1]))
    print("random structures: max MM error, max BR error", worst)
    # 5) timing: structured world, n=32 traders, kM=1, numpy single episode
    n = 32; F = rng.normal(size=(n+3, n+6)); Om = F @ F.T / F.shape[1]
    t0 = time.time()
    for _ in range(1000): lop(Om, [1]*n, 1)
    print("ms per episode (n=32, numpy, unbatched):", (time.time()-t0))
