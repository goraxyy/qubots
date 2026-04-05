"""
QUBO / Ising Portfolio Optimiser
=================================
Assets: A017 (Gov Bonds), A026 (IG Credit), A013 (HY Credit),
        A020 (Equities US), A023 (Equities Intl), A038 (Infrastructure),
        A022 (Real Estate), A048 (Cash)


Parameters: q=1, lambda=5, B=4 (pick 4 out of 8)
"""


import numpy as np
import random
import math


# ── 1. DATA ──────────────────────────────────────────────────────────────────


assets = ['A017', 'A026', 'A013', 'A020', 'A023', 'A038', 'A022', 'A048']
sectors = ['Gov Bonds', 'IG Credit', 'HY Credit', 'Equities US',
           'Equities Intl', 'Infrastructure', 'Real Estate', 'Cash']


mu = np.array([
    0.018326,  # A017 Gov Bonds
    0.033393,  # A026 IG Credit
    0.073167,  # A013 HY Credit
    0.078887,  # A020 Equities US
    0.083546,  # A023 Equities Intl
    0.062865,  # A038 Infrastructure
    0.060711,  # A022 Real Estate
    0.013631,  # A048 Cash
])


# Covariance submatrix extracted from investment_dataset_covariance.csv
cov = np.array([
    [0.001327, 0.000234, 0.000491, 0.000473, 0.000830, 0.000706, 0.000944, 4.70e-5],
    [0.000234, 0.002579, 0.000348, 0.000607, 0.001274, 0.001004, 0.000353, 2.28e-5],
    [0.000491, 0.000348, 0.014474, 0.003222, 0.001097, 0.001030, 0.000764, 1.07e-4],
    [0.000473, 0.000607, 0.003222, 0.030189, 0.003075, 0.001058, 0.001985, 2.51e-4],
    [0.000830, 0.001274, 0.001097, 0.003075, 0.040679, 0.001734, 0.001728, 2.43e-4],
    [0.000706, 0.001004, 0.001030, 0.001058, 0.001734, 0.013188, 0.002982, 2.06e-4],
    [0.000944, 0.000353, 0.000764, 0.001985, 0.001728, 0.002982, 0.018554, 2.41e-4],
    [4.70e-5,  2.28e-5,  1.07e-4,  2.51e-4,  2.43e-4,  2.06e-4,  2.41e-4, 1.41e-4],
])


# ── 2. BUILD Q MATRIX ────────────────────────────────────────────────────────


q_param  = 1
lam      = 5
B        = 4
n        = len(assets)


Q = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i == j:
            Q[i][j] = q_param * cov[i][i] - mu[i] + lam * (1 - 2*B)      # diagonal
                
        else:
            Q[i][j] = q_param * cov[i][j] + 2 * lam  # off-diagonal


# ── 3. ISING REPRESENTATION ──────────────────────────────────────────────────
# QUBO variables x_i in {0,1}.  Map to Ising spins s_i in {-1,+1}:
#   x_i = (1 + s_i) / 2
# The Ising Hamiltonian is:  H = sum_i h_i * s_i + sum_{i<j} J_ij * s_i * s_j
# Derived from Q:
#   h_i   = (Q[i][i] + sum_{j≠i} Q[i][j]) / 2  (linear biases)
#   J_ij  = Q[i][j] / 4                          (quadratic couplers, i<j)


h = np.zeros(n)
J = np.zeros((n, n))


for i in range(n):
    h[i] = (Q[i][i] + sum(Q[i][j] for j in range(n) if j != i)) / 2
for i in range(n):
    for j in range(i+1, n):
        J[i][j] = Q[i][j] / 4


# ── 4. ENERGY FUNCTION ───────────────────────────────────────────────────────


def qubo_energy(x, Q):
    """E = x^T Q x"""
    x = np.array(x, dtype=float)
    return float(x @ Q @ x)


def ising_energy(s, h, J):
    """H = sum_i h_i s_i + sum_{i<j} J_ij s_i s_j"""
    e = sum(h[i] * s[i] for i in range(n))
    for i in range(n):
        for j in range(i+1, n):
            e += J[i][j] * s[i] * s[j]
    return e


# ── 5. SIMULATED ANNEALING (QUBO) ────────────────────────────────────────────


def simulated_annealing(Q, n, B, T_start=10.0, T_end=1e-4, steps=100_000, seed=42):
    rng = random.Random(seed)
    # Start with exactly B assets selected randomly
    x = [0] * n
    chosen = rng.sample(range(n), B)
    for i in chosen:
        x[i] = 1


    best_x = x[:]
    best_e = qubo_energy(x, Q)
    current_e = best_e


    for step in range(steps):
        T = T_start * (T_end / T_start) ** (step / steps)
        # Propose a swap: flip one 1→0 and one 0→1 (keeps sum = B)
        ones  = [i for i in range(n) if x[i] == 1]
        zeros = [i for i in range(n) if x[i] == 0]
        flip_out = rng.choice(ones)
        flip_in  = rng.choice(zeros)
        x[flip_out] = 0
        x[flip_in]  = 1
        new_e = qubo_energy(x, Q)
        delta = new_e - current_e
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_e = new_e
            if new_e < best_e:
                best_e = new_e
                best_x = x[:]
        else:
            x[flip_out] = 1
            x[flip_in]  = 0


    return best_x, best_e


# ── 6. SOLVE & REPORT ────────────────────────────────────────────────────────


best_x, best_e = simulated_annealing(Q, n, B)
best_s = [2*xi - 1 for xi in best_x]   # convert to Ising spins


selected = [assets[i] for i in range(n) if best_x[i] == 1]
selected_sectors = [sectors[i] for i in range(n) if best_x[i] == 1]
selected_mu   = [mu[i] for i in range(n) if best_x[i] == 1]
portfolio_return = sum(selected_mu) / B  # equal weight


print("=" * 60)
print("  QUBO / ISING PORTFOLIO OPTIMISATION RESULTS")
print("=" * 60)
print(f"\nParameters:  q={q_param}, λ={lam}, B={B}")
print(f"QUBO energy (minimised): {best_e:.6f}")
print(f"Ising energy:            {ising_energy(best_s, h, J):.6f}")


print("\n── Q Matrix (diagonal highlighted) ──")
header = "       " + "  ".join(f"{a:>8}" for a in assets)
print(header)
for i in range(n):
    row = f"{assets[i]:>6} "
    for j in range(n):
        marker = "*" if i == j else " "
        row += f"{Q[i][j]:>8.4f}{marker} "
    print(row)


print("\n── Ising linear biases (h) ──")
for i in range(n):
    print(f"  h[{assets[i]}] = {h[i]:>9.4f}")


print("\n── Ising quadratic couplers (J, i<j, top 10 by |J|) ──")
J_pairs = [(i, j, J[i][j]) for i in range(n) for j in range(i+1, n)]
J_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
for i, j, v in J_pairs[:10]:
    print(f"  J[{assets[i]},{assets[j]}] = {v:.6f}")


print("\n── Solution vector ──")
print("  QUBO (x):  ", best_x)
print("  Ising (s): ", best_s)


print(f"\n── Optimal Portfolio (pick {B} of {n}) ──")
for a, sec, r in zip(selected, selected_sectors, selected_mu):
    print(f"  ✓  {a}  {sec:<18}  μ = {r:.4f} ({r*100:.2f}%)")


print(f"\n  Equal-weight portfolio expected return: {portfolio_return*100:.2f}%")


# Verify: portfolio variance (equal weight w = 1/B)
w = np.array([1/B if best_x[i] else 0 for i in range(n)])
port_var = float(w @ cov @ w)
port_std = math.sqrt(port_var)
sharpe = (portfolio_return - mu[7]) / port_std  # mu[7] = cash as risk-free
print(f"  Portfolio volatility (std dev):         {port_std*100:.2f}%")
print(f"  Sharpe ratio (vs cash):                 {sharpe:.3f}")
print("=" * 60)



