# QuBots — Quantum Portfolio Optimisation

**YQuantum 2026 | The Hartford & Capgemini Quantum Lab | QuEra**

A quantum-classical hybrid approach to insurance portfolio optimisation using QUBO/Ising formulations and QAOA executed on Bloqade's neutral atom simulator.

---

## Problem

Insurance companies must invest large premium volumes under strict solvency and liquidity constraints. Constructing a portfolio that maximises return while minimising correlated risk across asset classes is a highly non-trivial combinatorial problem. We model it as a Markowitz mean-variance optimisation, reformulate it as a QUBO, and solve it using quantum approaches.

---

## Approach

### 1. Data
- 8 assets selected from 50 (one per sector): Gov Bonds, IG Credit, HY Credit, Equities US, Equities Intl, Infrastructure, Real Estate, Cash
- Expected returns `mu` and covariance matrix `cov` extracted from `investment_dataset_covariance.csv`

### 2. QUBO Formulation
The Markowitz objective with budget constraint becomes:
min x^T Q x


where the Q matrix encodes:
- **Return reward**: `-mu[i]` on the diagonal
- **Risk penalty**: `q_risk * cov[i,j]` on off-diagonals  
- **Budget penalty**: `lambda * (sum(x_i) - B)^2` expanded into the matrix

Parameters: `q_risk = 1`, `lambda = 5`, `B = 4` (pick 4 of 8 assets)

### 3. Ising Mapping
QUBO binary variables `x_i ∈ {0,1}` are mapped to Ising spins `s_i ∈ {-1,+1}` via `x_i = (1 + s_i) / 2`, giving:
H = sum_i h_i * Z_i + sum_{i<j} J_ij * Z_i * Z_j


### 4. Quantum Circuit (Bloqade)
- 8 qubits in a 2×4 neutral atom grid, spacing 5μm, Rydberg blockade radius 15μm
- Fully connected graph — matches the dense QUBO structure of the portfolio problem
- QAOA circuit with p=1 and p=2 layers built using `@qasm2.main`
- Angles optimised via grid search + refinement
- Executed via Bloqade's PyQrack simulator (`state_vector` + `multi_run`)

### 5. Noise Analysis
Depolarising noise swept from 0% to 5% per-qubit per-layer. Results show energy remains stable up to ~1% gate error, consistent with current Rydberg hardware capabilities.

---

## Files

| File | Description |
|---|---|
| `01_simulated_annealing.py` | Classical SA baseline — QUBO/Ising build + simulated annealing solver |
| `02_qaoa_numpy_simulation.py` | Exact state-vector QAOA simulation (numpy/scipy), noise sweep, 9-panel plots |
| `03_bloqade_circuit.py` | Real Bloqade `@qasm2.main` circuit, PyQrack execution, shot sampling |
| `investment_dataset_covariance.csv` | Full 50×50 covariance matrix |
| `investment_dataset_assets.csv` | Asset metadata (sector, expected return, etc.) |
| `investment_dataset_scenarios.csv` | Stress scenario data |

---

## Setup

```bash
pip install bloqade bloqade-pyqrack[pyqrack-cpu] scipy numpy matplotlib
```

## Running

```bash
# Classical baseline
python 01_simulated_annealing.py

# Exact QAOA simulation + plots
python 02_qaoa_numpy_simulation.py

# Real Bloqade circuit execution
python 03_bloqade_circuit.py
```

---

## Results

*(To be updated with final numbers)*

- **Selected portfolio**: TBD
- **Expected return**: TBD
- **Volatility**: TBD
- **Sharpe ratio**: TBD
- **QAOA energy (p=2)**: TBD
- **Noise tolerance**: stable up to ~1% gate error

---

## Team

QuBots — YQuantum 2026




