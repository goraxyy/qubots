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

### 5. Classical Baselines
- **Simulated Annealing** (`01_simulated_annealing.py`) — classical combinatorial solver
- **Markowitz MVO** (`03_bloqade_circuit.py`) — continuous mean-variance optimisation via scipy SLSQP
- **Quantum Annealing** — adiabatic interpolation H(s) = (1-s)H_M + s*H_C for D-Wave comparison

### 6. Noise Analysis
Depolarising noise swept from 0% to 5% per-qubit per-layer using a proper density matrix channel:
`rho_noisy = (1 - p) * U rho U† + p * I/dim`
Results show energy remains stable up to ~1% gate error, consistent with current Rydberg hardware capabilities.

### 7. Stress Testing (Hartford-specific)
Portfolio performance evaluated under real market stress scenarios from `investment_dataset_scenarios.csv`.
CVaR (worst-5 average) computed for QAOA and brute-force portfolios to assess insurance solvency risk.

---

## Files

| File | Description |
|---|---|
| `01_simulated_annealing.py` | Classical SA baseline — QUBO/Ising build + simulated annealing solver |
| `02_qaoa_numpy_simulation.py` | Exact state-vector QAOA simulation (numpy/scipy), noise sweep, 9-panel plots |
| `03_bloqade_circuit.py` | Real Bloqade `@qasm2.main` circuit, PyQrack execution, shot sampling, MVO baseline, stress test |
| `investment_dataset_full.xlsx` | Full 50-asset return history |
| `investment_dataset_covariance.csv` | Full 50×50 covariance matrix |
| `investment_dataset_correlation.csv` | Full 50×50 correlation matrix |
| `investment_dataset_assets.csv` | Asset metadata (sector, expected return, etc.) |
| `investment_dataset_scenarios.csv` | Stress scenario data (used for CVaR analysis) |

---

## Setup

### Requirements
- **Python 3.10 or 3.11** (recommended — PyQrack wheels are not available for all platforms on 3.12+)
- All dependencies in `requirements.txt`

### Install

```bash
pip install -r requirements.txt
```

> ⚠️ **Bloqade note:** `bloqade-pyqrack[pyqrack-cpu]` is the simulator backend required
> by `03_bloqade_circuit.py`. If installation fails (e.g. no wheel for your platform/Python version),
> scripts `01` and `02` still run fully. Script `03` will automatically fall back to an exact
> numpy simulation producing identical results — you will see a clear message at startup.

### Verify Bloqade works (optional but recommended before running script 03)

```bash
python -c "from bloqade import qasm2; from bloqade.pyqrack import PyQrack; print('Bloqade OK')"
```

## Run Order

Run scripts in sequence — each builds on the previous:

```bash
# 1. Classical baseline (no quantum dependencies)
python 01_simulated_annealing.py

# 2. Exact QAOA simulation + plots (no Bloqade needed)
python 02_qaoa_numpy_simulation.py
# → saves outputs/qaoa_numpy_results.png

# 3. Full Bloqade circuit + four-way comparison + stress test
python 03_bloqade_circuit.py
# → saves bloqade_qaoa_results.png
```

## Key Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| N | 8 | Number of qubits / assets |
| B | 4 | Portfolio size (assets to select) |
| q_risk | 1 | Risk aversion |
| λ | 5 | Budget penalty strength |
| p | 2 | QAOA circuit depth |

---

## Results

*(Run `python 03_bloqade_circuit.py` to generate final numbers — outputs saved to `bloqade_qaoa_results.png`)*

- **Noise tolerance**: stable up to ~1% gate error
- **Connectivity**: fully connected via Rydberg blockade — ideal for dense QUBO graphs

---

## Slides
https://docs.google.com/presentation/d/1XIo6cwxRop6iZYg7M5266XKUhwO8SaDZ/edit?usp=sharing&ouid=100823284726496128305&rtpof=true&sd=true

## Team

QuBots — YQuantum 2026
 
