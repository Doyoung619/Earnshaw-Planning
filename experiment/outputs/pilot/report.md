# Harmonic navigation: first experimental prototype

Completed 1 maze/spec pairs in 0.10 minutes. CTMC specification pass: 1/1; SDE empirical pass using marginal 95% Wilson intervals: 1/1.

## Implementation

- Python/NumPy/SciPy/matplotlib, CPU only; grid=40x40, M=8, sigma=0.2 m/sqrt(s), gain=1.0 m^2/s, seed=20261101.
- A 4 m square workspace, 0.28 m goal radius, a point robot, and three fixed occupancy geometries.
- Smooth Gaussian wall partition of unity, zero goal potential, sparse five-point Laplace solves; coefficients restricted to [0, 8.0]. This is a chosen subfamily of the draft.
- Drift uses centered grid gradients. CTMC uses diffusion plus nonnegative upwind drift rates (draft Eq. 4.10).
- Optimization minimizes max(0, p-delta, 1-epsilon-q) with a tiny probability tie-breaker, random initial candidates and bounded L-BFGS-B. Search uses 72-step backward Euler; three final candidates are ranked by matrix-exponential probabilities.
- Final p is a sparse linear solve; final q is g-exp(TQ)g. These are numerical evaluations of the finite CTMC, not continuum certificates.
- Euler-Maruyama uses bilinear drift interpolation and occupancy-cell absorption; risk trials continue to max(3T,45s). Unabsorbed trials yield [wall fraction, wall fraction + timeout fraction] bounds on eventual risk. Mean exit among observed exits is conditional; the restricted mean includes censoring.
- No speed saturation or prescribed u_max constraint was imposed. Actual grid-node maximum commands are reported. No global infeasibility, continuum certification, or HJB proof is claimed.

## Main numerical results

| Maze | Spec (delta,epsilon,T) | p CTMC | Wall SDE | q CTMC | Arrival SDE | rho | CTMC / SDE CI pass |
|---|---|---:|---:|---:|---:|---:|---|
| open | S1 (0.1,0.1,15) | 0.00771 | 0.00000 | 0.99227 | 1.00000 | 0.00000 | True / True |

All coefficients, confidence intervals, horizons, timeout rates, speeds, and optimizer messages are in `data/summary.json` and `data/summary.csv`.

## Synthesized coefficients

- open/S1: theta=[8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]; max grid command=20.834 m/s; timeout=0.0000.

## Diagnosis and limits

- A nonzero local-search violation is **UNDETERMINED**, not evidence of mathematical infeasibility. Narrow passages, strong harmonic screening far from the goal, limited boundary basis, bounded coefficients, and finite search budget can each cause failure.
- Upwind CTMC adds drift-dependent artificial diffusion of order h. Its predicted risk can differ materially from SDE Monte Carlo even when the linear-system residual is tiny. Occupancy-cell face absorption vs absorbing grid nodes is another spatial error.
- Euler-Maruyama and line-segment contact checks can miss between-step Brownian crossings; Monte Carlo intervals quantify sampling error only. Grid/time-step audits are saved separately.
- In S1, S2, S4, and S5, delta=epsilon. Since q(T)<=1-p, satisfying the deadline condition already implies the risk condition. Only S3 imposes a strictly tighter separate risk budget. Consequently these specifications alone provide limited evidence that a joint objective outperforms time-only synthesis.
- The optimizer may stop just inside a constraint. Such a CTMC pass can fail an empirical confidence-interval test; a deployment-oriented implementation should synthesize a margin.

## Ablations

| Maze | Controller | p CTMC | q CTMC | rho | Wall SDE | Arrival SDE |
|---|---|---:|---:|---:|---:|---:|

## Numerical audits

```json
[]
```

## Files and reproduction

- Figures: `figures/*_potential.png`, `*_trajectories.png`, `specification_satisfaction.png`, `specification_sweep.png`, `ablations.png`, `runtime_scalability.png`.
- Detailed results: `data/summary.csv`, `summary.json`, `ablations.json`, `runtime.json`, `audits.json`.
- Reproduce: `python main.py --suite full --mc 5000` from `experiment/`.
- Default pilot: `python main.py --suite pilot` (one maze, 1,000 SDE trials).
