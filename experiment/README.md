# Harmonic navigation prototype

Latest U/narrow-only dense experiment:

```powershell
python refined_M500.py
python finalize_refined.py
```

Results: `outputs/refined_M500/final/report.md`, with 100x100 grids and M=500.
This version separates wall-contour basis support, excludes solid-cell gradients
from SDE interpolation, enforces repulsion at actual wall face endpoints/midpoints,
and imposes a 10 m/s grid command bound without clipping the command. Final
projection and Monte Carlo check exact corners, time-step sensitivity and
noise-free trajectories. Coefficient range [1,80] and speed 10 m/s are explicit
experiment choices. Earlier outputs are retained as development comparisons.

Run from this directory with Python, NumPy, SciPy, and matplotlib:

```powershell
python main.py --suite pilot
python main.py --suite full --mc 5000
python high_basis.py --basis 50 100 8 --maxiter 180 --mc 5000
```

The pilot makes the first potential/100-trajectory plots under `outputs/pilot/`.
The main CLI now defaults to M=8 for the open maze and M=200 localized wall
bases with adjoints for the two complex mazes. `--basis 8 --boundary spatial
--optimizer finite-difference --wall-safety free` reproduces the initial small-basis experiment.

**Wall correction:** Both CLIs now default to `--wall-safety repulsive`:
wall potential >= `--wall-floor 1`, wall potential >= its adjacent free-node
potential, and commanded velocity at wall-adjacent nodes has no outward normal
component. `--minimum-inward-speed` can require a strictly positive repulsion.
The previous probability-only runs allowed locally attractive wall segments;
they are retained as unconstrained comparisons, not wall-repulsive solutions.
`python check_wall_potential.py` audits them and runs a constrained M=200 case.
See `outputs/wall_safety/report.md` for the correction and the separate draft
Eq. (5.2) certificate check. No nonharmonic inverse-distance term was added.
The full suite evaluates three mazes and five specifications, compares fixed,
joint, risk-only, and time-only controllers on two mazes, checks exact CTMC jump
simulation and SDE time-step sensitivity, and measures rough runtime scaling.
All main figures, CSV/JSON metrics, and `report.md` are under `outputs/`.

The original unconstrained higher-basis follow-up is under `outputs/high_basis/`;
reproduce it with `--wall-safety free`. New default constrained runs write to
`outputs/high_basis_repulsive/`. It uses localized
functions centered on actual wall boundary nodes (full boundary rank checked),
M=50 and 100, and discrete adjoints so optimization cost does not require M
finite-difference evaluations per iteration. A negative-log-arrival warm-up
escapes the nearly zero-probability region before minimizing the direct joint
violation. An M=8 run of this same localized construction isolates basis size
from the original broad radial basis. M=8 results alone cannot establish a
limitation of the harmonic navigation method.

Options: `--grid 40 --basis 8 --budget 140 --sigma 0.20 --gain 1 --theta-max 8
--seed 20261101 --mc 5000 --output outputs/custom`.
For small bases the budget limits L-BFGS-B function evaluations approximately;
for adjoint search it controls iterations per initialization. SciPy can finish
the current finite-difference/line-search batch beyond that limit.

Code structure:

```
main.py                     CLI and CPU thread limits
configs/specifications.json Five risk/deadline specifications
src/maze.py                 Three geometries, custom NPZ loader
src/harmonic.py             Sparse Laplace extension and numerical gradients
src/boundary_contours.py     Continuous wall-contour hat functions
src/wall_safety.py           Wall potential and actual interpolated-normal constraints
src/adjoint.py               Discrete sensitivities and constrained probability search
src/ctmc.py                 Killed upwind generator and absorption rates
src/probabilities.py        Risk solve, exact expm and fast search time stepping
src/optimize_theta.py       Bounded local probability-guided synthesis
src/simulate.py             SDE and exact CTMC jump Monte Carlo
src/plotting.py             Potential, paths, specifications, sweeps, ablations
src/experiments.py          Suite, persistence, numerical audits, report
```

Geometry uses coordinates in meters on [0,4]^2, a point robot, sigma in
m/sqrt(s), gain in m^2/s, and zero goal potential. Boundary coefficients are
nonnegative and bounded; there is no imposed command-speed bound in this first
prototype. Reported maximum commands are sampled at interior grid nodes.
Custom occupancy grids can be loaded with `src.maze.load_maze`: provide an NPZ
containing boolean `occupied`, boolean `goal`, integer `start_rc`, and optional
`length`/`name`, then use the same basis/builder/evaluation APIs.

Final probabilities are exact finite-model equations evaluated numerically,
not continuum-certified probabilities. Upwind spatial diffusion, pixel boundary
placement, EM time stepping, local optimization, finite basis size, and finite
Monte Carlo samples are separate approximation sources. Nonzero violation is
UNDETERMINED; it is not an infeasibility proof. `report.md` explains event
definitions, censoring, confidence intervals, and negative results.

Reference: `../output/pdf/main.pdf`, especially Sections 4.1, 4.4, 4.5 and Appendix B.
