from pathlib import Path
import csv
import json
import time
import platform
import sys
import numpy as np
import scipy
from .maze import make_maze
from .harmonic import build_basis
from .ctmc import CTMCBuilder
from .probabilities import evaluate, violation
from .optimize_theta import synthesize
from .adjoint import synthesize_adjoint
from .wall_safety import audit_wall,repulsive_constraint
from .simulate import simulate_sde, simulate_ctmc
from .plotting import controller_figures, specification_figure, sweep_figure, ablation_figure, runtime_figure

ROOT = Path(__file__).resolve().parents[1]


def native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=native), encoding='utf-8')


def save_csv(path, records):
    rows = []
    for r in records:
        row = {k:v for k,v in r.items() if k not in ('sde','ctmc_mc','theta','history')}
        row['theta'] = json.dumps(r['theta'], default=native)
        for source in ('sde', 'ctmc_mc'):
            for k,v in r.get(source, {}).items():
                row[source+'_'+k] = json.dumps(v) if isinstance(v, list) else v
        rows.append(row)
    if rows:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with Path(path).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def make_record(name, spec, result, builder, args, output, stem, n_mc, make_figures=True):
    theta, predicted = result['theta'], result['prediction']
    wall_check=audit_wall(builder.basis,theta,args.sigma,args.gain,spec['delta'])
    if getattr(args,'wall_safety','free')=='repulsive':
        if (not wall_check['repulsive_at_grid_faces'] or
            wall_check['wall_min'] < args.wall_floor-1e-7 or
            wall_check['minimum_inward_command'] < args.minimum_inward_speed-1e-7):
            raise RuntimeError('Controller rejected: explicit wall-potential/repulsion constraints failed.')
    print(f'  {stem}: exact p={predicted["p"]:.5f}, q={predicted["q"]:.5f}; running {n_mc} SDE trials', flush=True)
    metrics, paths = simulate_sde(builder.basis, theta, args.sigma, args.gain, spec['T'],
                                  n=n_mc, seed=args.seed+17, save_paths=100 if make_figures else 0,
                                  interpolation=getattr(args,'interpolation','interior'))
    rho = violation(predicted['p'], predicted['q'], spec['delta'], spec['epsilon'])
    confidence_pass = metrics['risk_upper_ci'][1] <= spec['delta'] and metrics['arrival_ci'][0] >= 1-spec['epsilon']
    record = dict(maze=name, spec_name=spec['name'], delta=spec['delta'], epsilon=spec['epsilon'], T=spec['T'],
                  M=builder.basis.m,boundary_basis=builder.basis.boundary_kind,
                  wall_safety_mode=getattr(args,'wall_safety','free'),
                  mode=result['mode'], theta=theta, p=predicted['p'], q=predicted['q'], rho=rho,
                  ctmc_pass=bool(rho <= 1e-8), sde_point_pass=bool(metrics['risk_upper'] <= spec['delta'] and metrics['arrival_rate'] >= 1-spec['epsilon']),
                  sde_ci_pass=bool(confidence_pass), mean_exit_ctmc=predicted['mean_exit'],
                  max_command_speed=builder.build(theta).max_speed,
                  optimization_seconds=result['optimization_seconds'], evaluations=result['evaluations'],
                  optimizer_message=result.get('optimizer_message','Fixed boundary values'),
                  sde=metrics, residual=predicted['residual'],
                  wall_audit=wall_check)
    save_json(output/'data'/f'{stem}.json', record)
    save_json(output/'logs'/f'{stem}_optimization.json', result.get('history', []))
    if make_figures:
        controller_figures(builder.basis, theta, paths, metrics, spec, output/'figures', stem, args.gain)
        lengths = [len(p) for p in paths['paths']]
        np.savez_compressed(output/'data'/f'{stem}_trajectories.npz',
                            points=np.concatenate(paths['paths']) if sum(lengths) else np.empty((0,3)),
                            offsets=np.r_[0, np.cumsum(lengths)], status=paths['status'], exit_times=paths['exit_times'],
                            final_positions=paths['final_positions'],
                            collision_command_toward_wall=paths['collision_command_toward_wall'])
    print(f'  {stem}: SDE wall={metrics["collision_rate"]:.4f}, arrival={metrics["arrival_rate"]:.4f}, '
          f'timeout={metrics["timeout_rate"]:.4f}; CTMC pass={record["ctmc_pass"]}', flush=True)
    return record


def report(records, ablations, audits, runtime, args, seconds, output):
    n_pass = sum(r['ctmc_pass'] for r in records)
    n_emp = sum(r['sde_ci_pass'] for r in records)
    lines = ['# Harmonic navigation: first experimental prototype', '',
             f'Completed {len(records)} maze/spec pairs in {seconds/60:.2f} minutes. '
             f'CTMC specification pass: {n_pass}/{len(records)}; '
             f'SDE empirical pass using marginal 95% Wilson intervals: {n_emp}/{len(records)}.', '',
             '## Implementation', '',
             f'- Python/NumPy/SciPy/matplotlib, CPU only; grid={args.grid}x{args.grid}, M={args.basis or "open=8, complex=200"}, '
             f'sigma={args.sigma} m/sqrt(s), gain={args.gain} m^2/s, seed={args.seed}.',
             f'- A 4 m square workspace, 0.28 m goal radius, a point robot, and three fixed occupancy geometries.',
             f'- Smooth Gaussian wall partition of unity, zero goal potential, sparse five-point Laplace solves; '
             f'coefficients restricted to [0, {args.theta_max}]. This is a chosen subfamily of the draft.',
             '- Drift uses centered grid gradients. CTMC uses diffusion plus nonnegative upwind drift rates (draft Eq. 4.10).',
             '- Optimization minimizes max(0, p-delta, 1-epsilon-q) with a tiny probability tie-breaker and bounded L-BFGS-B. '
             'Small-basis search uses finite differences and 72-step backward Euler. Large-basis search uses discrete '
             'adjoints, log-arrival initialization and 96-step backward Euler. Final candidates are ranked by matrix-exponential probabilities.',
             '- Final p is a sparse linear solve; final q is g-exp(TQ)g. These are numerical evaluations of '
             'the finite CTMC, not continuum certificates.',
             '- Euler-Maruyama uses bilinear drift interpolation and occupancy-cell absorption; risk trials '
             'continue to max(3T,45s). Unabsorbed trials yield [wall fraction, wall fraction + timeout fraction] '
             'bounds on eventual risk. Mean exit among observed exits is conditional; the restricted mean includes censoring.',
             '- No speed saturation or prescribed u_max constraint was imposed. Actual grid-node maximum commands '
             'are reported. No global infeasibility, continuum certification, or HJB proof is claimed.', '',
             f'- Wall mode: {getattr(args,"wall_safety","free")}. Repulsive mode enforces a wall potential floor, '
             'an uphill wall edge, and a non-outward commanded normal velocity at every grid wall face. '
             'This local condition is separate from the draft Eq. (5.2) risk certificate, which is audited separately.', '',
             '## Main numerical results', '',
             '| Maze | Spec (delta,epsilon,T) | p CTMC | Wall SDE | q CTMC | Arrival SDE | rho | CTMC / SDE CI pass |',
             '|---|---|---:|---:|---:|---:|---:|---|']
    for r in records:
        lines.append(f'| {r["maze"]} | {r["spec_name"]} ({r["delta"]},{r["epsilon"]},{r["T"]:g}) '
                     f'| {r["p"]:.5f} | {r["sde"]["collision_rate"]:.5f} | {r["q"]:.5f} '
                     f'| {r["sde"]["arrival_rate"]:.5f} | {r["rho"]:.5f} '
                     f'| {r["ctmc_pass"]} / {r["sde_ci_pass"]} |')
    lines += ['', 'All coefficients, confidence intervals, horizons, timeout rates, speeds, and optimizer messages '
              'are in `data/summary.json` and `data/summary.csv`.', '', '## Synthesized coefficients', '']
    for r in records:
        lines.append(f'- {r["maze"]}/{r["spec_name"]}: theta={np.round(r["theta"],5).tolist()}; '
                     f'max grid command={r["max_command_speed"]:.3f} m/s; '
                     f'timeout={r["sde"]["timeout_rate"]:.4f}.')
    lines += ['', '## Diagnosis and limits', '',
              '- A nonzero local-search violation is **UNDETERMINED**, not evidence of mathematical infeasibility. '
              'Narrow passages, strong harmonic screening far from the goal, limited boundary basis, bounded '
              'coefficients, and finite search budget can each cause failure.',
              '- Upwind CTMC adds drift-dependent artificial diffusion of order h. Its predicted risk can differ '
              'materially from SDE Monte Carlo even when the linear-system residual is tiny. '
              'Occupancy-cell face absorption vs absorbing grid nodes is another spatial error.',
              '- Euler-Maruyama and line-segment contact checks can miss between-step Brownian crossings; '
              'Monte Carlo intervals quantify sampling error only. Grid/time-step audits are saved separately.',
              '- In S1, S2, S4, and S5, delta=epsilon. Since q(T)<=1-p, satisfying the deadline condition '
              'already implies the risk condition. Only S3 imposes a strictly tighter separate risk budget. '
              'Consequently these specifications alone provide limited evidence that a joint objective outperforms time-only synthesis.',
              '- The optimizer may stop just inside a constraint. Such a CTMC pass can fail an empirical '
              'confidence-interval test; a deployment-oriented implementation should synthesize a margin.', '',
              '## Ablations', '', '| Maze | Controller | p CTMC | q CTMC | rho | Wall SDE | Arrival SDE |',
              '|---|---|---:|---:|---:|---:|---:|']
    for r in ablations:
        lines.append(f'| {r["maze"]} | {r["mode"]} | {r["p"]:.5f} | {r["q"]:.5f} | {r["rho"]:.5f} '
                     f'| {r["sde"]["collision_rate"]:.5f} | {r["sde"]["arrival_rate"]:.5f} |')
    lines += ['', '## Numerical audits', '',
              '| Maze | Exact p | CTMC MC wall | Exact q | CTMC MC arrival | BE-72 q | Half-step SDE arrival |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for audit in audits:
        be72 = next(v['q'] for v in audit['backward_euler'] if v['steps']==72)
        lines.append(f'| {audit["maze"]} | {audit["exact_p"]:.6f} | '
                     f'{audit["ctmc_mc"]["collision_rate"]:.6f} | {audit["exact_q"]:.6f} | '
                     f'{audit["ctmc_mc"]["arrival_rate"]:.6f} | {be72:.6f} | '
                     f'{audit["sde_half_step"]["arrival_rate"]:.6f} |')
    lines += ['', 'Full confidence intervals and time-step audits are in `data/audits.json`. '
              'Four numerical tests reproduce Appendix B, check exact jump Monte Carlo, '
              'verify harmonic/time-monotonicity invariants, and compare adjoints with independent finite differences '
              '(`python -m unittest -v test_numerics`).', '',
              '## Files and reproduction', '',
              '- Figures: `figures/*_potential.png`, `*_trajectories.png`, `specification_satisfaction.png`, '
              '`specification_sweep.png`, `ablations.png`, `runtime_scalability.png`.',
              '- Detailed results: `data/summary.csv`, `summary.json`, `ablations.json`, `runtime.json`, `audits.json`.',
              f'- Reproduce this run: `python main.py --suite full --grid {args.grid} '+(f'--basis {args.basis} ' if args.basis else '')+
              f'--mc {args.mc or 5000} --budget {args.budget} --seed {args.seed} '
              f'--wall-safety {getattr(args,"wall_safety","free")}` from `experiment/`.',
              '- Default pilot: `python main.py --suite pilot` (one maze, 1,000 SDE trials).', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')


def run(args):
    began = time.perf_counter()
    output = Path(args.output) if args.output else ROOT/'outputs'/('pilot' if args.suite=='pilot' else '')
    for name in ('figures','data','logs'):
        (output/name).mkdir(parents=True, exist_ok=True)
    n_mc = args.mc or (1000 if args.suite=='pilot' else 5000)
    specs = json.loads((ROOT/'configs'/'specifications.json').read_text())
    names = ['open'] if args.suite=='pilot' else ['open','u_maze','narrow']
    selected_specs = specs[:1] if args.suite=='pilot' else specs
    records, builders = [], {}
    save_json(output/'logs'/'environment.json', dict(arguments=vars(args), python=sys.version,
               numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform()))
    total = len(names)*len(selected_specs)
    for maze_idx,name in enumerate(names):
        maze = make_maze(name, args.grid)
        m=args.basis or (8 if name=='open' else 200)
        boundary=getattr(args,'boundary','auto')
        if boundary=='auto':
            boundary='contour' if m>=500 else 'local' if m>=50 else 'spatial'
        basis = build_basis(maze, m,boundary)
        builder = CTMCBuilder(basis, args.sigma, args.gain)
        wall_constraint=None
        if getattr(args,'wall_safety','free')=='repulsive':
            wall_constraint=repulsive_constraint(basis,args.gain,args.wall_floor,args.minimum_inward_speed)
        builders[name] = builder
        np.savez_compressed(output/'data'/f'{name}_basis.npz', values=basis.values, gradients=basis.gradients,
                            occupied=maze.occupied, goal=maze.goal, start_rc=maze.start_rc, length=maze.length)
        print(f'[{name}] states={len(maze.rc)}, basis={basis.seconds:.3f}s, Laplace residual={basis.residual:.2e}', flush=True)
        warm = None
        for spec_idx,spec in enumerate(selected_specs):
            print(f'[{len(records)+1}/{total}] {name}/{spec["name"]}: synthesis starts, elapsed={(time.perf_counter()-began)/60:.1f} min', flush=True)
            use_adjoint=wall_constraint is not None or getattr(args,'optimizer','auto')=='adjoint' or (getattr(args,'optimizer','auto')=='auto' and m>=50)
            if use_adjoint:
                result=synthesize_adjoint(builder,spec,seed=args.seed+maze_idx*100+spec_idx,
                                          maxiter=args.budget,theta_max=args.theta_max,warm_start=warm,
                                          wall_constraint=wall_constraint)
            else:
                result = synthesize(builder, spec, seed=args.seed+maze_idx*100+spec_idx,
                                    max_evals=args.budget, theta_max=args.theta_max, warm_start=warm)
            warm = result['theta']
            record = make_record(name, spec, result, builder, args, output,
                                 f'{name}_{spec["name"]}', n_mc)
            records.append(record)
            save_json(output/'data'/'summary.json', records); save_csv(output/'data'/'summary.csv', records)
    specification_figure(records, output/'figures')
    ablations, audits, runtime = [], [], []
    if args.suite == 'full':
        sweep_figure(records, output/'figures')
        for name in ['open','u_maze']:
            builder = builders[name]
            spec = specs[1]
            main_joint = next(r for r in records if r['maze']==name and r['spec_name']==spec['name'])
            ablations.append(dict(main_joint))
            for mode in ['fixed','risk_only','time_only']:
                print(f'[ablation] {name}/{mode}', flush=True)
                if mode == 'fixed':
                    theta = np.full(builder.basis.m, 2.)
                    result = dict(theta=theta, prediction=evaluate(builder.build(theta), builder.maze.start, spec['T']),
                                  mode=mode, optimization_seconds=0., evaluations=0)
                else:
                    constraint=None
                    if getattr(args,'wall_safety','free')=='repulsive':
                        constraint=repulsive_constraint(builder.basis,args.gain,args.wall_floor,args.minimum_inward_speed)
                    if builder.basis.m>=50 or constraint is not None:
                        result=synthesize_adjoint(builder,spec,seed=args.seed+700,mode=mode,
                                                  maxiter=args.budget,theta_max=args.theta_max,wall_constraint=constraint)
                    else:
                        result = synthesize(builder,spec, seed=args.seed+700, mode=mode, max_evals=args.budget,
                                            theta_max=args.theta_max)
                ablations.append(make_record(name,spec,result,builder,args,output,f'{name}_{mode}',n_mc,False))
        ablation_figure(ablations, output/'figures')
        # Independent jump-process MC and temporal refinement audit on one controller per maze.
        for name in names:
            builder = builders[name]
            row = next(r for r in records if r['maze']==name and r['spec_name']=='S2')
            chain = builder.build(row['theta'])
            jump_mc = simulate_ctmc(chain,builder.maze.start,row['T'],n=max(n_mc,5000),
                                    seed=args.seed+900,horizon=90.)
            row['ctmc_mc'] = jump_mc
            temporal = []
            for steps in (36,72,144):
                value = evaluate(chain,builder.maze.start,row['T'],'backward_euler',steps)
                temporal.append(dict(steps=steps,p=value['p'],q=value['q']))
            sde_fine,_ = simulate_sde(builder.basis,row['theta'],args.sigma,args.gain,row['T'],
                                      n=min(n_mc,5000),seed=args.seed+17,dt=row['sde']['dt']/2,save_paths=0)
            audits.append(dict(maze=name, exact_p=row['p'],exact_q=row['q'],ctmc_mc=jump_mc,
                               backward_euler=temporal,sde_half_step=sde_fine))
        for n in (24,40,60):
            for m in (6,8,12):
                maze = make_maze('open',n)
                basis = build_basis(maze,m)
                chain = CTMCBuilder(basis,args.sigma,args.gain).build(np.full(m,2.))
                value = evaluate(chain,maze.start,15.)
                runtime.append(dict(n=n,M=m,states=len(maze.rc),basis_seconds=basis.seconds,
                                    evaluation_seconds=value['evaluation_seconds'],p=value['p'],q=value['q']))
        runtime_figure(runtime,output/'figures')
    save_json(output/'data'/'summary.json',records); save_csv(output/'data'/'summary.csv',records)
    save_json(output/'data'/'ablations.json',ablations); save_csv(output/'data'/'ablations.csv',ablations)
    save_json(output/'data'/'audits.json',audits); save_json(output/'data'/'runtime.json',runtime)
    seconds = time.perf_counter()-began
    report(records,ablations,audits,runtime,args,seconds,output)
    print(f'DONE in {seconds/60:.2f} min. CTMC passes {sum(r["ctmc_pass"] for r in records)}/{len(records)}. Output: {output}',flush=True)
