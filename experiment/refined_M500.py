"""Paper-aligned dense rerun: only U/narrow, 100x100 grid and M=500."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
from pathlib import Path
import argparse
import json
import time
import gc
import numpy as np
from scipy.optimize import lsq_linear
from scipy.ndimage import map_coordinates
from src.maze import make_maze
from src.harmonic import build_basis,geodesic_boundary_weights
from src.ctmc import CTMCBuilder
from src.adjoint import synthesize_adjoint
from src.wall_safety import repulsive_constraint,interpolated_face_normals,audit_wall
from src.simulate import simulate_sde,simulate_ctmc,DriftInterpolator
from src.experiments import save_json,save_csv,make_record
from src.plotting import maze_background,COLORS
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent


def run():
    parser=argparse.ArgumentParser()
    parser.add_argument('--grid',type=int,default=100)
    parser.add_argument('--basis',type=int,default=500)
    parser.add_argument('--maxiter',type=int,default=60)
    parser.add_argument('--mc',type=int,default=5000)
    parser.add_argument('--theta-max',type=float,default=80.)
    parser.add_argument('--u-max',type=float,default=10.)
    parser.add_argument('--steps',type=int,default=96)
    parser.add_argument('--output',default='outputs/refined_M500')
    parser.add_argument('--mazes',nargs='+',default=['u_maze','narrow'],choices=['u_maze','narrow'])
    args=parser.parse_args()
    args.sigma=.2; args.gain=1.; args.seed=20261101
    args.wall_safety='repulsive'; args.wall_floor=1.; args.minimum_inward_speed=0.
    output=Path(args.output)
    for folder in ('data','logs','figures'):
        (output/folder).mkdir(parents=True,exist_ok=True)
    save_json(output/'logs'/'configuration.json',vars(args))
    spec=dict(name='S2',delta=.05,epsilon=.05,T=15.)
    original=json.loads((ROOT/'outputs'/'wall_safety'/'data'/'repulsive_results.json').read_text())
    results=[]; began=time.perf_counter()
    for name in args.mazes:
        print(f'[{name}] BUILD {args.grid}x{args.grid}, M={args.basis}',flush=True)
        basis=build_basis(make_maze(name,args.grid),args.basis,'contour')
        maze=basis.maze
        builder=CTMCBuilder(basis,args.sigma,args.gain)
        mask=np.any(basis.boundary_weights>0,axis=2)
        boundary=basis.boundary_weights[mask]
        singular=np.linalg.svd(boundary,compute_uv=False)
        rank=int(np.sum(singular>singular[0]*1e-9))
        if rank!=args.basis:
            raise ArithmeticError(f'Boundary rank {rank} < M={args.basis}')
        oldrow=next(r for r in original if r['maze']==name)
        oldbasis=build_basis(make_maze(name,40),200,'local')
        oldpotential,_=oldbasis.field(oldrow['theta'])
        coordinate=np.array([maze.y[mask]/oldbasis.maze.h,maze.x[mask]/oldbasis.maze.h])
        oldtarget=map_coordinates(oldpotential,coordinate,order=1,mode='nearest')
        warm_old=lsq_linear(boundary,oldtarget,bounds=(1.,args.theta_max)).x
        distance=geodesic_boundary_weights(maze,2)[mask,1]
        warm_geo=lsq_linear(boundary,1.+min(14.,args.theta_max-1)*distance,bounds=(1.,args.theta_max)).x
        del oldbasis,oldpotential,boundary
        gc.collect()
        constraint=repulsive_constraint(basis,args.gain,1.,0.,interpolated=True)
        print(f'[{name}] basis residual={basis.residual:.2e}; rank={rank}; '
              f'contours={basis.boundary_metadata}; hard interpolated-wall constraints={len(constraint.lb)}, '
              f'u_max={args.u_max}',flush=True)
        result=synthesize_adjoint(builder,spec,seed=args.seed,theta_max=args.theta_max,
                                  maxiter=args.maxiter,steps=args.steps,wall_constraint=constraint,
                                  u_max=args.u_max,initial_guesses=[warm_old,warm_geo],progress_every=15,
                                  checkpoint_path=output/'logs'/f'{name}_checkpoint.json')
        record=make_record(name,spec,result,builder,args,output,f'{name}_M500',args.mc)
        theta=result['theta']
        potential,drift=basis.field(theta)
        rows,positions,normals=interpolated_face_normals(basis)
        actual=DriftInterpolator(maze,drift)(positions)
        outward=np.sum(actual*normals,axis=1)
        if outward.max()>1e-6 or builder.build(theta).max_speed>args.u_max+1e-6:
            raise ArithmeticError('Final actual interpolation/speed audit failed')
        record.update(grid=args.grid,M=args.basis,boundary_rank=rank,
                      boundary_condition_number=float(singular[0]/singular[-1]),
                      boundary_metadata=basis.boundary_metadata,u_max=args.u_max,theta_max=args.theta_max,
                      interpolated_outward_face_fraction=float(np.mean(outward>1e-7)),
                      interpolated_max_outward_speed=float(outward.max()),basis_residual=basis.residual,
                      protocol='contour hats; interior-only drift; hard wall and speed constraints')
        deterministic,detpath=simulate_sde(basis,theta,0.,args.gain,15.,n=1,seed=args.seed,save_paths=1)
        record['deterministic']=deterministic
        record['ctmc_mc']=simulate_ctmc(builder.build(theta),maze.start,15.,n=args.mc,seed=args.seed+900,horizon=90.)
        fine,_=simulate_sde(basis,theta,.2,1.,15.,n=min(2000,args.mc),seed=args.seed+901,
                            dt=record['sde']['dt']/2,save_paths=0)
        record['sde_half_step']=fine
        # Evaluate all deadlines on this controller; these are not independent resyntheses.
        from src.probabilities import evaluate,violation
        sweep=[]
        for deadline in (7.,10.,15.):
            val=evaluate(builder.build(theta),maze.start,deadline)
            sweep.append(dict(T=deadline,p=val['p'],q=val['q'],rho=violation(val['p'],val['q'],.05,.05)))
        record['deadline_evaluation']=sweep
        np.savez_compressed(output/'data'/f'{name}_field.npz',potential=potential,drift=drift,
                            theta=theta,occupied=maze.occupied,goal=maze.goal,start_rc=maze.start_rc,
                            length=maze.length,deterministic_path=detpath['paths'][0],
                            boundary_weights=basis.boundary_weights[mask],boundary_rc=np.column_stack(np.where(mask)))
        results.append(record)
        save_json(output/'data'/'summary.json',results); save_csv(output/'data'/'summary.csv',results)
        save_json(output/'data'/f'{name}_M500.json',record)
        print(f'[{name}] COMPLETE elapsed={(time.perf_counter()-began)/60:.2f}min; '
              f'interpolated outward={record["interpolated_outward_face_fraction"]}; '
              f'deterministic arrival={deterministic["arrival_rate"]}',flush=True)
        del constraint,basis,builder,rows,positions,normals,actual,potential,drift
        gc.collect()
    make_report(output,results,time.perf_counter()-began,args)
    print('REFINED M500 SUITE COMPLETE',flush=True)


def make_report(output,results,seconds,args):
    lines=['# U / Narrow: refined M=500 rerun','',
           f'Grid: {args.grid}x{args.grid}; M={args.basis}; sigma=.2; k=1; '
           f'coefficient bounds [0,{args.theta_max}]; sampled command bound {args.u_max} m/s. '
           f'Runtime: {seconds/60:.2f} min. Both mazes only.', '',
           '## Paper alignment and implementation correction','',
           '- Rechecked Sections 3, 4 and 5 plus numerical-verification appendix of `../../../output/pdf/main.pdf`.',
           '- Continuous periodic piecewise-linear functions along each actual wall contour, as allowed in Section 4.1. '
           'The two sides of an obstacle are not coupled solely by Euclidean proximity. Zero goal Dirichlet data; '
           'sparse Laplace harmonic extension; u=-k grad V; no inverse-distance/nonharmonic term.',
           '- Fixed a real interpolation defect: earlier SDE evaluation mixed gradients inside solid obstacles into free-space commands. '
           'Now only free-node gradients enter a normalized convex interpolation. This change is linear in theta.',
           '- Wall potential floor, uphill wall edge, and non-outward commands at endpoints and midpoints of every pixel wall face '
           'are hard linear constraints. The actual simulation interpolator is rechecked after optimization.',
           '- The paper command bound is now explicit and checked at all interior nodes. Convex free-only interpolation '
           'cannot exceed the maximum grid-node norm. Command clipping is not used.',
           '- Independent adjoints of the time-stepping equations guide optimization; final CTMC probabilities '
           'are recomputed using a matrix exponential. Exact CTMC jump MC, SDE half-step checks and deterministic '
           'trajectories separate model, stepping and field-direction effects.',
           '- Compared with previous runs, M, grid, boundary basis, interpolation, coefficient range and speed constraints changed. '
           'The performance change cannot be attributed to M alone.', '',
           '| Maze | p CTMC | Wall SDE | q(15) CTMC | Arrival SDE | Half-step arrival | Deterministic arrival | Actual outward wall commands |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f'| {r["maze"]} | {r["p"]:.6f} | {r["sde"]["collision_rate"]:.6f} | '
                     f'{r["q"]:.6f} | {r["sde"]["arrival_rate"]:.6f} | '
                     f'{r["sde_half_step"]["arrival_rate"]:.6f} | '
                     f'{r["deterministic"]["arrival_rate"]:.0f} | {r["interpolated_outward_face_fraction"]:.3g} |')
    lines+=['','S2 target: collision <=.05 and safe arrival by 15s >=.95. '
            'A failed local solve is UNDETERMINED, not infeasibility. Local repulsion is separate from '
            'the sufficient global wall-start potential margin in draft Eq. (5.2); its pass flag is in each wall audit. '
            'These are numerical discrete-model and empirical results, not verified continuum guarantees.', '',
            '## Collision interpretation','',
            'The deterministic trial tests whether the field itself reaches a wall or the goal. '
            'The stochastic collision diagnostic records the last precontact command component toward the hit cell; '
            'it is an instantaneous diagnostic, not proof of a complete opposite-wall causal mechanism. '
            'Half-step trials test sensitivity to numerical crossing/overshoot. See all values and intervals in `data/summary.json`.', '',
            '```powershell','python refined_M500.py','python -m unittest -v test_numerics','```','']
    for r in results:
        name=r['maze']; lines +=[f'![{name} potential](figures/{name}_M500_potential.png)','',
                                 f'![{name} 100 trajectories](figures/{name}_M500_trajectories.png)','']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    run()
