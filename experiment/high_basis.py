"""Higher-dimensional localized wall bases and adjoint probability synthesis."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
from pathlib import Path
import argparse
import json
import time
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.optimize import lsq_linear
from src.maze import make_maze
from src.harmonic import build_basis,geodesic_boundary_weights
from src.ctmc import CTMCBuilder
from src.adjoint import synthesize_adjoint
from src.experiments import make_record,save_json,save_csv
from src.probabilities import evaluate,violation
from src.simulate import simulate_ctmc
from src.wall_safety import repulsive_constraint
import matplotlib.pyplot as plt


def run():
    parser=argparse.ArgumentParser()
    parser.add_argument('--basis',type=int,nargs='+',default=[50,100,8])
    parser.add_argument('--mazes',nargs='+',default=['u_maze','narrow'])
    parser.add_argument('--grid',type=int,default=40)
    parser.add_argument('--maxiter',type=int,default=160)
    parser.add_argument('--mc',type=int,default=5000)
    parser.add_argument('--steps',type=int,default=96)
    parser.add_argument('--output',default=None)
    parser.add_argument('--wall-safety',choices=['repulsive','free'],default='repulsive')
    parser.add_argument('--wall-floor',type=float,default=1.)
    parser.add_argument('--minimum-inward-speed',type=float,default=0.)
    parser.add_argument('--theta-max',type=float,default=8.)
    args=parser.parse_args()
    args.sigma=.2; args.gain=1.; args.seed=20261101
    root=Path(__file__).resolve().parent
    output=Path(args.output) if args.output else root/'outputs'/('high_basis_repulsive' if args.wall_safety=='repulsive' else 'high_basis')
    for name in ('data','logs','figures'):
        (output/name).mkdir(parents=True,exist_ok=True)
    save_json(output/'logs'/'arguments.json',vars(args))
    spec=dict(name='S2',delta=.05,epsilon=.05,T=15.)
    records=[]; began=time.perf_counter()
    specs=json.loads((root/'configs'/'specifications.json').read_text())
    for name in args.mazes:
        for m in args.basis:
            stem=f'{name}_local_M{m}'
            basis=build_basis(make_maze(name,args.grid),m,'local')
            builder=CTMCBuilder(basis,args.sigma,args.gain)
            mask=basis.maze.occupied & binary_dilation(basis.maze.interior,
                     structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
            c=basis.boundary_weights[mask]
            singular=np.linalg.svd(c,compute_uv=False)
            rank=int(np.sum(singular>singular[0]*1e-8))
            target=.05*args.theta_max+.95*args.theta_max*geodesic_boundary_weights(basis.maze,2)[mask,1]
            warm=lsq_linear(c,target,bounds=(0.,args.theta_max)).x
            constraint=None
            if args.wall_safety=='repulsive':
                constraint=repulsive_constraint(basis,args.gain,args.wall_floor,args.minimum_inward_speed)
            print(f'[{stem}] wall nodes={mask.sum()}, effective boundary rank={rank}/{m}, '
                  f'basis={basis.seconds:.3f}s; optimization starts',flush=True)
            result=synthesize_adjoint(builder,spec,seed=args.seed,theta_max=args.theta_max,maxiter=args.maxiter,
                                      steps=args.steps,warm_start=warm,wall_constraint=constraint)
            record=make_record(name,spec,result,builder,args,output,stem,args.mc)
            record.update(M=m,boundary_basis='local',boundary_rank=rank,wall_safety_mode=args.wall_safety,
                          boundary_condition_number=float(singular[0]/singular[-1]),grid=args.grid,
                          basis_seconds=basis.seconds)
            record['ctmc_mc']=simulate_ctmc(builder.build(result['theta']),builder.maze.start,15.,
                                             n=args.mc,seed=args.seed+900,horizon=90.)
            # Reuse this S2-synthesized field to evaluate all requested specifications.
            controller_specs=[]
            for specification in specs:
                value=evaluate(builder.build(result['theta']),builder.maze.start,specification['T'])
                controller_specs.append(dict(**specification,p=value['p'],q=value['q'],
                        rho=violation(value['p'],value['q'],specification['delta'],specification['epsilon'])))
            record['fixed_controller_spec_evaluation']=controller_specs
            records.append(record)
            save_json(output/'data'/f'{stem}.json',record)
            save_json(output/'data'/'summary.json',records); save_csv(output/'data'/'summary.csv',records)
            print(f'[{stem}] DONE after {(time.perf_counter()-began)/60:.2f} minutes',flush=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4.))
    for name in args.mazes:
        rows=sorted([r for r in records if r['maze']==name],key=lambda r:r['M'])
        axes[0].plot([r['M'] for r in rows],[r['p'] for r in rows],'o-',label=name)
        axes[1].plot([r['M'] for r in rows],[r['q'] for r in rows],'o-',label=name)
    for ax,title,threshold in zip(axes,['Wall-first risk','Safe arrival by 15 s'],[.05,.95]):
        ax.axhline(threshold,color='black',ls='--',label='S2 threshold')
        ax.set(xlabel='Localized boundary basis size M',ylabel='CTMC probability',title=title,ylim=(-.03,1.03))
        ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output/'figures'/'basis_size_comparison.png'); plt.close(fig)
    lines=['# Higher-dimensional harmonic boundary experiment','',
           'This supersedes conclusions drawn only from the initial M=8 broad boundary basis. '
           'Localized wall functions are placed by farthest-point sampling on actual wall boundary nodes. '
           'Their bandwidth decreases with center spacing. All cases use exact discrete adjoints, '
           'negative-log-arrival initialization followed by direct maximum specification violation. '
           'Final probabilities use the matrix exponential; SDE and exact CTMC Monte Carlo use 5,000 trials each.',
           '',f'Total higher-basis suite runtime: {(time.perf_counter()-began)/60:.2f} minutes.',
           '', '| Maze | M | Boundary rank | p CTMC | Wall SDE | q CTMC | Arrival SDE | rho | CTMC pass |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in sorted(records,key=lambda r:(r['maze'],r['M'])):
        lines.append(f'| {r["maze"]} | {r["M"]} | {r["boundary_rank"]} | {r["p"]:.5f} '
                     f'| {r["sde"]["collision_rate"]:.5f} | {r["q"]:.5f} | {r["sde"]["arrival_rate"]:.5f} '
                     f'| {r["rho"]:.5f} | {r["ctmc_pass"]} |')
    lines+=['',f'These tests keep sigma=0.2, coefficient bounds [0,{args.theta_max}], the same geometry and T=15. '
            f'Wall safety mode: {args.wall_safety}; wall floor={args.wall_floor}, minimum inward speed={args.minimum_inward_speed}. '
            'M changes representation; the local optimizer is still not a global feasibility test. '
            'Grid probabilities and SDE empirical frequencies remain distinct models. '
            'No command-speed limit is imposed. Coefficients, actual grid maximum commands, '
            'confidence intervals and all five specification evaluations are saved in data/summary.json.',
            '', '![Basis size comparison](figures/basis_size_comparison.png)', '']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('HIGH BASIS SUITE COMPLETE',flush=True)


if __name__=='__main__':
    run()
