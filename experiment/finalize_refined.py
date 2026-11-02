"""Project dense M500 fields onto exact face-endpoint constraints and revalidate."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
from pathlib import Path
import argparse,json,time,gc,hashlib
import numpy as np
from scipy.optimize import minimize
from src.maze import make_maze
from src.harmonic import build_basis,DIRECTIONS
from src.ctmc import CTMCBuilder
from src.wall_safety import repulsive_constraint
from src.probabilities import evaluate,violation
from src.simulate import simulate_sde,simulate_ctmc,DriftInterpolator
from src.experiments import make_record,save_json,save_csv
from src.plotting import maze_background,COLORS
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
RAW=ROOT/'outputs'/'refined_M500'
OUT=RAW/'final'


def face_audit(maze,drift):
    interpolate=DriftInterpolator(maze,drift)
    values=[]
    for dr,dc in DIRECTIONS:
        nr,nc=maze.rc[:,0]+dr,maze.rc[:,1]+dc
        hit=maze.occupied[nr,nc]; r,c=maze.rc[hit].T; wr,wc=nr[hit],nc[hit]
        for tangent in np.linspace(-.5,.5,11):
            positions=np.column_stack(((c+wc)/2-tangent*dr,(r+wr)/2+tangent*dc))*maze.h
            velocity=interpolate(positions)
            values.extend(velocity[:,0]*dc+velocity[:,1]*dr)
    values=np.array(values)
    return dict(sample_count=len(values),max_outward=float(values.max()),
                outward_fraction=float(np.mean(values>1e-7)))


def run():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mazes',nargs='+',default=['u_maze','narrow'])
    args=parser.parse_args()
    for name in ('data','logs','figures'):
        (OUT/name).mkdir(parents=True,exist_ok=True)
    sources=['src/harmonic.py','src/boundary_contours.py','src/ctmc.py','src/probabilities.py',
             'src/simulate.py','src/wall_safety.py','finalize_refined.py']
    save_json(OUT/'logs'/'source_hashes.json',{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources})
    specs=dict(name='S2',T=15.,delta=.05,epsilon=.05)
    params=argparse.Namespace(sigma=.2,gain=1.,seed=20261101,wall_safety='repulsive',wall_floor=1.,minimum_inward_speed=0.)
    for name in args.mazes:
        raw=json.loads((RAW/'data'/f'{name}_M500.json').read_text())
        theta0=np.array(raw['theta'])
        basis=build_basis(make_maze(name,100),500,'contour'); maze=basis.maze
        builder=CTMCBuilder(basis,.2,1.)
        constraint=repulsive_constraint(basis,wall_floor=1.,interpolated=True)
        before=face_audit(maze,basis.field(theta0)[1])
        start=time.perf_counter()
        theta=np.maximum(theta0,1.)
        initial_u=-np.einsum('ndi,i->nd',builder.gradient,theta,optimize=False)
        active=set(np.argsort(np.linalg.norm(initial_u,axis=1))[-32:].tolist())
        # A few active velocity cones avoid materializing an N-by-M speed
        # Jacobian; all N grid speeds are rechecked after each convex solve.
        for velocity_round in range(8):
            gradient=builder.gradient[np.array(sorted(active))]
            def speed_value(t):
                u=-np.einsum('ndi,i->nd',gradient,t,optimize=False)
                return 1-np.sum(u*u,axis=1)/100.
            def speed_jac(t):
                u=-np.einsum('ndi,i->nd',gradient,t,optimize=False)
                return 2*np.einsum('ndi,nd->ni',gradient,u,optimize=False)/100.
            projection=minimize(lambda t:(.5*float(np.sum((t-theta0)**2)),t-theta0),
                                theta,jac=True,method='SLSQP',bounds=[(1.,80.)]*500,
                                constraints=[constraint,dict(type='ineq',fun=speed_value,jac=speed_jac)],
                                options=dict(maxiter=30,ftol=1e-12))
            theta=projection.x
            full_u=-np.einsum('ndi,i->nd',builder.gradient,theta,optimize=False)
            full_speed=np.linalg.norm(full_u,axis=1)
            if full_speed.max()<=10.+1e-6:
                break
            active.update(np.argsort(full_speed)[-32:].tolist())
        slack=constraint.physical_A@theta-constraint.physical_lower
        if slack.min() < -1e-7 or builder.build(theta).max_speed>10.+1e-6:
            raise ArithmeticError(f'Final projection failed: slack={slack.min()}')
        potential,drift=basis.field(theta)
        after=face_audit(maze,drift)
        if after['max_outward']>1e-7:
            raise ArithmeticError(f'Actual face audit failed: {after}')
        projected_seconds=time.perf_counter()-start
        prediction=evaluate(builder.build(theta),maze.start,15.)
        print(name,'projected:',before,after,'coefficient change',np.max(np.abs(theta-theta0)),
              'CTMC',prediction,flush=True)
        result=dict(theta=theta,prediction=prediction,mode='joint_exact_face_projection',
                    evaluations=raw['evaluations']+projection.nfev,
                    optimization_seconds=raw['optimization_seconds']+projected_seconds,
                    optimizer_message=raw['optimizer_message']+'; nearest-field convex wall projection: '+str(projection.message))
        record=make_record(name,specs,result,builder,params,OUT,f'{name}_M500',5000)
        deterministic,path=simulate_sde(basis,theta,0.,1.,15.,n=1,seed=params.seed,save_paths=1)
        fine,_=simulate_sde(basis,theta,.2,1.,15.,n=2000,seed=params.seed+901,
                            dt=record['sde']['dt']/2,save_paths=0)
        jump=simulate_ctmc(builder.build(theta),maze.start,15.,n=5000,seed=params.seed+900,horizon=90.)
        record.update(grid=100,M=500,u_max=10.,theta_max=80.,theta_min=1.,boundary_rank=500,
                      boundary_metadata=basis.boundary_metadata,basis_residual=basis.residual,
                      actual_wall_face_audit=after,preprojection_face_audit=before,
                      projection_seconds=projected_seconds,
                      projection_max_coefficient_change=float(np.max(np.abs(theta-theta0))),
                      deterministic=deterministic,sde_half_step=fine,ctmc_mc=jump,
                      collision_diagnostic='Last precontact command dotted with the occupied-square entry-face normal; not a complete causal attribution')
        record['deadlines']=[]
        for T in (7.,10.,15.):
            value=evaluate(builder.build(theta),maze.start,T)
            record['deadlines'].append(dict(T=T,p=value['p'],q=value['q'],rho=violation(value['p'],value['q'],.05,.05)))
        save_json(OUT/'data'/f'{name}_M500.json',record)
        np.savez_compressed(OUT/'data'/f'{name}_field.npz',potential=potential,drift=drift,theta=theta,
                            occupied=maze.occupied,goal=maze.goal,start_rc=maze.start_rc,length=maze.length,
                            deterministic_path=path['paths'][0])
        del constraint,basis,builder,potential,drift
        gc.collect()
    records=[json.loads(p.read_text()) for p in sorted((OUT/'data').glob('*_M500.json'))]
    save_json(OUT/'data'/'summary.json',records); save_csv(OUT/'data'/'summary.csv',records)
    create_overview(records)
    create_report(records)
    print('FINAL EXACT-FACE VALIDATION COMPLETE',flush=True)


def create_overview(records):
    fig,axes=plt.subplots(2,len(records),figsize=(6.2*len(records),10.1),squeeze=False)
    section,section_axes=plt.subplots(1,len(records),figsize=(6.2*len(records),4.),squeeze=False)
    for col,record in enumerate(records):
        name=record['maze']; maze=make_maze(name,100)
        data=np.load(OUT/'data'/f'{name}_field.npz')
        potential,drift=data['potential'],data['drift']
        from scipy.ndimage import binary_dilation
        wall=maze.occupied & binary_dilation(maze.interior,structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
        ax=axes[0,col]
        im=ax.imshow(np.ma.masked_where(~maze.interior,potential),origin='lower',
                      extent=[-maze.h/2,maze.length+maze.h/2]*2,cmap='viridis',vmin=0,
                      vmax=float(potential[wall | maze.interior].max()))
        maze_background(ax,maze)
        ax.scatter(maze.x[wall],maze.y[wall],c=potential[wall],cmap='viridis',norm=im.norm,s=8,marker='s',zorder=7)
        take=np.zeros_like(maze.interior); take[3::7,3::7]=True; take &= maze.interior
        u=drift[take]; u=u/np.maximum(np.linalg.norm(u,axis=1,keepdims=True),1e-12)
        ax.quiver(maze.x[take],maze.y[take],u[:,0],u[:,1],color='white',scale=25,alpha=.65,zorder=8)
        ax.set_title(f'{name}: 100x100, M=500\nColored wall edges show wall potential')
        fig.colorbar(im,ax=ax,label='Potential',fraction=.045,pad=.03)
        ax=axes[1,col]
        paths=np.load(OUT/'data'/f'{name}_M500_trajectories.npz')
        for j,(a,b) in enumerate(zip(paths['offsets'][:-1],paths['offsets'][1:])):
            points=paths['points'][a:b]; points=points[points[:,2]<=15.]
            if len(points):
                color=COLORS[{0:'timeout',1:'wall',2:'goal'}[int(paths['status'][j])]]
                ax.plot(points[:,0],points[:,1],color=color,alpha=.32,lw=.7)
        det=data['deterministic_path']; det=det[det[:,2]<=15.]
        if len(det): ax.plot(det[:,0],det[:,1],'k--',lw=1.8,label='No-noise trajectory',zorder=6)
        maze_background(ax,maze)
        ax.set_title(f'100 SDE trajectories; 5,000-trial arrival={record["sde"]["arrival_rate"]:.2%}\n'
                     f'wall={record["sde"]["collision_rate"]:.2%}; max command={record["max_command_speed"]:.2f} m/s')
        ax.legend(fontsize=8)
        # A cross-corridor cut displays potential and lateral drift separately.
        ax=section_axes[0,col]; r=int(np.argmin(np.abs(maze.y[:,0]-2.0)))
        visible=maze.interior[r] | wall[r]
        ax.plot(maze.x[r],np.where(visible,potential[r],np.nan),color='#315ca0',label='Potential')
        ax.scatter(maze.x[r,wall[r]],potential[r,wall[r]],marker='s',s=22,color='#303944',label='Wall boundary potential')
        other=ax.twinx()
        other.plot(maze.x[r],np.where(maze.interior[r],drift[r,:,0],np.nan),color='#ce683c',label='Lateral command u_x')
        other.axhline(0,color='#ce683c',ls=':',alpha=.5); other.set_ylabel('u_x (m/s)',color='#ce683c')
        ax.set(title=f'{name}: cut across opposing walls at y={maze.y[r,0]:.2f} m',xlabel='x (m)',ylabel='Potential')
        ax.grid(alpha=.2); ax.legend(fontsize=8,loc='upper left')
    fig.suptitle('Refined harmonic navigation\nWall potential, field direction and 100 trajectories',fontsize=15)
    fig.tight_layout(rect=[0,0,1,.94]); fig.savefig(OUT/'figures'/'overview.png',bbox_inches='tight'); plt.close(fig)
    section.tight_layout(); section.savefig(OUT/'figures'/'opposing_wall_cross_sections.png'); plt.close(section)


def create_report(records):
    lines=['# U maze / Narrow: final M=500 dense-grid validation','',
           'Only the two requested mazes were rerun. Grid=100x100, h=0.040404 m, M=500, sigma=0.2, k=1. '
           'The first prototype used 40x40 and M=8; these final results also correct the interpolation and boundary representation.', '',
           '## Concrete corrections after rereading the draft','',
           '- Sections 3 and 4.1: periodic continuous piecewise-linear wall boundary functions, separate along each contour, '
           'zero goal potential, sparse harmonic extensions. Both boundary matrices have rank 500. '
           'Opposite sides of a wall are not mixed by a Euclidean radial kernel.',
           '- The SDE formerly interpolated fictitious gradients inside solid cells. Now it uses only free-space gradients, '
           'with normalized nonnegative interpolation weights. No inverse-distance/nonharmonic barrier was added.',
           '- Hard wall constraints enforce wall >= adjacent free potential and non-outward actual interpolated commands. '
           'Final projection includes exact face endpoints/corners. Each final field is independently sampled at 11 points '
           'on every wall face, including endpoints. The earlier near-endpoint constraints left tiny corner leaks; '
           'those have been removed before the final Monte Carlo.',
           '- Final coefficients lie in [1,80], so the continuous hat-function wall trace is >=1, not merely sampled wall nodes. '
           'This coefficient range is a modeling choice, not a number specified by the paper.',
           '- Section 3 command bound: ||u||<=10 m/s at every interior node. Convex free-only interpolation preserves this bound. '
           'The 10 m/s value is an explicit experimental choice. The deployed command is not clipped.',
           '- Sections 4.4 and 4.5: killed upwind CTMC, separate eventual-wall and finite-deadline backward equations, '
           'discrete adjoints and direct rho minimization after log-arrival initialization. '
           'Final probabilities are recomputed with the matrix exponential.',
           '- The wall-start potential margin in Eq. (5.2) is a separate sufficient certificate. '
           'Its pass flag is reported honestly; local repulsion does not imply this certificate or the stochastic specifications.', '',
           '## Final quantitative results', '',
           '| Maze | p CTMC | Wall SDE | q(15) CTMC | Arrival SDE | Half-step arrival | No-noise goal by 15s | Max command | Actual outward wall samples |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f'| {r["maze"]} | {r["p"]:.6f} | {r["sde"]["collision_rate"]:.6f} | '
                     f'{r["q"]:.6f} | {r["sde"]["arrival_rate"]:.6f} | '
                     f'{r["sde_half_step"]["arrival_rate"]:.6f} | {r["deterministic"]["arrival_rate"]:.0f} | '
                     f'{r["max_command_speed"]:.5f} | {r["actual_wall_face_audit"]["outward_fraction"]:.3g} |')
    lines+=['','SDE main sample: 5,000 per maze. Half-step sensitivity: 2,000 per maze. '
            'Independent exact CTMC jump simulation: 5,000 per maze. All Wilson intervals, theta, timeouts, '
            'exit times, audit extrema, optimizer messages and 7/10/15-second evaluations are in `data/summary.json`.', '',
            '## Interpretation of opposite-wall collisions','',
            'The old interpolation defect could create wall-directed commands even when node-only tests passed. '
            'That was an implementation error. Final drift audits include actual physical pixel-face corners, '
            'and deterministic trajectories test the noise-free field. SDE step-size comparisons test numerical overshoot. '
            'A collision despite a non-outward boundary command is possible with Brownian disturbance and a finite barrier. '
            'A last-step directional diagnostic alone is not a complete attribution of a trajectory to an opposing wall.', '',
            '## Limits and conclusion','',
            'The requested S2 target remains wall risk <=5% and arrival by 15s >=95%. '
            'A failed candidate is UNDETERMINED, not a global impossibility result. '
            'Upwind CTMC artificial diffusion and node-vs-cell-face absorption still create spatial model error. '
            'Finite-step SDE and Monte Carlo intervals are not continuum certificates. '
            'M, grid, boundary basis, interpolation, wall constraints, coefficient range and speed bound all changed; '
            'these results must not be presented as an isolated M-only ablation.', '',
            '## Reproduction','',
            '```powershell','python refined_M500.py','python finalize_refined.py',
            'python polish_U_M500.py','python finalize_refined.py --mazes u_maze',
            'python -m unittest -v test_numerics','```','',
            'The saved final theta/fields reproduce evaluation directly; reoptimization can choose another local candidate. '
            'Source hashes for final validation are in `logs/source_hashes.json`.', '',
            '![Potential fields and 100 trajectories](figures/overview.png)','',
            '![Opposing-wall cross sections](figures/opposing_wall_cross_sections.png)','']
    lines+=['## Deterministic exit details','']
    for r in records:
        d=r['deterministic']
        lines.append(f'- {r["maze"]}: wall fraction={d["collision_rate"]:.0f}; '
                     f'eventual goal fraction within audit horizon={d["eventual_goal_rate"]:.0f}; '
                     f'exit time={d["mean_exit_observed"]}; deadline arrival={d["arrival_rate"]:.0f}.')
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    run()
