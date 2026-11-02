"""Generate compact overview, mesh audit and final report after the full suite."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
from pathlib import Path
import argparse
import json
import re
import copy
import numpy as np
import matplotlib.pyplot as plt
from src.maze import make_maze
from src.harmonic import build_basis
from src.ctmc import CTMCBuilder
from src.probabilities import evaluate,violation
from src.simulate import wilson
from src.experiments import save_json,save_csv,report
from src.plotting import maze_background,COLORS,ablation_figure

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'


def enrich(records):
    for record in records:
        for key in ('sde','ctmc_mc'):
            if key not in record:
                continue
            metric=record[key]
            metric['risk_upper_ci']=wilson(round(metric['risk_upper']*metric['n']),metric['n'])
            for ci in ('collision_ci','arrival_ci'):
                metric[ci]=[float(np.clip(v,0,1)) for v in metric[ci]]
        metric=record['sde']
        record['sde_ci_pass']=metric['risk_upper_ci'][1]<=record['delta'] and metric['arrival_ci'][0]>=1-record['epsilon']
    return records


def overview(records):
    fig,axes=plt.subplots(2,3,figsize=(14.5,9.0))
    for col,name in enumerate(('open','u_maze','narrow')):
        row=next(r for r in records if r['maze']==name and r['spec_name']=='S2')
        maze=make_maze(name,40)
        basis=build_basis(maze,8)
        potential,_=basis.field(row['theta'])
        ax=axes[0,col]
        im=ax.imshow(np.ma.masked_where(~maze.interior,potential),origin='lower',
                     extent=[-maze.h/2,maze.length+maze.h/2]*2,cmap='viridis',vmin=0,vmax=8)
        maze_background(ax,maze)
        ax.set_title(f'{name}: harmonic potential')
        ax=axes[1,col]
        data=np.load(OUT/'data'/f'{name}_S2_trajectories.npz')
        for j,(a,b) in enumerate(zip(data['offsets'][:-1],data['offsets'][1:])):
            points=data['points'][a:b]
            points=points[points[:,2]<=row['T']]
            if len(points):
                color=COLORS[{0:'timeout',1:'wall',2:'goal'}[int(data['status'][j])]]
                ax.plot(points[:,0],points[:,1],color=color,alpha=.4,lw=.7)
        maze_background(ax,maze)
        ax.set_title(f'100 paths; SDE arrival={row["sde"]["arrival_rate"]:.1%}\n'
                     f'CTMC p={row["p"]:.4f}, q(15)={row["q"]:.4f}')
    fig.suptitle('First-pass prototype: successes and failures\nS2: wall risk <= 5%, safe arrival by 15 s >= 95%',fontsize=15)
    fig.tight_layout(rect=[0,0,.95,.93])
    cax=fig.add_axes([.96,.55,.012,.33]); fig.colorbar(im,cax=cax,label='Potential')
    fig.savefig(OUT/'figures'/'overview.png',bbox_inches='tight'); plt.close(fig)


def mesh_audit(records):
    rows=[]
    for name in ('open','u_maze','narrow'):
        controller=next(r for r in records if r['maze']==name and r['spec_name']=='S2')
        for n in (40,60,80):
            basis=build_basis(make_maze(name,n),8)
            builder=CTMCBuilder(basis,.2,1.)
            predicted=evaluate(builder.build(controller['theta']),builder.maze.start,15.)
            rows.append(dict(maze=name,n=n,h=builder.maze.h,theta=controller['theta'],**predicted))
            print(f'mesh {name} {n}: p={predicted["p"]:.6g}, q={predicted["q"]:.6g}',flush=True)
    save_json(OUT/'data'/'mesh_audit.json',rows)
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    for name in ('open','u_maze','narrow'):
        subset=[r for r in rows if r['maze']==name]
        for ax,key in zip(axes,('p','q')):
            ax.plot([r['n'] for r in subset],[r[key] for r in subset],'o-',label=name)
    for ax,title in zip(axes,('Wall-first probability','Goal by 15 seconds')):
        ax.set(xlabel='Grid side length (nodes)',ylabel='Probability',title=title,ylim=(-.03,1.03)); ax.legend(); ax.grid(alpha=.2)
    fig.suptitle('Same S2 boundary coefficients on refined grids (no reoptimization)')
    fig.tight_layout(); fig.savefig(OUT/'figures'/'mesh_refinement.png'); plt.close(fig)
    return rows


def screening():
    rows=[]
    for name in ('open','u_maze','narrow'):
        basis=build_basis(make_maze(name,40),8)
        v,u=basis.field(np.full(8,8.))
        maze=basis.maze; v0=float(v[maze.start_rc]); a=50.
        proxy=np.exp(a*(v0-8))*(-np.expm1(-a*v0))/(-np.expm1(-a*8))
        rows.append(dict(maze=name,V_start=v0,wall_gap=8-v0,start_speed=float(np.linalg.norm(u[maze.start_rc])),
                         constant_wall_continuum_identity_proxy=float(proxy)))
    save_json(OUT/'diagnostics'/'data'/'screening.json',rows)
    return rows


def main():
    records=enrich(json.loads((OUT/'data'/'summary.json').read_text()))
    ablations=enrich(json.loads((OUT/'data'/'ablations.json').read_text()))
    # Reuse an already simulated identical controller at the same T=15. This
    # controls for the deliberately simple fixed=2 baseline's lower amplitude.
    if not any(r['mode']=='fixed_8' for r in ablations):
        for name in ('open',):
            source=next(r for r in records if r['maze']==name and r['spec_name']=='S1')
            assert np.array_equal(source['theta'],np.full(8,8.))
            strong=copy.deepcopy(source)
            strong.update(mode='fixed_8',spec_name='S2',delta=.05,epsilon=.05,
                          optimization_seconds=0.,evaluations=0,optimizer_message='Fixed uniform wall potential 8; reuses identical S1 trials')
            strong['rho']=violation(strong['p'],strong['q'],.05,.05)
            strong['ctmc_pass']=strong['rho']<=1e-8
            strong['sde_point_pass']=strong['sde']['risk_upper']<=.05 and strong['sde']['arrival_rate']>=.95
            ablations.append(strong)
        ablations=enrich(ablations)
    audits=json.loads((OUT/'data'/'audits.json').read_text())
    runtime=json.loads((OUT/'data'/'runtime.json').read_text())
    arguments=json.loads((OUT/'logs'/'environment.json').read_text())['arguments']
    args=argparse.Namespace(**arguments)
    old_report=(OUT/'report.md').read_text()
    elapsed=float(re.search(r'in ([\d.]+) minutes',old_report).group(1))*60
    for stem,data in (('summary',records),('ablations',ablations)):
        save_json(OUT/'data'/f'{stem}.json',data); save_csv(OUT/'data'/f'{stem}.csv',data)
    for row in records:
        save_json(OUT/'data'/f'{row["maze"]}_{row["spec_name"]}.json',row)
    overview(records)
    ablation_figure(ablations,OUT/'figures')
    mesh=mesh_audit(records)
    screen=screening()
    report(records,ablations,audits,runtime,args,elapsed,OUT)
    diagnostics=json.loads((OUT/'diagnostics'/'data'/'basis_diagnostics.json').read_text())
    logs=json.loads((OUT/'diagnostics'/'data'/'log_search.json').read_text())
    extra=['','## Additional failure diagnosis','','Uniform wall potential 8 produces the following start speeds:',
           '', '| Maze | Wall potential minus start potential | Start speed (m/s) |', '|---|---:|---:|']
    for row in screen:
        extra.append(f'| {row["maze"]} | {row["wall_gap"]:.4g} | {row["start_speed"]:.4g} |')
    extra+=['','The goal influence is strongly screened in the U and narrow geometries. '
            'This is a controller-family/search difficulty, not a failed sparse solve. '
            'A geodesic-distance boundary hat basis at M=8 and M=16 also failed S2 in four 5,000-trial runs. '
            'A separate negative-log-arrival search escaped part of the flat objective region, but did not satisfy S2:',
            '', '| Maze | Diagnostic search | CTMC p | CTMC q |', '|---|---|---:|---:|']
    for row in logs:
        extra.append(f'| {row["maze"]} | Geodesic M=16, -log q | {row["prediction"]["p"]:.6f} | {row["prediction"]["q"]:.6f} |')
    extra+=['', 'The log-search controllers above were evaluated by CTMC only; no SDE validation is claimed for them. '
            'Diagnostics are reproducible with `python diagnostics.py`; the exploratory log-search coefficients '
            'and exact predictions are stored in `diagnostics/data/log_search.json`.', '',
            '## Mesh refinement', '', '| Maze | Grid | p CTMC | q CTMC |', '|---|---:|---:|---:|']
    for row in mesh:
        extra.append(f'| {row["maze"]} | {row["n"]} | {row["p"]:.6f} | {row["q"]:.6f} |')
    extra+=['', 'These use the same boundary coefficients and do not reoptimize on each mesh. '
            'Rasterized walls/goals and the nearest-node start vary slightly with resolution. '
            'Refinement is a diagnostic, not a rigorous error bound.', '',
            '## Practical conclusion', '',
            'The amplitude-matched fixed wall potential 8 also satisfies every tested open-maze specification. '
            'The weak fixed=2 baseline fails, so improvement over that baseline alone cannot establish the '
            'benefit of adaptive boundary shaping. `fixed_8` reuses the identical already simulated S1 '
            'controller and deadline while applying S2 thresholds.', '',
            'The end-to-end pipeline works and the simple maze is successful. The current harmonic boundary '
            'family and local synthesis are inadequate for both demanding mazes at the tested noise and '
            'coefficient range. This experiment does not support a broad claim of maze navigation success. '
            'Improving boundary control on individual wall faces, search conditioning, and long-corridor '
            'signal strength should precede theorem-level certification or larger benchmarks.', '',
            '![Potential fields and 100 SDE trajectories](figures/overview.png)', '']
    with (OUT/'report.md').open('a',encoding='utf-8') as stream:
        stream.write('\n'.join(extra))
    print('REVIEW COMPLETE',flush=True)


if __name__=='__main__':
    main()
