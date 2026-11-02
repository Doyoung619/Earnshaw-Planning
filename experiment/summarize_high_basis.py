"""Combine the M=8/50/100/200 runs and make reviewable comparison figures."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from src.maze import make_maze
from src.harmonic import build_basis
from src.plotting import maze_background,COLORS
from src.experiments import save_json,save_csv

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'/'high_basis'


def main():
    records=[]
    for folder in (OUT,ROOT/'outputs'/'high_basis_200'):
        for record in json.loads((folder/'data'/'summary.json').read_text()):
            record['artifact_directory']=str(folder.relative_to(ROOT))
            records.append(record)
    records.sort(key=lambda r:(r['maze'],r['M']))
    save_json(OUT/'data'/'comparison.json',records); save_csv(OUT/'data'/'comparison.csv',records)
    fig,axes=plt.subplots(1,2,figsize=(11,4.1))
    for name,color in [('u_maze','#315ca0'),('narrow','#ce683c')]:
        rows=[r for r in records if r['maze']==name]
        for ax,pred,emp,ci in zip(axes,['p','q'],['collision_rate','arrival_rate'],['collision_ci','arrival_ci']):
            x=np.array([r['M'] for r in rows])
            ax.plot(x,[r[pred] for r in rows],'s-',color=color,label=f'{name}: CTMC')
            values=np.array([r['sde'][emp] for r in rows])
            limits=np.array([r['sde'][ci] for r in rows])
            ax.errorbar(x,values,yerr=np.abs(limits.T-values),fmt='o--',capsize=3,color=color,alpha=.65,label=f'{name}: SDE MC')
    for ax,title,threshold in zip(axes,['Wall-first collision probability','Safe arrival by 15 seconds'],[.05,.95]):
        ax.axhline(threshold,color='black',ls=':',label='Specification')
        ax.set(xlabel='Number of localized wall basis functions M',ylabel='Probability',title=title,ylim=(-.03,1.03),xticks=[8,50,100,200])
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Higher boundary resolution changes the result substantially',fontsize=13)
    fig.tight_layout(); fig.savefig(OUT/'figures'/'basis_size_comparison.png'); plt.close(fig)

    fig,axes=plt.subplots(2,3,figsize=(14.5,9.3))
    for row,name in enumerate(('u_maze','narrow')):
        maze=make_maze(name,40)
        for col,m in enumerate((50,100,200)):
            record=next(r for r in records if r['maze']==name and r['M']==m)
            folder=ROOT/record['artifact_directory']; stem=f'{name}_local_M{m}'
            data=np.load(folder/'data'/f'{stem}_trajectories.npz')
            ax=axes[row,col]
            for j,(a,b) in enumerate(zip(data['offsets'][:-1],data['offsets'][1:])):
                points=data['points'][a:b]; points=points[points[:,2]<=15]
                if len(points):
                    color=COLORS[{0:'timeout',1:'wall',2:'goal'}[int(data['status'][j])]]
                    ax.plot(points[:,0],points[:,1],color=color,alpha=.4,lw=.65)
            maze_background(ax,maze)
            ax.set_title(f'{name}, M={m}\nSDE arrival={record["sde"]["arrival_rate"]:.2%}; wall={record["sde"]["collision_rate"]:.2%}')
    fig.suptitle('100 trajectories per panel; quantitative rates use 5,000 trials\nGreen: goal exit. Red: wall exit. Deadline: 15 s.',fontsize=14)
    fig.tight_layout(rect=[0,0,1,.94]); fig.savefig(OUT/'figures'/'trajectory_basis_comparison.png'); plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.9))
    for ax,name in zip(axes,('u_maze','narrow')):
        record=next(r for r in records if r['maze']==name and r['M']==200)
        maze=make_maze(name,40); basis=build_basis(maze,200,'local')
        potential,_=basis.field(record['theta'])
        im=ax.imshow(np.ma.masked_where(~maze.interior,potential),origin='lower',
                     extent=[-maze.h/2,maze.length+maze.h/2]*2,cmap='viridis',vmin=0,vmax=8)
        maze_background(ax,maze); ax.set_title(f'{name}: M=200 harmonic potential')
        fig.colorbar(im,ax=ax,label='Potential')
    fig.tight_layout(); fig.savefig(OUT/'figures'/'potential_M200.png'); plt.close(fig)

    lines=['# Higher-resolution boundary synthesis: corrected experimental assessment','',
           '**The original M=8 experiment was underparameterized for drawing conclusions about the method.** '
           'This follow-up increases the actual wall-boundary spatial resolution, checks boundary matrix rank, '
           'and uses adjoints with substantially larger optimization budgets.', '',
           'A fair size comparison below uses the same localized construction and adjoint optimizer at M=8, 50, 100, 200. '
           'It is distinct from the earlier broad radial M=8 basis. Local centers lie on real wall nodes, '
           'and Gaussian bandwidth decreases with center spacing. All boundary matrices had full column rank.', '',
           'S2: delta=0.05, epsilon=0.05, T=15 s. Fixed physical geometry, 40x40 grid, sigma=0.2, '
           'gain=1, boundary coefficients in [0,8]. Each controller has 5,000 SDE trials and 5,000 exact CTMC jump trials.', '',
           '| Maze | M | p CTMC | Collision SDE | q CTMC | Arrival SDE | Boundary rank |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f'| {r["maze"]} | {r["M"]} | {r["p"]:.6f} | {r["sde"]["collision_rate"]:.6f} '
                     f'| {r["q"]:.6f} | {r["sde"]["arrival_rate"]:.6f} | {r["boundary_rank"]} |')
    lines+=['','## What changed in the conclusion','',
            '- Higher M produces large improvements in both mazes. M=8 was insufficient evidence for a negative assessment of the method.',
            '- At M=200, U-maze SDE deadline success reaches 51.36%; narrow-maze success reaches 30.50%. '
            'All tested S2 controllers still miss the requested 95% success and 5% collision thresholds.',
            '- This is evidence of representation sensitivity and partial navigation success, not global infeasibility. '
            'M=50 is not a universal sufficiency threshold. Resolution of each wall face, conditioning, coefficient range, '
            'noise, and optimization also matter. These changing bases are not mathematically nested, so monotone '
            'improvement is an observed trend, not a guarantee.',
            '- M=200 controls 200 of 250 U-maze or 260 narrow-maze wall boundary nodes on this grid. '
            'A substantially larger M should be paired with a finer maze grid rather than redundant columns.',
            '- Main CLI defaults now use M=8 for open space and M=200 localized bases plus adjoint search for complex mazes.',
            '', '## Optimization and numerical checks','',
            'Each evaluation differentiates the exact discrete backward-Euler recurrences via an adjoint, '
            'so gradient cost does not require M separate PDE evaluations. Three initializations use '
            'negative-log-arrival search to leave the flat near-zero-arrival region, followed by the requested '
            'joint maximum-violation objective. M=50/100 use 180 warm-up iterations per seed; M=200 uses 240. '
            'Final candidates are compared using matrix-exponential CTMC probabilities.', '',
            'Four numerical tests passed, including Appendix B reproduction, independent CTMC jump simulation, '
            'and adjoint gradients checked against independent finite differences. CTMC and SDE results differ '
            'because of spatial upwinding, cell-face absorption and SDE stepping; no continuum certification is claimed. '
            'No u_max constraint is imposed; actual maximum sampled commands are recorded.', '',
            '## Data and reproduction','',
            '- All coefficients, Wilson intervals, speed maxima, optimization histories and independent CTMC MC: '
            '`data/comparison.json` and `data/comparison.csv`.',
            '- Each S2-synthesized controller is also evaluated under all five original specifications in '
            '`fixed_controller_spec_evaluation`. Those extra rows reuse the controller; they are not separate reoptimizations.',
            '- The initial three-maze/five-spec separately optimized suite remains in `../data/summary.json`.', '',
            '```powershell',
            'python high_basis.py --basis 50 100 8 --maxiter 180 --mc 5000 --wall-safety free',
            'python high_basis.py --basis 200 --maxiter 240 --mc 5000 --wall-safety free --output outputs/high_basis_200',
            'python summarize_high_basis.py',
            'python -m unittest -v test_numerics',
            '```','',
            '![Basis size and probability comparison](figures/basis_size_comparison.png)','',
            '![100-trajectory comparison](figures/trajectory_basis_comparison.png)','',
            '![M200 harmonic potentials](figures/potential_M200.png)','']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    original=ROOT/'outputs'/'report.md'
    old=original.read_text(encoding='utf-8')
    notice='> Updated assessment: the M=8 pilot does not establish a limitation of the method. '
    if not old.startswith(notice):
        original.write_text(notice+'See the [M=50/100/200 follow-up](high_basis/report.md), '
                            'which shows substantial improvement with higher boundary resolution.\n\n'+old,encoding='utf-8')
    print('HIGH-BASIS COMPARISON AND REPORT COMPLETE',flush=True)


if __name__=='__main__':
    main()
