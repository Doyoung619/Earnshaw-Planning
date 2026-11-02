from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 120, 'savefig.dpi': 180})
COLORS = dict(wall='#dc5b50', goal='#268a63', timeout='#687887', ctmc='#315ca0', sde='#e49c35')


def maze_background(ax, maze):
    extent = [-maze.h/2, maze.length+maze.h/2]*2
    mask = np.ma.masked_where(~maze.occupied, maze.occupied)
    ax.imshow(mask, origin='lower', extent=extent, cmap=ListedColormap(['#303944']), zorder=4)
    goal = np.ma.masked_where(~maze.goal, maze.goal)
    ax.imshow(goal, origin='lower', extent=extent, cmap=ListedColormap([COLORS['goal']]), zorder=5)
    ax.scatter(*maze.start_xy, c='#f5b841', marker='*', s=140, edgecolor='black', zorder=10)
    ax.set(xlim=(-maze.h, maze.length+maze.h), ylim=(-maze.h, maze.length+maze.h),
           xlabel='x (m)', ylabel='y (m)', aspect='equal')


def controller_figures(basis, theta, trajectories, metrics, spec, out, stem, gain=1.0):
    out = Path(out)
    maze = basis.maze
    potential, drift = basis.field(theta, gain)
    from scipy.ndimage import binary_dilation
    wall_boundary=maze.occupied & binary_dilation(maze.interior,
                      structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    im = ax.imshow(np.ma.masked_where(~maze.interior, potential), origin='lower',
                   extent=[-maze.h/2, maze.length+maze.h/2]*2, cmap='viridis',
                   vmin=0,vmax=float(potential[maze.interior | wall_boundary].max()))
    stride=max(3,maze.n//14)
    take = np.zeros_like(maze.interior); take[2:-2:stride, 2:-2:stride] = True; take &= maze.interior
    u = drift[take]; norms = np.linalg.norm(u, axis=1)
    direction = u/np.maximum(norms[:, None], 1e-12)
    ax.quiver(maze.x[take], maze.y[take], direction[:, 0], direction[:, 1],
              color='white', alpha=0.55, scale=24, width=0.004)
    maze_background(ax, maze)
    ax.scatter(maze.x[wall_boundary],maze.y[wall_boundary],c=potential[wall_boundary],
               cmap='viridis',norm=im.norm,s=max(7,250/maze.n),marker='s',zorder=8,edgecolors='none')
    fig.colorbar(im, ax=ax, label='Harmonic potential')
    ax.set_title(f'{maze.name}: synthesized potential (M={basis.m})\nColored wall edges: Dirichlet values; arrows: direction')
    fig.tight_layout(); fig.savefig(out/f'{stem}_potential.png'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.1, 5.5))
    paths = trajectories['paths']
    for j, path in enumerate(paths):
        if not len(path):
            continue
        status = trajectories['status'][j]
        color = COLORS[{0:'timeout', 1:'wall', 2:'goal'}[status]]
        # Qualitative paths stop at T, even though risk MC continues after T.
        visible = path[path[:, 2] <= spec['T']]
        if len(visible):
            ax.plot(visible[:, 0], visible[:, 1], color=color, alpha=0.38, lw=0.8, zorder=2)
    maze_background(ax, maze)
    ax.set_title(f'{maze.name}: {len(paths)} SDE trajectories up to T={spec["T"]:g}s\n'
                 f'All {metrics["n"]:,} trials: wall={metrics["collision_rate"]:.3f}, '
                 f'arrival by T={metrics["arrival_rate"]:.3f}')
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0],[0],color=COLORS[k],label=v) for k,v in
                       [('goal','Goal exit'),('wall','Wall exit'),('timeout','No exit by horizon')]],
              loc='upper left', fontsize=8)
    fig.tight_layout(); fig.savefig(out/f'{stem}_trajectories.png'); plt.close(fig)


def specification_figure(records, out, filename='specification_satisfaction.png'):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.6), sharex='col', squeeze=False)
    for col, name in enumerate(dict.fromkeys(r['maze'] for r in records)):
        rows = [r for r in records if r['maze']==name]
        x = np.arange(len(rows))
        for row, pred_key, mc_key, threshold_key, label in [
                (0,'p','collision_rate','delta','Eventual wall-first probability'),
                (1,'q','arrival_rate','target','Safe arrival by deadline')]:
            ax = axes[row,col]
            predicted = [r[pred_key] for r in rows]
            empirical = [r['sde'][mc_key] for r in rows]
            cis = np.array([r['sde']['collision_ci' if row==0 else 'arrival_ci'] for r in rows])
            ax.scatter(x-.10, predicted, marker='s', color=COLORS['ctmc'], label='CTMC prediction', zorder=3)
            ax.errorbar(x+.10, empirical, yerr=np.abs(cis.T-np.array(empirical)),
                        fmt='o', capsize=3, color=COLORS['sde'], label='SDE MC (95% Wilson)')
            thresholds = [r['delta'] if row==0 else 1-r['epsilon'] for r in rows]
            ax.plot(x, thresholds, 'k--', lw=1, label='Required threshold')
            ax.set(ylim=(-0.025,1.025), ylabel=label, xticks=x,
                   xticklabels=[r['spec_name'] for r in rows])
            ax.grid(axis='y', alpha=.2)
            if row==0:
                ax.set_title(name)
    axes[0,0].legend(fontsize=8, loc='best')
    fig.suptitle('Discrete CTMC predictions vs continuous SDE simulation', fontsize=14)
    fig.tight_layout(); fig.savefig(Path(out)/filename); plt.close(fig)


def sweep_figure(records, out, maze_name='u_maze'):
    rows = [r for r in records if r['maze']==maze_name]
    fig, axes = plt.subplots(1,3, figsize=(13,3.9))
    x = np.arange(len(rows))
    labels = [f'{r["spec_name"]}\n$\\delta$={r["delta"]}, T={r["T"]:g}' for r in rows]
    for ax, key, title in zip(axes, ['p','q','rho'], ['Wall risk (lower is better)',
                                                      'Deadline success (higher is better)',
                                                      'Maximum specification violation']):
        ax.plot(x, [r[key] for r in rows], 'o-', color=COLORS['ctmc'])
        if key in ('p','q'):
            ax.plot(x, [r['delta'] if key=='p' else 1-r['epsilon'] for r in rows], 'k--')
        ax.set(title=title, xticks=x, xticklabels=labels)
        ax.grid(alpha=.2)
    fig.suptitle(f'{maze_name}: separately synthesized controller for each specification')
    fig.tight_layout(); fig.savefig(Path(out)/'specification_sweep.png'); plt.close(fig)


def ablation_figure(records, out):
    names = list(dict.fromkeys(r['maze'] for r in records))
    fig, axes = plt.subplots(1,len(names), figsize=(6*len(names),4.1), squeeze=False)
    for ax,name in zip(axes[0],names):
        rows = [r for r in records if r['maze']==name]
        x = np.arange(len(rows)); w=.34
        ax.bar(x-w/2, [r['p'] for r in rows], w, label='CTMC wall risk', color=COLORS['wall'])
        ax.bar(x+w/2, [r['q'] for r in rows], w, label='CTMC deadline success', color=COLORS['goal'])
        ax.axhline(rows[0]['delta'], ls='--', color=COLORS['wall'])
        ax.axhline(1-rows[0]['epsilon'], ls='--', color=COLORS['goal'])
        ax.set(title=name, ylim=(0,1.04), xticks=x, xticklabels=[r['mode'] for r in rows])
        ax.tick_params(axis='x', labelrotation=15); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(Path(out)/'ablations.png'); plt.close(fig)


def runtime_figure(rows, out):
    fig, axes = plt.subplots(1,2,figsize=(10,3.7))
    for m in sorted(set(r['M'] for r in rows)):
        subset = [r for r in rows if r['M']==m]
        for ax, key in zip(axes,['basis_seconds','evaluation_seconds']):
            ax.plot([r['states'] for r in subset], [r[key] for r in subset], 'o-', label=f'M={m}')
    for ax,title in zip(axes,['Harmonic basis construction','Exact probability evaluation']):
        ax.set(xlabel='Interior CTMC states', ylabel='Seconds', title=title); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(Path(out)/'runtime_scalability.png'); plt.close(fig)
