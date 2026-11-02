"""Audit previous wall potentials and test explicitly repulsive harmonic fields."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
from pathlib import Path
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.maze import make_maze
from src.harmonic import build_basis
from src.ctmc import CTMCBuilder
from src.adjoint import synthesize_adjoint
from src.wall_safety import audit_wall,wall_matrices,repulsive_constraint,best_paper_margin
from src.experiments import save_json,make_record
from src.plotting import maze_background


def main():
    root=Path(__file__).resolve().parent; out=root/'outputs'/'wall_safety'
    for name in ('data','figures','logs'):
        (out/name).mkdir(parents=True,exist_ok=True)
    original=json.loads((root/'outputs'/'high_basis'/'data'/'comparison.json').read_text())
    args=argparse.Namespace(sigma=.2,gain=1.,seed=20261101,wall_safety='repulsive',wall_floor=1.,minimum_inward_speed=0.)
    spec=dict(name='S2',delta=.05,epsilon=.05,T=15.)
    audits=[]; corrected=[]
    fig,axes=plt.subplots(2,2,figsize=(11.5,9.5))
    for col,name in enumerate(('u_maze','narrow')):
        basis=build_basis(make_maze(name,40),200,'local'); maze=basis.maze
        row=next(r for r in original if r['maze']==name and r['M']==200)
        old=audit_wall(basis,np.array(row['theta']))
        uniform=audit_wall(basis,np.full(200,8.))
        maximum=best_paper_margin(basis)
        audit=dict(maze=name,previous=old,uniform_wall_8=uniform,best_certificate_margin=maximum)
        audits.append(audit); save_json(out/'data'/'audit.json',audits)
        print(name,json.dumps(audit),flush=True)
        v,_=basis.field(row['theta'])
        faces=wall_matrices(basis)['faces']
        r,c,wr,wc=faces.T
        ax=axes[0,col]
        image=ax.imshow(np.ma.masked_where(~maze.interior,v),origin='lower',
                        extent=[-maze.h/2,maze.length+maze.h/2]*2,cmap='viridis',vmin=0,vmax=8)
        maze_background(ax,maze)
        ax.scatter(wc*maze.h,wr*maze.h,c=v[wr,wc],cmap='viridis',vmin=0,vmax=8,
                   s=18,marker='s',zorder=7,edgecolors='none')
        ax.set_title(f'{name}: actual previous wall potentials\nColored squares are wall Dirichlet values')
        fig.colorbar(image,ax=ax,label='Potential')
        ax=axes[1,col]
        ax.scatter(v[r,c],v[wr,wc],s=9,alpha=.5,label='Wall-adjacent faces')
        ax.plot([0,8],[0,8],'k--',label='Equal potential')
        ax.set(xlabel='Adjacent free-node potential',ylabel='Wall potential',xlim=(0,8.1),ylim=(0,8.1),
               title='Points below the line have a downhill edge into the wall')
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out/'figures'/'previous_wall_potentials.png'); plt.close(fig)
    # Keep the original harmonic family: impose wall floor, uphill edge into
    # each wall, and no outward commanded normal component. No 1/d term is added.
    for name in ('u_maze','narrow'):
        basis=build_basis(make_maze(name,40),200,'local')
        builder=CTMCBuilder(basis,.2,1.)
        constraint=repulsive_constraint(basis,wall_floor=1.,minimum_inward_speed=0.)
        print(name,'explicit wall-repulsive optimization starts',flush=True)
        result=synthesize_adjoint(builder,spec,maxiter=70,steps=96,wall_constraint=constraint)
        record=make_record(name,spec,result,builder,args,out,f'{name}_repulsive_M200',5000)
        record['wall_audit']=audit_wall(basis,result['theta'])
        corrected.append(record); save_json(out/'data'/'repulsive_results.json',corrected)
        print(name,'corrected wall audit',json.dumps(record['wall_audit']),flush=True)
    print('WALL AUDIT AND CONSTRAINED EXPERIMENT COMPLETE',flush=True)


if __name__=='__main__':
    main()
