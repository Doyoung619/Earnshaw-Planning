"""One bounded warm-start improvement of the already validated U/M500 field."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
from pathlib import Path
import json,shutil
import numpy as np
from src.maze import make_maze
from src.harmonic import build_basis
from src.ctmc import CTMCBuilder
from src.wall_safety import repulsive_constraint
from src.adjoint import synthesize_adjoint
from src.experiments import save_json

root=Path(__file__).resolve().parent
out=root/'outputs'/'refined_M500'
source=out/'final'/'data'/'u_maze_M500.json'
old=json.loads(source.read_text())
basis=build_basis(make_maze('u_maze',100),500,'contour')
builder=CTMCBuilder(basis,.2,1.)
constraint=repulsive_constraint(basis,wall_floor=1.,interpolated=True)
spec=dict(name='S2',T=15.,delta=.05,epsilon=.05)
result=synthesize_adjoint(builder,spec,theta_max=80.,maxiter=20,steps=96,
                          wall_constraint=constraint,u_max=10.,
                          initial_guesses=[np.array(old['theta'])],progress_every=10,
                          checkpoint_path=out/'logs'/'u_polish_checkpoint.json')
save_json(out/'logs'/'u_polish_result.json',result)
if result['prediction']['q']>old['q']:
    raw=out/'data'/'u_maze_M500.json'
    shutil.copy2(raw,out/'data'/'u_maze_M500_first_pass.json')
    row=dict(old)
    row.update(theta=result['theta'],p=result['prediction']['p'],q=result['prediction']['q'],
               evaluations=old['evaluations']+result['evaluations'],
               optimization_seconds=old['optimization_seconds']+result['optimization_seconds'],
               optimizer_message=old['optimizer_message']+'; warm-start polish: '+result['optimizer_message'])
    for key in ('sde','ctmc_mc','sde_half_step','deterministic','actual_wall_face_audit','wall_audit',
                'ctmc_pass','sde_point_pass','sde_ci_pass','rho'):
        row.pop(key,None)
    row['validation_status']='Polished candidate awaiting fresh final Monte Carlo; use final/data after finalize_refined.py.'
    save_json(raw,row)
    print('U POLISH IMPROVED. Re-run finalize_refined.py --mazes u_maze',row['p'],row['q'],flush=True)
else:
    print('U POLISH KEPT PREVIOUS VALIDATED FIELD',flush=True)
