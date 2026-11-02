"""Targeted basis/mesh diagnostics after the unmodified first-pass suite."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(name,'1')
from pathlib import Path
import json
import time
import argparse
import numpy as np
from src.maze import make_maze
from src.harmonic import build_basis
from src.ctmc import CTMCBuilder
from src.probabilities import evaluate
from src.optimize_theta import synthesize
from src.experiments import save_json, save_csv, make_record


def run():
    root = Path(__file__).resolve().parent
    out = root/'outputs'/'diagnostics'
    for name in ('data','logs','figures'):
        (out/name).mkdir(parents=True,exist_ok=True)
    args = argparse.Namespace(sigma=.2,gain=1.,seed=20261101)
    spec = dict(name='S2',delta=.05,epsilon=.05,T=15.)
    records=[]
    for name in ('u_maze','narrow'):
        for boundary,m in [('geodesic',8),('geodesic',16)]:
            began=time.perf_counter()
            basis=build_basis(make_maze(name,40),m,boundary)
            builder=CTMCBuilder(basis,args.sigma,args.gain)
            result=synthesize(builder,spec,seed=args.seed,max_evals=320,
                              warm_start=np.linspace(.4,8.,m))
            stem=f'{name}_{boundary}_M{m}'
            record=make_record(name,spec,result,builder,args,out,stem,5000)
            record.update(boundary_basis=boundary,M=m,total_seconds=time.perf_counter()-began)
            records.append(record)
            save_json(out/'data'/'basis_diagnostics.json',records)
            save_csv(out/'data'/'basis_diagnostics.csv',records)
    print('BASIS DIAGNOSTICS DONE',flush=True)


if __name__=='__main__':
    run()
