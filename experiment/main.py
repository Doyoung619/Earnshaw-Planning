import os
# Keep small sparse problems from paying large BLAS thread-pool overhead.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import argparse
from src.experiments import run


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Harmonic CTMC/SDE navigation prototype')
    parser.add_argument('--suite', choices=['pilot','full'], default='pilot')
    parser.add_argument('--grid', type=int, default=40)
    parser.add_argument('--basis', type=int, default=None,
                        help='Default: open=8, complex mazes=200')
    parser.add_argument('--boundary',choices=['auto','spatial','local','geodesic','contour'],default='auto')
    parser.add_argument('--interpolation',choices=['interior','legacy'],default='interior')
    parser.add_argument('--optimizer',choices=['auto','finite-difference','adjoint'],default='auto')
    parser.add_argument('--wall-safety',choices=['repulsive','free'],default='repulsive')
    parser.add_argument('--wall-floor',type=float,default=1.0)
    parser.add_argument('--minimum-inward-speed',type=float,default=0.0)
    parser.add_argument('--mc', type=int, default=None)
    parser.add_argument('--budget', type=int, default=140)
    parser.add_argument('--sigma', type=float, default=0.20)
    parser.add_argument('--gain', type=float, default=1.0)
    parser.add_argument('--theta-max', type=float, default=8.0)
    parser.add_argument('--seed', type=int, default=20261101)
    parser.add_argument('--output', default=None)
    run(parser.parse_args())
