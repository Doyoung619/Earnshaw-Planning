import time
import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import splu, expm_multiply


def violation(p, q, delta, epsilon):
    return float(max(0.0, p-delta, 1-epsilon-q))


def evaluate(chain, start, deadline, method='exact', steps=72, vectors=False):
    """Exact CTMC backward equations (expm) or fast BE search approximation.

    q(T) = g - exp(T Q)g, where (-Q)g=bG. No continuum certificate is claimed.
    """
    began = time.perf_counter()
    lu = splu(-chain.Q)
    both = lu.solve(np.column_stack((chain.wall, chain.goal, np.ones(chain.Q.shape[0]))))
    p, g, exit_mean = both.T
    if method == 'exact':
        remaining = expm_multiply(chain.Q*deadline, g, traceA=float(chain.Q.diagonal().sum()*deadline))
    elif method == 'backward_euler':
        step_lu = splu(eye(len(p), format='csc')-deadline/steps*chain.Q)
        remaining = g.copy()
        for _ in range(steps):
            remaining = step_lu.solve(remaining)
    else:
        raise ValueError(method)
    q = g-remaining
    residual = max(float(np.max(np.abs(-chain.Q@p-chain.wall))),
                   float(np.max(np.abs(p+g-1))))
    if residual > 1e-7 or min(p.min(), q.min()) < -1e-7 or max(p.max(), q.max()) > 1+1e-7:
        raise ArithmeticError(f'Probability invariant failure: residual={residual}')
    result = dict(p=float(p[start]), q=float(q[start]), mean_exit=float(exit_mean[start]),
                  residual=residual, evaluation_seconds=time.perf_counter()-began)
    if vectors:
        result.update(p_vector=p, q_vector=q)
    return result
