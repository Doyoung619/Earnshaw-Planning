import time
import numpy as np
from scipy.optimize import minimize
from .probabilities import evaluate, violation


def synthesize(builder, spec, seed=42, mode='joint', max_evals=140, theta_max=8.0,
               warm_start=None, steps=72):
    """Bounded local probability-guided search; nonzero rho means UNDETERMINED."""
    began = time.perf_counter()
    rng = np.random.default_rng(seed)
    m, start = builder.basis.m, builder.maze.start
    history, candidates, cache = [], [], {}

    def objective(theta):
        key = tuple(np.round(theta, 10))
        if key in cache:
            return cache[key][0]
        chain = builder.build(theta)
        values = evaluate(chain, start, spec['T'], 'backward_euler', steps)
        p, q = values['p'], values['q']
        rho = violation(p, q, spec['delta'], spec['epsilon'])
        if mode == 'joint':
            cost = rho + 1e-4*(p+1-q)
        elif mode == 'risk_only':
            cost = p
        elif mode == 'time_only':
            cost = 1-q
        else:
            raise ValueError(mode)
        history.append(dict(evaluation=len(history)+1, objective=cost, p=p, q=q, rho=rho))
        cache[key] = (cost, values)
        candidates.append((cost, np.array(theta).copy()))
        return cost

    initial = [np.full(m, a) for a in (0.5, 2.0, 5.0, theta_max)]
    if warm_start is not None:
        initial.insert(0, np.clip(warm_start, 0.0, theta_max))
    initial += [rng.uniform(0.1, theta_max, m) for _ in range(5)]
    for theta in initial:
        objective(theta)
    best = min(candidates, key=lambda v:v[0])[1]
    if mode != 'joint' or min(candidates, key=lambda v:v[0])[0] > 1e-4:
        result = minimize(objective, best, method='L-BFGS-B', bounds=[(0.0, theta_max)]*m,
                          options=dict(maxfun=max_evals, maxiter=max(10, max_evals//(m+1)),
                                       ftol=1e-8, gtol=2e-5, maxls=12, eps=2e-4))
        optimizer_message = str(result.message)
    else:
        optimizer_message = 'Initial candidate meets approximate specification'
    # Re-evaluate the strongest approximate candidates with the true exponential.
    ordered = sorted(candidates, key=lambda v:v[0])
    finalists, seen = [], []
    for _, theta in ordered:
        if any(np.linalg.norm(theta-old)<1e-3 for old in seen):
            continue
        seen.append(theta)
        values = evaluate(builder.build(theta), start, spec['T'])
        rho = violation(values['p'], values['q'], spec['delta'], spec['epsilon'])
        selection = rho if mode == 'joint' else (values['p'] if mode == 'risk_only' else 1-values['q'])
        finalists.append((selection, theta, values))
        if len(finalists) == 3:
            break
    _, theta, values = min(finalists, key=lambda v:v[0])
    return dict(theta=theta, prediction=values, history=history, mode=mode,
                optimization_seconds=time.perf_counter()-began, evaluations=len(history),
                optimizer_message=optimizer_message)
