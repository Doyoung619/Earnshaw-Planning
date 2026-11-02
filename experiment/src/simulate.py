import time
import numpy as np
from scipy.ndimage import map_coordinates


def wilson(successes, n, z=1.96):
    if n == 0:
        return [0., 1.]
    p = successes/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    radius = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(max(0.,center-radius)), float(min(1.,center+radius))]


def summarize(status, exit_times, deadline, horizon, seconds):
    n = len(status)
    wall = status == 1
    success = (status == 2) & (exit_times <= deadline)
    timeout = status == 0
    return dict(n=n, collision_rate=float(wall.mean()), collision_ci=wilson(wall.sum(), n),
                arrival_rate=float(success.mean()), arrival_ci=wilson(success.sum(), n),
                timeout_rate=float(timeout.mean()), eventual_goal_rate=float((status == 2).mean()),
                risk_lower=float(wall.mean()), risk_upper=float((wall | timeout).mean()),
                risk_upper_ci=wilson((wall | timeout).sum(), n),
                mean_exit_observed=float(exit_times[~timeout].mean()) if (~timeout).any() else None,
                restricted_mean_exit=float(np.minimum(exit_times, horizon).mean()),
                horizon=float(horizon), simulation_seconds=seconds)


class DriftInterpolator:
    """Convex interpolation of free-space drift only, excluding solid interiors.

    Gradients inside an occupied cell are not defined by the harmonic BVP and
    must not enter the continuous robot command. Legacy mode is for audits only.
    """
    def __init__(self,maze,drift,mode='interior'):
        self.maze,self.mode=maze,mode
        self.mask=maze.interior.astype(float)
        self.drift=drift*self.mask[...,None] if mode=='interior' else drift
    def __call__(self,positions):
        coords=np.array([positions[:,1]/self.maze.h,positions[:,0]/self.maze.h])
        velocity=np.column_stack([map_coordinates(self.drift[...,d],coords,order=1,mode='nearest',prefilter=False)
                                  for d in range(2)])
        if self.mode=='interior':
            denominator=map_coordinates(self.mask,coords,order=1,mode='nearest',prefilter=False)
            if np.any(denominator<1e-10):
                raise ValueError('Drift queried inside a solid/absorbing region')
            velocity/=denominator[:,None]
        return velocity


def simulate_sde(basis, theta, sigma, gain, deadline, n=5000, seed=1234,
                 dt=0.005, horizon=None, save_paths=100,interpolation='interior'):
    """Vectorized Euler-Maruyama; continue after T to estimate eventual wall risk.

    Drift is bilinearly interpolated; occupied/goal cells absorb at cell faces.
    Step and segment checks limit missed thin obstacles, but Brownian bridge
    crossings remain a discretization error, assessed by step refinement.
    """
    began = time.perf_counter()
    maze = basis.maze
    _, drift = basis.field(theta, gain)
    interpolate=DriftInterpolator(maze,drift,interpolation)
    max_speed = np.linalg.norm(drift[maze.interior], axis=1).max()
    dt = min(dt, 0.25*maze.h/max(max_speed, 1e-12), (0.18*maze.h/max(sigma,1e-12))**2)
    horizon = float(horizon or max(3*deadline,45.))
    rng = np.random.default_rng(seed)
    xy = np.tile(maze.start_xy, (n, 1))
    status = np.zeros(n, dtype=np.int8)
    exit_times = np.full(n,horizon,dtype=float)
    collision_command=np.full(n,np.nan)
    paths = [[] for _ in range(min(save_paths, n))]
    stride = max(1, int(0.035/dt))
    for step in range(int(np.ceil(horizon/dt))):
        live = np.flatnonzero(status == 0)
        if not len(live):
            break
        t = step*dt
        step_dt = min(dt, horizon-t)
        positions = xy[live]
        u = interpolate(positions)
        proposal = positions + u*step_dt + sigma*np.sqrt(step_dt)*rng.standard_normal((len(live), 2))
        # Check in path order so the first contact determines wall vs goal.
        hit = np.zeros(len(live), dtype=np.int8)
        hit_fraction = np.ones(len(live))
        for fraction in (0.25, 0.5, 0.75, 1.):
            point = positions+fraction*(proposal-positions)
            rc = np.rint(point[:, ::-1]/maze.h).astype(int)
            outside = np.any((rc<0) | (rc>=maze.n), axis=1)
            rc = np.clip(rc, 0, maze.n-1)
            w = outside | maze.occupied[rc[:, 0], rc[:, 1]]
            g = maze.goal[rc[:, 0], rc[:, 1]]
            first = (hit == 0) & (w | g)
            hit[first] = np.where(w[first], 1, 2)
            hit_fraction[first] = fraction
        xy[live] = positions+hit_fraction[:, None]*(proposal-positions)
        absorbed = hit != 0
        wall_hits=hit==1
        if wall_hits.any():
            wall_rc=np.clip(np.rint(xy[live[wall_hits],::-1]/maze.h).astype(int),0,maze.n-1)
            # Slab entry into the occupied square identifies the contacted
            # physical face. A vector to the cell center includes a tangential
            # component and is not a valid wall-normal diagnostic.
            displacement=proposal[wall_hits]-positions[wall_hits]
            side=wall_rc[:,::-1]*maze.h-np.sign(displacement)*maze.h/2
            entry=np.full_like(displacement,-np.inf)
            np.divide(side-positions[wall_hits],displacement,out=entry,where=np.abs(displacement)>1e-14)
            axis=np.argmax(entry,axis=1)
            direction=np.zeros_like(displacement)
            direction[np.arange(len(axis)),axis]=np.sign(displacement[np.arange(len(axis)),axis])
            collision_command[live[wall_hits]]=np.sum(u[wall_hits]*direction,axis=1)
        status[live[absorbed]] = hit[absorbed]
        exit_times[live[absorbed]] = t+step_dt*hit_fraction[absorbed]
        if step % stride == 0 or absorbed.any():
            for j in live[live < len(paths)]:
                if step % stride == 0 or status[j]:
                    paths[j].append([float(xy[j, 0]), float(xy[j, 1]), float(min(t+step_dt, horizon))])
    metrics = summarize(status, exit_times, deadline, horizon, time.perf_counter()-began)
    metrics['dt'] = float(dt)
    metrics['interpolation']=interpolation
    at_wall=collision_command[np.isfinite(collision_command)]
    metrics['collision_outward_command_fraction']=float(np.mean(at_wall>1e-7)) if len(at_wall) else None
    return metrics, dict(paths=[np.asarray(p).reshape(-1, 3) for p in paths],
                         status=status, exit_times=exit_times,final_positions=xy,
                         collision_command_toward_wall=collision_command)


def simulate_ctmc(chain, start, deadline, n=5000, seed=4321, horizon=90.):
    """Exact jump-chain Monte Carlo checks the backward equations independently."""
    began = time.perf_counter()
    rng = np.random.default_rng(seed)
    states = np.full(n, start, dtype=int)
    status = np.zeros(n, dtype=np.int8)
    times = np.zeros(n)
    totals = chain.rates.sum(axis=1)
    while True:
        live = np.flatnonzero((status == 0) & (times < horizon))
        if not len(live):
            break
        state = states[live]
        times[live] += rng.exponential(1/totals[state])
        active = times[live] < horizon
        live, state = live[active], state[active]
        cdf = np.cumsum(chain.rates[state], axis=1)
        draw = rng.random(len(live))*totals[state]
        directions = (draw[:, None] > cdf).sum(axis=1)
        next_states = chain.neighbors[state, directions]
        absorbed = next_states < 0
        status[live[absorbed]] = chain.targets[state[absorbed], directions[absorbed]]
        states[live[~absorbed]] = next_states[~absorbed]
    times = np.minimum(times, horizon)
    return summarize(status, times, deadline, horizon, time.perf_counter()-began)
