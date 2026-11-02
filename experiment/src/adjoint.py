"""Discrete adjoints of implicit-time CTMC backward equations, independent of M."""
import time
import json
from pathlib import Path
import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import splu
from scipy.optimize import minimize
from .probabilities import evaluate,violation


def backward_adjoint(builder,theta,deadline,steps=96):
    chain=builder.build(theta)
    size=len(chain.wall); start=builder.maze.start
    lu=splu(-chain.Q)
    p=lu.solve(chain.wall)
    unit=np.zeros(size); unit[start]=1.
    risk_adjoint=lu.solve(unit,trans='T')
    live=chain.neighbors>=0
    def difference(vector,wall_value,goal_value):
        neighbors=vector[np.maximum(chain.neighbors,0)]
        neighbors=np.where(live,neighbors,np.where(chain.targets==1,wall_value,goal_value))
        return neighbors-vector[:,None]
    wall_weights=risk_adjoint[:,None]*difference(p,1.,0.)
    dt=deadline/steps
    step_lu=splu(eye(size,format='csc')-dt*chain.Q)
    q=np.zeros(size)
    states=[]
    for _ in range(steps):
        q=step_lu.solve(q+dt*chain.goal)
        states.append(q)
    adj=unit.copy()
    goal_weights=np.zeros((size,4))
    for vector in reversed(states):
        adj=step_lu.solve(adj,trans='T')
        goal_weights+=dt*adj[:,None]*difference(vector,0.,1.)
    u=-builder.gain*np.einsum('ndi,i->nd',builder.gradient,theta,optimize=False)
    def contract(weights):
        w=np.column_stack((weights[:,0]*(u[:,0]>0)-weights[:,1]*(u[:,0]<0),
                           weights[:,2]*(u[:,1]>0)-weights[:,3]*(u[:,1]<0)))
        return -builder.gain/builder.maze.h*np.einsum('nd,ndi->i',w,builder.gradient,optimize=False)
    safe_exit=lu.solve(chain.goal)
    return dict(p=float(p[start]),q=float(q[start]),g=float(safe_exit[start]),
                gp=contract(wall_weights),gq=contract(goal_weights))


def synthesize_adjoint(builder,spec,seed=42,theta_max=8.,maxiter=160,steps=96,warm_start=None,mode='joint',wall_constraint=None,
                       u_max=None,initial_guesses=None,progress_every=0,checkpoint_path=None):
    """Log-arrival initialization followed by the requested direct rho objective.

    All reported candidates are re-evaluated using the exact CTMC exponential.
    Finite-time log search handles a numerically flat near-zero arrival region.
    """
    began=time.perf_counter(); m=builder.basis.m; rng=np.random.default_rng(seed)
    history=[]; candidates=[]; raw_candidates=[]
    speed_cache={}
    def speed_data(theta):
        if 'theta' not in speed_cache or not np.array_equal(theta,speed_cache['theta']):
            u=-builder.gain*np.einsum('ndi,i->nd',builder.gradient,theta,optimize=False)
            norms2=np.sum(u*u,axis=1); index=int(np.argmax(norms2))
            speed_cache.update(theta=theta.copy(),maximum=float(np.sqrt(norms2[index])),
                               jac=2*builder.gain*builder.gradient[index].T@u[index])
            if u_max is not None:
                # Smooth conservative maximum: logsumexp >= max. This avoids
                # a switching argmax derivative when several speed nodes bind.
                normalized=norms2/(u_max*u_max); peak=float(normalized.max()); beta=200.
                weights=np.exp(beta*(normalized-peak)); total=float(weights.sum())
                weights/=total
                speed_cache.update(smooth=peak+np.log(total)/beta,
                    smooth_jac=2*builder.gain/(u_max*u_max)*
                    np.einsum('n,ndi,nd->i',weights,builder.gradient,u,optimize=False))
        return speed_cache
    def feasible_theta(theta):
        linear=wall_constraint is None or np.min(
            getattr(wall_constraint,'physical_A',wall_constraint.A)@theta-
            getattr(wall_constraint,'physical_lower',wall_constraint.lb))>=-1e-7
        return linear and (u_max is None or speed_data(theta)['maximum']<=u_max+1e-6)
    def optimize(initial,phase,iterations):
        def fun(theta):
            value=backward_adjoint(builder,theta,spec['T'],steps)
            p,q,gp,gq=value['p'],value['q'],value['gp'],value['gq']
            rho=violation(p,q,spec['delta'],spec['epsilon'])
            if phase=='log_arrival':
                cost=-np.log(max(q,1e-300)); gradient=-gq/max(q,1e-300)
            elif phase=='log_safe_exit':
                cost=-np.log(max(value['g'],1e-300)); gradient=gp/max(value['g'],1e-300)
            elif phase=='risk_only':
                cost=p; gradient=gp
            elif phase=='time_only':
                cost=1-q; gradient=-gq
            else:
                violations=np.array([0.,p-spec['delta'],1-spec['epsilon']-q])
                active=int(np.argmax(violations))
                cost=rho+1e-5*(p+1-q)
                gradient=(np.zeros(m) if active==0 else gp if active==1 else -gq)+1e-5*(gp-gq)
            history.append(dict(evaluation=len(history)+1,phase=phase,p=p,q=q,rho=rho,objective=float(cost)))
            selection=rho if mode=='joint' else p if mode=='risk_only' else 1-q
            raw_candidates.append((selection,p+1-q,theta.copy()))
            feasible=feasible_theta(theta)
            if feasible:
                candidates.append((selection,p+1-q,theta.copy()))
            if progress_every and len(history)%progress_every==0:
                print(f'      eval {len(history)} {phase}: p={p:.5g} q={q:.5g}, constraints={feasible}',flush=True)
                if checkpoint_path:
                    wall_slack=None if wall_constraint is None else float(np.min(
                        getattr(wall_constraint,'physical_A',wall_constraint.A)@theta-
                        getattr(wall_constraint,'physical_lower',wall_constraint.lb)))
                    Path(checkpoint_path).write_text(json.dumps(dict(theta=theta.tolist(),phase=phase,p=p,q=q,
                        feasible=feasible,minimum_wall_slack=wall_slack,
                        maximum_speed=speed_data(theta)['maximum'] if u_max is not None else None,
                        evaluations=len(history))),encoding='utf-8')
            return cost,gradient
        if wall_constraint is None and u_max is None:
            return minimize(fun,np.clip(initial,0,theta_max),jac=True,method='L-BFGS-B',
                            bounds=[(0.,theta_max)]*m,
                            options=dict(maxiter=iterations,maxfun=iterations*3,maxls=20,ftol=1e-10,gtol=1e-7))
        constraints=[] if wall_constraint is None else [wall_constraint]
        if u_max is not None:
            constraints.append(dict(type='ineq',fun=lambda t:1-speed_data(t)['smooth'],
                                    jac=lambda t:speed_data(t)['smooth_jac']))
        return minimize(fun,np.clip(initial,0,theta_max),jac=True,method='SLSQP',
                        bounds=[(0.,theta_max)]*m,constraints=constraints,
                        options=dict(maxiter=iterations,ftol=1e-9))
    seeds=[np.full(m,theta_max*.75),rng.uniform(.15*theta_max,.85*theta_max,m)]
    if wall_constraint is not None:
        seeds=[np.full(m,theta_max),np.full(m,theta_max*.75)]
    if warm_start is not None:
        seeds.insert(0,warm_start)
    if initial_guesses is not None:
        seeds=list(initial_guesses)
    # Always preserve an explicitly checked uniform feasible fallback, rather
    # than accepting an infeasible optimizer iterate if a constrained run stalls.
    fallback=np.ones(m)
    if u_max is not None:
        fallback*=min(theta_max,.9*u_max/max(speed_data(fallback)['maximum'],1e-12))
    else:
        fallback*=theta_max
    if feasible_theta(fallback):
        v=backward_adjoint(builder,fallback,spec['T'],steps)
        base_score=violation(v['p'],v['q'],spec['delta'],spec['epsilon']) if mode=='joint' else v['p'] if mode=='risk_only' else 1-v['q']
        candidates.append((base_score,
                           v['p']+1-v['q'],fallback))
    messages=[]
    for index,initial in enumerate(seeds):
        result=optimize(initial,'log_safe_exit' if mode=='risk_only' else 'log_arrival',maxiter)
        print(f'    adjoint seed {index+1}/{len(seeds)}: evaluations={len(history)}, '
              f'best BE q={max(h["q"] for h in history):.5f}',flush=True)
        direct=optimize(result.x,mode,max(25,maxiter//2))
        messages.append(str(direct.message))
        if mode=='joint' and any(h['rho']<=1e-5 for h in history):
            break
    finalists=[]; seen=[]
    # Restore near-feasible speed-limited iterates along a convex segment to the
    # checked baseline. Both wall constraints and the speed-ball set are convex.
    # This never accepts the infeasible raw iterate or clips the deployed drift.
    if feasible_theta(fallback):
        for _,_,raw in sorted(raw_candidates,key=lambda v:(v[0],v[1]))[:5]:
            fraction=1.
            if wall_constraint is not None:
                physical=getattr(wall_constraint,'physical_A',wall_constraint.A)
                lower=getattr(wall_constraint,'physical_lower',wall_constraint.lb)
                base_slack=physical@fallback-lower; raw_slack=physical@raw-lower
                bad=raw_slack < -5e-8
                if np.any(bad):
                    fraction=min(fraction,float(np.min(np.maximum(base_slack[bad],0)/
                                            np.maximum(base_slack[bad]-raw_slack[bad],1e-300))))
            if u_max is not None:
                low,high=0.,fraction
                for _ in range(30):
                    mid=(low+high)/2; trial=fallback+mid*(raw-fallback)
                    if speed_data(trial)['maximum']<=u_max*(1-1e-9): low=mid
                    else: high=mid
                fraction=low
            trial=fallback+fraction*(raw-fallback)
            if feasible_theta(trial):
                value=backward_adjoint(builder,trial,spec['T'],steps)
                score=violation(value['p'],value['q'],spec['delta'],spec['epsilon']) if mode=='joint' else value['p'] if mode=='risk_only' else 1-value['q']
                candidates.append((score,value['p']+1-value['q'],trial))
    if not candidates:
        raise RuntimeError('No candidate satisfies explicit wall constraints; no controller is accepted.')
    for _,_,theta in sorted(candidates,key=lambda v:(v[0],v[1])):
        if any(np.linalg.norm(theta-t)<.05 for t in seen):
            continue
        seen.append(theta)
        value=evaluate(builder.build(theta),builder.maze.start,spec['T'])
        select=(violation(value['p'],value['q'],spec['delta'],spec['epsilon']) if mode=='joint' else
                value['p'] if mode=='risk_only' else 1-value['q'])
        finalists.append((select,
                          value['p']+1-value['q'],theta,value))
        if len(finalists)>=5:
            break
    rho,_,theta,prediction=min(finalists,key=lambda v:(v[0],v[1]))
    return dict(theta=theta,prediction=prediction,history=history,mode=mode,
                optimization_seconds=time.perf_counter()-began,evaluations=len(history),
                optimizer_message='; '.join(messages),u_max=u_max,
                best_raw_theta=min(raw_candidates,key=lambda v:(v[0],v[1]))[2])
