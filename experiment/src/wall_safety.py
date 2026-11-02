"""Explicit wall-potential, wall-repulsion and draft Eq. (5.2) checks."""
import numpy as np
from scipy.optimize import LinearConstraint,linprog
from .harmonic import DIRECTIONS


def wall_matrices(basis,gain=1.):
    maze=basis.maze
    walls=[]; inside=[]; inward=[]; points=[]
    for dr,dc in DIRECTIONS:
        nr,nc=maze.rc[:,0]+dr,maze.rc[:,1]+dc
        hit=maze.occupied[nr,nc]
        r,c=maze.rc[hit].T; wr,wc=nr[hit],nc[hit]
        walls.append(basis.values[wr,wc]); inside.append(basis.values[r,c])
        # Positive means commanded velocity away from the wall.
        inward.append(gain*(basis.gradients[r,c,0,:]*dc+basis.gradients[r,c,1,:]*dr))
        points.append(np.column_stack((r,c,wr,wc)))
    wall,interior,command=np.concatenate(walls),np.concatenate(inside),np.concatenate(inward)
    start=basis.values[maze.start_rc]
    return dict(wall=wall,interior=interior,jump=wall-interior,inward=command,
                certificate=wall-start,faces=np.concatenate(points))


def interpolated_face_normals(basis,gain=1.,fractions=(-.5,0.,.5)):
    """Linear normal-command rows for free-only interpolation along each face."""
    maze=basis.maze
    face=wall_matrices(basis,gain)['faces']
    r,c,wr,wc=face.T
    dc=wc-c; dr=wr-r
    rows=[]; positions=[]; normals=[]
    for tangent in fractions:
        rr=(r+wr)/2+tangent*dc
        cc=(c+wc)/2-tangent*dr
        r0=np.floor(rr).astype(int); c0=np.floor(cc).astype(int)
        fy=rr-r0; fx=cc-c0
        combined=np.zeros((len(face),basis.m))
        denominator=np.zeros(len(face))
        for ry,cx,weight in ((0,0,(1-fy)*(1-fx)),(1,0,fy*(1-fx)),
                              (0,1,(1-fy)*fx),(1,1,fy*fx)):
            rn,cn=r0+ry,c0+cx
            weighted=weight*maze.interior[rn,cn]
            combined+=gain*weighted[:,None]*(basis.gradients[rn,cn,0,:]*dc[:,None]+
                                             basis.gradients[rn,cn,1,:]*dr[:,None])
            denominator+=weighted
        combined/=np.maximum(denominator[:,None],1e-12)
        rows.append(combined); positions.append(np.column_stack((cc*maze.h,rr*maze.h)))
        normals.append(np.column_stack((dc,dr)))
    return np.vstack(rows),np.vstack(positions),np.vstack(normals)


def repulsive_constraint(basis,gain=1.,wall_floor=1.,minimum_inward_speed=0.,interpolated=False):
    matrices=wall_matrices(basis,gain)
    normal=interpolated_face_normals(basis,gain)[0] if interpolated else matrices['inward']
    A=np.vstack((matrices['wall'],matrices['jump'],normal))
    lower=np.r_[np.full(len(matrices['wall']),wall_floor),
                np.zeros(len(matrices['jump'])),
                np.full(len(normal),minimum_inward_speed)]
    # Normalize rows to improve SLSQP's feasibility conditioning.
    scale=np.maximum(np.linalg.norm(A,axis=1),1e-12)
    constraint=LinearConstraint(A/scale[:,None],lower/scale,np.inf)
    constraint.physical_A=A
    constraint.physical_lower=lower
    return constraint


def audit_wall(basis,theta,sigma=.2,gain=1.,delta=.05,tolerance=1e-7):
    matrix=wall_matrices(basis,gain)
    boundary=matrix['wall']@theta
    jump=matrix['jump']@theta
    inward=matrix['inward']@theta
    required=sigma*sigma/(2*gain)*np.log(1/delta)
    gap=matrix['certificate']@theta
    return dict(wall_min=float(boundary.min()),wall_max=float(boundary.max()),
                start_potential=float(basis.values[basis.maze.start_rc]@theta),
                minimum_wall_minus_adjacent=float(jump.min()),
                lower_wall_face_fraction=float(np.mean(jump < -tolerance)),
                minimum_inward_command=float(inward.min()),
                outward_command_face_fraction=float(np.mean(inward < -tolerance)),
                repulsive_at_grid_faces=bool(jump.min()>=-tolerance and inward.min()>=-tolerance),
                minimum_wall_minus_start=float(gap.min()),required_certificate_gap=float(required),
                discrete_paper_margin_pass=bool(gap.min()>=required-tolerance),
                face_count=len(jump))


def best_paper_margin(basis,theta_max=8.):
    """Maximize min_w(V(w)-V(start)) on the finite boundary transcription.

    This checks the sufficient certificate, not feasibility of the probabilities.
    """
    A=wall_matrices(basis)['certificate']
    objective=np.r_[np.zeros(basis.m),-1.]
    inequality=np.column_stack((-A,np.ones(len(A))))
    result=linprog(objective,A_ub=inequality,b_ub=np.zeros(len(A)),
                   bounds=[(0.,theta_max)]*basis.m+[(None,None)],method='highs')
    return dict(success=bool(result.success),message=result.message,
                best_margin=float(result.x[-1]) if result.success else None)
