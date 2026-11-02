"""Periodic piecewise-linear wall basis along each actual pixel-wall contour.

    Coefficients are not coupled across opposite sides merely because those
    sides are close in Euclidean distance. The harmonic solve remains nodal.
"""
import numpy as np
from collections import defaultdict


def contour_boundary_weights(maze,m):
    faces=[]; endpoints=defaultdict(list)
    for r,c in maze.rc:
        specifications=[(0,1,((2*r-1,2*c+1),(2*r+1,2*c+1))),
                        (0,-1,((2*r-1,2*c-1),(2*r+1,2*c-1))),
                        (1,0,((2*r+1,2*c-1),(2*r+1,2*c+1))),
                        (-1,0,((2*r-1,2*c-1),(2*r-1,2*c+1)))]
        for dr,dc,ends in specifications:
            if maze.occupied[r+dr,c+dc]:
                index=len(faces)
                faces.append((int(r+dr),int(c+dc),ends))
                for endpoint in ends:
                    endpoints[endpoint].append(index)
    if any(len(indices)!=2 for indices in endpoints.values()):
        raise ValueError('Wall contour is ambiguous at a diagonal contact; refine or repair occupancy.')
    visited=set(); contours=[]
    for first in range(len(faces)):
        if first in visited:
            continue
        ordered=[]; current=first; endpoint=faces[first][2][0]
        while current not in visited:
            visited.add(current); ordered.append(current)
            ends=faces[current][2]
            endpoint=ends[1] if endpoint==ends[0] else ends[0]
            choices=endpoints[endpoint]
            current=choices[1] if choices[0]==current else choices[0]
        if current!=first:
            raise ValueError('Wall contour did not close')
        contours.append(ordered)
    lengths=np.array([len(c) for c in contours])
    if m>len(set((f[0],f[1]) for f in faces)):
        raise ValueError('M exceeds distinct nodal wall boundary values; use a finer grid')
    allocations=np.maximum(3,np.floor(m*lengths/lengths.sum()).astype(int))
    while allocations.sum()<m:
        allocations[np.argmax(lengths/allocations)]+=1
    while allocations.sum()>m:
        valid=np.where(allocations>3,lengths/allocations,np.inf)
        allocations[np.argmin(valid)]-=1
    weights=np.zeros((maze.n,maze.n,m),dtype=float)
    counts=np.zeros((maze.n,maze.n),dtype=int)
    offset=0
    for ordered,knots in zip(contours,allocations):
        size=len(ordered)
        for j,face_index in enumerate(ordered):
            r,c,_=faces[face_index]
            coordinate=(j+.5)*knots/size
            left=int(np.floor(coordinate))%knots; fraction=coordinate-np.floor(coordinate)
            weights[r,c,offset+left]+=1-fraction
            weights[r,c,offset+(left+1)%knots]+=fraction
            counts[r,c]+=1
        offset+=int(knots)
    boundary=counts>0
    weights[boundary]/=counts[boundary,None]
    metadata=dict(contour_face_counts=lengths.tolist(),knots_per_contour=allocations.tolist(),
                  wall_face_count=len(faces),wall_node_count=int(boundary.sum()),
                  contour_partition_error=float(np.max(np.abs(weights[boundary].sum(axis=1)-1))))
    return weights,metadata
