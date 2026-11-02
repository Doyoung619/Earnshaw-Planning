from dataclasses import dataclass
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from collections import deque
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import binary_dilation

DIRECTIONS = ((0, 1), (0, -1), (1, 0), (-1, 0))


@dataclass
class HarmonicBasis:
    maze: object
    values: np.ndarray
    gradients: np.ndarray
    boundary_weights: np.ndarray
    residual: float
    seconds: float
    boundary_kind: str = 'spatial'

    @property
    def m(self):
        return self.values.shape[-1]

    def field(self, theta, gain=1.0):
        potential = np.einsum('yxi,i->yx', self.values, theta, optimize=False)
        drift = -gain * np.einsum('yxdi,i->yxd', self.gradients, theta, optimize=False)
        return potential, drift


def geodesic_boundary_weights(maze, m):
    """Diagnostic basis: hat functions of graph distance to goal along free space.

    Distinct wall faces can receive distinct values if the obstacle is at least
    two cells thick. Graph-distance kinks are accepted in this grid prototype.
    """
    distance = np.full((maze.n, maze.n), np.inf)
    queue = deque(map(tuple, np.column_stack(np.where(maze.goal))))
    distance[maze.goal] = 0.
    while queue:
        r,c = queue.popleft()
        for dr,dc in DIRECTIONS:
            nr,nc = r+dr,c+dc
            if 0 <= nr < maze.n and 0 <= nc < maze.n and not maze.occupied[nr,nc] and not np.isfinite(distance[nr,nc]):
                distance[nr,nc] = distance[r,c]+maze.h
                queue.append((nr,nc))
    wall_distance = np.full_like(distance,np.inf)
    for dr,dc in DIRECTIONS:
        shifted = np.roll(distance,(dr,dc),axis=(0,1))+maze.h
        wall_distance = np.minimum(wall_distance,shifted)
    missing = ~np.isfinite(wall_distance)
    nearest = distance_transform_edt(missing, return_distances=False, return_indices=True)
    wall_distance[missing] = wall_distance[tuple(nearest[:,missing])]
    coordinates = wall_distance/max(float(wall_distance[maze.occupied].max()),maze.h)*(m-1)
    weights = np.maximum(1-np.abs(coordinates[...,None]-np.arange(m)),0)
    weights /= np.maximum(weights.sum(axis=2,keepdims=True),1e-15)
    return weights


def build_basis(maze, m=8, boundary='spatial'):
    """Smooth positive wall partition of unity; goal Dirichlet values are zero.

    Gaussian boundary weights are a simple continuous alternative to discontinuous
    segment indicators. The same sparse Laplacian factorization serves every RHS.
    """
    started = time.perf_counter()
    metadata=None
    if boundary == 'contour':
        from .boundary_contours import contour_boundary_weights
        weights,metadata=contour_boundary_weights(maze,m)
    elif boundary == 'geodesic':
        weights = geodesic_boundary_weights(maze,m)
    elif boundary == 'local':
        wall_boundary = maze.occupied & binary_dilation(maze.interior,
                         structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
        boundary_rc = np.column_stack(np.where(wall_boundary))
        if m > len(boundary_rc):
            raise ValueError(f'M={m} exceeds {len(boundary_rc)} distinct wall boundary nodes')
        points = boundary_rc[:,::-1]*maze.h
        selected = [0]
        nearest = np.sum((points-points[0])**2,axis=1)
        for _ in range(m-1):
            choice = int(np.argmax(nearest)); selected.append(choice)
            nearest = np.minimum(nearest,np.sum((points-points[choice])**2,axis=1))
        centers = points[selected]
        center_d2 = np.sum((centers[:,None]-centers[None,:])**2,axis=2)
        np.fill_diagonal(center_d2,np.inf)
        width = max(.85*np.median(np.sqrt(np.min(center_d2,axis=1))),.65*maze.h)
        dist2 = ((maze.x[...,None]-centers[:,0])**2+(maze.y[...,None]-centers[:,1])**2)
        log_weights = -dist2/(2*width**2)
        log_weights -= log_weights.max(axis=2,keepdims=True)
        weights = np.exp(log_weights)
        weights /= weights.sum(axis=2,keepdims=True)
    elif boundary == 'spatial':
        angle = np.arange(m)*2*np.pi/m
        anchors = maze.length/2 + 0.60*maze.length*np.column_stack((np.cos(angle), np.sin(angle)))
        dist2 = ((maze.x[..., None]-anchors[:, 0])**2 +(maze.y[..., None]-anchors[:, 1])**2)
        weights = np.exp(-dist2/(2*(0.36*maze.length)**2))
        weights /= weights.sum(axis=2,keepdims=True)
    else:
        raise ValueError(boundary)
    values = weights.copy()
    values[maze.goal] = 0.0
    rows, cols, entries = [], [], []
    rhs = np.zeros((len(maze.rc), m))
    ids = np.arange(len(maze.rc))
    for dr, dc in DIRECTIONS:
        nr, nc = maze.rc[:, 0]+dr, maze.rc[:, 1]+dc
        neighbors = maze.index[nr, nc]
        live = neighbors >= 0
        rows.extend(ids[live]); cols.extend(neighbors[live]); entries.extend(-np.ones(live.sum()))
        rhs[~live] += values[nr[~live], nc[~live]]
    rows.extend(ids); cols.extend(ids); entries.extend(np.full(len(ids), 4.0))
    lap = coo_matrix((entries, (rows, cols)), shape=(len(ids), len(ids))).tocsc()
    interior_values = splu(lap).solve(rhs)
    values[maze.interior] = interior_values
    residual = float(np.max(np.abs(lap @ interior_values-rhs)))
    gy, gx = np.gradient(values, maze.h, axis=(0, 1))
    gradients = np.stack((gx, gy), axis=2)
    basis=HarmonicBasis(maze, values, gradients, weights, residual, time.perf_counter()-started,boundary)
    basis.boundary_metadata=metadata
    return basis
