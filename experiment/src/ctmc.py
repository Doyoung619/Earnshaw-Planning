from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix
from .harmonic import DIRECTIONS


@dataclass
class Chain:
    Q: object
    wall: np.ndarray
    goal: np.ndarray
    rates: np.ndarray
    neighbors: np.ndarray
    targets: np.ndarray
    max_speed: float


class CTMCBuilder:
    def __init__(self, basis, sigma=0.20, gain=1.0):
        self.basis, self.maze = basis, basis.maze
        self.sigma, self.gain = sigma, gain
        self.ids = np.arange(len(self.maze.rc))
        self.neighbors, self.targets = [], []
        for dr, dc in DIRECTIONS:
            nr, nc = self.maze.rc[:, 0]+dr, self.maze.rc[:, 1]+dc
            self.neighbors.append(self.maze.index[nr, nc])
            self.targets.append(np.where(self.maze.goal[nr, nc], 2, 1))
        self.neighbors = np.array(self.neighbors).T
        self.targets = np.array(self.targets).T
        self.gradient = basis.gradients[self.maze.interior]

    def build(self, theta):
        u = -self.gain*np.einsum('ndi,i->nd', self.gradient, theta, optimize=False)
        signed = np.column_stack((u[:, 0], -u[:, 0], u[:, 1], -u[:, 1]))
        rates = self.sigma**2/(2*self.maze.h**2) + np.maximum(signed, 0)/self.maze.h
        rows, cols, entries = [self.ids], [self.ids], [-rates.sum(axis=1)]
        wall, goal = np.zeros(len(u)), np.zeros(len(u))
        for d in range(4):
            live = self.neighbors[:, d] >= 0
            rows.append(self.ids[live]); cols.append(self.neighbors[live, d]); entries.append(rates[live, d])
            wall += np.where(~live & (self.targets[:, d] == 1), rates[:, d], 0)
            goal += np.where(~live & (self.targets[:, d] == 2), rates[:, d], 0)
        Q = coo_matrix((np.concatenate(entries), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(len(u), len(u))).tocsc()
        return Chain(Q, wall, goal, rates, self.neighbors, self.targets,
                     float(np.linalg.norm(u, axis=1).max()))
