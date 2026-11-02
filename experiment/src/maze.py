from dataclasses import dataclass
import numpy as np
from scipy.ndimage import label


@dataclass
class Maze:
    name: str
    occupied: np.ndarray
    goal: np.ndarray
    start_rc: tuple
    length: float = 4.0

    def __post_init__(self):
        self.n = self.occupied.shape[0]
        assert self.occupied.shape == (self.n, self.n)
        self.h = self.length / (self.n - 1)
        self.y, self.x = np.meshgrid(np.arange(self.n)*self.h,
                                     np.arange(self.n)*self.h, indexing='ij')
        self.interior = ~self.occupied & ~self.goal
        components, _ = label(self.interior)
        self.interior &= components == components[self.start_rc]
        self.occupied |= ~self.interior & ~self.goal
        self.rc = np.column_stack(np.where(self.interior))
        self.index = np.full((self.n, self.n), -1, dtype=int)
        self.index[self.interior] = np.arange(len(self.rc))
        self.start = int(self.index[self.start_rc])
        self.start_xy = np.array(self.start_rc[::-1]) * self.h
        assert self.start >= 0 and np.any(self.goal)


def make_maze(name='open', n=40, length=4.0):
    """Fixed physical geometries; row index increases in the positive y direction."""
    y, x = np.meshgrid(np.linspace(0, length, n), np.linspace(0, length, n), indexing='ij')
    occupied = (x == 0) | (y == 0) | (x == length) | (y == length)
    if name == 'open':
        start, center = (0.65, 0.8), (3.25, 3.15)
    elif name == 'u_maze':
        occupied |= (((x >= 1.25) & (x <= 1.48) & (y >= 1.0) & (y <= 3.0)) |
                     ((x >= 2.52) & (x <= 2.75) & (y >= 1.0) & (y <= 3.0)) |
                     ((x >= 1.25) & (x <= 2.75) & (y >= 1.0) & (y <= 1.23)))
        start, center = (2.0, 1.65), (2.0, 0.5)
    elif name == 'narrow':
        occupied |= (((x >= 1.2) & (x <= 1.45) & (y <= 2.9)) |
                     ((x >= 2.55) & (x <= 2.8) & (y >= 1.1)))
        start, center = (0.6, 0.65), (3.4, 3.35)
    else:
        raise ValueError(name)
    goal = (x-center[0])**2 + (y-center[1])**2 <= 0.28**2
    goal &= ~occupied
    start_rc = tuple(np.rint(np.array(start[::-1]) / (length/(n-1))).astype(int))
    return Maze(name, occupied, goal, start_rc, length)


def load_maze(path):
    """Custom .npz: occupied, goal, start_rc, optional length and name."""
    data = np.load(path, allow_pickle=False)
    return Maze(str(data.get('name', 'custom')), data['occupied'].astype(bool),
                data['goal'].astype(bool), tuple(data['start_rc']), float(data.get('length', 4.0)))
