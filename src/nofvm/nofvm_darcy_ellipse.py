"""D3 material geometry, sampled independently of its FV discretization."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class EllipseMaterial:
    parameters: np.ndarray

    def evaluate(self, n):
        a, b, theta = self.parameters
        if not (0 < b <= a < .5) or n < 1:
            raise ValueError('An ellipse must lie strictly inside the local square')
        x, y = np.meshgrid((np.arange(n)+.5)/n-.5, (np.arange(n)+.5)/n-.5)
        first = (x*np.cos(theta)+y*np.sin(theta))/a
        second = (-x*np.sin(theta)+y*np.cos(theta))/b
        return np.where(first**2+second**2 <= 1, 1000., 1.)


def sample_materials(count, seed, family='ellipse'):
    if family != 'ellipse':
        raise ValueError(family)
    random = np.random.default_rng(seed)
    values = random.uniform(size=(count, 3))
    values[:, 0] = .30 + .14*values[:, 0]
    values[:, 1] = .10 + .15*values[:, 1]
    values[:, 2] *= np.pi
    return [EllipseMaterial(row.copy()) for row in values]
