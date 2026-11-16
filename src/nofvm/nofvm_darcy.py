"""Darcy reference and conservative central-subdomain coupling for D1/D2.

The material function is fixed before choosing a mesh. Boundary traces are
ordered left, right, bottom, top, increasing along each side. Fluxes are
integrated and positive in the local outward normal direction.
"""
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .nofvm_2d import boundary_geometry, graph_matrix, rectangular_system


def interpolation_matrix(points, nodes):
    points, nodes = np.asarray(points), np.asarray(nodes)
    if np.any(points < nodes[0]) or np.any(points > nodes[-1]):
        raise ValueError("Interpolation query lies outside the material square")
    left = np.minimum(np.searchsorted(nodes, points, side="right") - 1, len(nodes) - 2)
    fraction = (points - nodes[left]) / (nodes[left + 1] - nodes[left])
    result = np.zeros((len(points), len(nodes)))
    result[np.arange(len(points)), left] = 1 - fraction
    result[np.arange(len(points)), left + 1] = fraction
    return result


@dataclass
class GaussianMaterial:
    nodal_w: np.ndarray
    family: str

    def evaluate(self, n):
        nodes = np.linspace(0., 1., len(self.nodal_w))
        interpolation = interpolation_matrix((np.arange(n) + .5) / n, nodes)
        w = interpolation @ self.nodal_w @ interpolation.T
        if self.family == "continuous":
            return np.exp(w)
        if self.family == "jump":
            return np.where(w <= 0., 3., 12.)
        raise ValueError(self.family)


def sample_materials(count, seed, family, node_count=33, correlation=.3):
    nodes = np.linspace(0., 1., node_count)
    covariance = np.exp(-.5 * ((nodes[:, None] - nodes[None, :]) / correlation) ** 2)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    if eigenvalues.min() < -1e-10:
        raise ValueError("Invalid GP covariance")
    root = vectors * np.sqrt(np.maximum(eigenvalues, 0.))
    random = np.random.default_rng(seed)
    return [GaussianMaterial(root @ random.normal(size=(node_count, node_count)) @ root.T, family)
            for _ in range(count)]


class DarcyBoundary:
    def __init__(self, seed, random_loads=3):
        self.y = np.linspace(0., 2., 65)
        covariance = np.exp(-.5 * ((self.y[:, None] - self.y[None, :]) / .3) ** 2)
        root = np.linalg.cholesky(covariance + 1e-10 * np.eye(len(self.y)))
        random = np.random.default_rng(seed)
        draws = random.normal(size=(random_loads, 2, len(self.y))) @ root.T
        contrast = np.stack((np.ones(65), -np.ones(65)))[None]
        self.samples = np.concatenate((contrast, draws, np.ones((1, 2, 65))))

    def __call__(self, xy, indices=None):
        samples = self.samples if indices is None else self.samples[np.atleast_1d(indices)]
        result = np.zeros((len(xy), len(samples)))
        for side, x in enumerate((0., 2.)):
            mask = np.isclose(xy[:, 0], x)
            for column, sample in enumerate(samples):
                result[mask, column] = np.interp(xy[mask, 1], self.y, sample[side])
        return result


def trace_prolongation(n, m):
    if n % m or min(n, m) < 1:
        raise ValueError("The number of ports per side must divide local n")
    rows = np.arange(4 * n)
    columns = (rows // n) * m + (rows % n) // (n // m)
    return sparse.csc_matrix((np.ones(4*n), (rows, columns)), shape=(4*n, 4*m))


def local_response(k, m=32):
    k = np.asarray(k)
    if k.shape != (len(k), len(k)):
        raise ValueError("Expected a square local material grid")
    system = rectangular_system(k)
    prolongation = trace_prolongation(len(k), m)
    lifting = (system.lifting @ prolongation).tocsc()
    trace_diagonal = np.asarray(prolongation.T @ system.tau).ravel()
    factor = splu(system.matrix, permc_spec="MMD_AT_PLUS_A")
    extension = factor.solve(lifting.toarray())
    schur = np.diag(trace_diagonal) - lifting.T @ extension
    return dict(H=extension, S=schur, B=lifting, T=np.diag(trace_diagonal), system=system)


def local_field_flux(k, field, trace, m=32):
    """Independently measure the reconstructed field's physical boundary flux."""
    n = len(k)
    boundary = boundary_geometry(n, n)
    values = trace_prolongation(n, m) @ trace
    flux = 2 * k.ravel()[boundary.owner, None] * (field[boundary.owner] - values)
    return np.asarray(trace_prolongation(n, m).T @ flux)


def full_material(material, n):
    if n % 2:
        raise ValueError("Central square requires an even local mesh size")
    k = np.ones((2*n, 2*n))
    k[n//2:3*n//2, n//2:3*n//2] = material.evaluate(n)
    return k


class FullDarcyFV:
    def __init__(self, material, n=64, m=32):
        if n % m:
            raise ValueError("Reference grid must resolve all interface ports")
        self.n, self.m = n, m
        self.k = full_material(material, n)
        self.system = rectangular_system(self.k, lengths=(2., 2.))
        s = self.system
        self.dirichlet = np.repeat([True, True, False, False], 2*n)
        remove = np.bincount(s.boundary.owner, weights=s.tau * ~self.dirichlet,
                             minlength=4*n*n)
        self.matrix = s.matrix - sparse.diags(remove, format="csc")
        self.lifting = s.lifting @ sparse.diags(self.dirichlet.astype(float), format="csc")
        self.factor = None

    def factorize(self):
        self.factor = splu(self.matrix, permc_spec="MMD_AT_PLUS_A")

    def interface_flux(self, field):
        n, m = self.n, self.m
        ids = np.arange(4*n*n).reshape(2*n, 2*n)
        a, b = n//2, 3*n//2
        inner = np.r_[ids[a:b, a], ids[a:b, b-1], ids[a, a:b], ids[b-1, a:b]]
        outer = np.r_[ids[a:b, a-1], ids[a:b, b], ids[a-1, a:b], ids[b, a:b]]
        ki = self.k.ravel()[inner]
        conductance = 2 * ki / (ki + 1)
        flux = conductance[:, None] * (field[inner] - field[outer])
        return flux.reshape(4*m, n//m, -1).sum(axis=1)

    def solve(self, boundary):
        g = boundary(self.system.boundary.xy)
        rhs = self.lifting @ g
        field = self.factor.solve(rhs)
        flux = self.interface_flux(field)
        exterior = self.system.tau[:, None] * self.dirichlet[:, None] * (
            field[self.system.boundary.owner] - g)
        return dict(field=field, flux=flux, boundary=g,
                    linear_residual=float(abs(self.matrix @ field - rhs).max()),
                    region_balance=float(abs(flux.sum(axis=0)).max()),
                    exterior_balance=float(abs(exterior.sum(axis=0)).max()))

    def reconstruct(self, result, target_n):
        if target_n == self.n:
            return result['field']
        n, size, h = self.n, 2*self.n, 1/self.n
        field = result['field']
        ids = np.arange(size*size).reshape(size, size)
        k = self.k.ravel()
        slopes = np.zeros((size*size, 2, field.shape[1]))
        count = np.zeros((size*size, 2))
        for dim, (a, b) in enumerate(((ids[:, :-1], ids[:, 1:]), (ids[:-1], ids[1:]))):
            a, b = a.ravel(), b.ravel()
            conductance = 2 * k[a] * k[b] / (k[a] + k[b])
            jump = conductance[:, None] * (field[b] - field[a]) / h
            slopes[a, dim] += jump / k[a, None]
            slopes[b, dim] += jump / k[b, None]
            count[a, dim] += 1
            count[b, dim] += 1
        for side, (dim, sign) in enumerate(((0, -1), (0, 1), (1, -1), (1, 1))):
            sl = slice(side*size, (side+1)*size)
            owner = self.system.boundary.owner[sl]
            gradient = sign * (result['boundary'][sl] - field[owner]) / (h/2) if side < 2 else 0.
            slopes[owner, dim] += gradient
            count[owner, dim] += 1
        slopes /= count[:, :, None]
        x, y = np.meshgrid((np.arange(2*target_n) + .5) / target_n,
                           (np.arange(2*target_n) + .5) / target_n)
        xy = np.column_stack((x.ravel(), y.ravel()))
        ij = np.minimum((xy / h).astype(int), size-1)
        owners = ij[:, 1]*size + ij[:, 0]
        delta = xy - (ij + .5)*h
        return field[owners] + np.einsum('nd,ndb->nb', delta, slopes[owners])


class DarcyHybrid:
    """Ordinary FV exterior and one learned or exact central response."""
    def __init__(self, response, n=64, m=32):
        if m % 2 or n % m or response['H'].shape != (n*n, 4*m):
            raise ValueError("Incompatible local response and exterior grid")
        self.response, self.n, self.m = response, n, m
        size = 2*m
        ordinary = np.ones((size, size), dtype=bool)
        ordinary[m//2:3*m//2, m//2:3*m//2] = False
        self.ids = np.full((size, size), -1, dtype=int)
        self.count = int(ordinary.sum())
        self.ids[ordinary] = np.arange(self.count)
        self.edges = []
        for a, b in ((self.ids[:, :-1], self.ids[:, 1:]), (self.ids[:-1], self.ids[1:])):
            keep = (a >= 0) & (b >= 0)
            self.edges.append((a[keep], b[keep], 1.))
        a, b = m//2, 3*m//2
        self.interface_owner = np.r_[self.ids[a:b, a-1], self.ids[a:b, b],
                                     self.ids[a-1, a:b], self.ids[b, a:b]]
        self.trace_ids = self.count + np.arange(4*m)
        self.exterior = boundary_geometry(size, size, (2., 2.))
        self.exterior_owner = self.ids.ravel()[self.exterior.owner]
        self.dirichlet = np.repeat([True, True, False, False], size)
        dofs = self.count + 4*m
        diagonal = np.bincount(self.exterior_owner, weights=2*self.dirichlet, minlength=dofs)
        self.matrix = graph_matrix(dofs, self.edges + [(self.interface_owner, self.trace_ids, 2.)], diagonal)
        self.matrix += sparse.block_diag((sparse.csc_matrix((self.count, self.count)),
                                          sparse.csc_matrix(response['S'])), format='csc')
        self.factor = None

    def factorize(self):
        self.factor = splu(self.matrix, permc_spec='MMD_AT_PLUS_A')

    def solve(self, boundary):
        g = boundary(self.exterior.xy)
        rhs = np.zeros((self.matrix.shape[0], g.shape[1]))
        np.add.at(rhs, self.exterior_owner, 2*self.dirichlet[:, None]*g)
        values = self.factor.solve(rhs)
        trace = values[self.trace_ids]
        local = self.response['H'] @ trace
        flux = -self.response['S'] @ trace
        outer_flux = 2*(values[self.interface_owner] - trace)
        exterior_flux = 2*self.dirichlet[:, None]*(values[self.exterior_owner] - g)
        return dict(ordinary=values[:self.count], trace=trace, local_field=local, flux=flux, boundary=g,
                    linear_residual=float(abs(self.matrix @ values - rhs).max()),
                    interface_balance=float(abs(flux + outer_flux).max()),
                    region_balance=float(abs(flux.sum(axis=0)).max()),
                    exterior_balance=float(abs(exterior_flux.sum(axis=0)).max()))

    def reconstruct(self, result):
        n, m, h = self.n, self.m, 1/self.m
        field = result['ordinary']
        slopes = np.zeros((self.count, 2, field.shape[1]))
        count = np.zeros((self.count, 2))
        for dim, (a, b, _) in enumerate(self.edges):
            gradient = (field[b] - field[a]) / h
            np.add.at(slopes[:, dim], a, gradient)
            np.add.at(slopes[:, dim], b, gradient)
            np.add.at(count[:, dim], a, 1)
            np.add.at(count[:, dim], b, 1)
        for side, (dim, sign) in enumerate(((0, 1), (0, -1), (1, 1), (1, -1))):
            sl = slice(side*m, (side+1)*m)
            owner = self.interface_owner[sl]
            gradient = sign*(result['trace'][sl] - field[owner]) / (h/2)
            np.add.at(slopes[:, dim], owner, gradient)
            np.add.at(count[:, dim], owner, 1)
        size = 2*m
        for side, (dim, sign) in enumerate(((0, -1), (0, 1), (1, -1), (1, 1))):
            sl = slice(side*size, (side+1)*size)
            owner = self.exterior_owner[sl]
            gradient = sign*(result['boundary'][sl] - field[owner]) / (h/2) if side < 2 else np.zeros_like(field[owner])
            np.add.at(slopes[:, dim], owner, gradient)
            np.add.at(count[:, dim], owner, 1)
        slopes /= count[:, :, None]
        x, y = np.meshgrid((np.arange(2*n) + .5) / n, (np.arange(2*n) + .5) / n)
        xy = np.column_stack((x.ravel(), y.ravel()))
        inside = (xy[:, 0] > .5) & (xy[:, 0] < 1.5) & (xy[:, 1] > .5) & (xy[:, 1] < 1.5)
        out = np.empty((4*n*n, field.shape[1]))
        out[inside] = result['local_field']
        ij = np.minimum((xy[~inside] / h).astype(int), size-1)
        owner = self.ids[ij[:, 1], ij[:, 0]]
        if np.any(owner < 0):
            raise RuntimeError("Ordinary field query entered the neural region")
        delta = xy[~inside] - (ij + .5)*h
        out[~inside] = field[owner] + np.einsum('nd,ndb->nb', delta, slopes[owner])
        return out
