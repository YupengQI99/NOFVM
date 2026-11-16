"""Cell-centered FV for insulated circular holes on a Cartesian raster.

The circle is represented by the union of removed center-inside cells. This is
an explicitly approximate staircase geometry; continuum accuracy must be checked
under refinement. Removed faces carry zero heat flux. No tiny conductivity or
Dirichlet condition is substituted for the insulated hole.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .nofvm_2d import boundary_geometry, graph_matrix


@dataclass
class HoleSystem:
    n: int
    length: float
    active: np.ndarray
    index: np.ndarray
    xy: np.ndarray
    matrix: sparse.csc_matrix
    lifting: sparse.csc_matrix
    owner: np.ndarray
    boundary_xy: np.ndarray
    normals: np.ndarray
    tau: np.ndarray
    dirichlet: np.ndarray
    edges: tuple
    hole_face_count: int
    radius: float

    @property
    def count(self):
        return len(self.xy)

    def factorize(self):
        return splu(self.matrix, permc_spec="MMD_AT_PLUS_A")

    def boundary_flux(self, u, g):
        q = self.tau[:, None]*(u[self.owner]-g)
        q[~self.dirichlet] = 0.
        return q

    def embed(self, values):
        shape = (self.n, self.n)+values.shape[1:]
        field = np.full(shape, np.nan)
        field[self.active] = values
        return field


def hole_system(n, length=.8, radius=.15, centers=None, dirichlet_sides=(True, True, True, True)):
    if n < 4 or length <= 0 or radius < 0:
        raise ValueError("Invalid mesh or geometry")
    h = length/n
    x, y = np.meshgrid((np.arange(n)+.5)*h, (np.arange(n)+.5)*h)
    centers = [(length/2, length/2)] if centers is None else centers
    active = np.ones((n, n), dtype=bool)
    for cx, cy in centers:
        if min(cx, cy, length-cx, length-cy) <= radius:
            raise ValueError("Each hole must be strictly interior")
        active &= (x-cx)**2+(y-cy)**2 >= radius**2
    if not (active[0].all() and active[-1].all() and active[:, 0].all() and active[:, -1].all()):
        raise ValueError("Outer boundary cells must stay active")
    index = np.full((n, n), -1, dtype=int)
    index[active] = np.arange(active.sum())
    ex = active[:, :-1] & active[:, 1:]
    ey = active[:-1] & active[1:]
    edges = ((index[:, :-1][ex], index[:, 1:][ex], 1.),
             (index[:-1][ey], index[1:][ey], 1.))
    outer = boundary_geometry(n, n, (length, length))
    owner = index.ravel()[outer.owner]
    dirichlet = np.repeat(dirichlet_sides, n)
    tau = np.full(4*n, 2.)
    diag = np.bincount(owner, weights=tau*dirichlet, minlength=active.sum())
    matrix = graph_matrix(int(active.sum()), edges, diag)
    lifting = sparse.csc_matrix((tau*dirichlet, (owner, np.arange(4*n))),
                                shape=(int(active.sum()), 4*n))
    hole_faces = int(np.count_nonzero(active[:, :-1] != active[:, 1:])
                     + np.count_nonzero(active[:-1] != active[1:]))
    return HoleSystem(n, length, active, index, np.column_stack((x[active], y[active])),
                      matrix, lifting, owner, outer.xy, outer.normal, tau, dirichlet,
                      edges, hole_faces, radius)


def circle_harmonic(xy, center=(.4, .4), radius=.15, mode=1, sine=False):
    """Analytical Laplace solution with zero normal derivative on r=radius."""
    xx, yy = xy[:, 0]-center[0], xy[:, 1]-center[1]
    z = xx+1j*yy
    value = z**mode+radius**(2*mode)/z**mode
    derivative = mode*z**(mode-1)-mode*radius**(2*mode)/z**(mode+1)
    if sine:
        # Im(z^m - a^(2m)/z^m) has the same radial Neumann cancellation.
        value = z**mode-radius**(2*mode)/z**mode
        derivative = mode*z**(mode-1)+mode*radius**(2*mode)/z**(mode+1)
        return value.imag, np.column_stack((derivative.imag, derivative.real))
    return value.real, np.column_stack((derivative.real, -derivative.imag))


def least_squares_gradient(system, values):
    """Independent center-to-center gradient on retained neighbors (diagnostic)."""
    values = np.asarray(values)
    scalar = values.ndim == 1
    u = values[:, None] if scalar else values
    gradient = np.zeros((system.count, 2, u.shape[1]))
    for dim, (left, right, _) in enumerate(system.edges):
        delta = (u[right]-u[left])/(system.length/system.n)
        counts = np.zeros(system.count)
        np.add.at(counts, left, 1); np.add.at(counts, right, 1)
        np.add.at(gradient[:, dim], left, delta); np.add.at(gradient[:, dim], right, delta)
        gradient[:, dim] /= np.maximum(counts[:, None], 1)
    return gradient[:, :, 0] if scalar else gradient


def boundary_prolongation(n, m):
    if n % m or m < 1:
        raise ValueError("Face count must divide fine boundary resolution")
    ids = np.arange(4*n)
    return sparse.csc_matrix((np.ones(4*n), (ids, (ids//n)*m+(ids%n)//(n//m))), shape=(4*n, 4*m))


def condensed_response(system, m):
    """Classical response and boundary-to-field basis, including source-free constants."""
    P = boundary_prolongation(system.n, m)
    B = (system.lifting@P).tocsc()
    H = system.factorize().solve(B.toarray())
    T = (P.T@sparse.diags(system.tau*system.dirichlet)@P).toarray()
    S = T-B.T@H
    return dict(H=H, S=S, B=B, T=T, P=P)


class SingleHoleHybrid:
    """Fine learned/cached center and real coarse ordinary FV exterior.

    Outer square [0,1.6]^2, local square [.4,1.2]^2. Its four interfaces
    retain m constant trace segments per side; flux restricts by P.T.
    """
    def __init__(self, local_system, response, m):
        if m % 2 or local_system.n % m:
            raise ValueError("An even dividing interface resolution is required")
        self.local_system, self.response, self.m = local_system, response, m
        self.n, self.h = 2*m, .8/m
        n, lo, hi = self.n, m//2, 3*m//2
        ordinary = np.ones((n, n), dtype=bool)
        ordinary[lo:hi, lo:hi] = False
        ids = np.full((n, n), -1, dtype=int)
        ids[ordinary] = np.arange(ordinary.sum())
        self.ordinary, self.ids, self.count = ordinary, ids, int(ordinary.sum())
        edges = []
        for a,b in ((ids[:, :-1], ids[:, 1:]), (ids[:-1], ids[1:])):
            valid = (a >= 0) & (b >= 0)
            edges.append((a[valid], b[valid], 1.))
        self.interface_owners = np.r_[ids[lo:hi, lo-1], ids[lo:hi, hi],
                                      ids[lo-1, lo:hi], ids[hi, lo:hi]]
        trace = self.count+np.arange(4*m)
        edges.append((self.interface_owners, trace, 2.))
        self.outer = boundary_geometry(n, n, (1.6, 1.6))
        self.outer_owner = ids.ravel()[self.outer.owner]
        self.d = np.repeat([True, True, False, False], n)
        size = self.count+4*m
        diagonal = np.bincount(self.outer_owner, weights=2.*self.d, minlength=size)
        self.matrix = graph_matrix(size, edges, diagonal)
        self.matrix += sparse.block_diag((sparse.csc_matrix((self.count, self.count)),
                                          sparse.csc_matrix(response["S"])), format="csc")
        self.factor = splu(self.matrix, permc_spec="MMD_AT_PLUS_A")

    def solve(self, boundary_function):
        g = np.asarray(boundary_function(self.outer.xy))
        if g.ndim == 1:
            g = g[:, None]
        rhs = np.zeros((self.matrix.shape[0], g.shape[1]))
        np.add.at(rhs, self.outer_owner, 2*self.d[:, None]*g)
        solution = self.factor.solve(rhs)
        trace = solution[self.count:]
        local_field = self.response["H"]@trace
        q = -self.response["S"]@trace
        qouter = 2*(solution[self.interface_owners]-trace)
        qexternal = 2*self.d[:, None]*(solution[self.outer_owner]-g)
        # Linear reconstructions of ordinary cell averages. Slopes use actual
        # neighbors, interface traces and exterior Dirichlet/zero-Neumann data.
        values = np.zeros((self.n, self.n, g.shape[1]))
        values[self.ordinary] = solution[:self.count]
        numer = np.zeros((self.count, 2, g.shape[1])); denom = np.zeros((self.count, 2))
        for dim, (a,b) in enumerate(((self.ids[:, :-1],self.ids[:, 1:]),(self.ids[:-1],self.ids[1:]))):
            valid = (a>=0)&(b>=0); a,b = a[valid],b[valid]
            slope = (solution[b]-solution[a])/self.h
            np.add.at(numer[:,dim],a,slope); np.add.at(numer[:,dim],b,slope)
            np.add.at(denom[:,dim],a,1); np.add.at(denom[:,dim],b,1)
        for side, (dim,sign) in enumerate(((0,1),(0,-1),(1,1),(1,-1))):
            sl = slice(side*self.m,(side+1)*self.m); owner = self.interface_owners[sl]
            np.add.at(numer[:,dim],owner,sign*(trace[sl]-solution[owner])/(self.h/2))
            np.add.at(denom[:,dim],owner,1)
        for side, (dim,sign) in enumerate(((0,-1),(0,1),(1,-1),(1,1))):
            sl = slice(side*self.n,(side+1)*self.n); owner = self.outer_owner[sl]
            slope = sign*(g[sl]-solution[owner])/(self.h/2) if side < 2 else np.zeros_like(g[sl])
            np.add.at(numer[:,dim],owner,slope); np.add.at(denom[:,dim],owner,1)
        slopes = numer/denom[:,:,None]
        return dict(ordinary=solution[:self.count], trace=trace, local_field=local_field, flux=q,
                    slopes=slopes, residual=float(np.max(abs(self.matrix@solution-rhs))),
                    interface_balance=float(np.max(abs(q+qouter))),
                    region_balance=float(np.max(abs(q.sum(axis=0)))),
                    exterior_balance=float(np.max(abs(qexternal.sum(axis=0)))))

    def reconstruct(self, solution, fine_system):
        x,y = fine_system.xy.T
        ci,cj = np.floor(x/self.h).astype(int), np.floor(y/self.h).astype(int)
        idx = self.ids[cj,ci]
        outside = idx>=0
        field = np.empty((fine_system.count,solution["trace"].shape[1]))
        delta = fine_system.xy[outside]-(np.c_[ci[outside],cj[outside]]+.5)*self.h
        field[outside] = (solution["ordinary"][idx[outside]]
            + np.einsum("nd,ndb->nb",delta,solution["slopes"][idx[outside]]))
        localxy = fine_system.xy[~outside]-.4
        nn, length = self.local_system.n,self.local_system.length
        ii = np.minimum(np.floor(localxy[:,0]/length*nn).astype(int),nn-1)
        jj = np.minimum(np.floor(localxy[:,1]/length*nn).astype(int),nn-1)
        loc = self.local_system.index[jj,ii]
        if np.any(loc<0):
            raise ValueError("Reconstruction requested inside a raster cell removed from the local model")
        field[~outside] = solution["local_field"][loc]
        return field


def full_interface_flux(system, field, m, local_length=.8):
    """Flux across [.4,1.2]^2 in the independent full uniform FV mesh."""
    n = int(round(local_length/(system.length/system.n)))
    lo = int(round(.4/(system.length/system.n))); hi = lo+n
    index = system.index
    inside = np.r_[index[lo:hi,lo],index[lo:hi,hi-1],index[lo,lo:hi],index[hi-1,lo:hi]]
    outside = np.r_[index[lo:hi,lo-1],index[lo:hi,hi],index[lo-1,lo:hi],index[hi,lo:hi]]
    q = field[inside]-field[outside]
    return np.asarray(boundary_prolongation(n,m).T@q)
