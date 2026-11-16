"""Orthogonal scalar-diffusion FVM and an exact local boundary response.

This development module is separate from the legacy four-port benchmark.
Boundary order: left, right, bottom, top; increasing y or x on each side.
Q always means physical heat flux integrated in the outward normal direction.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu, spsolve


@dataclass
class Boundary:
    owner: np.ndarray
    xy: np.ndarray
    normal: np.ndarray
    area: np.ndarray
    distance: np.ndarray


def boundary_geometry(nx: int, ny: int, lengths=(1., 1.)) -> Boundary:
    dx, dy = lengths[0] / nx, lengths[1] / ny
    x, y = (np.arange(nx) + .5) * dx, (np.arange(ny) + .5) * dy
    cells = np.arange(nx * ny).reshape(ny, nx)
    return Boundary(
        np.concatenate((cells[:, 0], cells[:, -1], cells[0], cells[-1])),
        np.concatenate((np.column_stack((np.zeros(ny), y)),
                        np.column_stack((np.full(ny, lengths[0]), y)),
                        np.column_stack((x, np.zeros(nx))),
                        np.column_stack((x, np.full(nx, lengths[1]))))),
        np.repeat([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]],
                  [ny, ny, nx, nx], axis=0),
        np.concatenate((np.full(2 * ny, dy), np.full(2 * nx, dx))),
        np.concatenate((np.full(2 * ny, dx / 2), np.full(2 * nx, dy / 2))),
    )


def graph_matrix(count, edges, diagonal=None):
    """Assemble positive edge energies t*(u_i-u_j)^2/2."""
    rows, columns, values = [], [], []
    diag = np.zeros(count) if diagonal is None else np.array(diagonal, copy=True)
    for i, j, t in edges:
        i, j = np.asarray(i).ravel(), np.asarray(j).ravel()
        t = np.asarray(t, dtype=float)
        t = np.full(i.shape, float(t)) if t.ndim == 0 else t.ravel()
        if len(t) != len(i) or len(j) != len(i):
            raise ValueError("Each edge requires two endpoints and one conductance")
        np.add.at(diag, i, t)
        np.add.at(diag, j, t)
        rows.extend((i, j))
        columns.extend((j, i))
        values.extend((-t, -t))
    ids = np.arange(count)
    rows.append(ids)
    columns.append(ids)
    values.append(diag)
    return sparse.coo_matrix((np.concatenate(values),
                              (np.concatenate(rows), np.concatenate(columns))),
                             shape=(count, count)).tocsc()


@dataclass
class FVSystem:
    matrix: sparse.csc_matrix
    lifting: sparse.csc_matrix
    boundary: Boundary
    tau: np.ndarray
    shape: tuple


def rectangular_system(k, lengths=(1., 1.)) -> FVSystem:
    k = np.asarray(k, dtype=float)
    if k.ndim != 2 or min(k.shape) < 1 or not np.isfinite(k).all() or np.any(k <= 0):
        raise ValueError("A finite positive two-dimensional conductivity is required")
    if min(lengths) <= 0:
        raise ValueError("Lengths must be positive")
    ny, nx = k.shape
    dx, dy = lengths[0] / nx, lengths[1] / ny
    ids = np.arange(nx * ny).reshape(ny, nx)
    tx = dy / (.5 * dx / k[:, :-1] + .5 * dx / k[:, 1:])
    ty = dx / (.5 * dy / k[:-1] + .5 * dy / k[1:])
    boundary = boundary_geometry(nx, ny, lengths)
    tau = k.ravel()[boundary.owner] * boundary.area / boundary.distance
    diagonal = np.bincount(boundary.owner, weights=tau, minlength=nx * ny)
    matrix = graph_matrix(nx * ny,
                          [(ids[:, :-1], ids[:, 1:], tx), (ids[:-1], ids[1:], ty)],
                          diagonal)
    lifting = sparse.csc_matrix((tau, (boundary.owner, np.arange(len(tau)))),
                                shape=(nx * ny, len(tau)))
    return FVSystem(matrix, lifting, boundary, tau, k.shape)


@dataclass
class LocalResponse:
    system: FVSystem
    extension: np.ndarray
    schur: np.ndarray

    def evaluate(self, trace):
        return self.extension @ trace, -self.schur @ trace


def exact_local_response(k) -> LocalResponse:
    system = rectangular_system(k)
    extension = splu(system.matrix).solve(system.lifting.toarray())
    schur = np.diag(system.tau) - system.lifting.T @ extension
    # Retain the raw response: do not hide skew, constant-mode, or sign errors.
    return LocalResponse(system, extension, schur)


def face_aggregation(n: int, m: int):
    """Coarse constant traces prolong by P; integrated fluxes restrict by P.T."""
    if m < 1 or n % m:
        raise ValueError("m must be a positive divisor of n")
    p = np.zeros((4 * n, 4 * m))
    indices = np.arange(4 * n)
    p[indices, (indices // n) * m + (indices % n) // (n // m)] = 1.
    return p


def material(n, kind):
    x, y = np.meshgrid((np.arange(n) + .5) / n, (np.arange(n) + .5) / n)
    if kind == "uniform":
        return np.ones((n, n))
    if kind == "layered":
        return np.where(x < .5, .2, 5.)
    if kind == "checkerboard":
        return np.where((np.floor(4 * x) + np.floor(4 * y)).astype(int) % 2, 3., .3)
    if kind == "smooth":
        return np.exp(.5 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
                      + .2 * np.cos(4 * np.pi * y))
    raise ValueError(kind)


def harmonic_temperature(x, y, length=1.):
    return (np.sin(np.pi * x / length) * np.sinh(np.pi * y / length) / np.sinh(np.pi)
            + .2 * x / length - .1 * y / length)


def harmonic_gradient(x, y, length=1.):
    scale = np.pi / length / np.sinh(np.pi)
    return np.column_stack((scale * np.cos(np.pi * x / length) * np.sinh(np.pi * y / length)
                            + .2 / length,
                            scale * np.sin(np.pi * x / length) * np.cosh(np.pi * y / length)
                            - .1 / length))


def boundary_function(name):
    if name == "constant":
        return lambda x, y: np.full_like(x, 3.)
    if name == "linear":
        return lambda x, y: 1. + .25 * x - .1 * y
    if name == "harmonic":
        return lambda x, y: harmonic_temperature(x, y, 3.)
    raise ValueError(name)


def interface_owners(n):
    ids = np.arange(9 * n * n).reshape(3 * n, 3 * n)
    inner = np.concatenate((ids[n:2*n, n], ids[n:2*n, 2*n-1],
                            ids[n, n:2*n], ids[2*n-1, n:2*n]))
    outer = np.concatenate((ids[n:2*n, n-1], ids[n:2*n, 2*n],
                            ids[n-1, n:2*n], ids[2*n, n:2*n]))
    return inner, outer


def full_reference(k, boundary_fn: Callable):
    n = len(k)
    if k.shape != (n, n):
        raise ValueError("The center block must be square")
    global_k = np.ones((3 * n, 3 * n))
    global_k[n:2*n, n:2*n] = k
    system = rectangular_system(global_k, (3., 3.))
    xy = system.boundary.xy
    g = boundary_fn(xy[:, 0], xy[:, 1])
    rhs = system.lifting @ g
    u = spsolve(system.matrix, rhs)
    inner, outer = interface_owners(n)
    ki = global_k.ravel()[inner]
    q = 2 / (1 / ki + 1) * (u[inner] - u[outer])
    trace = (ki * u[inner] + u[outer]) / (ki + 1)
    exterior_q = system.tau * (u[system.boundary.owner] - g)
    return dict(field=u.reshape(global_k.shape), interface_flux=q, trace=trace,
                residual=float(np.max(np.abs(system.matrix @ u - rhs))),
                external_balance=float(abs(exterior_q.sum())),
                dofs=len(u))


def hybrid_solve(response: LocalResponse, boundary_fn: Callable, *,
                 response_scale=1., flux_bias=None):
    n, nx = response.system.shape
    if n != nx:
        raise ValueError("The center block must be square")
    size = 3 * n
    ids = np.arange(size * size).reshape(size, size)
    outside = np.ones((size, size), dtype=bool)
    outside[n:2*n, n:2*n] = False
    mapping = np.full(size * size, -1, dtype=int)
    mapping[outside.ravel()] = np.arange(outside.sum())
    ordinary = int(outside.sum())
    count = ordinary + 4 * n
    edges = []
    for left, right in ((ids[:, :-1], ids[:, 1:]), (ids[:-1], ids[1:])):
        keep = outside.ravel()[left] & outside.ravel()[right]
        edges.append((mapping[left[keep]], mapping[right[keep]], 1.))
    _, outer = interface_owners(n)
    traces = ordinary + np.arange(4 * n)
    edges.append((mapping[outer], traces, 2.))
    exterior = boundary_geometry(size, size, (3., 3.))
    g = boundary_fn(exterior.xy[:, 0], exterior.xy[:, 1])
    exterior_owner = mapping[exterior.owner]
    diagonal = np.bincount(exterior_owner, weights=np.full(len(g), 2.), minlength=count)
    matrix = graph_matrix(count, edges, diagonal)
    schur = response_scale * response.schur
    matrix = matrix + sparse.block_diag((sparse.csc_matrix((ordinary, ordinary)),
                                         sparse.csc_matrix(schur)), format="csc")
    rhs = np.bincount(exterior_owner, weights=2 * g, minlength=count)
    bias = np.zeros(4 * n) if flux_bias is None else np.asarray(flux_bias)
    rhs[ordinary:] += bias
    solution = spsolve(matrix, rhs)
    trace = solution[ordinary:]
    inner_u = response.extension @ trace
    local_q = -schur @ trace + bias
    outer_q = 2 * (solution[mapping[outer]] - trace)
    field = np.empty((size, size))
    field[outside] = solution[:ordinary]
    field[n:2*n, n:2*n] = inner_u.reshape(n, n)
    exterior_q = 2 * (solution[exterior_owner] - g)
    return dict(field=field, trace=trace, interface_flux=local_q,
                interface_mismatch=float(np.max(np.abs(local_q + outer_q))),
                residual=float(np.max(np.abs(matrix @ solution - rhs))),
                external_balance=float(abs(exterior_q.sum())),
                external_signed_sum=float(exterior_q.sum()),
                local_balance=float(abs(local_q.sum())),
                dofs=count)
