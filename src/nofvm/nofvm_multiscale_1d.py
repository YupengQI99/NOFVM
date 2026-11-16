"""Fixed-physics affine neural subdomains with conservative 1-D FV assembly.

This is a boundary-to-field/flux operator, not a material-generalization model.
Flux orientation is always left to right: j = v*u - k*u_x and j_x = source.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve
import torch
from torch import nn


@dataclass(frozen=True)
class Problem:
    name: str
    length: float
    blocks: int
    velocity: float
    kmean: float
    source_integral: float

    @property
    def total_length(self):
        return self.length * self.blocks

    @property
    def field_scale(self):
        return self.length * self.source_integral / self.kmean

    @property
    def conductance_scale(self):
        return self.kmean / self.length

    def conductivity(self, x):
        x = np.asarray(x)
        return .8 + .5 * np.sin(32 * np.pi * x) if self.name == "B1" else np.ones_like(x)

    def source(self, x):
        x = np.asarray(x)
        s = x % 1
        return np.full_like(x, .5) if self.name == "B1" else s * (1 - s)

    def primitive(self, x):
        x = np.asarray(x)
        m, s = np.floor(x), x % 1
        return .5*x if self.name == "B1" else m/6 + s*s/2 - s*s*s/3


PROBLEMS = {
    "B1": Problem("B1", .125, 8, 0., .8, .0625),
    "B2": Problem("B2", 1., 10, 1., 1., 1/6),
}


def integrate_intervals(function, left, right, order=8):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    left, right = np.asarray(left), np.asarray(right)
    points = (left[..., None]+right[..., None])/2 + (right-left)[..., None]*nodes/2
    return np.sum(function(points)*weights, axis=-1)*(right-left)/2


def continuum_basis(problem, x, length=None, order=8):
    """Independent integrating-factor solution with resolved quadrature panels."""
    length = problem.length if length is None else length
    x = np.asarray(x, dtype=float)
    if np.any(x < 0) or np.any(x > length):
        raise ValueError("Queries must lie in the reference domain")
    # Include every query and every periodic breakpoint; never integrate a whole
    # oscillatory domain with a single small Gauss rule.
    panels = int(round(length/problem.length))*512
    knots = np.unique(np.r_[np.linspace(0, length, panels+1), x.ravel()])
    idx = np.searchsorted(knots, x)
    if problem.velocity == 0:
        r = np.r_[0., np.cumsum(integrate_intervals(
            lambda t: 1/problem.conductivity(t), knots[:-1], knots[1:], order))]
        w = np.r_[0., np.cumsum(integrate_intervals(
            lambda t: problem.primitive(t)/problem.conductivity(t),
            knots[:-1], knots[1:], order))]
        ccap, csrc = 1/r[-1], -w[-1]/r[-1]
        h, z = r[idx]/r[-1], -csrc*r[idx]-w[idx]
    else:
        # B2 has constant v=k=1. This path is intentionally separate from FV.
        a = -np.expm1(-knots)
        w = np.r_[0., np.cumsum(integrate_intervals(
            lambda t: np.exp(-t)*problem.primitive(t), knots[:-1], knots[1:], order))]
        ccap, csrc = 1/np.expm1(length), -w[-1]/a[-1]
        h = np.expm1(x)/np.expm1(length)
        z = np.exp(x)*(-csrc*a[idx]-w[idx])
    k = problem.conductivity(x)
    dh = (problem.velocity*h+ccap)/k
    dz = (problem.velocity*z-csrc-problem.primitive(x))/k
    return dict(h=h, z=z, dh=dh, dz=dz, C=float(ccap), c=float(csrc))


def bernoulli(z):
    z = np.asarray(z, dtype=float)
    small = np.abs(z) < 1e-5
    result = np.empty_like(z)
    result[small] = 1-z[small]/2+z[small]**2/12-z[small]**4/720
    result[~small] = z[~small]/np.expm1(z[~small])
    return result


def fv_geometry(problem, n, length=None):
    length = problem.length if length is None else length
    faces = np.linspace(0, length, n+1)
    centers = (faces[:-1]+faces[1:])/2
    points = np.r_[0., centers, length]
    resistance = integrate_intervals(lambda t: 1/problem.conductivity(t),
                                     points[:-1], points[1:])
    tau = 1/resistance
    pe = problem.velocity/tau
    left, right = tau*bernoulli(-pe), tau*bernoulli(pe)
    source = integrate_intervals(problem.source, faces[:-1], faces[1:])
    return dict(x=centers, faces=faces, left=left, right=right, source=source)


def fv_basis(problem, n, length=None):
    """Independent monolithic cell-centered FV; three actual RHS solves."""
    geom = fv_geometry(problem, n, length)
    left, right = geom["left"], geom["right"]
    matrix = diags((-left[1:-1], right[:-1]+left[1:], -right[1:-1]),
                   (-1, 0, 1), format="csc")
    rhs = np.zeros((n, 3))
    rhs[0, 0], rhs[-1, 1] = left[0], right[-1]
    rhs[:, 2] = geom["source"]
    u = spsolve(matrix, rhs)
    extended = np.vstack(([1., 0., 0.], u, [0., 1., 0.]))
    jface = left[:, None]*extended[:-1]-right[:, None]*extended[1:]
    jcenter = (jface[:-1]+jface[1:])/2
    du = (problem.velocity*u-jcenter)/problem.conductivity(geom["x"])[:, None]
    return dict(**geom, u=u, du=du, jface=jface, jcenter=jcenter,
                C=float(-jface[0, 1]), c=float(jface[0, 2]),
                solve_residual=float(np.max(np.abs(matrix@u-rhs))))


def training_labels(problem, n=256):
    fv = fv_basis(problem, n)
    h, z = fv["u"][:, 1], fv["u"][:, 2]
    k, x = problem.conductivity(fv["x"]), fv["x"]
    dh = (problem.velocity*h+fv["C"])/k
    dz = (problem.velocity*z-fv["c"]-problem.primitive(x))/k
    return dict(x=x, h=h, z=z, dh=dh, dz=dz, C=fv["C"], c=fv["c"])


class AffineLocalOperator(nn.Module):
    """Exactly affine boundary branch, learned smooth coordinate trunk/response."""
    def __init__(self, problem):
        super().__init__()
        self.problem = problem
        self.log_conductance = nn.Parameter(torch.tensor(0., dtype=torch.float64))
        self.source_response = nn.Parameter(torch.tensor(-.5, dtype=torch.float64))
        self.trunk = nn.ModuleList([nn.Linear(17, 64), nn.Linear(64, 64), nn.Linear(64, 2)])
        nn.init.zeros_(self.trunk[-1].weight)
        nn.init.zeros_(self.trunk[-1].bias)

    def responses(self):
        return (self.problem.conductance_scale*self.log_conductance.exp(),
                self.problem.source_integral*self.source_response)

    def correction(self, s):
        modes = 2*math.pi*torch.arange(1, 9, dtype=s.dtype, device=s.device)[None, :]
        phase = s*modes
        y = torch.cat((s, torch.sin(phase), torch.cos(phase)), dim=1)
        dy = torch.cat((torch.ones_like(s), modes*torch.cos(phase), -modes*torch.sin(phase)), dim=1)
        for i, layer in enumerate(self.trunk):
            y = layer(y)
            dy = nn.functional.linear(dy, layer.weight)
            if i < 2:
                y = torch.tanh(y)
                dy *= 1-y.square()
        return y, dy

    def forward(self, s):
        p = self.problem
        C, c = self.responses()
        k0, k1 = float(p.conductivity(0.)), float(p.conductivity(p.length))
        slopes0 = torch.stack((p.length*C/k0, -p.length*c/k0))
        slopes1 = torch.stack((p.length*(p.velocity+C)/k1,
                               -p.length*(c+p.source_integral)/k1))
        h01, h10, h11 = 3*s*s-2*s**3, s**3-2*s*s+s, s**3-s*s
        d01, d10, d11 = 6*s-6*s*s, 3*s*s-4*s+1, 3*s*s-2*s
        bubble, dbubble = s*s*(1-s)**2, 2*s*(1-s)*(1-2*s)
        correction, derivative = self.correction(s)
        scale = s.new_tensor([1., p.field_scale])
        field = torch.cat((h01, torch.zeros_like(s)), dim=1)
        field = field+h10*slopes0+h11*slopes1+bubble*correction*scale
        dfield = torch.cat((d01, torch.zeros_like(s)), dim=1)
        dfield = dfield+d10*slopes0+d11*slopes1+(dbubble*correction+bubble*derivative)*scale
        return field, dfield


def predict_basis(model, x):
    parameter = next(model.parameters())
    with torch.no_grad():
        s = torch.as_tensor(np.asarray(x)[:, None]/model.problem.length,
                            dtype=parameter.dtype, device=parameter.device)
        values, derivative = model(s)
        C, c = model.responses()
    values = values.cpu().numpy()
    derivative = derivative.cpu().numpy()/model.problem.length
    return dict(h=values[:, 0], z=values[:, 1], dh=derivative[:, 0], dz=derivative[:, 1],
                C=float(C.cpu()), c=float(c.cpu()))


def combine_basis(basis, loads):
    """loads columns = global left BC, global right BC, source multiplier."""
    loads = np.asarray(loads)
    field = loads[:, 0, None] + (loads[:, 1]-loads[:, 0])[:, None]*basis["h"]
    field += loads[:, 2, None]*basis["z"]
    derivative = (loads[:, 1]-loads[:, 0])[:, None]*basis["dh"]
    derivative += loads[:, 2, None]*basis["dz"]
    return field, derivative


def assemble_hybrid(problem, n, mask, learned):
    """Real FV cells + unknown interface traces. Never receives true traces.

    Returns three global basis responses. Reuse across the requested external
    loads is exact linear superposition, not extra PDE training data.
    """
    mask = np.asarray(mask, dtype=bool)
    if len(mask) != problem.blocks:
        raise ValueError("Wrong number of subdomains")
    bcount = problem.blocks
    ordinary_count = int(np.sum(~mask))*n
    size = ordinary_count+bcount-1
    cells, cursor = {}, 0
    for b in range(bcount):
        if not mask[b]:
            cells[b] = np.arange(cursor, cursor+n)
            cursor += n
    geom = fv_geometry(problem, n)
    # Expressions are (unknown-coefficient dictionary, known-load vector).
    def unknown(i):
        return ({int(i): 1.}, np.zeros(3))

    def face(b):
        if b == 0:
            return ({}, np.array([1., 0., 0.]))
        if b == bcount:
            return ({}, np.array([0., 1., 0.]))
        return unknown(ordinary_count+b-1)

    def expression(terms, source=0.):
        coeff, known = {}, np.array([0., 0., source])
        for weight, (values, load) in terms:
            for i, value in values.items():
                coeff[i] = coeff.get(i, 0.)+weight*value
            known += weight*load
        return coeff, known

    rows, cols, values = [], [], []
    rhs = np.zeros((size, 3))

    def add_row(row, terms, source=0.):
        coeff, known = expression(terms)
        for col, value in coeff.items():
            rows.append(row); cols.append(col); values.append(value)
        rhs[row] = -known
        rhs[row, 2] += source

    endflux, allflux = [], {}
    for b in range(bcount):
        if mask[b]:
            C, c, F = learned["C"], learned["c"], problem.source_integral
            terms = [(problem.velocity+C, face(b)), (-C, face(b+1))]
            endflux.append((expression(terms, c), expression(terms, c+F)))
        else:
            states = [face(b)] + [unknown(i) for i in cells[b]] + [face(b+1)]
            fluxes = [expression([(geom["left"][i], states[i]),
                                    (-geom["right"][i], states[i+1])]) for i in range(n+1)]
            allflux[b] = fluxes
            endflux.append((fluxes[0], fluxes[-1]))
            for i, row in enumerate(cells[b]):
                add_row(row, [(1., fluxes[i+1]), (-1., fluxes[i])], geom["source"][i])
    for b in range(1, bcount):
        add_row(ordinary_count+b-1, [(1., endflux[b][0]), (-1., endflux[b-1][1])])
    matrix = coo_matrix((values, (rows, cols)), shape=(size, size)).tocsc()
    solution = spsolve(matrix, rhs)

    def evaluate(expr):
        coeff, known = expr
        result = known.copy()
        for i, value in coeff.items():
            result += value*solution[i]
        return result

    traces = np.array([evaluate(face(b)) for b in range(bcount+1)])
    ends = np.array([[evaluate(item) for item in pair] for pair in endflux])
    field, gradient, jcenter = [], [], []
    for b in range(bcount):
        if mask[b]:
            g0, delta = traces[b], traces[b+1]-traces[b]
            u = g0+learned["h"][:, None]*delta
            u[:, 2] += learned["z"]
            du = learned["dh"][:, None]*delta
            du[:, 2] += learned["dz"]
            j = problem.velocity*u-problem.conductivity(geom["x"])[:, None]*du
        else:
            u = solution[cells[b]]
            jf = np.array([evaluate(f) for f in allflux[b]])
            j = (jf[:-1]+jf[1:])/2
            du = (problem.velocity*u-j)/problem.conductivity(geom["x"])[:, None]
        field.append(u); gradient.append(du); jcenter.append(j)
    target_balance = np.array([0., 0., problem.source_integral])
    return dict(x=np.concatenate([geom["x"]+b*problem.length for b in range(bcount)]),
                u=np.concatenate(field), du=np.concatenate(gradient), j=np.concatenate(jcenter),
                traces=traces, ends=ends, unknowns=size, ordinary_cells=ordinary_count,
                interface_unknowns=bcount-1,
                linear_residual=float(np.max(np.abs(matrix@solution-rhs))),
                region_balance=float(np.max(np.abs(ends[:, 1]-ends[:, 0]-target_balance))),
                interface_balance=float(np.max(np.abs(ends[:-1, 1]-ends[1:, 0]))))
