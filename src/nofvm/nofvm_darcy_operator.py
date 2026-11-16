"""Material-conditioned boundary-to-field/flux neural operator for D1/D2.

The homogeneous response is a fixed classical baseline. Variable-material
response matrices are training labels, never inference buffers. Boundary
projection enforces consistency of the learned field and integrated flux.
"""
import math

import numpy as np
import torch
from torch import nn

from .nofvm_darcy import trace_prolongation
from .nofvm_2d import rectangular_system
from .nofvm_hole_operator import coordinate_features


class DarcyBoundaryOperator(nn.Module):
    def __init__(self, base_h, base_s, n=64, m=32, rank=32):
        super().__init__()
        self.n, self.m, self.rank = n, m, rank
        faces = 4*m
        if base_h.shape != (n*n, faces) or base_s.shape != (faces, faces) or n <= m:
            raise ValueError("The projection requires finer interior cells than trace segments")
        i, j = np.triu_indices(faces, 1)
        weights = -base_s[i, j]
        if np.any(weights <= 0) or np.any(base_h <= 0):
            raise ValueError("The homogeneous harmonic response must have positive weights")
        system = rectangular_system(np.ones((n, n)))
        lifting = system.lifting @ trace_prolongation(n, m)
        boundary_ids = np.flatnonzero(np.asarray(abs(lifting).sum(axis=1)).ravel())
        pattern = lifting[boundary_ids].toarray().T / 2
        if np.linalg.matrix_rank(pattern) != faces:
            raise ValueError("Boundary projection is not full row rank")
        x, y = np.meshgrid((np.arange(n) + .5)/n, (np.arange(n) + .5)/n)
        features = coordinate_features(np.column_stack((x.ravel(), y.ravel())), length=1.)
        for name, value in (("log_base_h", np.log(base_h)),
                            ("log_base_weights", np.log(weights)),
                            ("features", features), ("boundary_pattern", pattern)):
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
        for name, value in (("edge_i", i), ("edge_j", j), ("boundary_ids", boundary_ids)):
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.long))
        convolution_size = (n+3)//4
        pooled_size = min(4, convolution_size)
        if convolution_size % pooled_size:
            raise ValueError("The declared sensor grid must admit nonoverlapping average pooling")
        pooling_stride = convolution_size//pooled_size
        self.branch = nn.Sequential(
            nn.Conv2d(1, 8, 5, stride=2, padding=2), nn.Tanh(),
            nn.Conv2d(8, 16, 5, stride=2, padding=2), nn.Tanh(),
            nn.AvgPool2d(pooling_stride, stride=pooling_stride), nn.Flatten(),
            nn.Linear(16*pooled_size*pooled_size, 128), nn.Tanh(),
            nn.Linear(128, len(i) + rank*faces),
        )
        self.trunk = nn.Sequential(nn.Linear(18, 64), nn.Tanh(),
                                   nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, rank))
        nn.init.zeros_(self.branch[-1].weight)
        nn.init.zeros_(self.branch[-1].bias)

    def encode(self, log_k):
        if log_k.shape[1:] != (self.n, self.n):
            raise ValueError("Material sensors must use the declared local grid")
        mean = log_k.mean(dim=(1, 2), keepdim=True)
        code = self.branch((log_k - mean)[:, None])
        return code, mean.reshape(-1, 1)

    def raw_field(self, code, ids=None):
        features = self.features if ids is None else self.features[ids]
        baseline = self.log_base_h if ids is None else self.log_base_h[ids]
        coefficients = code[:, len(self.edge_i):].reshape(-1, self.rank, 4*self.m)
        correction = torch.einsum('xr,brf->bxf', self.trunk(features), coefficients) / math.sqrt(self.rank)
        return torch.softmax(baseline[None] + correction, dim=2)

    def schur(self, code, mean):
        log_weights = self.log_base_weights + mean + code[:, :len(self.edge_i)]
        weights = log_weights.exp()
        matrix = weights.new_zeros((len(weights), 4*self.m, 4*self.m))
        matrix[:, self.edge_i, self.edge_j] = -weights
        matrix[:, self.edge_j, self.edge_i] = -weights
        matrix -= torch.diag_embed(matrix.sum(dim=2))
        return matrix, log_weights

    def boundary_matrices(self, log_k):
        k = log_k.flatten(1)[:, self.boundary_ids].exp()
        boundary = self.boundary_pattern[None] * (2*k[:, None, :])
        trace_matrix = torch.diag_embed(boundary.sum(dim=2))
        gram = boundary @ boundary.transpose(1, 2)
        projection = torch.linalg.solve(gram, boundary).transpose(1, 2)
        return boundary, trace_matrix, projection

    def projected_boundary(self, code, schur, matrices):
        boundary, trace_matrix, projection = matrices
        raw = self.raw_field(code, self.boundary_ids)
        return raw + projection @ (trace_matrix - schur - boundary @ raw)

    def forward(self, log_k, ids=None, matrices=None):
        code, mean = self.encode(log_k)
        schur, log_weights = self.schur(code, mean)
        matrices = self.boundary_matrices(log_k) if matrices is None else matrices
        boundary = self.projected_boundary(code, schur, matrices)
        field = self.raw_field(code, ids)
        if ids is None:
            field = field.index_copy(1, self.boundary_ids, boundary)
        return dict(H=field, S=schur, log_weights=log_weights, boundary_field=boundary)


def deploy_response(model, k):
    """Generate one material response; no target-material PDE solve is used."""
    k = np.asarray(k, dtype=float)
    parameter = next(model.parameters())
    inputs = torch.as_tensor(np.log(k)[None], dtype=parameter.dtype, device=parameter.device)
    with torch.no_grad():
        code, mean = model.encode(inputs)
        raw = model.raw_field(code)[0].cpu().numpy().astype(float)
        _, log_weights = model.schur(code, mean)
        weights = np.exp(log_weights[0].cpu().numpy().astype(float))
        i, j = model.edge_i.cpu().numpy(), model.edge_j.cpu().numpy()
        ids = model.boundary_ids.cpu().numpy()
        pattern = model.boundary_pattern.cpu().numpy().astype(float)
    raw /= raw.sum(axis=1, keepdims=True)
    schur = np.zeros((4*model.m, 4*model.m))
    schur[i, j] = schur[j, i] = -weights
    schur -= np.diag(schur.sum(axis=1))
    boundary = pattern * (2*k.ravel()[ids][None, :])
    trace_matrix = np.diag(boundary.sum(axis=1))
    raw[ids] += boundary.T @ np.linalg.solve(boundary @ boundary.T,
                                            trace_matrix - schur - boundary @ raw[ids])
    return dict(H=raw, S=schur)
