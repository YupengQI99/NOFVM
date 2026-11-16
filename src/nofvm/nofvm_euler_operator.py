"""Convolutional branch and fixed cosine query basis for an Euler local field.

Inputs are conservative mean functions on 16 internal cells and two exterior
context averages on each side. A mean-free query basis reconstructs eight fine
values per internal cell. The time integrator retains only the means.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn


def torch_primitive(q):
    rho = q[..., 0]
    vel = q[..., 1] / rho
    pressure = .4 * (q[..., 2] - .5 * q[..., 1] * vel)
    return torch.stack((rho, vel, pressure), dim=-1)


def torch_flux(q):
    w = torch_primitive(q)
    rho, vel, pressure = w.unbind(-1)
    flux = torch.stack((q[..., 1], q[..., 1] * vel + pressure, vel * (q[..., 2] + pressure)), dim=-1)
    speed = vel.abs() + (1.4 * pressure.clamp_min(1e-9) / rho.clamp_min(1e-9)).sqrt()
    return flux, speed


def torch_rusanov(left, right):
    fl, al = torch_flux(left)
    fr, ar = torch_flux(right)
    return .5 * (fl + fr) - .5 * torch.maximum(al, ar).unsqueeze(-1) * (right - left)


class EulerLocalDecoder(nn.Module):
    def __init__(self, ratio=8, width=48):
        super().__init__()
        self.ratio = ratio
        self.branch = nn.Sequential(nn.Conv1d(3, width, 5, padding=2), nn.Tanh(),
                                    nn.Conv1d(width, width, 5, padding=2), nn.Tanh(),
                                    nn.Conv1d(width, width, 5, padding=2), nn.Tanh(),
                                    nn.Conv1d(width, 3 * (ratio - 1), 1))
        nn.init.zeros_(self.branch[-1].weight)
        nn.init.zeros_(self.branch[-1].bias)
        coordinate = (torch.arange(ratio, dtype=torch.float32) + .5) / ratio
        basis = torch.cos(torch.pi * torch.arange(1, ratio, dtype=torch.float32)[:, None] * coordinate)
        basis -= basis.mean(dim=1, keepdim=True)
        self.register_buffer('basis', basis)
        self.register_buffer('coordinate', coordinate - .5)
        self.register_buffer('scale', torch.tensor([1., 2., 3.]))

    def forward(self, context):
        normalized = context / self.scale
        center = normalized[:, 2:-2]
        dl = center - normalized[:, 1:-3]
        dr = normalized[:, 3:-1] - center
        slope = .5 * (dl.sign() + dr.sign()) * torch.minimum(dl.abs(), dr.abs())
        linear = slope.unsqueeze(2) * self.coordinate[None, None, :, None]
        amplitude = torch.maximum(dl.abs(), dr.abs()).amax(dim=-1, keepdim=True)
        coeff = self.branch(normalized.transpose(1, 2))[:, :, 2:-2].transpose(1, 2)
        coeff = coeff.reshape(len(context), center.shape[1], 3, self.ratio - 1)
        correction = torch.einsum('bmck,kr->bmrc', coeff.tanh(), self.basis)
        delta = linear + amplitude.unsqueeze(2) * correction
        delta -= delta.mean(dim=2, keepdim=True)
        return (center.unsqueeze(2) + delta) * self.scale

    def numpy_decoder(self, context):
        with torch.no_grad():
            value = self(torch.as_tensor(context, dtype=torch.float32)[None])[0].cpu().numpy().astype(float)
        # Restore the exact float64 mean supplied by the coupled FV solver.
        return context[2:-2, None] + value - value.mean(axis=1, keepdims=True)


def decoder_loss(model, fine_context):
    batch, _, channels = fine_context.shape
    r = model.ratio
    coarse = fine_context.reshape(batch, 20, r, channels).mean(dim=2)
    pred = model(coarse)
    target = fine_context[:, 2 * r:-2 * r].reshape(batch, 16, r, channels)
    scaled = (pred - target) / model.scale
    # All internal points, with modest emphasis on the faces used for coupling.
    field = scaled.square().mean()
    edge = scaled[:, :, (0, -1)].square().mean()
    left = torch.cat((fine_context[:, 2 * r - 1:2 * r], pred[:, :, -1]), dim=1)
    right = torch.cat((pred[:, :, 0], fine_context[:, -2 * r:-2 * r + 1]), dim=1)
    truth_left = fine_context[:, torch.arange(2 * r - 1, 18 * r, r, device=fine_context.device)]
    truth_right = fine_context[:, torch.arange(2 * r, 18 * r + 1, r, device=fine_context.device)]
    flux_scale = model.scale * 2.
    flux = ((torch_rusanov(left, right) - torch_rusanov(truth_left, truth_right)) / flux_scale).square().mean()
    w = torch_primitive(pred)
    positive = torch.relu(1e-5 - w[..., 0]).square().mean() + torch.relu(1e-5 - w[..., 2]).square().mean()
    loss = field + .5 * edge + .1 * flux + 10 * positive
    return loss, dict(field=field.detach(), edge=edge.detach(), flux=flux.detach(), positive=positive.detach())
