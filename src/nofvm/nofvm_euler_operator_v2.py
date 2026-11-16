"""Positive conservative decoder: a mean-preserving limiter inside forward keeps every
reconstructed state admissible, so training and inference use the same decoder."""
import numpy as np
import torch
from nofvm.nofvm_euler_operator import EulerLocalDecoder as BaseDecoder
from nofvm.nofvm_euler_operator import decoder_loss, torch_rusanov, torch_flux


class EulerLocalDecoder(BaseDecoder):
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
        delta = linear + .1 * amplitude.unsqueeze(2) * correction
        delta -= delta.mean(dim=2, keepdim=True)
        mean = context[:, 2:-2, None]
        delta = delta * self.scale
        # A common factor for all queries and conserved components in each cell
        # preserves the mean. The convex Euler admissible set contains the mean.
        # Detach limiter selection; gradients still pass through limited deltas.
        with torch.no_grad():
            def admissible(theta):
                value = mean + theta[..., None, None] * delta
                rho = value[..., 0]
                p = .4 * (value[..., 2] - .5 * value[..., 1].square() / rho.clamp_min(1e-8))
                return ((rho >= 1e-4) & (p >= 1e-4)).all(dim=-1)
            theta = torch.ones_like(mean[..., 0, 0])
            if not bool(admissible(theta).all()):
                lo = torch.zeros_like(theta)
                hi = theta.clone()
                for _ in range(20):
                    mid = .5 * (lo + hi)
                    valid = admissible(mid)
                    lo = torch.where(valid, mid, lo)
                    hi = torch.where(valid, hi, mid)
                theta = torch.where(admissible(theta), theta, .99 * lo)
        return mean + theta[..., None, None] * delta

