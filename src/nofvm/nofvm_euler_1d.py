"""Euler finite volumes and a compressed conservative dynamic subdomain.

The learned decoder is supplied by the caller. Only coarse conservative means
are advanced inside the subdomain; reconstructed fine values are transient
queries, never a hidden fine-grid evolution. Every shared face has one flux.
"""
from __future__ import annotations
import time
import numpy as np
from scipy.optimize import brentq

GAMMA = 1.4


def conserved(primitive):
    w = np.asarray(primitive, dtype=float)
    rho, vel, pressure = np.moveaxis(w, -1, 0)
    return np.stack((rho, rho * vel, pressure / (GAMMA - 1) + .5 * rho * vel**2), axis=-1)


def primitive(state):
    q = np.asarray(state)
    rho = q[..., 0]
    vel = q[..., 1] / rho
    pressure = (GAMMA - 1) * (q[..., 2] - .5 * q[..., 1] * vel)
    return np.stack((rho, vel, pressure), axis=-1)


def admissible(state, floor=1e-10):
    w = primitive(state)
    return bool(np.isfinite(w).all() and (w[..., 0] > floor).all() and (w[..., 2] > floor).all())


def physical_flux(state):
    w = primitive(state)
    rho, vel, pressure = np.moveaxis(w, -1, 0)
    flux = np.stack((state[..., 1], state[..., 1] * vel + pressure,
                     vel * (state[..., 2] + pressure)), axis=-1)
    speed = abs(vel) + np.sqrt(GAMMA * pressure / rho)
    return flux, speed


def rusanov(left, right):
    fl, al = physical_flux(left)
    fr, ar = physical_flux(right)
    return .5 * (fl + fr) - .5 * np.maximum(al, ar)[..., None] * (right - left)


def initial_average(n, left, right, discontinuity=.5):
    fraction = np.clip(discontinuity * n - np.arange(n), 0., 1.)
    return fraction[:, None] * conserved(left) + (1 - fraction[:, None]) * conserved(right)


def minmod(a, b):
    return .5 * (np.sign(a) + np.sign(b)) * np.minimum(abs(a), abs(b))


def limit_reconstruction(values, mean):
    """Mean-preserving convex limiter; no clipping of individual conserved cells."""
    values = values.astype(float)
    delta = values - values.mean(axis=1, keepdims=True)
    theta = np.ones(len(mean))
    low = np.zeros(len(mean))
    high = theta.copy()
    bad = ~((values[..., 0] > 1e-9) & (primitive(values)[..., 2] > 1e-9)).all(axis=1)
    if bad.any():
        for _ in range(32):
            mid = .5 * (low + high)
            candidate = mean[:, None] + mid[:, None, None] * delta
            good = ((candidate[..., 0] > 1e-9) & (primitive(candidate)[..., 2] > 1e-9)).all(axis=1)
            low = np.where(good, mid, low)
            high = np.where(good, high, mid)
        theta[bad] = low[bad]
    result = mean[:, None] + theta[:, None, None] * delta
    return result, int(bad.sum()), float(theta.min())


def exact_riemann(x, t, left=(1., 0., 1.), right=(.125, 0., .1), discontinuity=.5):
    """Non-vacuum ideal-gas Riemann solution, independently implemented.

    Relations follow the Euler chapter of the Clawpack Riemann book. Root finding
    determines common star pressure; shocks and rarefaction fans are sampled
    explicitly. This function is a reference only, never a neural-module input.
    """
    x = np.asarray(x)
    l, r = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if t <= 0:
        return np.where((x < discontinuity)[..., None], conserved(l), conserved(r)), {}
    cl, cr = np.sqrt(GAMMA * l[2] / l[0]), np.sqrt(GAMMA * r[2] / r[0])
    if r[1] - l[1] >= 2 * (cl + cr) / (GAMMA - 1):
        raise ValueError('Vacuum states are outside this experiment')

    def wave(p, w, sound):
        rho, _, p0 = w
        if p > p0:
            return (p - p0) * np.sqrt(2 / ((GAMMA + 1) * rho) / (p + (GAMMA - 1) / (GAMMA + 1) * p0))
        return 2 * sound / (GAMMA - 1) * ((p / p0)**((GAMMA - 1) / (2 * GAMMA)) - 1)

    def residual(p):
        return wave(p, l, cl) + wave(p, r, cr) + r[1] - l[1]
    upper = max(l[2], r[2])
    while residual(upper) < 0:
        upper *= 2
    ps = brentq(residual, 1e-12, upper, xtol=1e-14)
    us = .5 * (l[1] + r[1] + wave(ps, r, cr) - wave(ps, l, cl))
    beta = (GAMMA - 1) / (GAMMA + 1)
    star_rho = lambda w: w[0] * ((ps / w[2] + beta) / (beta * ps / w[2] + 1) if ps > w[2] else (ps / w[2])**(1 / GAMMA))
    rl, rr = star_rho(l), star_rho(r)
    csl, csr = np.sqrt(GAMMA * ps / rl), np.sqrt(GAMMA * ps / rr)
    xi = (x - discontinuity) / t
    w = np.empty(x.shape + (3,))
    w[xi <= us] = (rl, us, ps)
    w[xi > us] = (rr, us, ps)
    info = dict(star_pressure=float(ps), star_velocity=float(us), star_density_left=float(rl), star_density_right=float(rr))
    if ps > l[2]:
        sl = l[1] - cl * np.sqrt((GAMMA + 1) / (2 * GAMMA) * ps / l[2] + (GAMMA - 1) / (2 * GAMMA))
        w[xi < sl] = l
        info['left_shock_speed'] = float(sl)
    else:
        head, tail = l[1] - cl, us - csl
        w[xi < head] = l
        fan = (xi >= head) & (xi <= tail)
        vf = 2 / (GAMMA + 1) * (cl + .5 * (GAMMA - 1) * l[1] + xi[fan])
        cf = 2 / (GAMMA + 1) * (cl + .5 * (GAMMA - 1) * (l[1] - xi[fan]))
        w[fan] = np.stack((l[0] * (cf / cl)**(2 / (GAMMA - 1)), vf,
                          l[2] * (cf / cl)**(2 * GAMMA / (GAMMA - 1))), axis=-1)
        info.update(left_fan_head=float(head), left_fan_tail=float(tail))
    if ps > r[2]:
        sr = r[1] + cr * np.sqrt((GAMMA + 1) / (2 * GAMMA) * ps / r[2] + (GAMMA - 1) / (2 * GAMMA))
        w[xi > sr] = r
        info['right_shock_speed'] = float(sr)
    else:
        tail, head = us + csr, r[1] + cr
        w[xi > head] = r
        fan = (xi >= tail) & (xi <= head)
        vf = 2 / (GAMMA + 1) * (-cr + .5 * (GAMMA - 1) * r[1] + xi[fan])
        cf = 2 / (GAMMA + 1) * (cr - .5 * (GAMMA - 1) * (r[1] - xi[fan]))
        w[fan] = np.stack((r[0] * (cf / cr)**(2 / (GAMMA - 1)), vf,
                          r[2] * (cf / cr)**(2 * GAMMA / (GAMMA - 1))), axis=-1)
        info.update(right_fan_tail=float(tail), right_fan_head=float(head))
    return conserved(w), info


class EulerFV:
    """Fine exterior FV and 16 conservative internal means in one local module."""
    def __init__(self, n=1024, start=None, coarse_cells=16, ratio=8, decoder=None):
        self.n, self.h, self.start = n, 1 / n, start
        self.m, self.ratio, self.decoder = coarse_cells, ratio, decoder
        self.end = None if start is None else start + coarse_cells * ratio
        if start is not None and (start < 2 * ratio or self.end > n - 2 * ratio):
            raise ValueError('Two external context blocks required at each side')
        self.widths = np.full(n if start is None else n - coarse_cells * (ratio - 1), self.h)
        if start is not None:
            self.widths[start:start + coarse_cells] *= ratio
        self.decode_calls = self.decode_limited_cells = self.step_limiter_calls = 0
        self.minimum_decode_theta = self.minimum_step_theta = 1.

    def encode(self, fine):
        if self.start is None:
            return fine.copy()
        s, e, m = self.start, self.end, self.m
        return np.concatenate((fine[:s], fine[s:e].reshape(m, self.ratio, 3).mean(axis=1), fine[e:]))

    def decode(self, state):
        s, m, r = self.start, self.m, self.ratio
        mean = state[s:s + m]
        if self.decoder is None:
            return np.repeat(mean[:, None], r, axis=1)
        context = np.concatenate((state[s - 2 * r:s].reshape(2, r, 3).mean(axis=1), mean,
                                  state[s + m:s + m + 2 * r].reshape(2, r, 3).mean(axis=1)))
        raw = self.decoder(context)
        decoded, limited, theta = limit_reconstruction(raw, mean)
        self.decode_calls += 1
        self.decode_limited_cells += limited
        self.minimum_decode_theta = min(self.minimum_decode_theta, theta)
        return decoded

    def reconstruct(self, state):
        if self.start is None:
            return state.copy()
        s, m = self.start, self.m
        return np.concatenate((state[:s], self.decode(state).reshape(-1, 3), state[s + m:]))

    def fluxes(self, state):
        padded = np.concatenate((state[:1], state, state[-1:]))
        low = rusanov(padded[:-1], padded[1:])
        speed = float(physical_flux(state)[1].max())
        if self.start is None or self.decoder is None:
            return low, low, speed
        local = self.decode(state)
        s, m = self.start, self.m
        left = np.concatenate((state[s - 1:s], local[:, -1]))
        right = np.concatenate((local[:, 0], state[s + m:s + m + 1]))
        high = low.copy()
        high[s:s + m + 1] = rusanov(left, right)
        return high, low, max(speed, float(physical_flux(local)[1].max()))

    def stage(self, state, high, low, dt):
        update = lambda flux: state - dt / self.widths[:, None] * np.diff(flux, axis=0)
        candidate = update(high)
        if admissible(candidate):
            return candidate, high
        base = update(low)
        if not admissible(base):
            raise FloatingPointError('Low-order positive update failed; reduce CFL')
        lo, hi = 0., 1.
        for _ in range(40):
            mid = .5 * (lo + hi)
            if admissible(base + mid * (candidate - base)):
                lo = mid
            else:
                hi = mid
        self.step_limiter_calls += 1
        self.minimum_step_theta = min(self.minimum_step_theta, lo)
        flux = low + lo * (high - low)
        return update(flux), flux

    def run(self, left=(1., 0., 1.), right=(.125, 0., .1), discontinuity=.5,
            final_time=.2, output_times=(.04, .08, .12, .16, .2), cfl=.35):
        started = time.perf_counter()
        state = self.encode(initial_average(self.n, left, right, discontinuity))
        initial = (state * self.widths[:, None]).sum(axis=0)
        budget = np.zeros(3)
        local_budget = np.zeros(3)
        s, m = self.start, self.m
        local_initial = None if s is None else (state[s:s + m] * self.widths[s:s + m, None]).sum(axis=0)
        snapshots, history, steps = {}, [], 0
        targets = sorted(set(float(t) for t in output_times if 0 < t <= final_time) | {float(final_time)})
        current = 0.
        minimum_rho = float(state[:, 0].min())
        minimum_pressure = float(primitive(state)[:, 2].min())
        while targets:
            high, low, speed = self.fluxes(state)
            dt = min(cfl * self.h / speed, targets[0] - current)
            stage1, f0 = self.stage(state, high, low, dt)
            high1, low1, _ = self.fluxes(stage1)
            stage2, f1 = self.stage(stage1, high1, low1, dt)
            state = .5 * (state + stage2)
            integrated = .5 * dt * (f0 + f1)
            budget += integrated[0] - integrated[-1]
            if s is not None:
                local_budget += integrated[s] - integrated[s + m]
            current += dt
            steps += 1
            minimum_rho = min(minimum_rho, float(state[:, 0].min()))
            minimum_pressure = min(minimum_pressure, float(primitive(state)[:, 2].min()))
            if abs(current - targets[0]) < 1e-12:
                field = self.reconstruct(state)
                total = (state * self.widths[:, None]).sum(axis=0)
                record = dict(time=current, global_budget_error=(total - initial - budget).tolist(),
                              integrated_external_flux=budget.tolist())
                if s is not None:
                    local = (state[s:s + m] * self.widths[s:s + m, None]).sum(axis=0)
                    record['region_budget_error'] = (local - local_initial - local_budget).tolist()
                    record['integrated_region_flux'] = local_budget.tolist()
                    record['reconstruction_mean_error'] = float(abs(field[s:self.end].reshape(m, self.ratio, 3).mean(axis=1) - state[s:s + m]).max())
                snapshots[f'{targets.pop(0):.8f}'] = field
                history.append(record)
        return dict(state=state, field=field, snapshots=snapshots, history=history,
                    wall_s=time.perf_counter() - started, steps=steps,
                    retained_conserved_scalars=int(state.size), fine_conserved_scalars=3 * self.n,
                    local_conserved_scalars=0 if s is None else 3 * m,
                    local_replaced_scalars=0 if s is None else 3 * m * self.ratio,
                    decode_calls=self.decode_calls, decode_limited_cells=self.decode_limited_cells,
                    minimum_decode_theta=self.minimum_decode_theta,
                    positivity_step_limiter_calls=self.step_limiter_calls,
                    minimum_step_theta=self.minimum_step_theta,
                    minimum_density=minimum_rho, minimum_pressure=minimum_pressure)
