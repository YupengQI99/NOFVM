"""Learn the harmonic response inside a known nonlinear Kirchhoff transform.

Geometry-only priors; no exact target solution in model buffers. Training and
deployment reuse the previously validated boundary-consistent architecture.
"""
import numpy as np
from .nofvm_2d import boundary_geometry, rectangular_system
from .nofvm_darcy import trace_prolongation
from .nofvm_hole_operator import HoleBoundaryOperator


def build_model(n=64,m=32):
    ports=boundary_geometry(m,m).xy
    x,y=np.meshgrid((np.arange(n)+.5)/n,(np.arange(n)+.5)/n)
    xy=np.column_stack((x.ravel(),y.ravel()))
    distance2=((xy[:,None,:]-ports[None,:,:])**2).sum(axis=-1)
    base_h=1/distance2
    base_h/=base_h.sum(axis=1,keepdims=True)
    port_distance2=((ports[:,None,:]-ports[None,:,:])**2).sum(axis=-1)
    np.fill_diagonal(port_distance2,np.inf)
    weights=(1/m)**2/(np.pi*port_distance2)
    base_s=np.diag(weights.sum(axis=1))-weights
    system=rectangular_system(np.ones((n,n)))
    prolongation=trace_prolongation(n,m)
    lifting=system.lifting@prolongation
    diagonal=np.diag(np.asarray(prolongation.T@system.tau).ravel())
    # The reused trunk feature helper divides coordinates by 0.8.
    return HoleBoundaryOperator(base_h,base_s,.8*xy,lifting,diagonal)
