"""Geometry admissibility and exact FV subdomain coupling on a genuine dual mesh."""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('gmsh', reason='gmsh is required to build the polygon mesh')
pytest.importorskip('shapely', minversion='2.1', reason='shapely>=2.1 is required for the ordered Voronoi construction')
from scipy import sparse
from scipy.sparse.linalg import splu, spsolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from nofvm.nofvm_polygon_mesh import (
    PolygonMesh, build_polygon_mesh, mesh_quality, fv_checks, fv_system,
)


@pytest.fixture(scope='module')
def mesh(tmp_path_factory):
    return build_polygon_mesh(tmp_path_factory.mktemp('gmsh_voronoi'))


def test_admissible_mesh_and_reference_heat_balance(mesh):
    quality = mesh_quality(mesh)
    assert quality['passed'], quality
    assert quality['interface_faces'] == 128
    result, _ = fv_checks(mesh)
    assert result['passed'], result
    assert result['harmonic_relative_l2'] < 1e-3


def test_export_preserves_solver_mesh(mesh, tmp_path):
    path = tmp_path/'cells.npz'
    mesh.save(path)
    restored = PolygonMesh.load(path)
    for name in mesh.__dict__:
        assert np.array_equal(getattr(mesh,name),getattr(restored,name))
    assert mesh_quality(restored)['passed']


def test_square_trace_condensation_matches_direct_fv(mesh):
    """An exact boundary-response block reproduces the same full FV solution.

    Splitting each crossed face into two positive half-transmissibilities gives
    precisely the trace coupling required by the response formulation.
    This checks the mesh/solver interface without a learned fit.
    """
    A, _, owner, midpoint, boundary_tau = fv_system(mesh)
    inside = mesh.inside
    li, ei = np.flatnonzero(inside), np.flatnonzero(~inside)
    local_ids=np.full(len(mesh.xy),-1,dtype=int); local_ids[li]=np.arange(len(li))
    outer_ids=np.full(len(mesh.xy),-1,dtype=int); outer_ids[ei]=np.arange(len(ei))
    internal=mesh.neighbour>=0
    i,j=mesh.owner[internal],mesh.neighbour[internal]
    crossed=inside[i]!=inside[j]
    i,j=i[crossed],j[crossed]
    inward=np.where(inside[i],i,j); outward=np.where(inside[i],j,i)
    tau=mesh.length[internal][crossed]/np.linalg.norm(mesh.xy[i]-mesh.xy[j],axis=1)
    n=len(tau); half_tau=2*tau
    Bl=sparse.csc_matrix((half_tau,(local_ids[inward],np.arange(n))),shape=(len(li),n))
    Bo=sparse.csc_matrix((half_tau,(outer_ids[outward],np.arange(n))),shape=(len(ei),n))
    Al=A[li][:,li]+sparse.diags(np.bincount(local_ids[inward],weights=tau,minlength=len(li)))
    Ko=A[ei][:,ei]+sparse.diags(np.bincount(outer_ids[outward],weights=tau,minlength=len(ei)))
    H=splu(Al.tocsc()).solve(Bl.toarray())
    S=np.diag(half_tau)-Bl.T@H
    assert np.max(abs(S-S.T))<1e-12
    assert np.max(abs(S.sum(axis=1)))<1e-12
    assert np.max(S-np.diag(np.diag(S)))<1e-12
    coupled=sparse.bmat([[Ko,-Bo],[-Bo.T,sparse.csc_matrix(np.diag(half_tau)+S)]],format='csc')
    prescribed=np.sin(2.1*midpoint[:,1])+.3*midpoint[:,0]
    rhs=np.bincount(owner,weights=boundary_tau*prescribed,minlength=len(mesh.xy))
    direct=spsolve(A,rhs)
    reduced=spsolve(coupled,np.r_[rhs[ei],np.zeros(n)])
    assert np.max(abs(reduced[:len(ei)]-direct[ei]))<1e-11
    assert np.max(abs(H@reduced[len(ei):]-direct[li]))<1e-11
