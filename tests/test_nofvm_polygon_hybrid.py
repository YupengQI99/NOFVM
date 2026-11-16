"""Multi-tile conservative assembly, exact reference and affine reconstruction."""
import sys
from pathlib import Path

import numpy as np
import pytest
pytest.importorskip('gmsh', reason='gmsh is required to build the polygon mesh')
pytest.importorskip('shapely', minversion='2.1', reason='shapely>=2.1 is required to build the polygon mesh')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from nofvm.nofvm_polygon_mesh import build_polygon_mesh
from nofvm.nofvm_polygon_hybrid import TiledFV,local_response,ExactTileReference


@pytest.fixture(scope='module')
def problem(tmp_path_factory):
    mesh=build_polygon_mesh(tmp_path_factory.mktemp('polygon_hybrid'))
    return mesh,local_response(mesh),ExactTileReference(mesh)


@pytest.mark.parametrize('p',[1,2])
def test_repeated_exact_subdomain_and_reference_match_full_fv(problem,p):
    mesh,response,reference=problem
    def bc(xy):return np.column_stack((np.ones(len(xy)),np.sin(2.3*xy[:,1])+.2*xy[:,0]))
    full=TiledFV(mesh,p);f=full.solve(bc)
    hybrid=TiledFV(mesh,p,response);h=hybrid.solve(bc)
    u=hybrid.reconstruct(h,mesh,bc,local_xy=response['xy'])
    exact=reference.solve(p,bc)
    assert np.max(abs(u-f['ordinary']))<1e-10
    assert np.max(abs(h['flux']-f['flux']))<1e-10
    assert np.max(abs(exact['field']-f['ordinary']))<1e-10
    assert np.max(abs(exact['flux']-f['flux']))<1e-10
    for name in ('linear_residual','global_balance','local_balance','interface_balance'):
        assert h[name]<1e-10


def test_exterior_gradient_preserves_affine_fields_across_tile_seams(problem):
    mesh,response,_=problem
    hybrid=TiledFV(mesh,2,response)
    def affine(xy):return (1.+2*xy[:,0])[:,None]
    state=dict(solution=affine(hybrid.global_xy))
    gradient=hybrid.gradients(state,affine)[:,:hybrid.nordinary,:,0]
    assert np.max(abs(gradient-[2.,0.]))<1e-11
