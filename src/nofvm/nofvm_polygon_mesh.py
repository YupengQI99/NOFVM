"""Gmsh-seeded, boundary-fitted Voronoi control volumes for the hole case.

Gmsh supplies the graded triangular background mesh. Boundary/interface collars
replace nearby sites with mirrored pairs; the subsequent Voronoi construction
is separate from Gmsh and preserves square interface faces for a local operator.
Every plotted polygon is exported with the face topology used by the FV check.
The circular CAD boundary is discretized by its actual Gmsh boundary segments.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sla


@dataclass
class PolygonMesh:
    xy: np.ndarray
    vertices: np.ndarray
    offsets: np.ndarray
    volume: np.ndarray
    faces: np.ndarray
    owner: np.ndarray
    neighbour: np.ndarray
    boundary_tag: np.ndarray
    triangle_xy: np.ndarray
    triangles: np.ndarray
    hole: np.ndarray
    interface_n: int

    @property
    def polygons(self):
        return [self.vertices[a:b] for a, b in zip(self.offsets[:-1], self.offsets[1:])]

    @property
    def length(self):
        return np.linalg.norm(self.faces[:, 1]-self.faces[:, 0], axis=1)

    @property
    def inside(self):
        return np.all((self.xy > .4) & (self.xy < 1.2), axis=1)

    def save(self, path):
        np.savez_compressed(path, **self.__dict__)

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            values = dict(data)
        values['interface_n'] = int(values['interface_n'])
        return cls(**values)


def gmsh_background(directory, scale=1):
    """Return the mesh of a 1.6 square, radius .15 hole and .8 square interface."""
    import gmsh
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    h_hole, h_far, h_trace = .01/scale, .1/scale, .025/scale
    n_circle = 96*scale
    geo = f'''// Gmsh background mesh; Voronoi cells are constructed separately.
SetFactory("Built-in");
Point(1) = {{0,0,0,{h_far}}}; Point(2) = {{1.6,0,0,{h_far}}};
Point(3) = {{1.6,1.6,0,{h_far}}}; Point(4) = {{0,1.6,0,{h_far}}};
Line(1) = {{1,2}}; Line(2) = {{2,3}}; Line(3) = {{3,4}}; Line(4) = {{4,1}};
Point(5) = {{.4,.4,0,{h_trace}}}; Point(6) = {{1.2,.4,0,{h_trace}}};
Point(7) = {{1.2,1.2,0,{h_trace}}}; Point(8) = {{.4,1.2,0,{h_trace}}};
Line(5) = {{5,6}}; Line(6) = {{6,7}}; Line(7) = {{7,8}}; Line(8) = {{8,5}};
Point(9) = {{.8,.8,0,{h_hole}}};
Point(10) = {{.95,.8,0,{h_hole}}}; Point(11) = {{.8,.95,0,{h_hole}}};
Point(12) = {{.65,.8,0,{h_hole}}}; Point(13) = {{.8,.65,0,{h_hole}}};
Circle(9) = {{10,9,11}}; Circle(10) = {{11,9,12}};
Circle(11) = {{12,9,13}}; Circle(12) = {{13,9,10}};
Curve Loop(1) = {{1,2,3,4}}; Curve Loop(2) = {{5,6,7,8}};
Curve Loop(3) = {{9,10,11,12}};
Plane Surface(1) = {{1,2}}; Plane Surface(2) = {{2,3}};
Transfinite Curve{{9:12}} = {n_circle//4+1};
Transfinite Curve{{5:8}} = {32*scale+1};
Field[1] = Distance; Field[1].CurvesList = {{9,10,11,12}}; Field[1].Sampling = 400;
Field[2] = Threshold; Field[2].InField = 1;
Field[2].SizeMin = {h_hole}; Field[2].SizeMax = {h_far};
Field[2].DistMin = .02; Field[2].DistMax = .45;
Field[3] = Distance; Field[3].CurvesList = {{5,6,7,8}}; Field[3].Sampling = 400;
Field[4] = Threshold; Field[4].InField = 3;
Field[4].SizeMin = {h_trace}; Field[4].SizeMax = {h_far};
Field[4].DistMin = .02; Field[4].DistMax = .35;
Field[5] = Min; Field[5].FieldsList = {{2,4}}; Background Field = 5;
Mesh.MeshSizeFromPoints = 0; Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeExtendFromBoundary = 0; Mesh.Algorithm = 6;
Mesh.Smoothing = 10; Mesh.ElementOrder = 1;
General.NumThreads = 1; Mesh.MaxNumThreads1D = 1; Mesh.MaxNumThreads2D = 1;
Mesh.RandomSeed = 1; Mesh.MshFileVersion = 4.1;
Physical Curve("outer",1) = {{1,2,3,4}};
Physical Curve("interface",2) = {{5,6,7,8}};
Physical Curve("insulated_hole",3) = {{9,10,11,12}};
Physical Surface("exterior",1) = {{1}}; Physical Surface("local",2) = {{2}};
'''
    geo_path = directory/'background.geo'
    geo_path.write_text(geo)
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.open(str(geo_path))
        gmsh.model.mesh.generate(2)
        gmsh.write(str(directory/'background.msh'))
        tags, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(coords).reshape(-1, 3)[:, :2]
        order = np.argsort(tags)
        tags, coords = tags[order], coords[order]
        _, tri_tags = gmsh.model.mesh.getElementsByType(2)
        triangles = np.searchsorted(tags, np.asarray(tri_tags).reshape(-1, 3))
        used = np.unique(triangles)
        remap = np.full(len(coords), -1, dtype=int)
        remap[used] = np.arange(len(used))
        points, triangles = coords[used], remap[triangles]
        boundary_nodes = []
        for curve in range(1, 13):
            t, _, _ = gmsh.model.mesh.getNodes(1, curve, includeBoundary=True)
            boundary_nodes.extend(t)
        boundary_tags = np.unique(boundary_nodes)
        interior = points[~np.isin(tags[used], boundary_tags)]
        r = np.linalg.norm(points-[.8,.8], axis=1)
        hole = points[abs(r-.15) < 1e-10]
        theta = np.arctan2(hole[:, 1]-.8, hole[:, 0]-.8)
        hole = hole[np.argsort(theta)]
        return points, triangles, interior, hole
    finally:
        gmsh.finalize()


def square_distance(xy, lo, hi):
    outside = np.maximum(np.maximum(lo-xy, xy-hi), 0)
    d = np.linalg.norm(outside, axis=1)
    inside = np.all((xy >= lo) & (xy <= hi), axis=1)
    d[inside] = np.min(np.minimum(xy[inside]-lo, hi-xy[inside]), axis=1)
    return d


def square_collar(lo, hi, n, both):
    h = (hi-lo)/n
    t = lo+(np.arange(n)+.5)*h
    rows = []
    for offset in ((.5, -.5) if both else (.5,)):
        for side in (lo+offset*h, hi-offset*h):
            rows.extend((np.column_stack((np.full(n, side), t)),
                         np.column_stack((t, np.full(n, side)))))
    if both:
        # Complete the four-site pattern at each interface corner.
        for a in (lo-h/2, hi+h/2):
            for b in (lo-h/2, hi+h/2):
                rows.append(np.array([[a,b]]))
    return np.concatenate(rows)


def build_polygon_mesh(directory, scale=1):
    import shapely
    from shapely.geometry import MultiPoint, Point, Polygon
    background, triangles, sites, hole = gmsh_background(directory, scale)
    n_trace, n_outer = 32*scale, 16*scale
    keep = ((square_distance(sites, 0., 1.6) > .95*(1.6/n_outer)) &
            (square_distance(sites, .4, 1.2) > .95*(.8/n_trace)))
    sites = np.concatenate((sites[keep], square_collar(0., 1.6, n_outer, False),
                            square_collar(.4, 1.2, n_trace, True)))
    sites = np.unique(np.round(sites, 13), axis=0)
    domain = Polygon([(0.,0.),(1.6,0.),(1.6,1.6),(0.,1.6)], holes=[hole])
    hole_boundary = Polygon(hole).boundary
    voronoi = shapely.voronoi_polygons(MultiPoint(sites), ordered=True)
    cells = [p.intersection(domain) for p in voronoi.geoms]
    if any(p.geom_type != 'Polygon' or not p.is_valid or p.area <= 0 for p in cells):
        raise ValueError('Voronoi clipping produced invalid or disconnected cells')
    polygons = [np.asarray(p.exterior.coords[:-1]) for p in cells]
    offsets = np.r_[0, np.cumsum([len(p) for p in polygons])]
    vertices = np.concatenate(polygons)
    edge_map = {}
    for owner, polygon in enumerate(polygons):
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            if np.linalg.norm(a-b) < 1e-11:
                continue
            key = tuple(sorted((tuple(np.round(a, 11)), tuple(np.round(b, 11)))))
            edge_map.setdefault(key, []).append((owner, a, b))
    faces, owners, neighbours, boundary_tags = [], [], [], []
    for records in edge_map.values():
        if len(records) > 2:
            raise ValueError('Non-manifold Voronoi edge')
        owner, a, b = records[0]
        neighbour = records[1][0] if len(records) == 2 else -1
        tag = 0
        if neighbour < 0:
            for test, value in ((np.max(abs(np.array([a[0],b[0]]))), 1),
                                (np.max(abs(np.array([a[0],b[0]])-1.6)), 2),
                                (np.max(abs(np.array([a[1],b[1]]))), 3),
                                (np.max(abs(np.array([a[1],b[1]])-1.6)), 4)):
                if test < 1e-9:
                    tag = value
            if tag == 0:
                if any(hole_boundary.distance(Point(p)) > 1e-9 for p in (a,b,(a+b)/2)):
                    raise ValueError('Unmatched internal face in polygon topology')
                tag = 5
        faces.append([a,b]); owners.append(owner); neighbours.append(neighbour); boundary_tags.append(tag)
    mesh = PolygonMesh(sites, vertices, offsets, np.array([p.area for p in cells]),
                       np.asarray(faces), np.asarray(owners), np.asarray(neighbours),
                       np.asarray(boundary_tags), background, triangles, hole, n_trace)
    mesh.save(Path(directory)/'control_volumes.npz')
    # Legacy VTK POLYDATA can be opened in ParaView without a Python dependency.
    lines = ['# vtk DataFile Version 3.0', 'Actual Voronoi finite-volume cells',
             'ASCII', 'DATASET POLYDATA', f'POINTS {len(vertices)} double']
    lines += [f'{x:.16g} {y:.16g} 0' for x, y in vertices]
    lines += [f'POLYGONS {len(sites)} {len(vertices)+len(sites)}']
    lines += [str(b-a)+' '+' '.join(map(str, range(a,b))) for a,b in zip(offsets[:-1],offsets[1:])]
    lines += [f'CELL_DATA {len(sites)}', 'SCALARS area double 1', 'LOOKUP_TABLE default']
    lines += [f'{a:.16g}' for a in mesh.volume]
    (Path(directory)/'control_volumes.vtk').write_text('\n'.join(lines)+'\n')
    return mesh


def mesh_quality(mesh):
    import shapely
    from shapely.geometry import Point, Polygon
    polygons = [Polygon(p) for p in mesh.polygons]
    domain = Polygon([(0.,0.),(1.6,0.),(1.6,1.6),(0.,1.6)], holes=[mesh.hole])
    union = shapely.union_all(polygons)
    internal = mesh.neighbour >= 0
    i, j = mesh.owner[internal], mesh.neighbour[internal]
    d = mesh.xy[j]-mesh.xy[i]
    tangent = mesh.faces[internal, 1]-mesh.faces[internal, 0]
    cos_error = abs(np.sum(d*tangent, axis=1))/(np.linalg.norm(d, axis=1)*np.linalg.norm(tangent, axis=1))
    angular = np.rad2deg(np.arcsin(np.clip(cos_error, 0, 1)))
    interface = internal.copy()
    interface[internal] = mesh.inside[i] != mesh.inside[j]
    faces = mesh.faces[interface]
    mid = faces.mean(axis=1)
    square_face_error = square_distance(mid, .4, 1.2)
    interface_lengths = mesh.length[interface]
    perimeters = np.array([p.length for p in polygons])
    compactness = 4*np.pi*mesh.volume/perimeters**2
    ratio = np.maximum(mesh.volume[i]/mesh.volume[j], mesh.volume[j]/mesh.volume[i])
    tau = mesh.length[internal]/np.linalg.norm(d, axis=1)
    outer = (mesh.boundary_tag > 0) & (mesh.boundary_tag < 5)
    a, b = mesh.faces[outer,0], mesh.faces[outer,1]
    center = mesh.xy[mesh.owner[outer]]
    t = np.sum((center-a)*(b-a), axis=1)/np.sum((b-a)**2, axis=1)
    projected = a+t[:,None]*(b-a)
    boundary_skew = np.linalg.norm(projected-(a+b)/2,axis=1)/np.linalg.norm(b-a,axis=1)
    tri = mesh.triangle_xy[mesh.triangles]
    edges = np.linalg.norm(np.roll(tri,-1,axis=1)-tri,axis=2)
    v, w = tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]
    tri_area = abs(v[:,0]*w[:,1]-v[:,1]*w[:,0])/2
    tri_quality = 4*np.sqrt(3)*tri_area/np.sum(edges**2,axis=1)
    point_inside = all(p.buffer(1e-12).covers(Point(x)) for p,x in zip(polygons,mesh.xy))
    checks = dict(valid_cells=all(p.is_valid for p in polygons),
                  positive_volumes=bool(np.min(mesh.volume)>0),
                  sites_in_own_cells=point_inside,
                  complete_domain=bool(union.symmetric_difference(domain).area < 1e-10),
                  no_overlap=bool(abs(mesh.volume.sum()-union.area)<1e-10),
                  complete_boundary_length=bool(abs(mesh.length[~internal].sum()-domain.length)<1e-10),
                  orthogonal_internal_faces=bool(angular.max()<1e-6),
                  positive_internal_transmissibilities=bool(np.min(tau)>0),
                  admissible_boundary_projections=bool(np.min(t)>-1e-8 and np.max(t)<1+1e-8),
                  boundary_midpoint_alignment=bool(boundary_skew.max()<1e-8),
                  aligned_square_interface=bool(square_face_error.max()<1e-10),
                  uniform_interface_faces=bool(np.max(abs(interface_lengths-.8/mesh.interface_n))<1e-9),
                  complete_interface=bool(len(faces)==4*mesh.interface_n))
    return dict(cells=len(mesh.xy), local_cells=int(mesh.inside.sum()),
                exterior_cells=int((~mesh.inside).sum()), background_triangles=len(tri),
                hole_segments=len(mesh.hole), interface_faces=int(len(faces)),
                minimum_cell_area=float(mesh.volume.min()),
                compactness_min=float(compactness.min()), compactness_median=float(np.median(compactness)),
                neighbour_area_ratio_p95=float(np.quantile(ratio,.95)),
                neighbour_area_ratio_max=float(ratio.max()),
                internal_nonorthogonality_max_degrees=float(angular.max()),
                boundary_skew_max=float(boundary_skew.max()),
                positive_tau_min=float(tau.min()),
                coverage_error=float(union.symmetric_difference(domain).area),
                triangle_quality_min=float(tri_quality.min()), triangle_quality_median=float(np.median(tri_quality)),
                polygonal_hole_area_error_relative=float(abs((2.56-domain.area)-np.pi*.15**2)/(np.pi*.15**2)),
                checks=checks, passed=all(checks.values()))


def fv_system(mesh, all_dirichlet=False):
    internal = mesh.neighbour >= 0
    i, j = mesh.owner[internal], mesh.neighbour[internal]
    tau = mesh.length[internal]/np.linalg.norm(mesh.xy[i]-mesh.xy[j],axis=1)
    count = len(mesh.xy)
    matrix = sparse.coo_matrix((np.r_[tau,tau,-tau,-tau],
                               (np.r_[i,j,i,j],np.r_[i,j,j,i])),shape=(count,count)).tocsc()
    selected = np.isin(mesh.boundary_tag, [1,2,3,4] if all_dirichlet else [1,2])
    owner = mesh.owner[selected]
    midpoint = mesh.faces[selected].mean(axis=1)
    t = mesh.length[selected]/np.linalg.norm(mesh.xy[owner]-midpoint,axis=1)
    matrix += sparse.diags(np.bincount(owner,weights=t,minlength=count),format='csc')
    return matrix, selected, owner, midpoint, t


def fv_checks(mesh):
    """Direct FV checks on exported cells; this does not train a neural model."""
    from scipy.sparse.csgraph import connected_components
    matrix, selected, owner, midpoint, tau = fv_system(mesh)
    factor = sla.splu(matrix)
    ones_rhs = np.bincount(owner,weights=tau,minlength=len(mesh.xy))
    constant = factor.solve(ones_rhs)
    prescribed = np.where(midpoint[:,0] < .8, 1., -1.)
    rhs = np.bincount(owner,weights=tau*prescribed,minlength=len(mesh.xy))
    u = factor.solve(rhs)
    outer_flux = tau*(u[owner]-prescribed)
    internal = mesh.neighbour >= 0
    i, j = mesh.owner[internal], mesh.neighbour[internal]
    inside = mesh.inside
    crossed = inside[i] != inside[j]
    q = mesh.length[internal]/np.linalg.norm(mesh.xy[j]-mesh.xy[i],axis=1)*(u[i]-u[j])
    local_balance = abs(np.sum(q[crossed]*np.where(inside[i[crossed]],1.,-1.)))
    # Smooth exact exterior-circle harmonic field with insulating circular hole.
    # The polygonal hole approximates that circle; its geometry error is retained.
    def harmonic(xy):
        x,y=(xy-[.8,.8]).T
        return x*(1+.15**2/(x*x+y*y))
    matrix2, _, owner2, midpoint2, tau2 = fv_system(mesh, all_dirichlet=True)
    rhs2 = np.bincount(owner2,weights=tau2*harmonic(midpoint2),minlength=len(mesh.xy))
    u2 = sla.spsolve(matrix2,rhs2)
    exact = harmonic(mesh.xy)
    relative_error = np.sqrt(np.dot(mesh.volume,(u2-exact)**2)/np.dot(mesh.volume,exact**2))
    offdiag = matrix-sparse.diags(matrix.diagonal())
    diff = matrix-matrix.T
    symmetry = float(abs(diff.data).max()) if diff.nnz else 0.
    checks = dict(constant_preservation=bool(abs(constant-1).max()<1e-10),
                  linear_residual=bool(abs(matrix@u-rhs).max()<1e-10),
                  global_heat_balance=bool(abs(outer_flux.sum())<1e-10),
                  local_heat_balance=bool(local_balance<1e-10),
                  maximum_principle=bool(u.min()>=-1-1e-10 and u.max()<=1+1e-10),
                  symmetric_matrix=bool(symmetry<1e-12),
                  nonpositive_offdiagonal=bool(offdiag.data.max()<=0),
                  connected_graph=bool(connected_components(matrix,directed=False,return_labels=False)==1))
    return dict(constant_error=float(abs(constant-1).max()),
                residual=float(abs(matrix@u-rhs).max()),
                global_balance=float(abs(outer_flux.sum())), local_balance=float(local_balance),
                harmonic_relative_l2=float(relative_error), checks=checks,
                passed=all(checks.values())), dict(temperature=u,harmonic=u2,harmonic_exact=exact)
