#!/usr/bin/env python3
"""Generate the Gmsh-seeded Voronoi polygon meshes for the hole problems (C1/C2), verify them and plot them.

Each scale writes its control volumes (scale_<s>/control_volumes.npz), mesh quality and a direct FV check
against an analytic harmonic field. Requires gmsh and shapely>=2.1.

Usage:
    python scripts/generate_nofvm_polygon_mesh.py --output runs/polygon_mesh
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Polygon as PolygonPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from nofvm.nofvm_polygon_mesh import build_polygon_mesh, mesh_quality, fv_checks


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.titlesize': 10, 'axes.labelsize': 9,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.linewidth': .6, 'xtick.labelsize': 8,
                         'ytick.labelsize': 8, 'savefig.facecolor': 'white'})


def domain_axes(ax, lim=(0,1.6)):
    ax.set_aspect('equal')
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$', labelpad=1)
    ax.tick_params(length=2.5, width=.6)
    if lim == (0,1.6):
        ax.set_xticks([0,.4,.8,1.2,1.6]); ax.set_yticks([0,.4,.8,1.2,1.6])
    else:
        ax.set_xticks([.6,.8,1.0]); ax.set_yticks([.6,.8,1.0])


def interface(ax):
    ax.add_patch(Rectangle((.4,.4),.8,.8, fill=False, edgecolor='#238983', lw=1.0, zorder=6))


def draw_fv(ax, mesh, zoom=False):
    facecolors = np.where(mesh.inside, '#eaf4f2', '#f6f7f8')
    ax.add_collection(PolyCollection(mesh.polygons, facecolors=facecolors, edgecolors='none'))
    ax.add_collection(LineCollection(mesh.faces, colors='#465463', linewidths=.35 if not zoom else .65))
    ax.add_patch(PolygonPatch(mesh.hole, closed=True, fc='white', ec='#253648', lw=1.05, zorder=5))
    interface(ax)
    domain_axes(ax, (.54,1.06) if zoom else (0,1.6))


def plot_mesh(mesh, quality, directory):
    style()
    fig, axes = plt.subplots(1,3,figsize=(12,4.9))
    ax = axes[0]
    tri_edges = np.sort(np.concatenate([mesh.triangles[:,[0,1]],mesh.triangles[:,[1,2]],
                                       mesh.triangles[:,[2,0]]]),axis=1)
    tri_edges = np.unique(tri_edges,axis=0)
    ax.add_collection(LineCollection(mesh.triangle_xy[tri_edges],colors='#697584',linewidths=.28))
    ax.add_patch(PolygonPatch(mesh.hole,closed=True,fc='white',ec='#253648',lw=1.05,zorder=5))
    interface(ax); domain_axes(ax)
    ax.set_title('a   Gmsh background triangulation',loc='left',pad=11)
    axes[1].set_title('b   Actual finite-volume mesh',loc='left',pad=11)
    axes[2].set_title('c   Hole-boundary detail',loc='left',pad=11)
    draw_fv(axes[1],mesh); draw_fv(axes[2],mesh,True)
    axes[1].add_patch(Rectangle((.54,.54),.52,.52,fill=False,ls=(0,(3,3)),lw=.75,ec='#a16a3e',zorder=6))
    # Sparse centre markers are not used: panel c includes every visible site.
    visible=np.all((mesh.xy>=.54)&(mesh.xy<=1.06),axis=1)
    axes[2].scatter(*mesh.xy[visible].T,s=1.9,c='#465463',zorder=4)
    labels=[f'{len(mesh.triangles):,} triangles · smooth size field',
            f'{len(mesh.xy):,} control volumes · {quality["interface_faces"]} interface faces',
            f'{len(mesh.hole)} boundary segments · all faces shown']
    for ax,label in zip(axes,labels):
        ax.text(.5,-.23,label,transform=ax.transAxes,ha='center',va='top',fontsize=8,color='#415261')
    fig.subplots_adjust(left=.045,right=.99,bottom=.25,top=.82,wspace=.3)
    fig.text(.046,.965,'Boundary-fitted mesh for the single-hole heat-transfer case',
             fontsize=13,weight='medium',va='top',color='#203447')
    fig.text(.046,.035,'Teal square: face-aligned operator interface. '
             'Gmsh supplies the background; boundary collars and Voronoi construction produce the FV cells.',
             fontsize=8,color='#415261')
    for ext in ('png','pdf','svg'):
        fig.savefig(directory/f'mesh_preview.{ext}',dpi=240)
    plt.close(fig)


def plot_solution(mesh, fields, reports, directory):
    style()
    fig,axes=plt.subplots(1,3,figsize=(11,4.1),layout='constrained')
    fig.get_layout_engine().set(rect=(0,.11,1,.89))
    for ax,value,title,cmap in zip(axes[:2],
        [fields['temperature'],fields['harmonic']-fields['harmonic_exact']],
        ['a   Direct FV: insulated hole','b   Harmonic verification error'],['RdBu_r','RdBu_r']):
        p=PolyCollection(mesh.polygons,array=value,cmap=cmap,edgecolors='none')
        limit=float(abs(value).max()); p.set_clim(-limit,limit)
        ax.add_collection(p); domain_axes(ax); interface(ax)
        ax.add_patch(PolygonPatch(mesh.hole,closed=True,fc='white',ec='#253648',lw=.8))
        ax.set_title(title,loc='left',pad=9)
        fig.colorbar(p,ax=ax,shrink=.8)
    n=np.array([r['quality']['cells'] for r in reports])
    error=np.array([r['fv']['harmonic_relative_l2'] for r in reports])
    axes[2].loglog(n,100*error,'o-',color='#238983',lw=1.5,ms=5)
    axes[2].set_xlabel('Number of control volumes'); axes[2].set_ylabel('Relative $L_2$ error (%)')
    axes[2].set_title('c   Smooth-solution refinement',loc='left',pad=9)
    axes[2].grid(alpha=.2,which='both')
    fig.text(.02,.025,'Verification: analytic exterior-circle harmonic field; volume-weighted point-value error; '
             'polygonal hole boundary refined with the mesh. No neural model is used.',fontsize=7.5,color='#415261')
    for ext in ('png','pdf'):
        fig.savefig(directory/f'fv_verification.{ext}',dpi=220)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output',type=Path,required=True,help='Output directory, e.g. runs/polygon_mesh')
    parser.add_argument('--scales',type=int,nargs='+',default=[1,2,4],help='Mesh refinement scales')
    args=parser.parse_args()
    if any(s<=0 for s in args.scales) or args.scales != sorted(set(args.scales)):
        parser.error('Scales must be unique positive increasing integers')
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():
        parser.error('Output already contains summary.json; choose a new directory')
    import gmsh,shapely,scipy
    manifest=dict(geometry=dict(square=[0,1.6],hole_center=[.8,.8],hole_radius=.15,interface=[.4,1.2]),
                  scales=args.scales,gmsh_algorithm='Frontal-Delaunay (6)',
                  construction='Gmsh interior sites; regular boundary collars and mirrored interface collars; clipped Voronoi polygons.',
                  field_check='(x-.8)*(1+.15**2/r**2), Dirichlet at outer square and zero flux at polygonal approximation of the circle.',
                  refinement_check='Relative volume-weighted nodal L2 error decreases at each mesh level.',
                  versions=dict(python=platform.python_version(),gmsh=gmsh.__version__,
                                shapely=shapely.__version__,numpy=np.__version__,scipy=scipy.__version__))
    save_json(out/'manifest.json',manifest)
    reports=[]; first=None
    for scale in args.scales:
        directory=out/f'scale_{scale}'
        mesh=build_polygon_mesh(directory,scale)
        quality=mesh_quality(mesh)
        result,fields=fv_checks(mesh)
        np.savez_compressed(directory/'verification_fields.npz',**fields)
        save_json(directory/'quality.json',quality); save_json(directory/'fv_checks.json',result)
        reports.append(dict(scale=scale,quality=quality,fv=result))
        print(json.dumps(dict(scale=scale,cells=quality['cells'],mesh_pass=quality['passed'],
                              fv_pass=result['passed'],harmonic_error=result['harmonic_relative_l2'])),flush=True)
        if first is None:
            first=(mesh,fields,quality)
    decreasing=all(b['fv']['harmonic_relative_l2']<a['fv']['harmonic_relative_l2'] for a,b in zip(reports[:-1],reports[1:]))
    summary=dict(reports=reports,refinement_error_decreases=decreasing,
                 passed=decreasing and all(r['quality']['passed'] and r['fv']['passed'] for r in reports))
    save_json(out/'summary.json',summary)
    plot_mesh(first[0],first[2],out); plot_solution(first[0],first[1],reports,out)
    snapshot=out/'source_snapshot'; snapshot.mkdir(exist_ok=True)
    for path in [Path(__file__),ROOT/'src/nofvm/nofvm_polygon_mesh.py']:
        shutil.copy2(path,snapshot/path.name)
    manifest={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'}
    save_json(out/'artifact_hashes.json',manifest)
    if not summary['passed']:
        raise SystemExit('Mesh or direct FV checks failed; see summary.json.')


if __name__=='__main__':
    main()
