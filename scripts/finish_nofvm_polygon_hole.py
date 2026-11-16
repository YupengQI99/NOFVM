#!/usr/bin/env python3
"""Collect the polygon hole training and evaluation outputs, re-evaluate them independently and plot them.

Every method/layout combination is solved again and its saved errors, fluxes and fields are
compared with the evaluation records (verification.json). Meshes are read from results/polygon_mesh.

Usage:
    python scripts/finish_nofvm_polygon_hole.py --training runs/polygon_training \
        --evaluation runs/polygon_reuse --single runs/polygon_single --output runs/polygon_hole
"""
from __future__ import annotations
import argparse,hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection,LineCollection
from matplotlib.patches import Rectangle,Polygon as PolygonPatch
from nofvm.nofvm_polygon_mesh import PolygonMesh
from nofvm.nofvm_polygon_hybrid import TiledFV,local_response,ExactTileReference
from evaluate_nofvm_polygon_hole import Boundary

NAMES={'neural':'Neural FV','exact_cached':'Exact cached','fv1':'Full FV1','fv2':'Full FV2'}
COLORS={'neural':'#16836b','exact_cached':'#787878','fv1':'#dc8b2e','fv2':'#326ba4'}
METHODS=list(NAMES)
MESH=ROOT/'results/polygon_mesh'


def save(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def condensed_case(case):
    result={k:case[k] for k in ('p','holes','loads','reference_cells')}
    result['methods']={}
    for method in METHODS:
        rows=[r for r in case['records'] if r['method']==method]
        r=rows[0]
        result['methods'][method]=dict(metrics=r['metrics'],diagnostics=r['diagnostics'],unknowns=r['unknowns'],
              online_median_s=float(np.median([v['online_s'] for v in rows])),
              online_samples_s=[v['online_s'] for v in rows],
              assembly_median_s=float(np.median([v['assembly_s'] for v in rows])),
              factorization_median_s=float(np.median([v['factorization_s'] for v in rows])),
              solve_output_median_s=float(np.median([v['solve_and_complete_output_s'] for v in rows])),
              internal_residual=r['internal_residual'],maximum_principle_violation_loads=r['maximum_principle_violation_loads'])
    return result


def verify(output,meshes,neural,cached,reference):
    checks=[]
    def check(name,condition,value=None):
        checks.append(dict(name=name,passed=bool(condition),value=value))
        if not condition:raise RuntimeError(f'Verification failed {name}: {value}')
    for path in sorted((output/'cases').glob('*_p*.json')):
        case=json.loads(path.read_text());p=case['p'];single=path.name.startswith('single')
        seed=202610041 if single else 202610050+p
        bc=Boundary(p,100 if single else 20,seed)
        truth=reference.solve(p,bc)
        n=truth['flux'].shape[1]//128
        qt=np.stack([truth['flux'][:,j*n:(j+1)*n].sum(axis=1) for j in range(128)],axis=1)
        saved=np.load(output/'data'/f'{path.stem}_flux_and_traces.npz')
        check(path.stem+'/boundary',np.array_equal(saved['boundary_samples'],bc.samples))
        check(path.stem+'/reference_flux',np.max(abs(qt-saved['reference_flux']))<1e-10)
        for method in METHODS:
            rows=[r for r in case['records'] if r['method']==method];row=rows[0]
            solver=TiledFV(meshes[1] if method in ('neural','exact_cached') else meshes[int(method[-1])],p,
                           (neural if method=='neural' else cached) if method in ('neural','exact_cached') else None)
            result=solver.solve(bc)
            field=solver.reconstruct(result,meshes[4],bc,local_xy=cached['xy'])
            ratio=result['flux'].shape[1]//128
            q=np.stack([result['flux'][:,j*ratio:(j+1)*ratio].sum(axis=1) for j in range(128)],axis=1)
            num=np.zeros(case['loads']);den=np.zeros_like(num)
            for tile in range(p*p):
                num+=np.sum((field[tile]-truth['field'][tile])**2*meshes[4].volume[:,None],axis=0)
                den+=np.sum(truth['field'][tile]**2*meshes[4].volume[:,None],axis=0)
            field_error=np.sqrt(num/np.maximum(den,1e-28))[:-1]
            flux_error=np.sqrt(np.sum((q-qt)**2,axis=(0,1))/np.maximum(np.sum(qt**2,axis=(0,1)),1e-28))[:-1]
            prefix=path.stem+'/'+method
            check(prefix+'/all_field_errors',np.max(abs(field_error-row['field_errors']))<1e-10)
            check(prefix+'/all_flux_errors',np.max(abs(flux_error-row['flux_errors']))<1e-10)
            check(prefix+'/saved_flux',np.max(abs(q-saved[method+'_flux']))<1e-10)
            check(prefix+'/balance',max(result[k] for k in ('linear_residual','global_balance','local_balance'))<1e-8)
            check(prefix+'/constant',np.max(abs(field[:,:,-1]-1))<1e-8)
            for repeated in rows:
                check(prefix+f'/repeat_{repeated["repeat"]}_accuracy',
                      np.max(abs(np.array(repeated['field_errors'])-field_error))<1e-10)
                check(prefix+f'/repeat_{repeated["repeat"]}_cost',
                      abs(repeated['online_s']-sum(repeated[k] for k in ('assembly_s','factorization_s','solve_and_complete_output_s')))<1e-12)
            selected_path=output/'data'/f'{path.stem}_selected_fields.npz'
            if selected_path.exists():
                with np.load(selected_path) as selected:
                    check(prefix+'/selected_field',np.max(abs(selected[method+'_field']-field[:,:,:2]))<1e-10)
            print('VERIFIED',prefix,flush=True)
            del field,solver,result
        saved.close()
    return dict(passed=True,checks=checks,count=len(checks),
                scope='Independent rerun of all 24 method/layout combinations; every nonconstant field/flux error and constant control, all saved fluxes and selected fields. Runtime samples are checked arithmetically, not reproduced as equal wall times.')


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
                         'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
                         'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.6})


def savefig(fig,out,name):
    fig.savefig(out/f'{name}.png',dpi=220,bbox_inches='tight')
    fig.savefig(out/f'{name}.pdf',bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)


def field_panel(ax,mesh,values,p,clim,title):
    polygons=mesh.polygons
    collection=PolyCollection([poly+[1.6*x,1.6*y] for y in range(p) for x in range(p) for poly in polygons],
                              array=values.ravel(),cmap='RdBu_r',edgecolors='none',antialiaseds=False,rasterized=True)
    collection.set_clim(*clim);ax.add_collection(collection)
    for y in range(p):
        for x in range(p):
            offset=np.array([1.6*x,1.6*y])
            ax.add_patch(PolygonPatch(mesh.hole+offset,fc='white',ec='#38495a',lw=.4))
            ax.add_patch(Rectangle(offset+[.4,.4],.8,.8,fill=False,ec='#465b65',lw=.45,ls=(0,(3,3))))
    ax.set(xlim=(0,1.6*p),ylim=(0,1.6*p),xlabel='$x$',ylabel='$y$')
    ax.set_aspect('equal');ax.set_title(title,loc='left',pad=8)
    return collection


def plot_single(output,mesh):
    fields=np.load(output/'data/single_p1_selected_fields.npz')
    traces=np.load(output/'data/single_p1_flux_and_traces.npz')
    summary=json.loads((output/'summary.json').read_text())
    fig=plt.figure(figsize=(10.4,8.8));grid=fig.add_gridspec(3,3,height_ratios=[1,1,.62],hspace=.30,wspace=.28)
    for row,load in enumerate((0,1)):
        ref=fields['reference_field'][:,:,load];pred=fields['neural_field'][:,:,load]
        limits=(min(ref.min(),pred.min()),max(ref.max(),pred.max()));limit=float(abs(pred-ref).max())
        for col,(values,title,clim) in enumerate(((ref,'Reference FV',limits),(pred,'Neural FV',limits),(pred-ref,'Signed error',(-limit,limit)))):
            ax=fig.add_subplot(grid[row,col])
            pic=field_panel(ax,mesh,values,1,clim,f'{chr(97+row*3+col)}   {title}')
            fig.colorbar(pic,ax=ax,shrink=.74,pad=.025)
    ax=fig.add_subplot(grid[2,:2])
    for key,label,color in [('reference_flux','Reference FV','#326ba4'),('neural_flux','Neural FV','#16836b')]:
        ax.plot(np.arange(128),traces[key][0,:,1],color=color,label=label,lw=1.1,
                ls='-' if key.startswith('reference') else '--')
    for x in (31.5,63.5,95.5):ax.axvline(x,color='.8',lw=.6)
    ax.set(xlabel='Interface segment (left / right / bottom / top)',ylabel='Integrated outward flux')
    ax.set_title('g   Interface flux, first GP load',loc='left');ax.legend(frameon=False,ncol=2,fontsize=8)
    ax=fig.add_subplot(grid[2,2])
    case=json.loads((output/'cases/single_p1.json').read_text())
    row=next(r for r in case['records'] if r['method']=='neural')
    ax.boxplot([100*np.array(row['field_errors']),100*np.array(row['flux_errors'])],tick_labels=['Field','Flux'],whis=(5,95),showfliers=True)
    ax.set_ylabel('Relative error (%)');ax.set_title('h   101 held-out loads',loc='left')
    fig.suptitle('Single-hole heat conduction on boundary-fitted polygon meshes',y=.99,fontsize=13)
    fig.text(.05,.025,'Rows 1–2: fixed 1/−1 contrast and the first GP boundary load. '
             'Full fields shown at all 31,554 reference sites; colour scales shared within each pair.',fontsize=8)
    fig.subplots_adjust(bottom=.09,top=.95)
    savefig(fig,output/'figures','single_hole')
    fields.close();traces.close()


def plot_reuse(output,mesh):
    fields=np.load(output/'data/reuse_p3_selected_fields.npz')
    ref=fields['reference_field'][:,:,1];pred=fields['neural_field'][:,:,1]
    limits=(min(ref.min(),pred.min()),max(ref.max(),pred.max()));limit=float(abs(pred-ref).max())
    fig,axes=plt.subplots(1,3,figsize=(12,4.2),layout='constrained')
    fig.get_layout_engine().set(rect=(0,.08,1,.92))
    for k,(ax,values,title,clim) in enumerate(zip(axes,[ref,pred,pred-ref],['Reference FV','Trained neural operator × 9','Signed error'],[limits,limits,(-limit,limit)])):
        pic=field_panel(ax,mesh,values,3,clim,f'{chr(97+k)}   {title}');fig.colorbar(pic,ax=ax,shrink=.75,pad=.025)
    fig.text(.03,.02,'First GP load; 283,986 reference locations. The same trained hole operator is used nine times, without retraining.',fontsize=8)
    savefig(fig,output/'figures','nine_hole_fields');fields.close()


def plot_mesh(output,mesh,local_hole=None):
    fig,axes=plt.subplots(1,2,figsize=(10,5),layout='constrained')
    for mode,ax in enumerate(axes):
        for y in range(3):
            for x in range(3):
                offset=np.array([1.6*x,1.6*y])
                if mode==0:
                    ax.add_collection(LineCollection(mesh.faces+offset,colors='#596777',linewidths=.23,rasterized=True))
                else:
                    polys=[poly+offset for poly,inside in zip(mesh.polygons,mesh.inside) if not inside]
                    ax.add_collection(PolyCollection(polys,facecolors='#f5f6f8',edgecolors='#596777',linewidths=.3,rasterized=True))
                    ax.add_patch(Rectangle(offset+[.4,.4],.8,.8,fc='#e1f0eb',ec='#16836b',lw=.9))
                    ax.text(offset[0]+.8,offset[1]+1.08,'NO',ha='center',fontsize=7,color='#16836b')
                hole=mesh.hole if mode==0 or local_hole is None else local_hole
                ax.add_patch(PolygonPatch(hole+offset,fc='white',ec='#38495a',lw=.6))
        ax.set(xlim=(0,4.8),ylim=(0,4.8),xlabel='$x$',ylabel='$y$');ax.set_aspect('equal')
    axes[0].set_title('a   Full FV1: 17,379 polygon cells',loc='left')
    axes[1].set_title('b   Neural FV: 6,876 cells + 1,152 traces',loc='left')
    savefig(fig,output/'figures','nine_hole_mesh')


def plot_scaling(output,summary):
    cases=summary['reuse'];holes=np.array([c['holes'] for c in cases])
    fig,axes=plt.subplots(2,2,figsize=(10,7.8),layout='constrained')
    fig.get_layout_engine().set(rect=(0,.065,1,.935))
    for k,(ax,metric,title) in enumerate(zip(axes[0],['field','interface_flux'],['Complete-field error','Interface-flux error'])):
        for method in METHODS:
            med=[100*c['methods'][method]['metrics'][metric]['median'] for c in cases]
            p95=[100*c['methods'][method]['metrics'][metric]['p95'] for c in cases]
            ax.plot(holes,med,'o-',label=NAMES[method],color=COLORS[method],lw=1.2,ms=4)
            ax.plot(holes,p95,':',color=COLORS[method],lw=.8)
        ax.set(xscale='log',xlabel='Number of holes',ylabel='Relative error (%)')
        ax.set_title(f'{chr(97+k)}   {title}',loc='left');ax.grid(alpha=.15)
    axes[0,0].legend(frameon=False,fontsize=8,ncol=2)
    ax=axes[1,0]
    for method in METHODS:
        times=[c['methods'][method]['online_median_s'] for c in cases]
        ax.loglog(holes,times,'o-',label=NAMES[method],color=COLORS[method],lw=1.2,ms=4)
    ax.set(xlabel='Number of holes',ylabel='Assembly + factorization + output (s)')
    ax.set_title('c   CPU online cost, 22 loads',loc='left');ax.grid(alpha=.15)
    ax=axes[1,1];last=cases[-1]
    heights=[last['methods'][m]['online_median_s'] for m in METHODS]
    ax.bar(np.arange(4),heights,color=[COLORS[m] for m in METHODS],alpha=.8)
    for i,m in enumerate(METHODS):
        raw=last['methods'][m]['online_samples_s'];ax.scatter(i+np.linspace(-.09,.09,len(raw)),raw,color='black',s=10,zorder=4)
    ax.set_xticks(np.arange(4),[NAMES[m] for m in METHODS],rotation=15,ha='right')
    ax.set_ylabel('Time (s)');ax.set_title('d   100 holes: all three repetitions',loc='left')
    ax.set_ylim(0,max(max(last['methods'][m]['online_samples_s']) for m in METHODS)*1.15)
    fig.suptitle('Reuse of the trained operator: accuracy and measured computation cost',fontsize=13)
    fig.text(.02,.018,'Errors: solid = median, dotted = P95. Costs exclude offline training. '
             'Both full FV baselines are more accurate; these are not matched-error speedups.',fontsize=7.5)
    savefig(fig,output/'figures','accuracy_and_cost')


def main():
    parser=argparse.ArgumentParser(description='Collect, verify and plot the polygon hole results.')
    for name,text in (('training','Output directory of train_nofvm_polygon_hole.py'),
                      ('evaluation','Output directory of evaluate_nofvm_polygon_hole.py'),
                      ('single','Output directory of evaluate_nofvm_polygon_hole.py --single-only'),
                      ('output','New output directory, e.g. runs/polygon_hole')):
        parser.add_argument('--'+name,type=Path,required=True,help=text)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    for name in ('training','cases','data','figures','source_snapshot'): (out/name).mkdir()
    for name in ('best.pt','deployed_response.npz','training.json','manifest.json','preflight.json','history.csv'):
        shutil.copy2(args.training/name,out/'training'/name)
    history=out/'training/history.csv'
    history.write_text(history.read_text())  # Normalize CSV line endings; preserve every value.
    for path in (args.training/'source_snapshot').iterdir():shutil.copy2(path,out/'source_snapshot'/('training_'+path.name))
    evaluation=json.loads((args.evaluation/'summary.json').read_text())
    single=json.loads((args.single/'single_p1.json').read_text())
    cases=[('single',single,args.single)]+[('reuse',c,args.evaluation) for c in evaluation['reuse']]
    for label,case,source in cases:
        name=f'{label}_p{case["p"]}'
        shutil.copy2(source/f'{name}.json',out/'cases'/f'{name}.json')
        shutil.copy2(source/f'{name}_flux_and_traces.npz',out/'data'/f'{name}_flux_and_traces.npz')
        if (source/f'{name}_selected_fields.npz').exists():shutil.copy2(source/f'{name}_selected_fields.npz',out/'data'/f'{name}_selected_fields.npz')
    for prefix,source in [('reuse',args.evaluation),('single',args.single)]:
        for name in ('manifest.json','preflight.json','resource_samples.json'):
            shutil.copy2(source/name,out/f'{prefix}_{name}')
        for path in (source/'source_snapshot').iterdir():shutil.copy2(path,out/'source_snapshot'/(prefix+'_'+path.name))
    meshes={s:PolygonMesh.load(MESH/f'scale_{s}/control_volumes.npz') for s in (1,2,4)}
    with np.load(args.training/'deployed_response.npz') as values:neural=dict(values)
    cached=local_response(meshes[4]);reference=ExactTileReference(meshes[4])
    verification=verify(out,meshes,neural,cached,reference);save(out/'verification.json',verification)
    summary=dict(training=json.loads((args.training/'training.json').read_text()),single=condensed_case(single),
                 reuse=[condensed_case(c) for c in evaluation['reuse']],verification_passed=verification['passed'],
                 optimizer_updates_during_reuse=0,
                 timing_scope=evaluation['timing_scope'],
                 notes=['FV1 and FV2 both have lower field and flux errors than the learned hybrid.',
                        'The finest FV4 accuracy reference uses exact tile condensation, not a timed direct full-domain solve.',
                        'Single-hole timings use a separate three-repeat run (--single-only).'])
    save(out/'summary.json',summary)
    style();plot_single(out,meshes[4]);plot_reuse(out,meshes[4]);plot_mesh(out,meshes[1],meshes[4].hole);plot_scaling(out,summary)
    shutil.copy2(Path(__file__),out/'source_snapshot'/Path(__file__).name)
    save(out/'artifact_hashes.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'})
    print('COMPLETE',out,flush=True)


if __name__=='__main__':main()
