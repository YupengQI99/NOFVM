#!/usr/bin/env python3
"""Evaluate the trained polygon hole operator: single hole and repeated-hole layouts (C1/C2).

The deployed response of the training run is used without further updates. Each layout is
solved with NOFVM, the exact cached local response and direct full FV on the scale-1 and
scale-2 polygon meshes (FV1, FV2); errors are measured against the exact static condensation
of the scale-4 polygon FV tile. Meshes are read from results/polygon_mesh.

Usage:
    python scripts/evaluate_nofvm_polygon_hole.py --training runs/polygon_training --output runs/polygon_reuse
    python scripts/evaluate_nofvm_polygon_hole.py --training runs/polygon_training --output runs/polygon_single --single-only

finish_nofvm_polygon_hole.py collects both outputs into one directory.
"""
from __future__ import annotations
import argparse,gc,hashlib,json,os,platform,shutil,sys,time,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from nofvm.nofvm_polygon_mesh import PolygonMesh
from nofvm.nofvm_polygon_hybrid import TiledFV,ExactTileReference,local_response,interface_faces
from nofvm.runtime import Monitor,resources

MESH=ROOT/'results/polygon_mesh'


def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stats(x):return dict(median=float(np.median(x)),p95=float(np.quantile(x,.95)),maximum=float(np.max(x)))


class Boundary:
    def __init__(self,p,random_loads,seed):
        self.length=1.6*p
        self.y=np.linspace(0,self.length,64*p+1)
        kernel=np.exp(-.5*((self.y[:,None]-self.y[None,:])/.3)**2)
        factor=np.linalg.cholesky(kernel+1e-10*np.eye(len(self.y)))
        samples=np.random.default_rng(seed).normal(size=(random_loads,2,len(self.y)))@factor.T
        self.samples=np.concatenate((np.stack((np.ones(len(self.y)),-np.ones(len(self.y))))[None],
                                     samples,np.ones((1,2,len(self.y)))),axis=0)

    def __call__(self,xy):
        result=np.zeros((len(xy),len(self.samples)))
        for side,x in enumerate((0.,self.length)):
            mask=np.isclose(xy[:,0],x)
            for k,sample in enumerate(self.samples):result[mask,k]=np.interp(xy[mask,1],self.y,sample[side])
        return result


def aggregate_flux(q,ports=32):
    n=q.shape[1]//4
    if n%ports:raise ValueError('Interface aggregation must divide face count')
    return q.reshape(q.shape[0],4,ports,n//ports,q.shape[-1]).sum(axis=3).reshape(q.shape[0],4*ports,q.shape[-1])


def errors(pred,truth,q,qt,volume):
    difference=pred-truth
    numerator=np.einsum('bnk,n,bnk->k',difference,volume,difference,optimize=True)
    denominator=np.einsum('bnk,n,bnk->k',truth,volume,truth,optimize=True)
    eu=np.sqrt(numerator/np.maximum(denominator,1e-28))
    eq=np.sqrt(np.sum((q-qt)**2,axis=(0,1))/np.maximum(np.sum(qt*qt,axis=(0,1)),1e-28))
    return eu,eq


def preflight(meshes,cached,reference):
    checks=[]
    for scale in (1,2):
        mesh=meshes[scale];local=local_response(mesh,mesh.interface_n)
        for p in (1,2):
            bc=Boundary(p,2,123+p)
            f=TiledFV(mesh,p);fr=f.solve(bc)
            h=TiledFV(mesh,p,local);hr=h.solve(bc)
            u=h.reconstruct(hr,mesh,bc,local_xy=local['xy'])
            field_error=float(abs(u-fr['ordinary']).max())
            flux_error=float(abs(hr['flux']-fr['flux']).max())
            exact=ExactTileReference(mesh).solve(p,bc)
            tile_error=float(abs(exact['field']-fr['ordinary']).max())
            row=dict(scale=scale,p=p,hybrid_field_absolute=field_error,hybrid_flux_absolute=flux_error,
                     condensed_tile_reference_absolute=tile_error)
            if max(field_error,flux_error,tile_error)>1e-10:raise RuntimeError(row)
            checks.append(row)
    # Fine reference agreement on a direct scale-4 single-hole solve.
    bc=Boundary(1,2,987)
    direct=TiledFV(meshes[4]);result=direct.solve(bc)
    exact=reference.solve(1,bc)
    error=float(abs(result['ordinary']-exact['field']).max())
    if error>1e-10:raise RuntimeError(f'Fine reference mismatch {error}')
    return dict(coupling_checks=checks,fine_reference_absolute=error,passed=True)


def run_case(p,bc,methods,meshes,neural,cached,reference,output,repetitions=1,single=False):
    target=meshes[4]
    print('REFERENCE',p,flush=True);t=time.perf_counter();truth=reference.solve(p,bc)
    reference_s=time.perf_counter()-t
    qt=aggregate_flux(truth['flux'])
    data=dict(reference_flux=qt,reference_trace=truth['trace'],boundary_y=bc.y,boundary_samples=bc.samples)
    selected=dict(reference_field=truth['field'][:,:,:2],xy=target.xy,volume=target.volume)
    records=[]
    for method in methods:
        for repeat in range(repetitions):
            print('CASE',json.dumps(dict(p=p,method=method,repeat=repeat)),flush=True)
            t=time.perf_counter();cpu=time.process_time()
            if method in ('neural','exact_cached'):
                case=TiledFV(meshes[1],p,neural if method=='neural' else cached)
            else:case=TiledFV(meshes[int(method[-1])],p)
            assembly=time.perf_counter()-t
            t=time.perf_counter();case.factorize();factor=time.perf_counter()-t
            t=time.perf_counter();result=case.solve(bc)
            field=case.reconstruct(result,target,bc,local_xy=cached['xy'])
            solve=time.perf_counter()-t
            cpu_elapsed=time.process_time()-cpu
            q=aggregate_flux(result['flux'])
            eu,eq=errors(field,truth['field'],q,qt,target.volume)
            metrics=dict(field=stats(eu[:-1]),interface_flux=stats(eq[:-1]))
            diagnostics={key:float(result[key]) for key in ('linear_residual','global_balance','local_balance')}
            if 'interface_balance' in result:diagnostics['interface_balance']=float(result['interface_balance'])
            diagnostics.update(constant_field=float(abs(field[:,:,-1]-1).max()),
                               constant_flux=float(abs(q[:,:,-1]).max()))
            if method in ('neural','exact_cached'):
                trace=result['trace']
                local_field=case.response['H']@trace.transpose(1,0,2).reshape(128,-1)
                physical=cached['B'].T@local_field-cached['T']@trace.transpose(1,0,2).reshape(128,-1)
                implied=q.transpose(1,0,2).reshape(128,-1)
                diagnostics['field_flux_mismatch']=float(abs(physical-implied).max())
                resid=cached['A']@local_field-cached['B']@trace.transpose(1,0,2).reshape(128,-1)
                denom=cached['B']@trace.transpose(1,0,2).reshape(128,-1)
                local_residual=np.linalg.norm(resid,axis=0)/np.maximum(np.linalg.norm(denom,axis=0),1e-14)
                residual_stats=stats(local_residual.reshape(p*p,-1)[:,:-1].ravel())
                data[method+'_trace']=trace
            else:residual_stats=None
            lower=bc.samples.min(axis=(1,2));upper=bc.samples.max(axis=(1,2))
            outside_bounds=(field.min(axis=(0,1))<lower-1e-6)|(field.max(axis=(0,1))>upper+1e-6)
            rec=dict(p=p,holes=p*p,method=method,repeat=repeat,unknowns=case.matrix.shape[0],
                     matrix_nnz=case.matrix.nnz,factor_nnz=case.factor.L.nnz+case.factor.U.nnz,
                     assembly_s=assembly,factorization_s=factor,solve_and_complete_output_s=solve,
                     online_s=assembly+factor+solve,online_cpu_s=cpu_elapsed,metrics=metrics,
                     field_errors=eu[:-1].tolist(),flux_errors=eq[:-1].tolist(),diagnostics=diagnostics,
                     internal_residual=residual_stats,maximum_principle_violation_loads=int(outside_bounds[:-1].sum()),
                     resources=resources())
            records.append(rec)
            if repeat==0:
                data[method+'_flux']=q
                if single or p==3:selected[method+'_field']=field[:,:,:2].copy()
            print('RESULT',json.dumps({k:rec[k] for k in ('p','method','repeat','online_s','metrics','diagnostics')}),flush=True)
            save(output/f'{"single" if single else "reuse"}_p{p}_records.json',records)
            del case,field,result
            gc.collect()
    label='single' if single else 'reuse'
    np.savez_compressed(output/f'{label}_p{p}_flux_and_traces.npz',**data)
    if single or p==3:np.savez_compressed(output/f'{label}_p{p}_selected_fields.npz',**selected)
    info=dict(p=p,holes=p*p,loads=len(bc.samples),records=records,reference_s=reference_s,
              reference_cells=p*p*len(target.xy),reference_free_trace_residual=truth['free_trace_residual'],
              reference_local_balance=truth['local_balance'],
              reference_kind='Algebraically exact static condensation of the full scale-4 polygon FV tile; verified against direct FV.')
    save(output/f'{label}_p{p}.json',info)
    return info


def main():
    parser=argparse.ArgumentParser(description='Evaluate the trained polygon hole operator.')
    parser.add_argument('--training',type=Path,required=True,help='Output directory of train_nofvm_polygon_hole.py')
    parser.add_argument('--output',type=Path,required=True,help='New output directory')
    parser.add_argument('--single-only',action='store_true',
                        help='Evaluate only the single hole, timed three times')
    args=parser.parse_args()
    training=args.training;out=args.output;out.mkdir(parents=True,exist_ok=False)
    if not (training/'training.json').exists():raise RuntimeError('Training has not completed')
    manifest=dict(training=str(training),checkpoint_sha256=sha(training/'best.pt'),
                  response_sha256=sha(training/'deployed_response.npz'),
                  python=platform.python_version(),numpy=np.__version__,affinity=sorted(os.sched_getaffinity(0)),
                  blas_threads=os.environ.get('OPENBLAS_NUM_THREADS'),optimizer_updates=0,single_only=args.single_only,
                  sparse_ordering='COLAMD for every method and reference')
    save(out/'manifest.json',manifest)
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for path in [Path(__file__),ROOT/'src/nofvm/nofvm_polygon_hybrid.py',ROOT/'src/nofvm/nofvm_polygon_mesh.py']:
        shutil.copy2(path,snapshot/path.name)
    try:
        with Monitor(out):
            meshes={s:PolygonMesh.load(MESH/f'scale_{s}/control_volumes.npz') for s in (1,2,4)}
            with np.load(training/'deployed_response.npz') as data:neural=dict(data)
            cached=local_response(meshes[4],32)
            t=time.perf_counter();reference=ExactTileReference(meshes[4]);offline=time.perf_counter()-t
            np.savez_compressed(out/'reference_tile_response.npz',H=reference.H,S=reference.S)
            print('PREFLIGHT',flush=True)
            save(out/'preflight.json',preflight(meshes,cached,reference))
            bc=Boundary(1,100,202610041)
            single=run_case(1,bc,['neural','exact_cached','fv1','fv2'],meshes,neural,cached,reference,out,
                            single=True,repetitions=3 if args.single_only else 1)
            reuse=[]
            for p in (() if args.single_only else (1,2,3,5,10)):
                bc=Boundary(p,20,202610050+p)
                reuse.append(run_case(p,bc,['neural','exact_cached','fv1','fv2'],meshes,neural,cached,reference,out,repetitions=3))
                gc.collect()
            summary=dict(single=single,reuse=reuse,reference_preparation_s=offline,
                         checkpoint_sha256=manifest['checkpoint_sha256'],optimizer_updates=0,
                         timing_scope='CPU affinity and BLAS threads as recorded in manifest.json; ordinary/response assembly + global LU + all loads + complete field output. Model training/preparation excluded and recorded separately.')
            save(out/'summary.json',summary)
            save(out/'artifact_hashes.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'})
            print('EVALUATION_COMPLETE',out,flush=True)
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
