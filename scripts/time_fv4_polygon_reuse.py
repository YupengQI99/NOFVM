#!/usr/bin/env python3
"""Direct full-FV4 timing for the repeated-hole layouts (C2), with the other methods timed on the same host.

No training and no model update: the deployed response in results/polygon_hole/training is read as is, and
``evaluate_nofvm_polygon_hole.run_case`` times NOFVM, the exact local response and full FV1/FV2 exactly as in
the evaluation. In addition, the full FV4 system, the fine discretization whose exact tile condensation is the
accuracy reference, is solved directly and timed.

FV4 timing: assembly, COLAMD sparse LU and all right-hand sides, as for FV1/FV2. Its cell values are already at
the output sites, so the least-squares reconstruction step that maps FV1/FV2 to those sites is not applied; this
favours FV4. The direct solution is compared with the condensed reference after timing.

Usage, pinned to one logical CPU with one BLAS thread:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 19 \
        python scripts/time_fv4_polygon_reuse.py --output runs/polygon_fv4_timing --count-factor-up-to 10

``--count-factor-up-to 10`` also counts the L and U nonzeros of FV4 at 100 holes, which needs about 15 GiB of memory;
the default (5) keeps the 100-hole run below 8 GiB.
"""
from __future__ import annotations
import argparse,gc,hashlib,json,os,platform,resource,shutil,sys,time,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import scipy
import evaluate_nofvm_polygon_hole as evaluation
from nofvm.nofvm_polygon_mesh import PolygonMesh
from nofvm.nofvm_polygon_hybrid import TiledFV,ExactTileReference,local_response
from nofvm.runtime import Monitor,resources

METHODS=['neural','exact_cached','fv1','fv2']
TRAINING=ROOT/'results/polygon_hole/training'
MESH=ROOT/'results/polygon_mesh'
# Stop only when available memory falls below 1 GiB or free disk below 4 GiB: FV4 at 100 holes needs a large LU.
MONITOR_LIMITS=dict(max_cpu_c=None,max_rss_gib=None,min_available_gib=1.)


def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def peak_rss_gib():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20


def time_fv4(p,bc,mesh,reference,repetitions,count_factor=True):
    records=[];field=flux=None
    for repeat in range(repetitions):
        gc.collect()
        print('CASE',json.dumps(dict(p=p,method='fv4',repeat=repeat)),flush=True)
        t=time.perf_counter();cpu=time.process_time()
        case=TiledFV(mesh,p);assembly=time.perf_counter()-t
        t=time.perf_counter();case.factorize();factor=time.perf_counter()-t
        t=time.perf_counter();result=case.solve(bc);solve=time.perf_counter()-t
        cpu_elapsed=time.process_time()-cpu
        rec=dict(p=p,holes=p*p,method='fv4',repeat=repeat,unknowns=int(case.matrix.shape[0]),
                 matrix_nnz=int(case.matrix.nnz),superlu_nnz=int(case.factor.nnz),
                 # L and U are copied out of SuperLU to count them as for the other methods; skipped where that
                 # copy would not fit in memory.
                 factor_nnz=int(case.factor.L.nnz+case.factor.U.nnz) if count_factor else None,
                 assembly_s=assembly,factorization_s=factor,solve_s=solve,online_s=assembly+factor+solve,
                 online_cpu_s=cpu_elapsed,
                 diagnostics={key:float(result[key]) for key in ('linear_residual','global_balance','local_balance')},
                 peak_rss_gib=peak_rss_gib(),resources=resources())
        records.append(rec)
        print('RESULT',json.dumps({k:rec[k] for k in ('p','method','repeat','online_s','superlu_nnz','factor_nnz','peak_rss_gib')}),flush=True)
        if repeat==repetitions-1:
            field=np.array(result['ordinary'],copy=True);flux=np.array(result['flux'],copy=True)
        del case,result
        gc.collect()
    # Agreement of the direct FV4 solution with the condensed reference used for every reported error.
    truth=reference.solve(p,bc)
    agreement=dict(field_absolute=float(abs(field-truth['field']).max()),
                   flux_absolute=float(abs(flux-truth['flux']).max()),
                   field_scale=float(abs(truth['field']).max()))
    del truth,field,flux
    gc.collect()
    return records,agreement


def median(records,method,key='online_s'):
    return float(np.median([r[key] for r in records if r['method']==method]))


def main():
    parser=argparse.ArgumentParser(description='Direct full-FV4 timing for the repeated-hole layouts.')
    parser.add_argument('--output',type=Path,required=True,help='Output directory')
    parser.add_argument('--sizes',type=int,nargs='+',default=[1,2,3,5,10],help='layouts p (p x p holes)')
    parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--count-factor-up-to',type=int,default=5,
                        help='largest p for which the L and U factors of FV4 are copied out to count their nonzeros')
    args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    training=TRAINING
    cpu_model=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                    if line.startswith('model name')),'unknown')
    manifest=dict(purpose='Direct full-FV4 timing for the C2 layouts; NOFVM, exact local response and full FV1/FV2 timed on the same host.',
                  response_sha256=sha(training/'deployed_response.npz'),checkpoint_sha256=sha(training/'best.pt'),
                  optimizer_updates=0,sizes=args.sizes,repetitions=args.repetitions,
                  python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                  platform=platform.platform(),cpu=cpu_model,affinity=sorted(os.sched_getaffinity(0)),
                  blas_threads=os.environ.get('OPENBLAS_NUM_THREADS'),sparse_ordering='COLAMD for every method',
                  host_note='Absolute times depend on the cpu, affinity and blas_threads recorded here; every ratio '
                            'in this run is formed within the run.',
                  fv4_timing='assembly + COLAMD LU + all right-hand sides; no reconstruction step, because FV4 '
                             'cell values are already at the output sites',
                  method_timing='evaluate_nofvm_polygon_hole.run_case')
    save(out/'manifest.json',manifest)
    snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for path in [Path(__file__),ROOT/'scripts/evaluate_nofvm_polygon_hole.py',
                 ROOT/'src/nofvm/nofvm_polygon_hybrid.py',ROOT/'src/nofvm/nofvm_polygon_mesh.py']:
        shutil.copy2(path,snapshot/path.name)
    try:
        with Monitor(out,**MONITOR_LIMITS):
            meshes={s:PolygonMesh.load(MESH/f'scale_{s}/control_volumes.npz') for s in (1,2,4)}
            with np.load(training/'deployed_response.npz') as data:neural=dict(data)
            cached=local_response(meshes[4],32)
            reference=ExactTileReference(meshes[4])
            for p in args.sizes:
                bc=evaluation.Boundary(p,20,202610050+p)
                info=evaluation.run_case(p,bc,METHODS,meshes,neural,cached,reference,out,repetitions=args.repetitions)
                gc.collect()
                fv4,agreement=time_fv4(p,bc,meshes[4],reference,args.repetitions,count_factor=p<=args.count_factor_up_to)
                records=info['records']+fv4
                times={m:median(records,m) for m in METHODS+['fv4']}
                ranges={m:[min(r['online_s'] for r in records if r['method']==m),
                           max(r['online_s'] for r in records if r['method']==m)] for m in METHODS+['fv4']}
                accuracy={}
                for m in METHODS:
                    here=next(r for r in records if r['method']==m and r['repeat']==0)['metrics']
                    accuracy[m]=dict(field_median=here['field']['median'],flux_median=here['interface_flux']['median'])
                row=dict(p=p,holes=p*p,loads=len(bc.samples),
                         unknowns={m:next(r['unknowns'] for r in records if r['method']==m) for m in METHODS+['fv4']},
                         online_median_s=times,online_range_s=ranges,
                         fv4_over={m:times['fv4']/times[m] for m in METHODS},
                         fv4_factor_nnz=fv4[0]['factor_nnz'],fv4_superlu_nnz=fv4[0]['superlu_nnz'],fv4_stage_median_s={k:median(fv4,'fv4',k) for k in
                             ('assembly_s','factorization_s','solve_s')},
                         fv4_agreement_with_condensed_reference=agreement,accuracy=accuracy,
                         peak_rss_gib=peak_rss_gib())
                save(out/f'fv4_p{p}_records.json',fv4)
                save(out/f'timing_p{p}.json',row)
                print('SUMMARY',json.dumps(row),flush=True)
            rows=[json.loads((out/f'timing_p{p}.json').read_text()) for p in args.sizes]
            save(out/'summary.json',dict(sizes=rows,manifest=manifest))
            save(out/'artifact_hashes.json',{str(q.relative_to(out)):sha(q) for q in sorted(out.rglob('*'))
                                             if q.is_file() and q.name!='artifact_hashes.json'})
            print('FV4_TIMING_COMPLETE',out,flush=True)
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
