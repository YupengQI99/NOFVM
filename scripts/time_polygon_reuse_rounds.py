#!/usr/bin/env python3
"""Interleaved timing of the repeated-hole layouts (C2) over many rounds.

At 100 holes the stage that writes the full field varies by up to a factor of two between repetitions of the
same method, so the methods are interleaved and sampled across the whole run. The timed code is the same as in
time_fv4_polygon_reuse.py; only the sampling differs:

- one round times, in this order, NOFVM, the exact local response, full FV1 and full FV2 with
  ``evaluate_nofvm_polygon_hole.run_case`` (one repetition each), then the direct full-FV4 solve;
- the round is repeated ``--rounds`` times, so every method is sampled across the whole run;
- every sample is kept and reported; nothing is discarded.

No training and no model update. The deployed response is read from results/polygon_hole/training and the
meshes from results/polygon_mesh.

Usage, pinned to one logical CPU with one BLAS thread:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 19 \
        python scripts/time_polygon_reuse_rounds.py --output runs/polygon_timing --count-factor-up-to 10
"""
from __future__ import annotations
import argparse,gc,json,os,platform,shutil,sys,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import scipy
import evaluate_nofvm_polygon_hole as evaluation
from time_fv4_polygon_reuse import (METHODS as BASE_METHODS,MESH,MONITOR_LIMITS,TRAINING,
                                    peak_rss_gib,save,sha,time_fv4)
from nofvm.nofvm_polygon_mesh import PolygonMesh
from nofvm.nofvm_polygon_hybrid import ExactTileReference,local_response
from nofvm.runtime import Monitor

METHODS=BASE_METHODS+['fv4']
STAGES=dict(assembly='assembly_s',factorization='factorization_s')


def stage(record,name):
    if name in STAGES:return record[STAGES[name]]
    return record['solve_s'] if record['method']=='fv4' else record['solve_and_complete_output_s']


def main():
    parser=argparse.ArgumentParser(description='Interleaved timing of the repeated-hole layouts.')
    parser.add_argument('--output',type=Path,required=True,help='Output directory, e.g. runs/polygon_timing')
    parser.add_argument('--sizes',type=int,nargs='+',default=[1,2,3,5,10],help='layouts p (p x p holes)')
    parser.add_argument('--rounds',type=int,default=10)
    parser.add_argument('--count-factor-up-to',type=int,default=5,
                        help='largest p for which the L and U factors of FV4 are copied out to count their nonzeros')
    args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    training=TRAINING
    cpu_model=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                    if line.startswith('model name')),'unknown')
    manifest=dict(purpose='Interleaved timing of the C2 layouts: NOFVM, exact local response, full FV1/FV2 and '
                          'the direct full-FV4 solve, one sample per method per round.',
                  response_sha256=sha(training/'deployed_response.npz'),checkpoint_sha256=sha(training/'best.pt'),
                  optimizer_updates=0,sizes=args.sizes,rounds=args.rounds,round_order=METHODS,
                  python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                  platform=platform.platform(),cpu=cpu_model,affinity=sorted(os.sched_getaffinity(0)),
                  blas_threads=os.environ.get('OPENBLAS_NUM_THREADS'),sparse_ordering='COLAMD for every method',
                  sampling='Every round and every sample is kept; medians are taken over all rounds.',
                  fv4_timing='assembly + COLAMD LU + all right-hand sides; no reconstruction step, because FV4 '
                             'cell values are already at the output sites',
                  method_timing='evaluate_nofvm_polygon_hole.run_case, one repetition per round')
    save(out/'manifest.json',manifest)
    snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for path in [Path(__file__),ROOT/'scripts/time_fv4_polygon_reuse.py',ROOT/'scripts/evaluate_nofvm_polygon_hole.py',
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
                records=[];agreements=[]
                for k in range(args.rounds):
                    print('ROUND',json.dumps(dict(p=p,round=k)),flush=True)
                    info=evaluation.run_case(p,bc,BASE_METHODS,meshes,neural,cached,reference,out,repetitions=1)
                    gc.collect()
                    fv4,agreement=time_fv4(p,bc,meshes[4],reference,1,count_factor=k==0 and p<=args.count_factor_up_to)
                    for r in info['records']+fv4:r['round']=k
                    records+=info['records']+fv4;agreements.append(agreement)
                    save(out/f'rounds_p{p}_records.json',records)
                    gc.collect()
                samples={m:[r['online_s'] for r in records if r['method']==m] for m in METHODS}
                medians={m:float(np.median(v)) for m,v in samples.items()}
                accuracy={}
                for m in BASE_METHODS:
                    here=next(r for r in records if r['method']==m and r['round']==0)['metrics']
                    accuracy[m]=dict(field_median=here['field']['median'],flux_median=here['interface_flux']['median'])
                first=next(r for r in records if r['method']=='fv4' and r['round']==0)
                row=dict(p=p,holes=p*p,loads=len(bc.samples),rounds=args.rounds,
                         unknowns={m:next(r['unknowns'] for r in records if r['method']==m) for m in METHODS},
                         online_samples_s=samples,online_median_s=medians,
                         online_min_s={m:min(v) for m,v in samples.items()},
                         online_max_s={m:max(v) for m,v in samples.items()},
                         stage_median_s={m:{name:float(np.median([stage(r,name) for r in records if r['method']==m]))
                                            for name in ('assembly','factorization','solve_and_output')} for m in METHODS},
                         fv4_over={m:medians['fv4']/medians[m] for m in BASE_METHODS},
                         exact_cached_over_neural=medians['exact_cached']/medians['neural'],
                         fv4_factor_nnz=first['factor_nnz'],fv4_superlu_nnz=first['superlu_nnz'],
                         fv4_agreement_with_condensed_reference=dict(
                             field_absolute=max(a['field_absolute'] for a in agreements),
                             flux_absolute=max(a['flux_absolute'] for a in agreements),
                             field_scale=agreements[0]['field_scale']),
                         accuracy=accuracy,
                         peak_rss_gib=peak_rss_gib())
                save(out/f'timing_p{p}.json',row)
                print('SUMMARY',json.dumps({k:row[k] for k in ('p','holes','rounds','online_median_s','online_min_s',
                                                                'online_max_s','fv4_over','exact_cached_over_neural')}),flush=True)
            rows=[json.loads((out/f'timing_p{p}.json').read_text()) for p in args.sizes]
            save(out/'summary.json',dict(sizes=rows,manifest=manifest))
            save(out/'artifact_hashes.json',{str(q.relative_to(out)):sha(q) for q in sorted(out.rglob('*'))
                                             if q.is_file() and q.name!='artifact_hashes.json'})
            print('ROUNDS_TIMING_COMPLETE',out,flush=True)
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
