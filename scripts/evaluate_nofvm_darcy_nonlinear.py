#!/usr/bin/env python3
"""Evaluate the trained D4 operator: independent complete fields, nonlinear balances and costs.

The training directory must contain best.pt and summary.json written by
run_nofvm_darcy_nonlinear.py.

Usage:
    python scripts/evaluate_nofvm_darcy_nonlinear.py --training runs/darcy_nonlinear/training --output runs/darcy_nonlinear/evaluation
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from nofvm.nofvm_2d import boundary_geometry
from nofvm.nofvm_darcy import local_response
from nofvm.nofvm_darcy_nonlinear import phi,inverse_phi,FullNonlinearFV,NonlinearHybrid
from nofvm.nofvm_darcy_nonlinear_operator import build_model
from nofvm.nofvm_hole_operator import deploy_response
from run_nofvm_darcy_ellipse import ResourceMonitor,save,sha,progress


def stats(values):
    return dict(median=float(np.median(values)),p95=float(np.quantile(values,.95)),maximum=float(np.max(values)))


def relative(a,b):
    return np.linalg.norm(a-b,axis=0)/np.maximum(np.linalg.norm(b,axis=0),1e-14)


class Cohort:
    def __init__(self):
        self.y=np.linspace(0,2,65)
        covariance=np.exp(-.5*((self.y[:,None]-self.y[None,:])/.3)**2)
        root=np.linalg.cholesky(covariance+1e-10*np.eye(65))
        random=np.random.default_rng(2026100344)
        draws=5+random.normal(size=(20,2,65))@root.T
        constants=np.broadcast_to(np.array([[6,4],[5,5],[-5,-5],[0,0],[1,-1]])[:,:,None],(5,2,65))
        self.samples=np.concatenate((constants[:1],draws,constants[1:]))

    def __call__(self,xy,ids=None):
        samples=self.samples if ids is None else self.samples[np.atleast_1d(ids)]
        values=np.zeros((len(xy),len(samples)))
        for side,x in enumerate((0.,2.)):
            mask=np.isclose(xy[:,0],x)
            for j,row in enumerate(samples):values[mask,j]=np.interp(xy[mask,1],self.y,row[side])
        return values


def evaluate_record(result,field,truth,qtruth):
    eu,eq=relative(field[:,:21],truth[:,:21]),relative(result['flux'][:,:21],qtruth[:,:21])
    constant=max(float(abs(field[:,j]-v).max()) for j,v in ((21,5.),(22,-5.),(23,0.)))
    constant_flux=float(abs(result['flux'][:,21:24]).max())
    balance=max(result.get(k,0.) for k in ('region_balance','interface_balance','exterior_balance'))
    return dict(field=stats(eu),interface_flux=stats(eq),field_errors=eu.tolist(),flux_errors=eq.tolist(),
        cross_zero=dict(field=float(relative(field[:,24:25],truth[:,24:25])[0]),
                        flux=float(relative(result['flux'][:,24:25],qtruth[:,24:25])[0])),
        constant_error=constant,constant_flux=constant_flux,balance=balance,
        nonlinear_residual=result['nonlinear_residual'],
        nonlinear_iterations=[len(h)-1 for h in result['histories']])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--training',type=Path,required=True,
                        help='Training directory with best.pt and summary.json, e.g. runs/darcy_nonlinear/training')
    parser.add_argument('--output',type=Path,required=True,help='New output directory, e.g. runs/darcy_nonlinear/evaluation')
    args=parser.parse_args();training=args.training.resolve();out=args.output.resolve()
    training_summary=json.loads((training/'summary.json').read_text())
    if sha(training/'best.pt')!=training_summary['checkpoint_sha256']:raise RuntimeError('Checkpoint does not match the training summary')
    out.mkdir(parents=True,exist_ok=False)
    paths=[str(p) for p in (ROOT/'scripts/run_nofvm_darcy_nonlinear.py',
           ROOT/'src/nofvm/nofvm_darcy_nonlinear_operator.py',ROOT/'src/nofvm/nofvm_hole_operator.py',
           ROOT/'src/nofvm/nofvm_darcy_nonlinear.py',ROOT/'src/nofvm/nofvm_darcy.py',
           ROOT/'src/nofvm/nofvm_2d.py',ROOT/'scripts/run_nofvm_darcy_ellipse.py',Path(__file__).resolve())]
    (out/'source_snapshot').mkdir()
    for path in paths:shutil.copy2(path,out/'source_snapshot'/Path(path).name)
    save(out/'manifest.json',dict(pid=os.getpid(),started=time.time(),training_directory=str(training),
        training_summary_sha256=sha(training/'summary.json'),checkpoint_sha256=sha(training/'best.pt'),
        sources={p:sha(p) for p in paths},affinity=sorted(os.sched_getaffinity(0)),
        optimizer_updates=0,global_seed=2026100344,local_seed=2026100345,exclusive_timing=False))
    try:
        with ResourceMonitor(out):
            torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
            start=time.perf_counter();model=build_model().cuda()
            model.load_state_dict(torch.load(training/'best.pt',weights_only=True))
            torch.cuda.synchronize();model_load_time=time.perf_counter()-start
            start=time.perf_counter();exact=local_response(np.ones((64,64)),32)
            exact_generation_time=time.perf_counter()-start
            learned=deploy_response(model)
            np.savez_compressed(out/'neural_response.npz',**learned)
            cohort=Cohort();np.savez_compressed(out/'global_boundary.npz',y=cohort.y,samples=cohort.samples)
            save(out/'shared_preparation.json',dict(model_load_s=model_load_time,exact_cache_generation_s=exact_generation_time,
                 exact_rhs=128,neural_response_sha256=sha(out/'neural_response.npz')))
            records=[];reference=None;qreference=None
            schedule=[('fv64',0)]+[(name,repeat) for repeat in range(3) for name in ('neural','exact_cached','fv32','fv64') if (name,repeat)!=('fv64',0)]
            rest=schedule[1:];np.random.default_rng(2026100346).shuffle(rest);schedule=schedule[:1]+rest
            save(out/'schedule.json',schedule)
            first_solutions={}
            for name,repeat in schedule:
                progress(out,'method',method=name,repeat=repeat)
                start=time.perf_counter()
                if name=='neural':
                    response=deploy_response(model);case=NonlinearHybrid(response)
                elif name=='exact_cached':
                    response=exact;case=NonlinearHybrid(response)
                else:case=FullNonlinearFV(int(name[2:]),32)
                preparation=time.perf_counter()-start
                start=time.perf_counter();result=case.solve(cohort)
                field=case.reconstruct(result) if isinstance(case,NonlinearHybrid) else case.reconstruct(result,64)
                solve=time.perf_counter()-start
                if reference is None:reference=field.copy();qreference=result['flux'].copy()
                record=evaluate_record(result,field,reference,qreference)
                record.update(method=name,repeat=repeat,preparation_s=preparation,solve_and_full_output_25_s=solve,
                    complete_wall_s=preparation+solve,unknowns=3200 if isinstance(case,NonlinearHybrid) else 4*case.n**2,
                    condensed_nonlinear_unknowns=128 if isinstance(case,NonlinearHybrid) else 4*case.n**2)
                lower=cohort.samples[:21].min(axis=(1,2));upper=cohort.samples[:21].max(axis=(1,2))
                record['maximum_principle_violation']=float(max(0.,np.max(lower-field[:,:21].min(axis=0)),np.max(field[:,:21].max(axis=0)-upper)))
                if isinstance(case,NonlinearHybrid):
                    physical=exact['B'].T@phi(result['local_field'])-exact['T']@phi(result['trace'])
                    residual=exact['system'].matrix@phi(result['local_field'])-exact['B']@phi(result['trace'])
                    rhs=exact['B']@phi(result['trace'])
                    record['physical_flux_mismatch']=float(abs(physical-result['flux']).max())
                    record['internal_relative_residual']=stats(np.linalg.norm(residual[:,:21],axis=0)/np.linalg.norm(rhs[:,:21],axis=0))
                if repeat==0:
                    arrays={k:v for k,v in result.items() if isinstance(v,np.ndarray)}
                    arrays['complete_field']=field
                    np.savez_compressed(out/f'{name}_fields.npz',**arrays)
                    save(out/f'{name}_newton_history.json',result['histories'])
                    first_solutions[name]=result
                records.append(record);save(out/f'sample_{name}_{repeat}.json',record)
                progress(out,'sample_complete',method=name,repeat=repeat,field=record['field'],flux=record['interface_flux'])
            save(out/'samples.json',records)
            progress(out,'reference_refinement')
            fine=FullNonlinearFV(128,32)
            fine_sol=fine.solve(lambda xy:cohort(xy,[0,1,24]))
            fine_to_coarse=fine_sol['field'].reshape(256,256,3).reshape(128,2,128,2,3).mean(axis=(1,3)).reshape(16384,3)
            # FV cell averages: compare fine subcell average with coarse pressure.
            refinement=dict(load_ids=[0,1,24],field_errors=relative(reference[:,[0,1,24]],fine_to_coarse).tolist(),
                            flux_errors=relative(qreference[:,[0,1,24]],fine_sol['flux']).tolist(),
                            fine_unknowns=65536,comparison='Four fine pressure centers averaged per coarse FV cell')
            np.savez_compressed(out/'refinement_fields.npz',field=fine_sol['field'],flux=fine_sol['flux'])
            save(out/'reference_refinement.json',refinement)
            ports=boundary_geometry(32,32).xy
            distance=((ports[:,None,:]-ports[None,:,:])**2).sum(axis=-1)
            root=np.linalg.cholesky(np.exp(-.5*distance/.3**2)+1e-10*np.eye(128))
            local_g=5+root@np.random.default_rng(2026100345).normal(size=(128,32))
            local_truth=inverse_phi(exact['H']@phi(local_g));local_pred=inverse_phi(learned['H']@phi(local_g))
            local_flux=-exact['S']@phi(local_g);pred_flux=-learned['S']@phi(local_g)
            np.savez_compressed(out/'local_independent_fields.npz',boundary=local_g,reference=local_truth,predicted=local_pred,
                                reference_flux=local_flux,predicted_flux=pred_flux)
            local_stats=dict(field=stats(relative(local_pred,local_truth)),flux=stats(relative(pred_flux,local_flux)))
            progress(out,'initial_guess_checks')
            guesses=[];hybrid=NonlinearHybrid(learned)
            for value in (-5.,0.,10.):
                try:
                    sol=hybrid.solve(lambda xy:cohort(xy,[0,1,24]),initial=value)
                    error=float(abs(sol['trace']-first_solutions['neural']['trace'][:,[0,1,24]]).max())
                    guesses.append(dict(initial=value,converged=True,trace_max_difference=error,
                                        iterations=[len(h)-1 for h in sol['histories']]))
                except RuntimeError as exc:guesses.append(dict(initial=value,converged=False,error=str(exc)))
            save(out/'initial_guess_checks.json',guesses)
            methods={}
            for name in ('neural','exact_cached','fv32','fv64'):
                selected=[r for r in records if r['method']==name]
                m=dict(next(r for r in selected if r['repeat']==0))
                m['complete_wall_s_median']=float(np.median([r['complete_wall_s'] for r in selected]))
                m['cached_solve_s_median']=float(np.median([r['solve_and_full_output_25_s'] for r in selected]))
                m['all_observed_complete_s']=[r['complete_wall_s'] for r in selected]
                methods[name]=m
            save(out/'summary.json',dict(completed=time.time(),checkpoint_sha256=sha(training/'best.pt'),
                optimizer_updates=0,main_nonconstant_loads=21,all_loads=25,methods=methods,
                local_independent=local_stats,reference_refinement=refinement,initial_guesses=guesses,
                minimum_harmonic_weight=float(learned['H'].min()),
                limitations=['Known Kirchhoff transform specialized to this coefficient',
                             'Checkpoint selected on the complete training basis; independent GP test loads generated after training',
                             'Timings measured on a shared host'],shared_preparation=json.loads((out/'shared_preparation.json').read_text())))
            progress(out,'complete')
        save(out/'artifact_hashes.json',{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='artifact_hashes.json'})
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
