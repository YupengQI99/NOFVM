#!/usr/bin/env python3
"""Train the D4 (nonlinear Darcy) cell operator with a fixed budget of 4000 steps.

The labels are the exact local response of one 64 x 64 cell; no test inputs are
generated here.

Usage:
    python scripts/run_nofvm_darcy_nonlinear.py --output runs/darcy_nonlinear/training
"""
import argparse
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
from nofvm.nofvm_darcy import local_response
from nofvm.nofvm_darcy_nonlinear_operator import build_model
from nofvm.nofvm_hole_operator import deploy_response
from run_nofvm_darcy_ellipse import ResourceMonitor,save,sha,progress


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True,help='New output directory, e.g. runs/darcy_nonlinear/training')
    args=parser.parse_args();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__).resolve(),
             ROOT/'src/nofvm/nofvm_darcy_nonlinear_operator.py',ROOT/'src/nofvm/nofvm_hole_operator.py',
             ROOT/'src/nofvm/nofvm_darcy_nonlinear.py',ROOT/'src/nofvm/nofvm_darcy.py',
             ROOT/'src/nofvm/nofvm_2d.py',ROOT/'scripts/run_nofvm_darcy_ellipse.py']
    (out/'source_snapshot').mkdir()
    for path in sources:shutil.copy2(path,out/'source_snapshot'/path.name)
    save(out/'manifest.json',dict(pid=os.getpid(),started=time.time(),sources={str(p):sha(p) for p in sources},
                                 seed=1403,affinity=sorted(os.sched_getaffinity(0)),test_inputs_generated=0))
    try:
        with ResourceMonitor(out):
            torch.set_num_threads(1);torch.set_num_interop_threads(1)
            torch.manual_seed(1403);torch.use_deterministic_algorithms(True)
            start=time.perf_counter();labels=local_response(np.ones((64,64)),32)
            np.savez_compressed(out/'labels.npz',H=labels['H'],S=labels['S'])
            label_time=time.perf_counter()-start
            model=build_model().cuda()
            initial=deploy_response(model)
            np.savez_compressed(out/'initial_response.npz',**initial)
            H=torch.tensor(labels['H'],dtype=torch.float32,device='cuda')
            logs=torch.tensor(np.log(-labels['S'][model.edge_i.cpu(),model.edge_j.cpu()]),dtype=torch.float32,device='cuda')
            interior=np.setdiff1d(np.arange(4096),model.boundary_ids.cpu().numpy())
            interior=torch.tensor(interior,device='cuda')
            opt=torch.optim.Adam(model.parameters(),lr=1e-3)
            schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,4000,eta_min=1e-5)
            history=[];best=float('inf');selected=None
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
            for step in range(1,4001):
                ids=interior[torch.randint(len(interior),(1024,),device='cuda')]
                opt.zero_grad(set_to_none=True)
                h=model.raw_field(ids);s,logw=model.schur();hb=model.boundary_field(s)
                loss=((h-H[ids]).square().mean()/H[ids].square().mean()+
                      .2*(hb-H[model.boundary_ids]).square().mean()/H[model.boundary_ids].square().mean()+
                      .2*(logw-logs).square().mean())
                if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite D4 loss')
                loss.backward()
                if step==1:
                    if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                        raise RuntimeError('Nonfinite D4 gradient')
                opt.step();schedule.step()
                if step%200==0:
                    with torch.no_grad():
                        full=model.raw_field();s,logw=model.schur()
                        full[model.boundary_ids]=model.boundary_field(s)
                        field=(full-H).square().sum()/H.square().sum()
                        edges=(logw-logs).square().mean()
                        score=float((field+.2*edges).cpu())
                    if score<best:
                        best=score;selected=step;torch.save(model.state_dict(),out/'best.pt')
                    row=dict(step=step,objective=score,field_basis_relative=float(field.sqrt().cpu()),
                             log_weight_mse=float(edges.cpu()),elapsed_s=time.perf_counter()-start)
                    history.append(row);save(out/'history.json',history);progress(out,'training',**row)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            model.load_state_dict(torch.load(out/'best.pt',weights_only=True))
            deployed=deploy_response(model)
            np.savez_compressed(out/'deployed_response.npz',**deployed)
            initial_errors={k:float(np.linalg.norm(initial[k]-labels[k])/np.linalg.norm(labels[k])) for k in ('H','S')}
            learned_errors={k:float(np.linalg.norm(deployed[k]-labels[k])/np.linalg.norm(labels[k])) for k in ('H','S')}
            save(out/'summary.json',dict(completed=time.time(),steps=4000,selected_step=selected,
                seed=1403,training_s=elapsed,label_preparation_s=label_time,label_rhs=128,geometry_count=1,
                initial_relative_errors=initial_errors,deployed_relative_errors=learned_errors,
                parameters=sum(p.numel() for p in model.parameters()),
                fixed_buffer_elements=sum(p.numel() for p in model.buffers()),
                peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                checkpoint_sha256=sha(out/'best.pt'),labels_sha256=sha(out/'labels.npz'),
                deployed_response_sha256=sha(out/'deployed_response.npz'),test_inputs_generated=0,
                minimum_harmonic_weight=float(deployed['H'].min()),
                field_flux_identity_error=float(abs(labels['B'].T@deployed['H']-(labels['T']-deployed['S'])).max()),
                selection_scope='Full training basis objective; not an independent validation cohort',
                inference_contains_target_classical_response=False))
            progress(out,'complete')
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
