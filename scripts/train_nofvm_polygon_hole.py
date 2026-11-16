#!/usr/bin/env python3
"""Train the polygon hole operator (C1/C2) on a GPU with a fixed budget of 4000 updates.

Labels are the exact local responses of the scale-4 polygon mesh around the insulated hole,
read from results/polygon_mesh (see generate_nofvm_polygon_mesh.py).

Usage:
    python scripts/train_nofvm_polygon_hole.py --output runs/polygon_training
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,platform,shutil,sys,time,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from scipy.interpolate import RegularGridInterpolator
from nofvm.nofvm_polygon_mesh import PolygonMesh
from nofvm.nofvm_polygon_hybrid import local_response
from nofvm.nofvm_hole import hole_system,condensed_response
from nofvm.nofvm_hole_operator import HoleBoundaryOperator,deploy_response
from nofvm.runtime import guard


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description='Train the polygon hole operator.')
    parser.add_argument('--output',type=Path,required=True,help='New output directory')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    setup=dict(geometry='Square [.4,1.2]^2 minus radius .15 insulated circular hole; 384 polygonal hole segments.',
        local_mesh_scale=4,ordinary_mesh_scale=1,local_cells=19065,retained_trace_ports=128,
        label_generation='128 deterministic Dirichlet basis solves on the fine polygonal local domain.',
        model='HoleBoundaryOperator: coordinate trunk and positive graph-response weights; projected field/flux agreement.',
        base_response='Solid square (no hole), FV128 with 128 trace ports; bilinear interpolation of its positive field basis to polygon sites.',
        training=dict(steps=4000,seed=13,batch=2048,lr=.001,min_lr=.00001,loss_weights=[1,.2,.2],
                      interior_sampling='Probability proportional to FV cell volume',selection='Training basis volume error + .2 log-edge MSE every 200 updates'))
    snapshot=out/'source_snapshot';snapshot.mkdir()
    sources=[Path(__file__),ROOT/'src/nofvm/nofvm_polygon_mesh.py',ROOT/'src/nofvm/nofvm_polygon_hybrid.py',ROOT/'src/nofvm/nofvm_hole_operator.py']
    for path in sources:shutil.copy2(path,snapshot/path.name)
    meshpath=ROOT/'results/polygon_mesh/scale_4/control_volumes.npz'
    save(out/'manifest.json',dict(setup=setup,sources={str(p.relative_to(ROOT)):sha(p) for p in sources},
         mesh_sha256=sha(meshpath),torch=torch.__version__,numpy=np.__version__,python=platform.python_version(),
         gpu=torch.cuda.get_device_name(),initial_resources=guard(out)))
    try:
        start=time.perf_counter()
        mesh=PolygonMesh.load(meshpath);labels=local_response(mesh,32)
        np.savez_compressed(out/'local_labels.npz',H=labels['H'],S=labels['S'],T=labels['T'],xy=labels['xy'],volume=labels['volume'])
        solid=hole_system(128,radius=0.);base=condensed_response(solid,32)
        axis=(np.arange(128)+.5)*(.8/128)
        interp=RegularGridInterpolator((axis,axis),base['H'].reshape(128,128,128),bounds_error=True)
        query=np.clip(labels['xy'][:,::-1],axis[0],axis[-1])
        base_h=interp(query)
        base_h/=base_h.sum(axis=1,keepdims=True)
        torch.manual_seed(13)
        model=HoleBoundaryOperator(base_h,base['S'],labels['xy'],labels['B'],labels['T']).cuda()
        initial=deploy_response(model)
        preflight=dict(label_constant_field=float(abs(labels['H'].sum(axis=1)-1).max()),
                       label_constant_flux=float(abs(labels['S'].sum(axis=1)).max()),
                       label_symmetry=float(abs(labels['S']-labels['S'].T).max()),
                       initial_field_flux=float(abs(labels['B'].T@initial['H']-labels['T']+initial['S']).max()),
                       initial_constant_field=float(abs(initial['H'].sum(axis=1)-1).max()),
                       boundary_map_rank=int(np.linalg.matrix_rank(labels['B'].toarray())),
                       required_rank=128)
        assert preflight['boundary_map_rank']==128
        assert max(preflight[k] for k in ('label_constant_field','label_constant_flux','label_symmetry','initial_field_flux','initial_constant_field'))<1e-8
        save(out/'preflight.json',preflight)
        preparation=time.perf_counter()-start
        print('PREPARED',json.dumps(dict(seconds=preparation,**preflight)),flush=True)
        H=torch.as_tensor(labels['H'],dtype=torch.float32,device='cuda')
        i,j=model.edge_i.cpu().numpy(),model.edge_j.cpu().numpy()
        logs=torch.tensor(np.log(-labels['S'][i,j]),dtype=torch.float32,device='cuda')
        weights=torch.as_tensor(labels['volume'],dtype=torch.float32,device='cuda')
        interior=torch.tensor(np.setdiff1d(np.arange(len(H)),model.boundary_ids.cpu().numpy()),device='cuda')
        probabilities=weights[interior]/weights[interior].sum()
        opt=torch.optim.Adam(model.parameters(),lr=.001)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,4000,eta_min=1e-5)
        torch.manual_seed(13);best=float('inf');history=[];resources=[guard(out)]
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
        for step in range(1,4001):
            ids=interior[torch.multinomial(probabilities,2048,replacement=True)]
            opt.zero_grad(set_to_none=True)
            prediction=model.raw_field(ids);S,logw=model.schur();hb=model.boundary_field(S)
            field=(prediction-H[ids]).square().mean()/H[ids].square().mean()
            boundary=(hb-H[model.boundary_ids]).square().mean()/H[model.boundary_ids].square().mean()
            edge=(logw-logs).square().mean();loss=field+.2*boundary+.2*edge
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward();opt.step();schedule.step()
            if step%200==0:
                with torch.no_grad():
                    full=model.raw_field();S,logw=model.schur();full[model.boundary_ids]=model.boundary_field(S)
                    ferr=((full-H).square()*weights[:,None]).sum()/(H.square()*weights[:,None]).sum()
                    logerr=(logw-logs).square().mean();score=float((ferr+.2*logerr).cpu())
                if score<best:
                    best=score;selected=step;torch.save(model.state_dict(),out/'best.pt')
                row=dict(step=step,loss=float(loss.detach().cpu()),selection=score,
                         volume_weighted_basis_error=float(ferr.sqrt().cpu()),log_weight_mse=float(logerr.cpu()),
                         elapsed_s=time.perf_counter()-start)
                history.append(row);print('TRAIN',json.dumps(row),flush=True);resources.append(guard(out))
                save(out/'progress.json',row)
        torch.cuda.synchronize();training_time=time.perf_counter()-start
        model.load_state_dict(torch.load(out/'best.pt',weights_only=True))
        t=time.perf_counter();deployed=deploy_response(model);deployment=time.perf_counter()-t
        np.savez_compressed(out/'deployed_response.npz',**deployed)
        with (out/'history.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=history[0],lineterminator='\n');writer.writeheader();writer.writerows(history)
        final=dict(steps=4000,selected_step=selected,training_s=training_time,preparation_s=preparation,
                   deployment_s=deployment,checkpoint_sha256=sha(out/'best.pt'),label_sha256=sha(out/'local_labels.npz'),
                   peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,parameters=sum(p.numel() for p in model.parameters()),
                   constant_field_error=float(abs(deployed['H'].sum(axis=1)-1).max()),
                   field_flux_agreement=float(abs(labels['B'].T@deployed['H']-labels['T']+deployed['S']).max()),
                   resources=resources)
        save(out/'training.json',final);print('TRAINING_COMPLETE',json.dumps(final),flush=True)
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
