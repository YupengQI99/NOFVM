#!/usr/bin/env python3
"""Generate D1 (continuous) or D2 (jump) training and validation labels and train the operator.

Test materials are not generated here; evaluate_nofvm_darcy_linear.py loads the
selected checkpoint and its training metadata.

Usage:
    python scripts/run_nofvm_darcy_linear.py --family continuous
    python scripts/run_nofvm_darcy_linear.py --family jump

Outputs are written to runs/darcy_<family>/training unless --output is given.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch

from nofvm.nofvm_darcy import sample_materials, local_response
from nofvm.nofvm_darcy_operator import DarcyBoundaryOperator

N, M = 64, 32
SEEDS = dict(continuous=dict(train=2026100311, validation=2026100312, test=2026100313, model=1301),
             jump=dict(train=2026100321, validation=2026100322, test=2026100323, model=1302))


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(4*1024*1024):
            digest.update(chunk)
    return digest.hexdigest()


def resources():
    memory = {line.split(':')[0]:float(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    lines = Path('/proc/self/status').read_text().splitlines()
    rss = float(next(line.split()[1] for line in lines if line.startswith('VmRSS:')))/2**20
    cpu = []
    for directory in Path('/sys/class/hwmon').glob('hwmon*'):
        name = directory/'name'
        if name.exists() and name.read_text().strip() in ('k10temp', 'coretemp'):
            cpu.extend(float(path.read_text())/1000 for path in directory.glob('temp*_input'))
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu,memory.used',
                          '--format=csv,noheader,nounits'], text=True, capture_output=True, timeout=10)
    if gpu.returncode:
        raise RuntimeError('Cannot read GPU temperature: '+gpu.stderr.strip())
    temperature, used = (float(x.strip()) for x in gpu.stdout.splitlines()[0].split(','))
    return dict(time=time.time(), cpu_max_c=max(cpu) if cpu else None, gpu_c=temperature,
                gpu_used_mib=used, rss_gib=rss, available_gib=memory['MemAvailable']/2**20,
                disk_free_gib=shutil.disk_usage(ROOT).free/2**30, load_average=list(os.getloadavg()))


class ResourceMonitor:
    def __init__(self, output):
        self.output = output
        self.stop = threading.Event()
        self.records = []
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        try:
            while not self.stop.is_set():
                row = resources()
                self.records.append(row)
                save(self.output/'resource_samples.json', self.records)
                bad = ((row['cpu_max_c'] is not None and row['cpu_max_c'] >= 90) or row['gpu_c'] >= 90
                       or row['rss_gib'] > 40 or row['available_gib'] < 8 or row['disk_free_gib'] < 4)
                if bad:
                    save(self.output/'resource_stop.json', row)
                    os._exit(90)
                self.stop.wait(5)
        except BaseException:
            (self.output/'monitor_failure.txt').write_text(traceback.format_exc())
            os._exit(91)

    def __enter__(self):
        first = resources()
        if ((first['cpu_max_c'] is not None and first['cpu_max_c'] >= 90) or first['gpu_c'] >= 90
                or first['rss_gib'] > 40 or first['available_gib'] < 8 or first['disk_free_gib'] < 4):
            save(self.output/'resource_stop.json', first)
            raise RuntimeError('Resource limit reached before starting this stage')
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=15)


def progress(output, stage, **values):
    record = dict(stage=stage, time=time.time(), **values)
    save(output/'progress.json', record)
    print(json.dumps(record), flush=True)


def labels(family, split, count, output):
    seed = SEEDS[family][split]
    materials = sample_materials(count, seed, family)
    i, j = np.triu_indices(4*M, 1)
    fields = np.empty((count, N*N, 4*M), dtype=np.float32)
    schurs = np.empty((count, 4*M, 4*M))
    log_inputs = np.empty((count, N, N), dtype=np.float32)
    log_weights = np.empty((count, len(i)), dtype=np.float32)
    rows = []
    started = time.perf_counter()
    for index, material in enumerate(materials):
        start = time.perf_counter()
        k = material.evaluate(N)
        response = local_response(k, M)
        h, s, system, b = response['H'], response['S'], response['system'], response['B']
        weights = -s[i, j]
        if np.any(h <= 0) or np.any(weights <= 0):
            raise RuntimeError(f'Nonpositive harmonic label at {split} {index}')
        row = dict(index=index, k_min=float(k.min()), k_max=float(k.max()),
                   symmetry=float(abs(s-s.T).max()), constant_field=float(abs(h.sum(axis=1)-1).max()),
                   constant_flux=float(abs(s.sum(axis=1)).max()),
                   equation_residual=float(abs(system.matrix @ h-b.toarray()).max()),
                   wall_s=time.perf_counter()-start)
        if max(row[key] for key in ('symmetry', 'constant_field', 'constant_flux', 'equation_residual')) > 1e-8:
            raise RuntimeError(f'Incorrect classical label: {row}')
        fields[index], schurs[index] = h, s
        log_inputs[index], log_weights[index] = np.log(k), np.log(weights)
        rows.append(row)
        if index % 8 == 7 or index == count-1:
            progress(output, 'development_labels', split=split, completed=index+1, total=count,
                     elapsed_s=time.perf_counter()-started)
            save(output/(split+'_generation.json'), rows)
    arrays = dict(H=fields, S=schurs, log_k=log_inputs, log_weights=log_weights,
                  nodal_w=np.stack([material.nodal_w for material in materials]))
    np.savez_compressed(output/(split+'_labels.npz'), **arrays)
    metadata = dict(seed=seed, material_count=count, rhs_per_material=4*M,
                    rhs_count=count*4*M, elapsed_s=time.perf_counter()-started,
                    label_sha256=sha(output/(split+'_labels.npz')),
                    k_min=min(r['k_min'] for r in rows), k_max=max(r['k_max'] for r in rows))
    save(output/(split+'_metadata.json'), metadata)
    return arrays, metadata


def device_data(arrays, model):
    data = {key:torch.as_tensor(arrays[key], dtype=torch.float32, device='cuda')
            for key in ('H', 'log_k', 'log_weights')}
    with torch.no_grad():
        data['matrices'] = model.boundary_matrices(data['log_k'])
    return data


def validation_objective(model, data):
    with torch.no_grad():
        prediction = model(data['log_k'], matrices=data['matrices'])
        errors = ((prediction['H']-data['H']).square().sum(dim=(1, 2))
                  / data['H'].square().sum(dim=(1, 2)))
        edge = (prediction['log_weights']-data['log_weights']).square().mean()
        objective = errors.mean() + .2*edge
    return dict(objective=float(objective.cpu()), field_basis_rms=float(errors.mean().sqrt().cpu()),
                field_basis_median=float(errors.sqrt().median().cpu()), log_weight_mse=float(edge.cpu()))


def train(model, arrays, validation_arrays, family, output, steps=4000):
    training = device_data(arrays, model)
    validation = device_data(validation_arrays, model)
    boundary_ids = model.boundary_ids
    interior_ids = np.setdiff1d(np.arange(N*N), boundary_ids.cpu().numpy())
    interior_ids = torch.tensor(interior_ids, dtype=torch.long, device='cuda')
    random = torch.Generator(device='cuda').manual_seed(SEEDS[family]['model']+1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=1e-5)
    history, best, selected = [], float('inf'), None
    initial = validation_objective(model, validation)
    save(output/'initial_validation.json', initial)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for step in range(1, steps+1):
        batch = torch.randint(len(arrays['H']), (8,), generator=random, device='cuda')
        ids = interior_ids[torch.randint(len(interior_ids), (512,), generator=random, device='cuda')]
        matrices = tuple(value[batch] for value in training['matrices'])
        optimizer.zero_grad(set_to_none=True)
        prediction = model(training['log_k'][batch], ids=ids, matrices=matrices)
        target = training['H'][batch[:, None], ids[None, :]]
        target_boundary = training['H'][batch[:, None], boundary_ids[None, :]]
        field = ((prediction['H']-target).square().sum(dim=(1, 2))/target.square().sum(dim=(1, 2))).mean()
        boundary = ((prediction['boundary_field']-target_boundary).square().sum(dim=(1, 2))
                    / target_boundary.square().sum(dim=(1, 2))).mean()
        edges = (prediction['log_weights']-training['log_weights'][batch]).square().mean()
        loss = field + .2*boundary + .2*edges
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f'Nonfinite objective at step {step}')
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % 200 == 0 or step == steps:
            selection = validation_objective(model, validation)
            if selection['objective'] < best:
                best, selected = selection['objective'], step
                torch.save(model.state_dict(), output/'best.pt')
            torch.cuda.synchronize()
            row = dict(step=step, training_loss=float(loss.detach().cpu()),
                       elapsed_s=time.perf_counter()-start, **selection)
            history.append(row)
            progress(output, 'training', **row)
            save(output/'history.json', history)
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    model.load_state_dict(torch.load(output/'best.pt', weights_only=True))
    with (output/'history.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    metadata = dict(steps=steps, selected_step=selected, selection_objective=best,
                    seed=SEEDS[family]['model'], batch_size=8, internal_points_per_step=512,
                    train_materials=len(arrays['H']), validation_materials=len(validation_arrays['H']), test_materials_generated=0,
                    parameters=sum(p.numel() for p in model.parameters()),
                    fixed_buffer_elements=sum(p.numel() for p in model.buffers()),
                    training_s=elapsed, peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    checkpoint_sha256=sha(output/'best.pt'))
    save(output/'training.json', metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=SEEDS, required=True)
    parser.add_argument('--output', type=Path, help='Output directory (default: runs/darcy_<family>/training)')
    parser.add_argument('--smoke', action='store_true',
                        help='Quick pipeline test with two training and two validation materials and three optimizer updates')
    args = parser.parse_args()
    output = (args.output or ROOT/'runs'/f'darcy_{args.family}'/'training').resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(SEEDS[args.family]['model'])
    sources = [Path(__file__).resolve(), ROOT/'src/nofvm/nofvm_darcy.py',
               ROOT/'src/nofvm/nofvm_darcy_operator.py', ROOT/'src/nofvm/nofvm_2d.py',
               ROOT/'src/nofvm/nofvm_hole_operator.py']
    snapshot = output/'source_snapshot'
    snapshot.mkdir()
    for path in sources:
        shutil.copy2(path, snapshot/path.name)
    save(output/'manifest.json', dict(pid=os.getpid(), started=time.time(), family=args.family,
        seeds=SEEDS[args.family], n=N, m=M, affinity=sorted(os.sched_getaffinity(0)),
        sources={str(path):sha(path) for path in sources},
        python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        timing_scope='Shared host; observational timing only',
        run_kind='pipeline_smoke' if args.smoke else 'full',
        scope='Development labels and training only; test materials are not generated'))
    try:
        with ResourceMonitor(output):
            progress(output, 'homogeneous_baseline')
            start = time.perf_counter()
            baseline = local_response(np.ones((N, N)), M)
            baseline_s = time.perf_counter()-start
            np.savez_compressed(output/'homogeneous_baseline.npz', H=baseline['H'], S=baseline['S'])
            arrays, train_metadata = labels(args.family, 'train', 2 if args.smoke else 128, output)
            val_arrays, val_metadata = labels(args.family, 'validation', 2 if args.smoke else 16, output)
            model = DarcyBoundaryOperator(baseline['H'], baseline['S'], N, M).cuda()
            training = train(model, arrays, val_arrays, args.family, output, steps=3 if args.smoke else 4000)
            smoke = None
            if args.smoke:
                import copy
                from evaluate_nofvm_darcy_linear import one_material, summarize
                coupling = output/'coupling_smoke'
                coupling.mkdir()
                material = sample_materials(1, SEEDS[args.family]['validation'], args.family)[0]
                context = one_material(material, args.family, 'validation', 0, model,
                                       copy.deepcopy(model).cpu(), coupling)
                smoke = summarize([context])
                if (smoke['methods']['neural']['maximum_conservation'] > 1e-8 or
                    smoke['methods']['neural']['maximum_constant_error'] > 1e-8 or
                    smoke['maximum_physical_flux_mismatch'] > 1e-8):
                    raise RuntimeError('Pipeline smoke failed conservation, constants, or field-flux consistency')
                save(coupling/'summary.json', smoke)
            save(output/'summary.json', dict(completed=time.time(), family=args.family, training=training,
                train_labels=train_metadata, validation_labels=val_metadata,
                homogeneous_baseline_rhs=4*M, homogeneous_baseline_s=baseline_s,
                run_kind='pipeline_smoke' if args.smoke else 'full', coupling_smoke_completed=smoke is not None,
                test_materials_generated=0))
        save(output/'artifact_hashes.json', {str(path.relative_to(output)):sha(path) for path in sorted(output.rglob('*'))
                                            if path.is_file() and path.name != 'artifact_hashes.json'})
        print('COMPLETE', output, flush=True)
    except BaseException:
        (output/'failure.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
