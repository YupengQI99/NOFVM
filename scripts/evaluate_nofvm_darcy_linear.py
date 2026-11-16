#!/usr/bin/env python3
"""Full-field D1/D2 evaluation of a trained operator on validation and test materials.

Usage:
    python scripts/evaluate_nofvm_darcy_linear.py --training runs/darcy_continuous/training
    python scripts/evaluate_nofvm_darcy_linear.py --training runs/darcy_jump/training

Outputs are written to runs/darcy_<family>/evaluation unless --output is given.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch

from nofvm.nofvm_darcy import (sample_materials, DarcyBoundary, local_response,
                                 FullDarcyFV, DarcyHybrid, local_field_flux)
from nofvm.nofvm_darcy_operator import DarcyBoundaryOperator, deploy_response
from run_nofvm_darcy_linear import N, M, SEEDS, ResourceMonitor, progress, resources, save, sha

METHODS = ('neural', 'exact_condensed', 'fv32', 'fv64')


def timed(function, gpu=False):
    if gpu:
        torch.cuda.synchronize()
    wall, cpu = time.perf_counter(), time.process_time()
    result = function()
    if gpu:
        torch.cuda.synchronize()
    return result, dict(wall_s=time.perf_counter()-wall, cpu_s=time.process_time()-cpu)


def stats(values):
    return dict(median=float(np.median(values)), p95=float(np.quantile(values, .95)),
                maximum=float(np.max(values)))


def norms(prediction, truth):
    error = np.linalg.norm(prediction-truth, axis=0)
    reference = np.linalg.norm(truth, axis=0)
    return error/np.maximum(reference, 1e-14), error, reference


def array_hash(value):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(value).data).hexdigest()


def prepare(method, material, gpu_model):
    if method == 'neural':
        response, preparation = timed(lambda:deploy_response(gpu_model, material.evaluate(N)), gpu=True)
        case, assembly = timed(lambda:DarcyHybrid(response, N, M))
    elif method == 'exact_condensed':
        response, preparation = timed(lambda:local_response(material.evaluate(N), M))
        case, assembly = timed(lambda:DarcyHybrid(response, N, M))
    else:
        preparation = dict(wall_s=0., cpu_s=0.)
        case, assembly = timed(lambda:FullDarcyFV(material, int(method[2:]), M))
    _, factor = timed(case.factorize)
    return case, dict(material_response=preparation, assembly=assembly, factorization=factor)


def sample(method, material, boundary, repeat, model):
    case, timing = prepare(method, material, model)
    def output():
        result = case.solve(boundary)
        field = case.reconstruct(result) if isinstance(case, DarcyHybrid) else case.reconstruct(result, N)
        return result, field
    (result, field), timing['solve_and_full_output_5'] = timed(output)
    row = dict(method=method, repeat=repeat, **timing,
               complete_wall_s=sum(t['wall_s'] for t in timing.values()),
               complete_cpu_s=sum(t['cpu_s'] for t in timing.values()),
               unknowns=case.matrix.shape[0], matrix_nnz=case.matrix.nnz,
               factor_nnz=case.factor.L.nnz+case.factor.U.nnz,
               diagnostics={key:result[key] for key in ('linear_residual', 'region_balance', 'exterior_balance')},
               field_sha256=array_hash(field), flux_sha256=array_hash(result['flux']))
    if 'interface_balance' in result:
        row['diagnostics']['interface_balance'] = result['interface_balance']
    return case, result, field, row


def evaluate_record(row, field, result, truth, flux_truth, boundary):
    field_error, field_norm, truth_norm = norms(field, truth)
    flux_error, flux_norm, flux_reference_norm = norms(result['flux'], flux_truth)
    row.update(field_errors=field_error[:4].tolist(), flux_errors=flux_error[:4].tolist(),
               field_error_norms=field_norm.tolist(), field_reference_norms=truth_norm.tolist(),
               flux_error_norms=flux_norm.tolist(), flux_reference_norms=flux_reference_norm.tolist(),
               field=stats(field_error[:4]), interface_flux=stats(flux_error[:4]),
               constant_field_absolute=float(abs(field[:, -1]-1).max()),
               constant_flux_absolute=float(abs(result['flux'][:, -1]).max()))
    low, high = boundary.samples.min(axis=(1, 2)), boundary.samples.max(axis=(1, 2))
    row['maximum_principle_violation'] = float(max(np.maximum(low-field.min(axis=0), 0).max(),
                                                 np.maximum(field.max(axis=0)-high, 0).max()))


def reference_trace(material, field):
    n = N
    ids = np.arange(4*n*n).reshape(2*n, 2*n)
    a, b = n//2, 3*n//2
    inner = np.r_[ids[a:b, a], ids[a:b, b-1], ids[a, a:b], ids[b-1, a:b]]
    outer = np.r_[ids[a:b, a-1], ids[a:b, b], ids[a-1, a:b], ids[b, a:b]]
    k = material.evaluate(n)
    ki = np.r_[k[:, 0], k[:, -1], k[0], k[-1]]
    trace = (ki[:, None]*field[inner]+field[outer])/(ki[:, None]+1)
    return trace.reshape(4*M, n//M, -1).mean(axis=1)


def diagnostics(material, neural_case, neural_result, exact_case, truth):
    # Only post-hoc diagnostics use a reference trace; the coupled solves above
    # have already solved their own unknown trace values.
    projected_trace = reference_trace(material, truth)
    learned, exact = neural_case.response, exact_case.response
    pu, tq = learned['H'] @ projected_trace, exact['H'] @ projected_trace
    lfield, _, _ = norms(pu, tq)
    lflux, _, _ = norms(-learned['S'] @ projected_trace, -exact['S'] @ projected_trace)
    field = neural_result['local_field']
    rhs = exact['B'] @ neural_result['trace']
    residual = exact['system'].matrix @ field-rhs
    pde = np.linalg.norm(residual, axis=0)/np.maximum(np.linalg.norm(rhs, axis=0), 1e-14)
    physical = local_field_flux(material.evaluate(N), field, neural_result['trace'], M)
    return dict(local_field_errors=lfield[:4].tolist(), local_flux_errors=lflux[:4].tolist(),
                local_field=stats(lfield[:4]), local_flux=stats(lflux[:4]),
                internal_discrete_residual=stats(pde[:4]), internal_residual_by_load=pde[:4].tolist(),
                physical_flux_mismatch=float(abs(physical-neural_result['flux']).max()),
                minimum_harmonic_weight=float(learned['H'].min()),
                diagnostic_trace_sha256=array_hash(projected_trace),
                reference_trace_used_only_for_posthoc_local_diagnostic=True)


def refinement(material, boundary, truth, flux_truth):
    fine = FullDarcyFV(material, 128, M)
    fine.factorize()
    result = fine.solve(lambda xy:boundary(xy, indices=[0, 1]))
    raster = result['field'].reshape(2*N, 2, 2*N, 2, 2).mean(axis=(1, 3))
    field_error, _, _ = norms(truth[:, :2], raster.reshape(4*N*N, 2))
    flux_error, _, _ = norms(flux_truth[:, :2], result['flux'])
    return dict(coarse_unknowns=4*N*N, fine_unknowns=4*128*128, load_ids=[0, 1],
                field_errors=field_error.tolist(), interface_flux_errors=flux_error.tolist(),
                scope='First four test materials, two loads each; differences are not rigorous error bounds')


def one_material(material, family, split, index, model, cpu_model, output):
    stem = f'{split}_{index:03d}'
    seed = SEEDS[family][split]+10000+index
    boundary = DarcyBoundary(seed)
    np.savez_compressed(output/(stem+'_inputs.npz'), nodal_w=material.nodal_w,
                        boundary_nodes=boundary.y, boundary_samples=boundary.samples)
    records = []
    progress(output, 'reference', split=split, material=index)
    reference_case, reference_result, truth, row = sample('fv64', material, boundary, 0, model)
    flux_truth = reference_result['flux'].copy()
    evaluate_record(row, truth, reference_result, truth, flux_truth, boundary)
    records.append(row)
    retain_fields = split == 'test' and index < 4
    if retain_fields:
        np.savez_compressed(output/(stem+'_fv64_fields.npz'), field=truth, flux=flux_truth)
    del reference_case, reference_result
    schedule = [(method, rep) for rep in range(3) for method in METHODS if (method, rep) != ('fv64', 0)]
    random = np.random.default_rng(SEEDS[family][split]+20000+index)
    random.shuffle(schedule)
    save(output/(stem+'_schedule.json'), dict(reference_first=['fv64', 0], remaining=schedule))
    saved_cases, saved_results = {}, {}
    for method, repeat in schedule:
        case, result, field, row = sample(method, material, boundary, repeat, model)
        evaluate_record(row, field, result, truth, flux_truth, boundary)
        records.append(row)
        save(output/(stem+'_samples.json'), records)
        if repeat == 0 and method in ('neural', 'exact_condensed'):
            saved_cases[method], saved_results[method] = case, result
        if repeat == 0 and retain_fields:
            np.savez_compressed(output/(stem+'_'+method+'_fields.npz'), field=field, flux=result['flux'])
        del case, result, field
    extra = diagnostics(material, saved_cases['neural'], saved_results['neural'],
                        saved_cases['exact_condensed'], truth)
    if retain_fields:
        extra['reference_refinement'] = refinement(material, boundary, truth, flux_truth)
        k = material.evaluate(N)
        cpu_response, cpu_time = timed(lambda:deploy_response(cpu_model, k))
        gpu_response, gpu_time = timed(lambda:deploy_response(model, k), gpu=True)
        extra['cpu_gpu'] = dict(cpu_preparation=cpu_time, gpu_preparation=gpu_time,
                               H_max_absolute=float(abs(cpu_response['H']-gpu_response['H']).max()),
                               S_max_absolute=float(abs(cpu_response['S']-gpu_response['S']).max()))
    else:
        extra['reference_refinement'] = None
    exact_trace = saved_results['exact_condensed']['trace']
    neural_trace = saved_results['neural']['trace']
    extra['exact_coupled_trace_sha256'] = array_hash(exact_trace)
    extra['neural_coupled_trace_sha256'] = array_hash(neural_trace)
    save(output/(stem+'_diagnostics.json'), extra)
    progress(output, 'material_complete', split=split, material=index,
             field=next(r['field'] for r in records if r['method']=='neural'),
             flux=next(r['interface_flux'] for r in records if r['method']=='neural'))
    del saved_cases, saved_results, truth, flux_truth
    gc.collect()
    return dict(split=split, index=index, boundary_seed=seed, records=records, diagnostics=extra)


def summarize(materials):
    methods = {}
    for method in METHODS:
        rows = [next(r for r in context['records'] if r['method']==method and r['repeat']==0) for context in materials]
        all_rows = [r for context in materials for r in context['records'] if r['method']==method]
        field = np.concatenate([r['field_errors'] for r in rows])
        flux = np.concatenate([r['flux_errors'] for r in rows])
        timing = [float(np.median([r['complete_wall_s'] for r in context['records'] if r['method']==method])) for context in materials]
        warm = [float(np.median([r['solve_and_full_output_5']['wall_s'] for r in context['records'] if r['method']==method])) for context in materials]
        balance = max(max(r['diagnostics'].get(k, 0.) for k in ('region_balance', 'interface_balance', 'exterior_balance')) for r in rows)
        constant = max(max(r['constant_field_absolute'], r['constant_flux_absolute']) for r in rows)
        uf, qf = stats(field), stats(flux)
        methods[method] = dict(field=uf, interface_flux=qf, field_errors=field.tolist(), flux_errors=flux.tolist(),
            unknowns=rows[0]['unknowns'], material_count=len(rows), nonconstant_loads=len(field),
            complete_per_material_median_s=timing, warm_per_material_median_s=warm,
            median_complete_wall_s=float(np.median(timing)), median_warm_5_load_wall_s=float(np.median(warm)),
            complete_all_observations_s=[r['complete_wall_s'] for r in all_rows],
            maximum_conservation=balance, maximum_constant_error=constant,
            maximum_principle_violation=max(r['maximum_principle_violation'] for r in rows),
            repeated_outputs_identical=all(len({r['field_sha256'] for r in c['records'] if r['method']==method})==1
                                           and len({r['flux_sha256'] for r in c['records'] if r['method']==method})==1 for c in materials))
    eligible = []
    for name, row in methods.items():
        if name == 'neural':
            continue
        row['matches_neural_accuracy'] = all(row[key][stat] <= methods['neural'][key][stat]
                                            for key in ('field', 'interface_flux') for stat in ('median', 'p95'))
        if row['matches_neural_accuracy']:
            eligible.append(name)
    best = min(eligible, key=lambda name:methods[name]['median_complete_wall_s'])
    return dict(methods=methods, fastest_matching_classical=best,
                classical_over_neural_complete_ratio=methods[best]['median_complete_wall_s']/methods['neural']['median_complete_wall_s'],
                local_field=stats(np.concatenate([m['diagnostics']['local_field_errors'] for m in materials])),
                local_flux=stats(np.concatenate([m['diagnostics']['local_flux_errors'] for m in materials])),
                maximum_physical_flux_mismatch=max(m['diagnostics']['physical_flux_mismatch'] for m in materials),
                minimum_harmonic_weight=min(m['diagnostics']['minimum_harmonic_weight'] for m in materials),
                maximum_internal_discrete_residual=max(m['diagnostics']['internal_discrete_residual']['maximum'] for m in materials),
                comparison_scope='Only tested classical methods, cohort accuracy matching, shared physical core')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', type=Path, required=True,
                        help='Training directory written by run_nofvm_darcy_linear.py')
    parser.add_argument('--output', type=Path, help='Output directory (default: runs/darcy_<family>/evaluation)')
    args = parser.parse_args()
    training = args.training.resolve()
    trained = json.loads((training/'summary.json').read_text())
    manifest = json.loads((training/'manifest.json').read_text())
    for path, expected in manifest['sources'].items():
        if sha(path) != expected:
            raise RuntimeError('Training source changed: '+path)
    for filename, expected in json.loads((training/'artifact_hashes.json').read_text()).items():
        if sha(training/filename) != expected:
            raise RuntimeError('Training artifact changed: '+filename)
    checkpoint = trained['training']['checkpoint_sha256']
    if (sha(training/'best.pt') != checkpoint or trained['test_materials_generated'] != 0
            or trained.get('run_kind') != 'full' or trained['training']['steps'] != 4000
            or trained['training']['train_materials'] != 128 or trained['training']['validation_materials'] != 16):
        raise RuntimeError('Checkpoint is not from a full training run (4000 steps, 128 training and 16 validation materials)')
    family = trained['family']
    output = (args.output or ROOT/'runs'/f'darcy_{family}'/'evaluation').resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    sources = [Path(__file__).resolve(), Path(__file__).with_name('run_nofvm_darcy_linear.py'),
               ROOT/'src/nofvm/nofvm_darcy.py',
               ROOT/'src/nofvm/nofvm_darcy_operator.py', ROOT/'src/nofvm/nofvm_2d.py',
               ROOT/'src/nofvm/nofvm_hole_operator.py']
    snapshot = output/'source_snapshot'
    snapshot.mkdir()
    for path in sources:
        shutil.copy2(path, snapshot/path.name)
    save(output/'manifest.json', dict(pid=os.getpid(), started=time.time(), family=family,
        training_directory=str(training), training_summary_sha256=sha(training/'summary.json'),
        checkpoint_sha256=checkpoint, optimizer_updates=0, sources={str(p):sha(p) for p in sources},
        seeds=SEEDS[family], repeats=3, affinity=sorted(os.sched_getaffinity(0)),
        shared_physical_core=True, reference_local_n=N, ordinary_m=M,
        timing_scope='Shared host; observational cost only',
        reference_refined_test_materials=[0, 1, 2, 3], reference_refined_loads=[0, 1]))
    try:
        with ResourceMonitor(output):
            def load_models():
                base = np.load(training/'homogeneous_baseline.npz')
                gpu_model = DarcyBoundaryOperator(base['H'], base['S'], N, M).cuda()
                gpu_model.load_state_dict(torch.load(training/'best.pt', weights_only=True))
                return gpu_model, copy.deepcopy(gpu_model).cpu()
            (model, cpu_model), preparation = timed(load_models, gpu=True)
            save(output/'shared_preparation.json', dict(checkpoint_and_models=preparation,
                offline_training=trained['training'], train_labels=trained['train_labels'],
                validation_labels=trained['validation_labels'], homogeneous_baseline_s=trained['homogeneous_baseline_s']))
            summaries = {}
            for split, count in (('validation', 16), ('test', 32)):
                materials = sample_materials(count, SEEDS[family][split], family)
                np.savez_compressed(output/(split+'_materials.npz'), nodal_w=np.stack([m.nodal_w for m in materials]))
                contexts = []
                for index, material in enumerate(materials):
                    contexts.append(one_material(material, family, split, index, model, cpu_model, output))
                summaries[split] = summarize(contexts)
                save(output/(split+'_summary.json'), summaries[split])
            save(output/'summary.json', dict(completed=time.time(), family=family, checkpoint_sha256=checkpoint,
                optimizer_updates=0, **summaries, test_material_count=32, test_nonconstant_load_count=128,
                shared_physical_core=True, resources=resources()))
        save(output/'artifact_hashes.json', {p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name!='artifact_hashes.json'})
        print('COMPLETE', output, flush=True)
    except BaseException:
        (output/'failure.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
