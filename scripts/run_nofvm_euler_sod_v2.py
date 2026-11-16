#!/usr/bin/env python3
"""Train the NOFVM Euler local decoder and evaluate coupled shock-tube problems.

The decoder is trained with a fixed budget on windows of reference Euler
solutions, then coupled with a finite-volume solver and evaluated on the Sod
problem and on held-out Riemann problems.

Usage:
    python scripts/run_nofvm_euler_sod_v2.py --output runs/euler_sod
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from nofvm.nofvm_euler_1d import (EulerFV, conserved, primitive, rusanov, initial_average,
                                    exact_riemann, physical_flux, limit_reconstruction)
from nofvm.nofvm_euler_operator_v2 import EulerLocalDecoder, decoder_loss, torch_rusanov, torch_flux
from run_nofvm_darcy_ellipse import ResourceMonitor, save, sha

TIMES = (0., .02, .04, .06, .08, .10, .12, .15, .18, .20)
SEEDS = dict(train=2026100341, validation=2026100342, test=2026100353, model=1304, batches=2026100344)


BaseEulerFV = EulerFV
class EulerFV(BaseEulerFV):
    def stage(self, *args, **kwargs):
        self._stage_count = getattr(self, '_stage_count', 0) + 1
        if self._stage_count > 40000:
            raise RuntimeError('Maximum 20000 Euler steps exceeded; CFL collapse')
        return super().stage(*args, **kwargs)


def progress(output, stage, **values):
    row = dict(stage=stage, time=time.time(), **values)
    save(output / 'progress.json', row)
    print(json.dumps(row), flush=True)


def sample_conditions(count, seed):
    rng = np.random.default_rng(seed)
    high = np.column_stack((rng.uniform(.8, 1.2, count), rng.uniform(-.08, .08, count), rng.uniform(.8, 1.2, count)))
    low = np.column_stack((rng.uniform(.08, .20, count), rng.uniform(-.08, .08, count), rng.uniform(.06, .18, count)))
    common = rng.uniform(-.15, .15, count)
    high[:, 1] += common
    low[:, 1] += common
    left, right = high.copy(), low.copy()
    mirror = np.arange(count) % 2 == 1
    left[mirror], right[mirror] = low[mirror], high[mirror]
    left[mirror, 1] *= -1
    right[mirror, 1] *= -1
    position = rng.uniform(.35, .65, count)
    return left, right, position


def gpu_trajectories(left, right, position, n=1024, times=TIMES):
    initial = np.stack([initial_average(n, l, r, x) for l, r, x in zip(left, right, position)])
    q = torch.tensor(initial, device='cuda', dtype=torch.float32)
    result = [initial.astype(np.float32)]
    current = 0.
    with torch.no_grad():
        for target in times[1:]:
            while target - current > 1e-12:
                dt = min(.35 / n / float(torch_flux(q)[1].max()), target - current)
                def update(u):
                    padded = torch.cat((u[:, :1], u, u[:, -1:]), dim=1)
                    flux = torch_rusanov(padded[:, :-1], padded[:, 1:])
                    return u - dt * n * (flux[:, 1:] - flux[:, :-1])
                q = .5 * (q + update(update(q)))
                current += dt
            result.append(q.cpu().numpy())
    return np.stack(result, axis=1)


def precheck():
    checks = []
    _, exact = exact_riemann(np.array([.5]), .2)
    checks.append(dict(name='Sod exact star state', passed=abs(exact['star_pressure'] - .3031301780506468) < 1e-12 and abs(exact['star_velocity'] - .9274526200489499) < 1e-12))
    star = conserved([exact['star_density_right'], exact['star_velocity'], exact['star_pressure']])
    right = conserved([.125, 0, .1])
    rh = physical_flux(star)[0] - physical_flux(right)[0] - exact['right_shock_speed'] * (star - right)
    checks.append(dict(name='Rankine Hugoniot', error=float(abs(rh).max()), passed=float(abs(rh).max()) < 1e-12))
    errors = []
    for n in (256, 512, 1024):
        result = EulerFV(n=n).run(output_times=(.2,))
        truth, _ = exact_riemann((np.arange(n) + .5) / n, .2)
        error = np.sum(abs(primitive(result['field']) - primitive(truth)), axis=0) / np.sum(abs(primitive(truth)), axis=0)
        errors.append(error)
        checks.append(dict(name=f'FV{n} physical budget', error=max(abs(v) for v in result['history'][-1]['global_budget_error']), passed=max(abs(v) for v in result['history'][-1]['global_budget_error']) < 1e-11))
    checks.append(dict(name='reference refinement', errors=np.array(errors).tolist(), passed=bool((np.diff(errors, axis=0) < 0).all())))
    rng = np.random.default_rng(31)
    w = np.column_stack((rng.uniform(.1, 2, 100), rng.uniform(-1, 1, 100), rng.uniform(.1, 2, 100)))
    q = conserved(w)
    tf = torch_rusanov(torch.tensor(q[:-1]), torch.tensor(q[1:])).numpy()
    difference = float(abs(tf - rusanov(q[:-1], q[1:])).max())
    checks.append(dict(name='Torch and NumPy flux', error=difference, passed=difference < 1e-12))
    model = EulerLocalDecoder().eval()
    with torch.no_grad():
        model.branch[-1].weight.normal_(0, .01)
    constant = conserved([.8, .3, .7])
    context = np.repeat(constant[None], 20, axis=0)
    decoded = model.numpy_decoder(context)
    checks.append(dict(name='nonzero network constant preservation', passed=float(abs(decoded - constant).max()) < 1e-12))
    bad = np.repeat(q[:16, None], 8, axis=1)
    bad[:, 0, 0] -= 3
    bad[:, 1, 0] += 3
    limited, count, theta = limit_reconstruction(bad, q[:16])
    checks.append(dict(name='positive conservative reconstruction limiter', limited=count, theta=theta,
                       passed=bool((primitive(limited)[..., (0, 2)] > 0).all() and abs(limited.mean(axis=1) - q[:16]).max() < 1e-12)))
    model = EulerLocalDecoder().eval()
    coupled = EulerFV(start=640, decoder=model.numpy_decoder).run(output_times=(.2,))
    row = coupled['history'][-1]
    checks.append(dict(name='coupled regional and global budgets', values=row,
                       passed=max(abs(v) for k in ('global_budget_error', 'region_budget_error') for v in row[k]) < 1e-11 and row['reconstruction_mean_error'] < 1e-12))
    left, right, x = sample_conditions(1, 42)
    gpu = gpu_trajectories(left, right, x, n=256, times=(0., .04))[0, -1]
    cpu = EulerFV(n=256).run(left[0], right[0], x[0], final_time=.04, output_times=(.04,))['field']
    error = float(abs(gpu - cpu).max())
    checks.append(dict(name='GPU label and CPU reference consistency', error=error, passed=error < 2e-4))
    batch = torch.tensor(np.stack([initial_average(160, l, r, .473) for l, r in zip(*sample_conditions(4, 41)[:2])]), dtype=torch.float32)
    probe = EulerLocalDecoder()
    with torch.no_grad():
        probe.branch[-1].weight.normal_(0, 2.)
        context = batch.reshape(4, 20, 8, 3).mean(dim=2)
        positive = probe(context)
        w = primitive(positive.numpy())
        mean_error = float(abs(positive.mean(dim=2) - context[:, 2:-2]).max())
    checks.append(dict(name='training decoder positive conservative stress check',
                       mean_error=mean_error, minimum_density=float(w[..., 0].min()), minimum_pressure=float(w[..., 2].min()),
                       passed=bool((w[..., (0, 2)] > 0).all() and mean_error < 5e-7)))
    loss, _ = decoder_loss(model, batch)
    loss.backward()
    checks.append(dict(name='nonzero supervised loss and finite gradients', loss=float(loss.detach()), passed=bool(loss > 0 and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))))
    return dict(all_passed=all(row['passed'] for row in checks), checks=checks)


def labels(split, count, output):
    started = time.perf_counter()
    left, right, position = sample_conditions(count, SEEDS[split])
    trajectories = gpu_trajectories(left, right, position)
    w = primitive(trajectories)
    if not np.isfinite(w).all() or w[..., 0].min() <= 0 or w[..., 2].min() <= 0:
        raise RuntimeError('Nonphysical reference labels')
    rng = np.random.default_rng(SEEDS[split] + 17)
    windows, ids = [], []
    for case in range(count):
        for index, t in enumerate(TIMES):
            field = trajectories[case, index]
            jump = np.linalg.norm(np.diff(field / np.array([1., 2., 3.]), axis=0), axis=1) + 1e-9
            jump /= jump.sum()
            for sample in range(8):
                if sample < 6:
                    center = rng.choice(1023, p=jump)
                    start = int(np.clip(center - rng.integers(24, 136), 0, 1024 - 160))
                else:
                    start = int(rng.integers(0, 1024 - 160 + 1))
                windows.append(field[start:start + 160])
                ids.append((case, index, start))
    data = np.asarray(windows, dtype=np.float32)
    np.savez_compressed(output / (split + '_data.npz'), fine=data, window_origin=np.array(ids),
                        left=left, right=right, discontinuity=position, times=np.array(TIMES))
    elapsed = time.perf_counter() - started
    meta = dict(seed=SEEDS[split], independent_initial_conditions=count, windows=len(data), wall_s=elapsed,
                minimum_density=float(w[..., 0].min()), minimum_pressure=float(w[..., 2].min()))
    save(output / (split + '_generation.json'), meta)
    progress(output, 'labels_complete', split=split, **meta)
    return data, meta


def train(training, validation, output, steps=6000):
    torch.manual_seed(SEEDS['model'])
    model = EulerLocalDecoder().cuda()
    train_data = torch.tensor(training, device='cuda')
    val_data = torch.tensor(validation, device='cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=.0001)
    generator = torch.Generator(device='cuda').manual_seed(SEEDS['batches'])
    with torch.no_grad():
        best = sum(float(decoder_loss(model, batch)[0]) * len(batch) for batch in val_data.split(256)) / len(val_data)
    initial_validation_loss = best
    selected, records = 0, []
    save(output / 'initial_validation.json', dict(validation_loss=best))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step in range(1, steps + 1):
        model.train()
        batch = train_data[torch.randint(len(train_data), (128,), generator=generator, device='cuda')]
        optimizer.zero_grad(set_to_none=True)
        loss, terms = decoder_loss(model, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        scheduler.step()
        if step % 200 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                total = 0.
                for batch in val_data.split(256):
                    val, _ = decoder_loss(model, batch)
                    total += float(val) * len(batch)
                score = total / len(val_data)
            if score < best:
                best, selected = score, step
                torch.save(dict(model={k:v.detach().cpu() for k, v in model.state_dict().items()},
                                selected_step=step, validation_loss=score, config=dict(ratio=8, width=48)), output / 'best.pt')
            row = dict(step=step, training_loss=float(loss.detach()), validation_loss=score,
                       selected_step=selected, elapsed_s=time.perf_counter() - started,
                       **{k:float(v) for k, v in terms.items()})
            records.append(row)
            progress(output, 'training', **row)
            save(output / 'training_curve.json', records)
    torch.cuda.synchronize()
    if selected == 0:
        raise RuntimeError('No checkpoint improved on the initial validation loss')
    metadata = dict(steps=steps, selected_step=selected, best_validation_loss=best, initial_validation_loss=initial_validation_loss,
                    training_s=time.perf_counter() - started, parameters=sum(p.numel() for p in model.parameters()),
                    peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                    checkpoint_sha256=sha(output / 'best.pt'))
    save(output / 'training.json', metadata)
    del model, train_data, val_data
    torch.cuda.empty_cache()
    return metadata


def field_error(field, truth, start=None):
    w, ref = primitive(field), primitive(truth)
    relative = lambda a, b: dict(l1=(np.sum(abs(a - b), axis=0) / np.maximum(np.sum(abs(b), axis=0), 1e-12)).tolist(),
                               l2=(np.linalg.norm(a - b, axis=0) / np.maximum(np.linalg.norm(b, axis=0), 1e-12)).tolist())
    result = relative(w, ref)
    if start is not None:
        result['local'] = relative(w[start:start + 128], ref[start:start + 128])
    return result


def shock_location(field, time_value, case):
    _, info = exact_riemann(np.array([.5]), time_value, case['left'], case['right'], case['discontinuity'])
    p = primitive(field)[:, 2]
    x = (np.arange(len(p)) + .5) / len(p)
    if 'right_shock_speed' in info:
        target = .5 * (info['star_pressure'] + case['right'][2])
        ids = np.flatnonzero((p[:-1] >= target) & (p[1:] < target))
        index = int(ids[-1]) if len(ids) else None
        speed = info['right_shock_speed']
    elif 'left_shock_speed' in info:
        target = .5 * (info['star_pressure'] + case['left'][2])
        ids = np.flatnonzero((p[:-1] < target) & (p[1:] >= target))
        index = int(ids[0]) if len(ids) else None
        speed = info['left_shock_speed']
    else:
        return None
    if index is None:
        return None
    position = x[index] + (x[index + 1] - x[index]) * (target - p[index]) / (p[index + 1] - p[index])
    exact = case['discontinuity'] + speed * time_value
    return dict(position=float(position), exact_position=float(exact), exact_speed=float(speed),
                position_error=float(abs(position - exact)))


def metadata(result):
    return {k:v for k, v in result.items() if k not in ('field', 'state', 'snapshots')}


def evaluate(output):
    checkpoint = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
    model = EulerLocalDecoder(**checkpoint['config']).eval()
    model.load_state_dict(checkpoint['model'])
    base = dict(left=[1., 0., 1.], right=[.125, 0., .1], discontinuity=.5)
    cases = [dict(name=f'sod_start_{s}', start=s, **base) for s in (576, 640, 704)]
    left, right, position = sample_conditions(6, SEEDS['test'])
    # Test positions are generated after checkpoint selection, with a prescribed
    # central jump range so the shock can traverse the chosen local region.
    position = .5 + (position - .5) / 3
    for i, (l, r, x) in enumerate(zip(left, right, position)):
        cases.append(dict(name=f'heldout_{i}', start=640 if l[2] > r[2] else 256,
                          left=l.tolist(), right=r.tolist(), discontinuity=float(x)))
    save(output / 'test_cases.json', dict(checkpoint_sha256=sha(output / 'best.pt'), seed=SEEDS['test'], cases=cases))
    records = []
    fields = {}
    for case in cases:
        progress(output, 'evaluation', case=case['name'])
        args = {k:case[k] for k in ('left', 'right', 'discontinuity')}
        reference = EulerFV(n=2048).run(**args)
        truth = reference['field'].reshape(1024, 2, 3).mean(axis=1)
        field_arrays = dict(reference=truth)
        for method in ('full_fv1024', 'coarse_subdomain', 'neural'):
            solver = EulerFV() if method == 'full_fv1024' else EulerFV(start=case['start'], decoder=model.numpy_decoder if method == 'neural' else None)
            result = solver.run(**args)
            errors = field_error(result['field'], truth, case['start'])
            curve = [dict(time=float(t), **shock_location(q, float(t), case)) for t, q in result['snapshots'].items() if shock_location(q, float(t), case) is not None]
            fit = float(np.polyfit([v['time'] for v in curve], [v['position'] for v in curve], 1)[0]) if len(curve) > 1 else None
            row = dict(case=case['name'], method=method, errors=errors, shock=curve, fitted_shock_speed=fit, **metadata(result))
            records.append(row)
            field_arrays[method] = result['field']
            field_arrays[method + '_state'] = result['state']
            field_arrays[method + '_widths'] = solver.widths
            if case['name'] == 'sod_start_640':
                field_arrays.update({method + '_t' + t:q for t, q in result['snapshots'].items()})
            save(output / 'evaluation_progress.json', records)
            progress(output, 'case_method_complete', case=case['name'], method=method, l1=errors['l1'], wall_s=result['wall_s'])
        exact, info = exact_riemann((np.arange(1024) + .5) / 1024, .2, **args)
        field_arrays['exact'] = exact
        field_arrays['exact_star'] = np.array([info['star_pressure'], info['star_velocity']])
        np.savez_compressed(output / (case['name'] + '_fields.npz'), **field_arrays)
        fields[case['name']] = field_arrays
    constant = [.9, .2, .8]
    constant_run = EulerFV(start=640, decoder=model.numpy_decoder).run(left=constant, right=constant, output_times=(.2,))
    constant_error = float(abs(constant_run['field'] - conserved(constant)).max())
    save(output / 'constant_control.json', dict(maximum_absolute_error=constant_error, **metadata(constant_run)))
    # Complete online solve and field output; model loading and training separate.
    timing = {}
    for method in ('full_fv1024', 'neural'):
        timing[method] = []
        for repeat in range(3):
            solver = EulerFV() if method == 'full_fv1024' else EulerFV(start=640, decoder=model.numpy_decoder)
            result = solver.run()
            timing[method].append(result['wall_s'])
    save(output / 'timings.json', dict(samples=timing, backend='single-thread CPU',
                                     scope='Complete SSPRK2 evolution to 0.2 and five full-field outputs; trained model already loaded'))
    neural = [row for row in records if row['method'] == 'neural']
    l1 = np.array([row['errors']['l1'] for row in neural])
    l2 = np.array([row['errors']['l2'] for row in neural])
    conservation = max(abs(v) for row in neural for h in row['history'] for key in ('global_budget_error', 'region_budget_error') for v in h[key])
    means = max(h['reconstruction_mean_error'] for row in neural for h in row['history'])
    final_shock = [row['shock'][-1]['position_error'] for row in neural[:3]]
    summary = dict(case_count=len(cases), checkpoint_sha256=sha(output / 'best.pt'),
                   variable_order=['density', 'velocity', 'pressure'],
                   neural_l1_median=np.median(l1, axis=0).tolist(), neural_l1_p95=np.quantile(l1, .95, axis=0).tolist(),
                   neural_l2_median=np.median(l2, axis=0).tolist(),
                   maximum_conservation_error=conservation, maximum_mean_error=means,
                   maximum_sod_shock_position_error=max(final_shock),
                   online_speed_ratio=float(np.median(timing['full_fv1024']) / np.median(timing['neural'])),
                   full_fv1024_median_s=float(np.median(timing['full_fv1024'])), neural_median_s=float(np.median(timing['neural'])),
                   local_state_compression=8, global_retained_scalars=2736, full_fv1024_scalars=3072,
                   positivity_step_limiter_calls=sum(row['positivity_step_limiter_calls'] for row in neural),
                   decoder_limited_cells=sum(row['decode_limited_cells'] for row in neural),
                   extra_classical_comparison='coarse_subdomain is an additional classical comparison; full_fv1024 is the main baseline')
    save(output / 'evaluation.json', records)
    save(output / 'summary.json', summary)
    plots(output, cases, fields, records)
    return summary


def plots(output, cases, fields, records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = (np.arange(1024) + .5) / 1024
    data = fields['sod_start_640']
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for j, ax in enumerate(axes):
        ax.axvspan(640/1024, 768/1024, color='#dcefdc', alpha=.75, label='Neural subdomain')
        for key, label, color, style in (('exact', 'Exact Riemann', 'black', '-'),
                                          ('full_fv1024', 'Full FV 1024', '#3479ba', '--'),
                                          ('neural', 'Neural hybrid', '#cc4b37', '-')):
            ax.plot(x, primitive(data[key])[:, j], color=color, ls=style, lw=1.3, label=label)
        ax.set(xlabel='x', ylabel=('Density', 'Velocity', 'Pressure')[j], title='Sod at t = 0.2')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(output/'sod_fields.png', dpi=180)
    fig.savefig(output/'sod_fields.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for ax, start in zip(axes, (576, 640, 704)):
        ax.axvspan(start/1024, (start+128)/1024, color='#dcefdc', alpha=.75)
        for method, color in (('full_fv1024','#3479ba'),('neural','#cc4b37')):
            row = next(r for r in records if r['case']==f'sod_start_{start}' and r['method']==method)
            ax.plot([v['position'] for v in row['shock']], [v['time'] for v in row['shock']], 'o-', color=color, ms=3, label=method)
        ax.plot([.5,.5+1.752155732*.2],[0,.2],'k--',label='Exact shock')
        ax.set(xlabel='Shock position',ylabel='Time',title=f'Local region [{start/1024:.4f}, {(start+128)/1024:.4f}]')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(output/'shock_transit.png',dpi=180)
    fig.savefig(output/'shock_transit.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    neural=[r for r in records if r['method']=='neural']
    for j,label in enumerate(('Density','Velocity','Pressure')):
        axes[0].plot(np.arange(len(neural)),[100*r['errors']['l1'][j] for r in neural],'o-',label=label)
    axes[0].set(xlabel='Held-out coupled case',ylabel='Relative L1 error (%)',title='Against full FV 2048')
    axes[0].legend(fontsize=8)
    row=next(r for r in neural if r['case']=='sod_start_640')
    for j,label in enumerate(('Mass','Momentum','Energy')):
        axes[1].semilogy([h['time'] for h in row['history']],[max(abs(h['global_budget_error'][j]),1e-17) for h in row['history']],'o-',label=label)
    axes[1].set(xlabel='Time',ylabel='Absolute budget defect',title='External flux accounted for')
    axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(output/'errors_and_budgets.png',dpi=180)
    fig.savefig(output/'errors_and_budgets.pdf')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description='Train and evaluate the Euler local decoder.')
    parser.add_argument('--output',type=Path,required=True,help='New output directory, e.g. runs/euler_sod')
    parser.add_argument('--precheck-only',action='store_true',help='Run only the numerical consistency checks')
    args=parser.parse_args()
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    sources=[Path(__file__).resolve(),ROOT/'src/nofvm/nofvm_euler_1d.py',ROOT/'src/nofvm/nofvm_euler_operator.py',ROOT/'src/nofvm/nofvm_euler_operator_v2.py']
    try:
        with ResourceMonitor(output):
            progress(output,'numerical_precheck')
            checked=precheck()
            save(output/'precheck.json',checked)
            if not checked['all_passed']:raise RuntimeError('Numerical precheck failed')
            if args.precheck_only:return
            snapshot=output/'source_snapshot';snapshot.mkdir()
            for p in sources:shutil.copy2(p,snapshot/p.name)
            save(output/'manifest.json',dict(started=time.time(),pid=os.getpid(),sources={p.relative_to(ROOT).as_posix():sha(p) for p in sources},
                 seeds=SEEDS,affinity=sorted(os.sched_getaffinity(0)),device=torch.cuda.get_device_name(),initial_conditions_train=256,
                 initial_conditions_validation=32,optimization_steps=6000))
            training,train_meta=labels('train',256,output)
            validation,val_meta=labels('validation',32,output)
            train(training,validation,output)
            progress(output,'checkpoint_selected',checkpoint_sha256=sha(output/'best.pt'))
            summary=evaluate(output)
            progress(output,'complete',**summary)
        save(output/'artifact_hashes.json',{p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name!='artifact_hashes.json'})
    except BaseException:
        (output/'failure.txt').write_text(traceback.format_exc())
        raise


if __name__=='__main__':main()
