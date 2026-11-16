#!/usr/bin/env python3
"""Train and evaluate the one-dimensional NOFVM local operators for B1 and B2.

For each problem one local operator is trained with a fixed budget, coupled
with finite-volume subdomains in several layouts, and compared with the
continuum reference over 101 boundary loads.

Usage:
    python scripts/run_nofvm_multiscale_1d.py --output runs/multiscale_1d
    python scripts/run_nofvm_multiscale_1d.py --output runs/multiscale_1d --evaluate-only
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nofvm.nofvm_multiscale_1d import (
    PROBLEMS, AffineLocalOperator, assemble_hybrid, combine_basis,
    continuum_basis, fv_basis, predict_basis, training_labels,
)

MODULE = ROOT / "src/nofvm/nofvm_multiscale_1d.py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def guard(output):
    cpu = []
    for directory in Path("/sys/class/hwmon").glob("hwmon*"):
        namepath = directory / "name"
        if namepath.exists() and namepath.read_text().strip() in ("k10temp", "coretemp"):
            for path in directory.glob("temp*_input"):
                cpu.append(float(path.read_text().strip())/1000)
    gpu = float(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
        text=True, timeout=15).strip().splitlines()[0])
    free = shutil.disk_usage(output).free/2**30
    record = dict(cpu_max_c=max(cpu) if cpu else None, gpu_c=gpu, free_gib=free)
    if gpu >= 90 or any(t >= 90 for t in cpu) or free < 4:
        raise RuntimeError(f"Resource guard stopped this experiment: {record}")
    return record


def rel(a, b):
    return np.linalg.norm(a-b, axis=-1)/np.maximum(np.linalg.norm(b, axis=-1), 1e-14)


def stats(values):
    return dict(median=float(np.median(values)), p95=float(np.quantile(values, .95)),
                maximum=float(np.max(values)))


def numerical_preflight(problem, output, device):
    torch.manual_seed(13)
    model = AffineLocalOperator(problem).to(device=device, dtype=torch.float64)
    # Exercise nonzero network derivatives; zero final weights alone is too easy.
    with torch.no_grad():
        model.trunk[-1].weight.normal_(0, .03)
        model.trunk[-1].bias.normal_(0, .01)
    s = torch.linspace(.013, .987, 19, device=device, dtype=torch.float64)[:, None]
    s.requires_grad_(True)
    values, derivatives = model(s)
    discrepancy = []
    for i in range(2):
        ad = torch.autograd.grad(values[:, i].sum(), s, retain_graph=True)[0][:, 0]
        discrepancy.append(float((ad-derivatives[:, i]).abs().max().detach().cpu()))
    # Check parameter gradients too: derivative supervision passes through trunk.
    loss = values.square().mean()+derivatives.square().mean()
    grads = torch.autograd.grad(loss, tuple(model.parameters()))
    grads_finite = all(bool(torch.isfinite(g).all()) for g in grads)
    endpoints = predict_basis(model, np.array([0., problem.length]))
    endpoint_flux_error = max(
        abs(-problem.conductivity(0.)*endpoints["dh"][0]+endpoints["C"]),
        abs(problem.velocity-problem.conductivity(problem.length)*endpoints["dh"][1]+endpoints["C"]),
        abs(-problem.conductivity(0.)*endpoints["dz"][0]-endpoints["c"]),
        abs(-problem.conductivity(problem.length)*endpoints["dz"][1]-endpoints["c"]-problem.source_integral))
    n = 256
    x = (np.arange(n)+.5)*problem.length/n
    exact = continuum_basis(problem, x)
    split = assemble_hybrid(problem, n, [False]*problem.blocks, exact)
    monolithic = fv_basis(problem, n*problem.blocks, problem.total_length)
    split_error = float(np.linalg.norm(split["u"]-monolithic["u"])/np.linalg.norm(monolithic["u"]))
    analytical = assemble_hybrid(problem, n, [True]*problem.blocks, exact)
    reference = continuum_basis(problem, analytical["x"], problem.total_length)
    reference16 = continuum_basis(problem, analytical["x"], problem.total_length, order=16)
    exact_global_error = max(float(np.max(np.abs(analytical["u"][:, 1]-reference["h"]))),
                             float(np.max(np.abs(analytical["u"][:, 2]-reference["z"]))))
    quadrature_error = max(float(np.max(np.abs(reference[key]-reference16[key])))
                           for key in ("h", "z", "dh", "dz", "C", "c"))
    convergence = []
    for nlocal in (128, 256, 512):
        fv = fv_basis(problem, nlocal*problem.blocks, problem.total_length)
        ref = continuum_basis(problem, fv["x"], problem.total_length)
        field, derivative = combine_basis(ref, [[0., 0., 1.]])
        flux = problem.velocity*field-problem.conductivity(fv["x"])*derivative
        convergence.append(dict(cells_per_block=nlocal,
            field_relative=float(rel(fv["u"][:, 2], field[0])),
            gradient_relative=float(rel(fv["du"][:, 2], derivative[0])),
            flux_relative=float(rel(fv["jcenter"][:, 2], flux[0]))))
    record = dict(coordinate_derivative_max=max(discrepancy), parameter_gradients_finite=grads_finite,
                  endpoint_flux_absolute=float(endpoint_flux_error),
                  partition_vs_monolithic_relative=split_error,
                  exact_response_global_absolute=exact_global_error,
                  quadrature_8_vs_16_absolute=quadrature_error,
                  ordinary_region_balance=split["region_balance"],
                  ordinary_interface_balance=split["interface_balance"], convergence=convergence)
    record["passed"] = bool(max(discrepancy) < 1e-10 and grads_finite and endpoint_flux_error < 1e-10
        and split_error < 1e-8 and exact_global_error < 1e-8 and quadrature_error < 1e-8
        and split["region_balance"] < 1e-8 and split["interface_balance"] < 1e-8
        and convergence[-1]["field_relative"] < convergence[0]["field_relative"]
        # B1's constant source gives an exactly linear flux: its gradient
        # error is already roundoff and need not decrease under refinement.
        and (convergence[-1]["gradient_relative"] < convergence[0]["gradient_relative"]
             or max(r["gradient_relative"] for r in convergence) < 1e-8))
    save_json(output / f"{problem.name}_preflight.json", record)
    if not record["passed"]:
        raise RuntimeError(f"Numerical preflight failed: {record}")
    return record


def tensors(problem, labels, device):
    scale = np.array([1., problem.field_scale])
    values = np.column_stack((labels["h"], labels["z"]))/scale
    deriv = np.column_stack((labels["dh"], labels["dz"]))*problem.length/scale
    convert = lambda v: torch.as_tensor(v, device=device, dtype=torch.float64)
    return (convert(labels["x"][:, None]/problem.length), convert(values), convert(deriv),
            convert(scale), convert([labels["C"]/problem.conductance_scale,
                                      labels["c"]/problem.source_integral]))


def objective(model, data, response=True):
    x, y, dy, scale, target = data
    pred, dpred = model(x)
    field = ((pred/scale-y)**2).mean()
    gradient = ((dpred/scale-dy)**2).mean()
    C, c = model.responses()
    normalized = torch.stack((C/model.problem.conductance_scale, c/model.problem.source_integral))
    responses = ((normalized-target)**2).mean()
    loss = field+.1*gradient+(.1*responses if response else 0.)
    return loss, (field, gradient, responses)


def train(problem, output, device):
    resource = [guard(output)]
    numerical_preflight(problem, output, device)
    start = time.perf_counter()
    labels = training_labels(problem)
    labels_path = output / f"{problem.name}_labels.npz"
    np.savez_compressed(labels_path, **labels)
    preparation = time.perf_counter()-start
    # Validation coordinates differ from every cell-center training coordinate.
    xv = np.linspace(0, problem.length, 514)
    validation = dict(x=xv, **continuum_basis(problem, xv))
    data, valid = tensors(problem, labels, device), tensors(problem, validation, device)
    torch.manual_seed(13)
    model = AffineLocalOperator(problem).to(device=device, dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4000, eta_min=1e-5)
    history, best, selected = [], float("inf"), None
    checkpoint = output / f"{problem.name}_best.pt"
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for step in range(1, 4001):
        optimizer.zero_grad(set_to_none=True)
        loss, parts = objective(model, data)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Nonfinite training objective at {problem.name} {step}")
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % 200 == 0:
            with torch.no_grad():
                vloss, vparts = objective(model, valid, response=False)
            value = float(vloss.cpu())
            if value < best:
                best, selected = value, step
                torch.save(model.state_dict(), checkpoint)
            resource.append(guard(output))
            row = dict(step=step, training_loss=float(loss.detach().cpu()),
                       validation_loss=value, validation_field=float(vparts[0].cpu()),
                       validation_gradient=float(vparts[1].cpu()), elapsed_s=time.perf_counter()-start)
            history.append(row)
            print(problem.name, json.dumps(row), flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    with (output / f"{problem.name}_history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)
    metadata = dict(selected_step=selected, updates=4000, seed=13,
        selection="normalized independent spatial field MSE + 0.1 coordinate derivative MSE",
        unique_local_rhs=3, labels_sha256=sha(labels_path), checkpoint_sha256=sha(checkpoint),
        parameters=sum(p.numel() for p in model.parameters()), train_points=256, validation_points=514,
        label_generation_s=preparation, training_and_selection_s=elapsed,
        peak_allocated_gpu_gib=torch.cuda.max_memory_allocated()/2**30, resources=resource)
    save_json(output / f"{problem.name}_training.json", metadata)
    return model


def layouts(problem):
    if problem.name == "B1":
        return [("all_neural", [True]*8), ("alternating_4", [b % 2 == 0 for b in range(8)]),
                ("all_fv", [False]*8)]
    rng = np.random.default_rng(202610033)
    result = [("all_neural", [True]*10)]
    for count in (2, 4, 8):
        for repeat in range(2):
            mask = np.zeros(10, dtype=bool)
            mask[rng.choice(10, count, replace=False)] = True
            result.append((f"mixed_{count}_{repeat+1}", mask.tolist()))
    return result+[("all_fv", [False]*10)]


def residual_diagnostics(problem, model):
    # Interior strong residual is independent of exact endpoint/region balance.
    parameter = next(model.parameters())
    s = torch.linspace(.0003, .9997, 2049, dtype=parameter.dtype,
                       device=parameter.device)[:, None].requires_grad_(True)
    u, dus = model(s)
    x = s[:, 0]*problem.length
    k = .8+.5*torch.sin(32*np.pi*x) if problem.name == "B1" else torch.ones_like(x)
    j = problem.velocity*u-k[:, None]*dus/problem.length
    residual = []
    source = torch.full_like(x, .5) if problem.name == "B1" else x*(1-x)
    for i in range(2):
        dj = torch.autograd.grad(j[:, i].sum(), s, retain_graph=True)[0][:, 0]/problem.length
        expected = torch.zeros_like(source) if i == 0 else source
        residual.append(float(torch.sqrt(((dj-expected)**2).mean()).detach().cpu()))
    return dict(homogeneous_rms=residual[0], unit_source_rms=residual[1],
                definition="RMS(dj/dx-source) on 2049 interior points; dimensional absolute residual",
                scope="Diagnostic only; hard constraints ensure region balance, not this residual")


def evaluate(problem, model, output):
    n = 256
    start = time.perf_counter()
    xlocal = (np.arange(n)+.5)*problem.length/n
    learned = predict_basis(model, xlocal)
    # Test loads are generated after checkpoint selection.
    seed = 202610031 if problem.name == "B1" else 202610032
    rng = np.random.default_rng(seed)
    loads = np.ones((101, 3)); loads[0, :2] = 0.; loads[1:, :2] = rng.uniform(-.1, .1, (100, 2))
    loadpath = output / f"{problem.name}_test_loads.npz"
    np.savez_compressed(loadpath, loads=loads)
    xglobal = (np.arange(n*problem.blocks)+.5)*problem.length/n
    ref = continuum_basis(problem, xglobal, problem.total_length)
    truth, dtruth = combine_basis(ref, loads)
    qtruth = problem.velocity*truth-problem.conductivity(xglobal)*dtruth
    gleft = problem.velocity*loads[:, 0]-ref["C"]*(loads[:, 1]-loads[:, 0])+loads[:, 2]*ref["c"]
    interface_x = np.linspace(0, problem.total_length, problem.blocks+1)
    interface_truth = gleft[:, None]+loads[:, 2, None]*problem.primitive(interface_x)
    results, records, saved = [], [], {}
    for name, mask in layouts(problem):
        t0 = time.perf_counter()
        coupled = assemble_hybrid(problem, n, mask, learned)
        elapsed = time.perf_counter()-t0
        u, du, q = (loads@coupled[key].T for key in ("u", "du", "j"))
        interface = np.vstack((coupled["ends"][:, 0], coupled["ends"][-1, 1]))
        predicted_interface = loads@interface.T
        eu, ed, eq, ei = rel(u, truth), rel(du, dtruth), rel(q, qtruth), rel(predicted_interface, interface_truth)
        # Actual-load balances as well as basis balances, with no normalizer hiding zero loads.
        target = np.array([0., 0., problem.source_integral])
        reg = float(np.max(np.abs(loads@(coupled["ends"][:, 1]-coupled["ends"][:, 0]-target).T)))
        iface = float(np.max(np.abs(loads@(coupled["ends"][:-1, 1]-coupled["ends"][1:, 0]).T)))
        constants = np.array([[0., 0., 0.], [1., 1., 0.]])
        cu = constants@coupled["u"].T
        cq = constants@coupled["j"].T
        constant_field = float(np.max(np.abs(cu-constants[:, :1])))
        constant_flux = float(np.max(np.abs(cq-problem.velocity*constants[:, :1])))
        metrics = dict(field=stats(eu), gradient=stats(ed), total_flux=stats(eq), interface_flux=stats(ei))
        record = dict(layout=name, neural_regions=np.flatnonzero(mask).tolist(), loads=101,
                      unknowns=coupled["unknowns"], ordinary_cells=coupled["ordinary_cells"],
                      interface_unknowns=coupled["interface_unknowns"],
                      reference_full_fv_unknowns=n*problem.blocks, metrics=metrics,
                      region_balance_absolute=reg, interface_balance_absolute=iface,
                      constant_field_absolute=constant_field, constant_flux_absolute=constant_flux,
                      linear_residual=coupled["linear_residual"], assembly_three_rhs_s=elapsed,
                      original_zero_bc={"field": float(eu[0]), "gradient": float(ed[0]), "total_flux": float(eq[0])},
                      original_zero_bc_min_field=float(u[0].min()))
        results.append(record)
        for i in range(len(loads)):
            records.append(dict(layout=name, load=i, gleft=loads[i, 0], gright=loads[i, 1],
                                field=eu[i], gradient=ed[i], total_flux=eq[i], interface_flux=ei[i]))
        saved[name+"_u_basis"] = coupled["u"]
        saved[name+"_du_basis"] = coupled["du"]
        saved[name+"_j_basis"] = coupled["j"]
        saved[name+"_trace_basis"] = coupled["traces"]
        saved[name+"_endflux_basis"] = coupled["ends"]
        saved[name+"_mask"] = mask
        print(problem.name, name, json.dumps(record), flush=True)
    saved.update(x=xglobal, loads=loads, reference_u=truth, reference_du=dtruth, reference_j=qtruth)
    np.savez_compressed(output / f"{problem.name}_global_fields.npz", **saved)
    with (output / f"{problem.name}_load_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys()); writer.writeheader(); writer.writerows(records)
    # Fine FV balance computed from reconstructed point values is *not* identically
    # imposed; report strong residual too, to avoid conflating region and cell balance.
    interior_residual = residual_diagnostics(problem, model)
    local_ref = continuum_basis(problem, xlocal)
    local_metrics = {key: float(rel(learned[key], local_ref[key])) for key in ("h", "z", "dh", "dz")}
    local_metrics.update(C_relative=abs(learned["C"]/local_ref["C"]-1),
                         c_relative=abs(learned["c"]/local_ref["c"]-1))
    report = dict(problem=asdict(problem), layouts=results,
                  local_basis_errors=local_metrics, interior_strong_residual=interior_residual,
                  test_seed=seed, test_loads_sha256=sha(loadpath), distinct_test_boundary_pairs=101,
                  source_variations=1, unseen_materials=0, total_evaluation_s=time.perf_counter()-start,
                  timing_scope="Observed cost only, not a controlled speedup comparison")
    save_json(output / f"{problem.name}_evaluation.json", report)
    make_figure(problem, saved, results, output)
    return report


def make_figure(problem, data, results, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), constrained_layout=True)
    x = data["x"]
    for ax, key, label in zip(axes.ravel()[:3], ("u", "du", "j"),
                             ("Field u", "Gradient du/dx", "Total flux j")):
        ax.plot(x, data["reference_"+key][0], "k-", linewidth=1.4, label="Continuum reference")
        ax.plot(x, data["all_neural_"+key+"_basis"][:, 2], "--", linewidth=1.1, label="All neural subdomains")
        mixed = results[1]["layout"]
        ax.plot(x, data[mixed+"_"+key+"_basis"][:, 2], ":", linewidth=1.1, label=mixed)
        ax.set(xlabel="x", ylabel=label)
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    last = axes[1, 1]
    positions = np.arange(len(results))
    last.bar(positions-.16, [100*r["metrics"]["field"]["p95"] for r in results], .32, label="Field P95")
    last.bar(positions+.16, [100*r["metrics"]["total_flux"]["p95"] for r in results], .32, label="Flux P95")
    last.set_xticks(positions, [r["layout"] for r in results], rotation=35, ha="right", fontsize=8)
    last.set_ylabel("Relative L2 error (%) over 101 loads")
    last.legend(fontsize=8); last.grid(axis="y", alpha=.2)
    description = "Oscillatory conductivity" if problem.name == "B1" else "Periodic-source steady advection-diffusion"
    fig.suptitle(f"{problem.name}: {description}; fixed reusable local operator")
    fig.savefig(output / f"{problem.name}_fields_and_errors.png", dpi=170)
    fig.savefig(output / f"{problem.name}_fields_and_errors.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the B1/B2 local operators.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory, e.g. runs/multiscale_1d (must not exist unless --evaluate-only)")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Evaluate the checkpoints already saved in --output without training")
    args = parser.parse_args()
    output = args.output.resolve()
    if not args.evaluate_only:
        output.mkdir(parents=True, exist_ok=False)
    elif not (output / "manifest.json").exists():
        raise ValueError("--evaluate-only requires an existing manifest.json in --output")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    device = "cuda"
    try:
        if not args.evaluate_only:
            save_json(output / "manifest.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
                module_sha256=sha(MODULE), runner_sha256=sha(__file__),
                module=MODULE.relative_to(ROOT).as_posix(),
                runner=Path(__file__).resolve().relative_to(ROOT).as_posix(),
                python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                gpu=torch.cuda.get_device_name(), dtype="float64", steps_per_model=4000,
                pre_run_resources=guard(output), problems={k:asdict(v) for k,v in PROBLEMS.items()},
                note="Two fixed local problems; the operator maps finite-dimensional boundary data, materials are not varied"))
        summary = {}
        for p in PROBLEMS.values():
            if args.evaluate_only:
                model = AffineLocalOperator(p).to(device=device, dtype=torch.float64)
                model.load_state_dict(torch.load(output / f"{p.name}_best.pt", map_location=device, weights_only=True))
            else:
                model = train(p, output, device)
            summary[p.name] = evaluate(p, model, output)
        save_json(output / "summary.json", dict(completed_utc=datetime.now(timezone.utc).isoformat(),
            problems=summary,
            evaluation_sources=dict(module=sha(MODULE), runner=sha(__file__)), resources=guard(output)))
        save_json(output / "artifact_hashes.json", {p.name: sha(p) for p in sorted(output.iterdir())
                  if p.is_file() and p.name != "artifact_hashes.json"})
        print("COMPLETE", output, flush=True)
    except BaseException:
        (output / "failure.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
