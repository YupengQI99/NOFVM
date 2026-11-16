# NOFVM: Neural-operator finite volume method

This repository contains the code and result summaries for the paper

> **Neural-operator finite volume method: Conservative and efficient simulation with reusable local neural operators**

NOFVM couples pretrained neural operators with a conventional finite-volume solver. A neural-operator control volume (NOCV) replaces a subdomain that would otherwise need a fine mesh. It reconstructs the field inside the subdomain and exchanges fluxes with the surrounding cells through shared interfaces, so the coupled scheme is conservative for any number of NOCVs. A trained NOCV can be reused in new configurations without retraining.

## Repository layout

| Path | Contents |
|---|---|
| `src/nofvm/` | Finite-volume discretizations, local responses, neural operators and the NOCV coupling |
| `scripts/` | Scripts that train, evaluate and time each experiment |
| `tests/` | Tests for the polygonal mesh and the coupled polygon solver |
| `results/` | Metrics, training histories and timings of the paper (JSON and CSV) |

## Experiments

| Case | Problem | Scripts | Results |
|---|---|---|---|
| B1, B2 | One-dimensional multiscale diffusion (B1) and advection–diffusion (B2) | `run_nofvm_multiscale_1d.py` | `results/multiscale_1d` |
| C1, C2 | Heat conduction with one hole (C1) and with 1 to 100 repeated holes (C2) on polygonal meshes | `generate_nofvm_polygon_mesh.py`, `train_nofvm_polygon_hole.py`, `evaluate_nofvm_polygon_hole.py`, `finish_nofvm_polygon_hole.py`, `time_polygon_reuse_rounds.py` | `results/polygon_mesh`, `results/polygon_hole`, `results/polygon_timing` |
| D1, D2 | Darcy flow with continuous (D1) and two-valued (D2) random permeability | `run_nofvm_darcy_linear.py`, `evaluate_nofvm_darcy_linear.py` | `results/darcy_continuous`, `results/darcy_jump` |
| D3 | Darcy flow with elliptical high-permeability inclusions | `run_nofvm_darcy_ellipse.py`, `evaluate_nofvm_darcy_ellipse.py` | `results/darcy_ellipse` |
| D4 | Darcy flow with pressure-dependent permeability | `run_nofvm_darcy_nonlinear.py`, `evaluate_nofvm_darcy_nonlinear.py` | `results/darcy_nonlinear` |
| Euler | Sod shock tube and six further Riemann problems | `run_nofvm_euler_sod_v2.py` | `results/euler_sod` |

## Installation

```bash
git clone https://github.com/YupengQI99/NOFVM.git
cd NOFVM
python -m pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version (see [pytorch.org](https://pytorch.org)). The scripts add `src/` to the Python path themselves, so no package installation is needed.

The training scripts require a CUDA GPU and `nvidia-smi`. While running, they monitor temperature, memory and free disk space and stop when a limit is reached (see `src/nofvm/runtime.py` and `scripts/run_nofvm_darcy_ellipse.py`). Set `NOFVM_OUTPUT_VOLUME` to monitor a disk other than the one holding the repository.

The results in `results/` were obtained with an NVIDIA RTX 4090 GPU and an AMD Ryzen 9 7950X3D CPU, using Python 3.11, PyTorch 2.7, NumPy 2.3 and SciPy 1.16.

## Data

Trained weights, meshes and field arrays (`.pt`, `.npz`, `.msh`, `.vtk`) are too large for git. Download `NOFVM_data.tar.gz` from the [Releases](https://github.com/YupengQI99/NOFVM/releases) page and unpack it in the repository root:

```bash
tar -xzf NOFVM_data.tar.gz
```

The files are placed under `results/`, next to the JSON summaries. The polygon scripts read the meshes from `results/polygon_mesh`, and the timing script reads the trained hole operator from `results/polygon_hole/training`.

## Running the experiments

Each script writes to a new output directory and stops if that directory already exists. The examples below write to `runs/`, which is ignored by git.

**B1, B2**

```bash
python scripts/run_nofvm_multiscale_1d.py --output runs/multiscale_1d
```

**D1–D3**

```bash
python scripts/run_nofvm_darcy_linear.py --family continuous    # D1, writes runs/darcy_continuous/training
python scripts/run_nofvm_darcy_linear.py --family jump          # D2, writes runs/darcy_jump/training
python scripts/run_nofvm_darcy_ellipse.py --family ellipse      # D3, writes runs/darcy_ellipse/training

python scripts/evaluate_nofvm_darcy_linear.py --training runs/darcy_continuous/training
python scripts/evaluate_nofvm_darcy_linear.py --training runs/darcy_jump/training
python scripts/evaluate_nofvm_darcy_ellipse.py --training runs/darcy_ellipse/training
```

The evaluation scripts check the files of the training directory against the hashes recorded during training, so run them from the same checkout.

**D4**

```bash
python scripts/run_nofvm_darcy_nonlinear.py --output runs/darcy_nonlinear/training
python scripts/evaluate_nofvm_darcy_nonlinear.py --training runs/darcy_nonlinear/training \
    --output runs/darcy_nonlinear/evaluation
```

**Euler**

```bash
python scripts/run_nofvm_euler_sod_v2.py --output runs/euler_sod
```

**C1, C2**

The meshes are included in the data archive. To regenerate them (requires Gmsh and Shapely):

```bash
python scripts/generate_nofvm_polygon_mesh.py --output runs/polygon_mesh
```

Training, evaluation and figures:

```bash
python scripts/train_nofvm_polygon_hole.py --output runs/polygon_training
python scripts/evaluate_nofvm_polygon_hole.py --training runs/polygon_training --output runs/polygon_reuse
python scripts/evaluate_nofvm_polygon_hole.py --training runs/polygon_training --output runs/polygon_single --single-only
python scripts/finish_nofvm_polygon_hole.py --training runs/polygon_training \
    --evaluation runs/polygon_reuse --single runs/polygon_single --output runs/polygon_hole
```

Timing of the repeated-hole layouts (ten interleaved rounds, one CPU core and one BLAS thread):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 19 \
    python scripts/time_polygon_reuse_rounds.py --output runs/polygon_timing --count-factor-up-to 10
```

With `--count-factor-up-to 10`, the direct fine-grid solve of the 100-hole layout needs about 15 GiB of memory.

## Tests

```bash
python -m pytest tests
```

The tests require Gmsh and Shapely and are skipped when these are not installed.

## License

This project is released under the [MIT License](LICENSE).
