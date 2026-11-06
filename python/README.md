# makutu — Python seismic modelling & imaging on FUnTiDES

`makutu` is the Python part of FUnTiDES-App: a library for seismic forward
modelling, imaging (RTM) and full-waveform inversion, and the applications built
on it. It drives the [FUnTiDES](https://github.com/FUnTiDES-sim/FUnTiDES)
spectral-element solvers through their `pyfuntides` Python bindings (zero-copy
Kokkos view injection) and adds the acquisition, mesh/model, I/O and workflow
layers on top.

It began as a [GEOS-DEV/makutu](https://github.com/GEOS-DEV/makutu) `pygeosx`
interface, then lived in the `makutu-lib` repository (proxy-FUn, then FUnTiDES)
before moving here; its history is kept.

## Layout

| Path | Contents |
|---|---|
| `makutu/` | the library (`import makutu…`): `mesh`, `model`, `solvers` (acoustic/elastic SEM), `input`, `acquisition`, `output`, `imaging`, `forwardBackward`, `display`, `tools` |
| `apps/` | applications: `modeling/examples`, `fullWaveformInversion`, `display`, `modelProcessing`, and `run_funtides.py` (YAML launcher of the C++ drivers of `../examples`) |
| `tests/` | `unitTests` (acquisition), `integratedTests` (forward shots on FUnTiDES: arrival times, elastic, MPI, 2-D against the exact solution) |
| `docs/` | `funtides_input_parameters.md` + Sphinx source |
| `scripts/`, `validation/` | developer helper scripts, DAS validation scripts |

The applications import the library, never the reverse. Example data and
models (SEAM acquisition, Marmousi SEP/Cubit models, …) live in the sibling
`FUnTiDES-data/` directory, next to FUnTiDES-App, out of the source tree.

## Requirements

- **FUnTiDES** installed with its Python bindings (`-DENABLE_PYWRAP=ON`): it
  provides `pyfuntides` and, through FUnTiDES-TPL, the `kokkos` module and the
  Python virtual environment (`tpl-venv`). These are **not** pip packages; see
  the FUnTiDES README (Quick Start).
- Python >= 3.10 with `numpy`, `scipy`, `matplotlib`, `mpi4py`, `h5py`, `segyio`,
  `vtk`, `pyevtk`, `numba` (installed by `pip install` below) and `pytest` for
  the tests.

> **GPU.** makutu allocates its Kokkos arrays in the memory space and layout of
> the FUnTiDES build (`CudaUVMSpace` + `LayoutLeft` with CUDA, see
> `makutu/tools/kokkosarrays.py`); it is tested with the CUDA build. The
> polynomial orders available to the solvers are those compiled in FUnTiDES
> (`MAX_SOLVER_ORDER`). Pin the GPU with `CUDA_VISIBLE_DEVICES` on multi-GPU
> machines (the applications default it to 0).

## Installation

```bash
source <FUnTiDES install>/funtides_env.sh   # kokkos, pyfuntides, tpl-venv
cd FUnTiDES-App/python
pip install -e ".[test]"                     # once: makutu in editable mode
```

## Tests

From `FUnTiDES-App/python/`, with the environment above:

```bash
make unitTests          # acquisition, pure Python
make integratedTests    # forward shots on FUnTiDES (about 2 min on a GPU; MPI tests launch mpirun)
```

## Quick start

Run an acoustic one-shot forward model from a SEP velocity file:

```bash
python apps/modeling/examples/acoustic_one_shot_unstruct_sep.py \
    --vp-file <model.H> --f0 10 --order 2
```

The application builds a heterogeneous Cartesian mesh, runs the SEM acoustic
solver through `pyfuntides`, and writes the seismogram to HDF5. See
`apps/modeling/examples/` for more (multi-shot acquisition, `model_from_sep.py`, …).

The Python API in a nutshell:

```python
import kokkos; kokkos.initialize()
import pyfuntides.solver as FTSolver
from makutu.mesh.InternalMesh import StructuredMesh
from makutu.solvers.AcousticSolver import AcousticSolver

mesh = StructuredMesh(lx, ly, lz, ex, ey, ez, order=2, vp_file="model.H")
solver = AcousticSolver(
    model=mesh.model, order=2,
    implem_type=FTSolver.ImplemType.MAKUTU,
    mesh_type=FTSolver.MeshType.UNSTRUCT,
    dt=dt, maxTime=tmax, sourceType="ricker2", sourceFreq=f0,
    kk_model_vp=mesh.kk_model_vp, kk_model_rho=mesh.kk_model_rho)
solver.initialize()
solver.updateSourceAndReceiversFromCoords(src_coords, rcv_coords)
solver.evaluateSource()
for t in range(solver.n_time_steps):
    solver.execute(t)
seismogram = solver.getPressureAtReceivers()   # (n_samples, n_receivers)
```

**2-D.** With a FUnTiDES built with `COMPILE_SEM_2D`, `makutu.mesh.CartesianMesh2D`
builds a 2-D (x, z) quadrilateral mesh (z up, free surface on top), homogeneous,
from arrays or from 2-D SEP files; the solvers detect the dimension from the model
(acoustic only, single process). Coordinates stay `[x, y, z]` lists, y being ignored:

```python
from makutu.mesh import CartesianMesh2D
mesh = CartesianMesh2D(lx, lz, ex, ez, order=4, vp_file="Vp_xz.H")
solver = AcousticSolver(implem_type=FTSolver.ImplemType.MAKUTU, mesh_type=FTSolver.MeshType.UNSTRUCT,
                        dt=dt, maxTime=tmax, sourceType="ricker2", sourceFreq=f0, **mesh.solver_kwargs())
```

See `apps/modeling/marmousi_2d/` (README: data preparation, mesh choice, modelling).

MPI is supported in 3-D (1-D X-axis domain decomposition) via `mpi4py`; pass a
communicator to the mesh/solver constructors. A partitioned `StructuredMesh`
uses the `ModelUnstruct` backend (`mesh.heterogeneous` is True), so pass
`MeshType.UNSTRUCT` to the solver.

## Documentation

- [`docs/funtides_input_parameters.md`](docs/funtides_input_parameters.md) —
  FUnTiDES solver input parameters.
- Sphinx source in `docs/source/` — build with `make docs` (needs `sphinx`);
  output lands in `docs/build/`.
