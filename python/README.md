# makutu — Python seismic modelling & imaging on FUnTiDES

`makutu` is the Python part of FUnTiDES-App: a library for seismic forward
modelling, adjoint-state gradients and acoustic full-waveform inversion (FWI), and the
applications built on it. It drives the [FUnTiDES](https://github.com/FUnTiDES-sim/FUnTiDES)
spectral-element solvers through their `pyfuntides` Python bindings (zero-copy
Kokkos view injection) and adds the acquisition, mesh/model, I/O and workflow
layers on top.

It began as a [GEOS-DEV/makutu](https://github.com/GEOS-DEV/makutu) `pygeosx`
interface, then lived in the `makutu-lib` repository (proxy-FUn, then FUnTiDES)
before moving here; its history is kept.

## Layout

| Path | Contents |
|---|---|
| `makutu/` | the library (`import makutu…`): `mesh` (including `seismesh`), `mortar` (2-D blocks glued by SIPG, hybrid SEM + DG triangles; see its README), `solvers` (acoustic/elastic SEM), `input`, `acquisition`, `output`, `imaging`, `forwardBackward`, `display`, `tools` |
| `apps/` | applications and their examples and documents: `modeling` (examples and tools), `mesh`, `mortar` (comparisons, meshes and costs, FUnTiDES shots, reference document, `reproduce.py`), `fullWaveformInversion`, `display`, and `run_funtides.py` (YAML launcher of the C++ drivers of `../examples`) |
| `tests/` | `unitTests` (acquisition, cost, filter, optimisers, conditioning, seismesh, mortar and hybrid prototypes; no FUnTiDES solver), `integratedTests` (FUnTiDES on the GPU: shots, 2-D against the exact solution, mortar blocks, elastic, gradient, adjoint, FWI functions, MPI) |
| `docs/` | Sphinx documentation (user guide, examples, API, status) and `funtides_input_parameters.md` |
| `validation/` | DAS validation: analytical solution of an elastic point source (displacement and DAS signal) |

The applications import the library, never the reverse. Example data and
models (SEAM acquisition, Marmousi SEP/Cubit models, …) live in the sibling
`FUnTiDES-data/` directory, next to FUnTiDES-App, out of the source tree.

## Requirements

- **FUnTiDES** installed with its Python bindings (`-DENABLE_PYWRAP=ON`): it
  provides `pyfuntides` and, through FUnTiDES-TPL, the `kokkos` module and the
  Python virtual environment (`tpl-venv`). These are **not** pip packages; see
  the FUnTiDES README (Quick Start).
- Python >= 3.10 with `numpy`, `scipy`, `matplotlib`, `mpi4py`, `h5py`, `segyio`
  (installed by `pip install` below) and `pytest` for the tests.

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
make unitTests          # pure Python, about 2 s
make integratedTests    # FUnTiDES on the GPU, about 3 min (the MPI tests launch mpirun)
```

## Quick start

Model acoustic shots on a SEP velocity file, 2-D or 3-D:

```bash
python apps/modeling/shot_modeling/run_shot_modeling.py \
    --vp-file <model.H> --f0 10 --order 2 [--n-shots 5 | --decision <rootname>] --plot
```

The application builds a heterogeneous Cartesian mesh, runs the SEM acoustic
solver through `pyfuntides`, and writes one gather per shot (SEG-Y or HDF5). See
`apps/modeling/README.md` for the examples (Marmousi 2-D data preparation) and the
data-preparation tools.

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

**Coordinates.** makutu uses the seismic convention: z = 0 at the surface (the free
surface, top of the mesh) and z positive downwards.
- Coordinates are `[x, y, z]`: acquisitions, `updateSourceAndReceiversFromCoords`,
  the solver `coordTo*` helpers.
- Grids have z layer 0 at the surface: the SEP models, `CartesianMesh2D.vp`/`rho`,
  the solver `get`/`updateVelocityModel` and `get`/`updateDensityModel` (flat, x
  fastest), `getPressureField()` and `getDisplacementField()`.
- The elastic z component (source and displacement) is positive downwards.
- In the SEG-Y headers, the source z is `sdepth` (byte 49), and makutu writes the
  receiver z in `gelev` (byte 41).

Inside FUnTiDES the mesh has z up, with the free surface on its face z = lz; makutu
converts in one place (`WaveSolver`, model builders). The Kokkos arrays
`kk_model_*` and the raw solver fields keep the FUnTiDES order.

**2-D.** With a FUnTiDES built with `COMPILE_SEM_2D`, `makutu.mesh.CartesianMesh2D`
builds a 2-D (x, z) quadrilateral mesh (z downwards, free surface on top), homogeneous,
from arrays or from 2-D SEP files; the solvers detect the dimension from the model
(acoustic only; no domain decomposition, but the shots can be dealt out to MPI ranks). Coordinates stay `[x, y, z]` lists, y being ignored:

```python
from makutu.mesh import CartesianMesh2D
mesh = CartesianMesh2D(lx, lz, ex, ez, order=4, vp_file="Vp_xz.H")
solver = AcousticSolver(implem_type=FTSolver.ImplemType.MAKUTU, mesh_type=FTSolver.MeshType.UNSTRUCT,
                        dt=dt, maxTime=tmax, sourceType="ricker2", sourceFreq=f0, **mesh.solver_kwargs())
```

See `apps/modeling/examples/marmousi_2d/` (README: data preparation, mesh choice, modelling).

**2-D blocks and hybrid SEM + DG.** `makutu.mortar` tiles a SEP model with regular blocks, each at the
size N v / (fmax ppw), glued by a symmetric interior penalty (run on the GPU by the FUnTiDES 2-D SEM
solver of the `dev-2d` branch), and has a Python prototype that adds DG triangles along the model
interfaces. Applications, the reference document and the reproduction of the modelling cases on the
SEP models are in `apps/mortar/` (README there):

```bash
python apps/mortar/reproduce.py --list
python apps/mortar/reproduce.py --model seg2d disque
```

MPI, via `mpi4py`: in 3-D, 1-D X-axis domain decomposition (pass a communicator to the
mesh/solver constructors); in 2-D and 3-D, shots dealt out to the ranks when the solver
has no communicator (modelling and FWI, `makutu/forwardBackward/parallel.py`).

**FWI.** `apps/fullWaveformInversion/fwi.py` (README there): SEP initial model, SEG-Y
observed data, multi-scale, optimisers of `makutu.imaging.Minimizer`, gradient
conditioning; synthetic examples `anomaly_2d` and `marmousi_2d`. A partitioned `StructuredMesh`
uses the `ModelUnstruct` backend (`mesh.heterogeneous` is True), so pass
`MeshType.UNSTRUCT` to the solver.

## Documentation

- [`docs/funtides_input_parameters.md`](docs/funtides_input_parameters.md) —
  FUnTiDES solver input parameters.
- Sphinx documentation in `docs/source/` — `pip install sphinx sphinx_rtd_theme sphinx-prompt`,
  then `make docs`; open `docs/build/index.html` (FUnTiDES is not needed to build it).
