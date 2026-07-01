# makutu — Python seismic modelling & imaging on FUnTiDES

`makutu` is a Python library for seismic forward modelling, imaging (RTM) and
full-waveform inversion. It drives the **FUnTiDES** spectral-element solvers —
provided by [proxy-FUn](../proxy-FUn)'s `FUn-Libs` through the `pyfuntides`
Python bindings (zero-copy Kokkos view injection) — and adds the acquisition,
mesh/model, I/O and workflow layers on top.

It began as a [GEOS-DEV/makutu](https://github.com/GEOS-DEV/makutu) `pygeosx`
interface; it has since been **migrated off GEOS / pysabl / XML** to a
pure-Python stack on FUnTiDES (the old upstream stays reachable as the
`upstream` git remote).

## Repository layout

| Path | Contents |
|---|---|
| `utilities/` | the core package (`import utilities…`): `mesh`, `model`, `solvers` (acoustic/elastic SEM), `input`, `acquisition`, `output`, `imaging`, `forwardBackward`, `display` |
| `workflows/` | runnable end-to-end scripts, e.g. `workflows/modeling/examples/` |
| `tests/` | `unitTests`, `integratedTests`, `manualTests` |
| `scripts/` | developer inspection / helper scripts |
| `docs/` | `funtides_input_parameters.md` + Sphinx source |

Example data/models live in the sibling `../FUn-examples/` directory (SEAM
acquisition test data, Marmousi SEP/Cubit models, …), kept out of the source
tree.

## Requirements

- **Python 3.10+** with: `numpy`, `scipy`, `matplotlib`, `mpi4py`, `h5py`,
  `segyio` (plus `vtk` / `pyevtk` / `numba` for a few workflows).
- **FUnTiDES** = `proxy-FUn`'s `FUn-Libs`, built with the `pyfuntides` wrappers
  (`-DENABLE_PYWRAP=ON`). This provides the `pyfuntides` modules and the `kokkos`
  runtime — **these are not pip packages**, they come from the FUn-Libs build and
  the FUn-TPL prefix. Clone `proxy-FUn` as a **sibling** of this repo:

  ```
  numpex/
  ├── proxy-FUn/     # FUnTiDES: FUn-TPL + FUn-Libs (+ pyfuntides)
  └── makutu-lib/    # this repo
  ```

> **SEM modelling needs the CUDA build.** makutu's Cartesian mesh builder
> allocates `CudaUVMSpace` Kokkos views, so the acoustic/elastic solvers require
> a **CUDA** FUn-Libs build (`build-cuda`) on a GPU. The finite-difference path
> (`pyfuntides.fd`) also runs on a serial (CPU) build.

## Build FUnTiDES (once)

Build `FUn-Libs` + the `pyfuntides` wrappers from the sibling `proxy-FUn` repo
(see its `README.md` / `CLAUDE.md` for the full option reference):

```bash
cd ../proxy-FUn
FUN_BACKEND=cuda FUN_CUDA_ARCH=<your-arch> ./bootstrap.sh   # SEM (GPU)
# ./bootstrap.sh                                            # serial CPU (FD only)
```

## Environment

Source `env.sh` — it wires `pyfuntides` + `kokkos` (via
`../proxy-FUn/FUn-Libs/envpython.sh`) and puts this repo on `PYTHONPATH` so
`import utilities` / `import workflows` resolve:

```bash
source env.sh
```

To target a specific FUn-Libs build (the CUDA build for SEM):

```bash
FUN_LIBS_BUILD=../proxy-FUn/FUn-Libs/build-cuda source env.sh
```

`env.sh` also pins `CUDA_VISIBLE_DEVICES=0` (single-GPU UVM); override it before
sourcing if needed. There is **no `pip install` / `setup.py`** any more (the old
editable install was removed with the GEOS migration) — `env.sh` alone makes the
package importable.

## Quick start

Run an acoustic one-shot forward model from a SEP velocity file:

```bash
FUN_LIBS_BUILD=../proxy-FUn/FUn-Libs/build-cuda source env.sh
python workflows/modeling/examples/acoustic_one_shot_unstruct_sep.py \
    --vp-file <model.H> --f0 10 --order 2
```

The driver builds a heterogeneous Cartesian mesh, runs the SEM acoustic solver
through `pyfuntides`, and writes the seismogram to HDF5. See
`workflows/modeling/examples/` for more (elastic, multi-shot acquisition,
`model_from_sep.py`, …).

The Python API in a nutshell:

```python
import kokkos; kokkos.initialize()
import pyfuntides.solver as FTSolver
from utilities.mesh.InternalMesh import StructuredMesh
from utilities.solvers.AcousticSolver import AcousticSolver

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

MPI is supported (1-D X-axis domain decomposition) via `mpi4py`; pass a
communicator to the mesh/solver constructors.

## Documentation

- [`docs/funtides_input_parameters.md`](docs/funtides_input_parameters.md) —
  FUnTiDES solver input parameters.
- Sphinx source in `docs/source/` — build with `make -C docs html` (needs
  `sphinx`); output lands in `docs/build/`.
