# makutu — Python seismic modelling & imaging on FUnTiDES

`makutu` is a Python library for seismic forward modelling, imaging (RTM) and
full-waveform inversion. It drives the
[FUnTiDES](https://github.com/FUnTiDES-sim/FUnTiDES) spectral-element solvers
through their `pyfuntides` Python bindings (zero-copy Kokkos view injection) and
adds the acquisition, mesh/model, I/O and workflow layers on top.

It began as a [GEOS-DEV/makutu](https://github.com/GEOS-DEV/makutu) `pygeosx`
interface; it has since been **migrated off GEOS / pysabl / XML** to a
pure-Python stack on FUnTiDES (the old upstream stays reachable as the
`upstream` git remote).

## Repository layout

| Path | Contents |
|---|---|
| `utilities/` | the core package (`import utilities…`): `mesh`, `model`, `solvers` (acoustic/elastic SEM), `input`, `acquisition`, `output`, `imaging`, `forwardBackward`, `display` |
| `workflows/` | runnable end-to-end scripts, e.g. `workflows/modeling/examples/` |
| `tests/` | `unitTests` (acquisition), `integratedTests` (forward shots on FUnTiDES: arrival times, elastic, MPI); run with `make unitTests` / `make integratedTests` after `source env.sh` |
| `scripts/` | developer inspection / helper scripts |
| `docs/` | `funtides_input_parameters.md` + Sphinx source |

Example data/models live in the sibling `../FUn-examples/` directory (SEAM
acquisition test data, Marmousi SEP/Cubit models, …), kept out of the source
tree.

## Requirements

- **FUnTiDES**, installed with its Python bindings (`-DENABLE_PYWRAP=ON`). It
  provides the `pyfuntides` modules and, through FUnTiDES-TPL, the `kokkos`
  module and the Python virtual environment (`tpl-venv`). **These are not pip
  packages.** Expected layout (default of `env.sh`):

  ```
  FUnTiDES-dir/
  ├── FUnTiDES-TPL/      # third-party libraries (Kokkos, pykokkos-base, ADIOS2, ...)
  ├── FUnTiDES/          # FUnTiDES sources
  ├── FUnTiDES-install/  # TPL + FUnTiDES install, with funtides_env.sh
  └── makutu-lib/        # this repo
  ```

- **Python packages** (in `tpl-venv`): `numpy`, `scipy`, `matplotlib`, `mpi4py`,
  `h5py`, `segyio`, `vtk`, `pyevtk`, `numba` (`utilities.model` imports `vtk` and
  `numba`), and `pytest` for the tests:

  ```bash
  pip install scipy h5py segyio vtk pyevtk numba pytest
  ```

> **GPU.** makutu allocates its Kokkos arrays in the memory space and layout of
> the FUnTiDES build (`CudaUVMSpace` + `LayoutLeft` with CUDA, `HostSpace` +
> `LayoutRight` on CPU, see `utilities/tools/kokkosarrays.py`). It is tested with
> the CUDA build. The polynomial orders available to the solvers are those
> compiled in FUnTiDES (`MAX_SOLVER_ORDER`).

## Build FUnTiDES (once)

Follow the FUnTiDES README (Quick Start): install FUnTiDES-TPL with its virtual
environment, then FUnTiDES with `-DENABLE_PYWRAP=ON`, both into
`FUnTiDES-install`:

```bash
FUnTiDES-TPL/install.sh --prefix=$(pwd)/FUnTiDES-install --enable-cuda --enable-venv
source FUnTiDES-install/setup_env.sh
cmake -S FUnTiDES -B FUnTiDES/build -DCMAKE_BUILD_TYPE=Release -DENABLE_PYWRAP=ON \
      -DCMAKE_INSTALL_PREFIX=$(pwd)/FUnTiDES-install
cmake --build FUnTiDES/build -j 8 && cmake --install FUnTiDES/build
```

## Environment

Source `env.sh` — it sources `FUnTiDES-install/funtides_env.sh` (Kokkos,
`pyfuntides`, `tpl-venv`) and puts this repo on `PYTHONPATH` so
`import utilities` / `import workflows` resolve:

```bash
source env.sh
```

To use another FUnTiDES installation:

```bash
FUNTIDES_ENV=/path/to/funtides_env.sh source env.sh
```

`env.sh` also pins `CUDA_VISIBLE_DEVICES=0` (single-GPU UVM); override it before
sourcing if needed. It can be sourced several times. There is **no
`pip install` / `setup.py`** any more (the old editable install was removed with
the GEOS migration) — `env.sh` alone makes the package importable.

## Quick start

Run an acoustic one-shot forward model from a SEP velocity file:

```bash
source env.sh
python workflows/modeling/examples/acoustic_one_shot_unstruct_sep.py \
    --vp-file <model.H> --f0 10 --order 2
```

The driver builds a heterogeneous Cartesian mesh, runs the SEM acoustic solver
through `pyfuntides`, and writes the seismogram to HDF5. See
`workflows/modeling/examples/` for more (multi-shot acquisition,
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
communicator to the mesh/solver constructors. A partitioned `StructuredMesh`
uses the `ModelUnstruct` backend (`mesh.heterogeneous` is True), so pass
`MeshType.UNSTRUCT` to the solver.

## Documentation

- [`docs/funtides_input_parameters.md`](docs/funtides_input_parameters.md) —
  FUnTiDES solver input parameters.
- Sphinx source in `docs/source/` — build with `make -C docs html` (needs
  `sphinx`); output lands in `docs/build/`.
