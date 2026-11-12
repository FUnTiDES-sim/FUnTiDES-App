# FUnTiDES-App

Applications built on top of the FUnTiDES library:

- **C++ drivers** (`examples/`, below): one example program per numerical method;
- **Python** (`python/`): the `makutu` library (seismic modelling, imaging, FWI on
  `pyfuntides`), its applications (`python/apps/`) and tests. See
  [`python/README.md`](python/README.md).

## C++ drivers

Example programs built on top of the FUnTiDES library, one per numerical method.
Each example is a single file that reads top to bottom: mesh, solver, source and
receiver, time loop, outputs. Copy the one closest to your use case as a starting
point.

| Executable              | Source                                          | Method                                         | Physics                            |
|-------------------------|-------------------------------------------------|------------------------------------------------|------------------------------------|
| `funtides-sem`          | `examples/sem/sem_example.cc`                   | spectral elements                              | acoustic, elastic, acousto-elastic |
| `funtides-dg`           | `examples/dg/dg_example.cc`                     | discontinuous Galerkin                         | acoustic                           |
| `funtides-dg-sem`       | `examples/dg_sem/dg_sem_example.cc`             | DG below a horizontal plane, SEM above         | acoustic                           |
| `funtides-dg-padaptive` | `examples/dg_padaptive/dg_padaptive_example.cc` | DG, low order below a plane, high order above  | acoustic                           |
| `funtides-sem2d`        | `examples/sem2d/sem2d_example.cc`               | spectral elements in 2-D, (x, z) plane         | acoustic                           |

Only `funtides-sem` supports MPI (`-DUSE_MPI=ON`, domain split along x), except for
acousto-elastic runs; the other examples are single-process.

`examples/common/` holds what the examples share and is not specific to a method:
command-line options, cartesian mesh construction, source and receiver location,
text outputs and timing.

### Requirements

- An installed FUnTiDES (`cmake --install` of the FUnTiDES repository). An example
  is built only if FUnTiDES was built with its method (`COMPILE_SEM`, `COMPILE_DG`,
  `COMPILE_DG_SEM`, `COMPILE_DG_PADAPTIVE`, `COMPILE_SEM_2D` for `funtides-sem2d`).
- MPI, only with `-DUSE_MPI=ON`.

Kokkos is found through the FUnTiDES package configuration. Use the same C++
compiler as the one FUnTiDES was built with.

### Build

```bash
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=<compiler used for FUnTiDES> \
  -Dfuntides_DIR=<funtides install>/lib/cmake/funtides
cmake --build build -j4
```

The executables are written to `build/bin/`.

### Run

```bash
./build/bin/funtides-sem --help
./build/bin/funtides-sem --ex 100 --ey 100 --ez 100 --mesh=cartesian -o 2 --dt 0.001 --timemax 1.5
./build/bin/funtides-dg-sem --dg-sem-boundary-z 1000 --rcvz 690
./build/bin/funtides-sem2d --ex 100 --ez 100 -o 3 --dt 0.001 --timemax 0.7   # 2-D (x, z); y options ignored
```

Each example folder has a YAML file listing its options. `python/apps/run_funtides.py`
runs the executable named by the `method` key of such a file:

```bash
python python/apps/run_funtides.py examples/dg/dg_acoustic.yaml
```

### Outputs

All outputs are plain text, written to the working directory.

- `receiver_trace.txt`: one line per time step, `time value...`. The values are the
  pressure, or `ux uy uz` for an elastic receiver; line n holds the field at time n * dt.
- `das_trace.txt`: DAS signal (`funtides-sem --is-elastic --das-type dipole|strain`).
- With `--snapshots`, `slice_*_<step>.dat` every `--snap-interval` steps: a plane of
  the field, one line per row of nodes (gnuplot: `plot 'file' matrix with image`).

### Validation

The validation compares the receiver traces of `funtides-sem` with analytical
solutions for the acoustic and elastic cases, on cartesian and unstructured cartesian
meshes, orders 1 to 3.

```bash
cmake --build build --target validate
```

`validate_methods` runs the acoustic case of every built example (SEM, DG, DG-SEM,
p-adaptive DG) against the exact solution of the homogeneous medium,
p = rho f(t - r/c) / (4 pi r), without normalization. The coupled solvers have their
source below the interface and their receiver above it, so the wave crosses it.
`funtides-sem2d` (built when FUnTiDES has `COMPILE_SEM_2D`) is compared with the 2-D
solution p = (rho c / 2 pi) int f(tau) / sqrt(c^2 (t - tau)^2 - r^2) dtau.

```bash
cmake --build build --target validate_methods
cd build && ctest -L analytical_dg-sem --output-on-failure   # one method
```

## Python: makutu library and applications

```sh
source <FUnTiDES install>/funtides_env.sh   # kokkos, pyfuntides, tpl-venv
cd python
pip install -e ".[test]"                     # once
make unitTests integratedTests               # about 1 min on a GPU
```

| Path | Contents |
|---|---|
| `python/makutu/` | the library: acquisition, mesh/model (including `mesh/seismesh`, meshes adapted to a SEP model), solvers (acoustic/elastic SEM), `mortar` (2-D non-conforming blocks glued by SIPG, hybrid SEM + DG triangles), I/O, imaging, FWI |
| `python/apps/` | applications (modeling, mesh, mortar, FWI, display, model processing) and `run_funtides.py`; `apps/mortar/reproduce.py` reruns the 2-D block and hybrid modelling cases and checks their values |
| `python/tests/` | unit and integrated tests |

The applications import the library, never the reverse. Example data and models
live in the sibling `FUnTiDES-data/` directory.

## License

FUnTiDES-App, including the Python `makutu` library, is distributed under the
CeCILL-C free software license, like FUnTiDES: see [`LICENCE.md`](LICENCE.md)
(English) and [`LICENCE_fr.md`](LICENCE_fr.md) (French).
