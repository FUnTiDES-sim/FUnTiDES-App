# FUnTiDES-App

Example driver program built on top of the FUnTiDES library. It produces the
`funtides-sem` executable, which runs the SEM, DG, DG-SEM and p-adaptive DG
solvers from the command line.

## Requirements

- An installed FUnTiDES (`cmake --install` of the FUnTiDES repository). The
  methods available in `funtides-sem` are the ones FUnTiDES was built with
  (`COMPILE_SEM`, `COMPILE_DG`, `COMPILE_DG_SEM`, `COMPILE_DG_PADAPTIVE`).
- ADIOS2, used to write receivers and snapshots.
- MPI, only with `-DUSE_MPI=ON`.

Kokkos is found through the FUnTiDES package configuration. Use the same C++
compiler as the one FUnTiDES was built with.

## Build

```bash
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=<compiler used for FUnTiDES> \
  -Dfuntides_DIR=<funtides install>/lib/cmake/funtides \
  -DADIOS2_DIR=<adios2 install>/lib/cmake/adios2
cmake --build build -j4
```

The executable is written to `build/bin/funtides-sem`.

## Run

```bash
./build/bin/funtides-sem --help
./build/bin/funtides-sem --ex 100 --ey 100 --ez 100 --method=sem --mesh=cartesian -o 2 --dt 0.001 --timemax 1.5
```

`examples/example_config.yaml` lists the command line options;
`python/run_funtides.py` runs `funtides-sem` from such a YAML file.

## Python examples

`examples/python/` holds small scripts calling the FUnTiDES Python bindings
(`pyfuntides`) directly, without the `funtides-sem` executable:

| Script                           | Content                                         |
|----------------------------------|-------------------------------------------------|
| `solver_cartesian.py`            | acoustic SEM, cartesian mesh                    |
| `solver_cartesian_bilayer.py`    | acoustic SEM, two-layer cartesian model         |
| `solver_cartesian_dg.py`         | acoustic DG, cartesian mesh                     |
| `solver_cartesian_mpi.py`        | acoustic SEM split across MPI ranks             |
| `solver_multigpu.py`             | acoustic SEM, one GPU per MPI rank              |
| `solver_multigpu_multinode.py`   | same on several nodes, strong/weak scaling      |
| `qualitative_gradient_test.py`   | one FWI gradient on an unstructured model       |

Requirements:

- FUnTiDES configured with `-DENABLE_PYWRAP=ON` and installed; its `python`
  directory provides `pyfuntides`.
- `pykokkos` (the `kokkos` module), built on the same Kokkos as FUnTiDES.
- The packages of `examples/python/requirements.txt`, and `mpi4py` for the MPI
  scripts.
- For now, GPU runs need a Kokkos built with CUDA UVM
  (`Kokkos_ENABLE_CUDA_UVM=ON`): the scripts allocate their views in
  `CudaUVMSpace`.

```bash
export PYTHONPATH=<funtides install>/python:$PYTHONPATH
python3 examples/python/solver_cartesian.py --ex 100 --ey 100 --ez 100 --order 2
mpirun -n 4 python3 examples/python/solver_cartesian_mpi.py --ex 200 --snap_interval 20
```

`scripts/environments/plafrim_{strong,weak}_scaling.sbatch` run the scaling
studies of `solver_multigpu_multinode.py` on PlaFRIM:

```bash
FUNTIDES_INSTALL=<funtides install> \
FUNTIDES_APP_PATH=<this repository> \
FUNTIDES_VENV=<Python env providing pykokkos> \
sbatch scripts/environments/plafrim_strong_scaling.sbatch
```

## Validation

The validation compares receiver traces with analytical solutions for the
acoustic and elastic cases, on cartesian and unstructured cartesian meshes,
orders 1 to 3. It needs Python 3 with the `adios2` module to extract the traces.

```bash
cmake --build build --target validate
```
