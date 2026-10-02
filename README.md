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

## Validation

The validation compares receiver traces with analytical solutions for the
acoustic and elastic cases, on cartesian and unstructured cartesian meshes,
orders 1 to 3. It needs Python 3 with the `adios2` module to extract the traces.

```bash
cmake --build build --target validate
```
