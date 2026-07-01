# Modeling examples

Example workflows showing how to drive the FUnTiDES SEM solvers (through
`pyfuntides`) with makutu. They use simplified setups and are **not** meant to be
physically accurate. The runnable scripts live under `adt/`.

Source/receiver geometry is read from a native **decision file** (no XML/GEOS).
Source the environment first — see the top-level `README.md`; for SEM use the
CUDA build, e.g. `FUN_LIBS_BUILD=../../../../proxy-FUn/FUn-Libs/build-cuda source ../../../env.sh`.

## Acoustic modeling — `adt/acoustic_modeling.py`
One forward acoustic shot per source; pressure at receivers written as SEG-Y
(or HDF5).

```bash
mpirun -np 4 python3 adt/acoustic_modeling.py \
    --decision inputfiles/acquisition/decision.mapped \
    --database identity --swords 22 23 15 \
    --lx 5000 --ly 5000 --lz 2000 --ex 50 --ey 50 --ez 20 \
    --vp 1500 --dt 0.001 --tmax 2.0 --f0 5 --output-dir seismoTraces
```

## Elastic modeling — `adt/elastic_modeling.py`
One forward elastic shot per source; three-component (X, Y, Z) displacement
seismograms.

```bash
mpirun -np 4 python3 adt/elastic_modeling.py \
    --decision inputfiles/acquisition/decision.mapped \
    --database identity \
    --lx 5000 --ly 5000 --lz 2000 --ex 50 --ey 50 --ez 20 \
    --vp 3000 --vs 1700 --dt 0.001 --tmax 2.0 --f0 5 \
    --source-components z --output-dir seismoTraces
```
Run `python3 adt/elastic_modeling.py --help` for all options.

## Limited-aperture acquisition — `adt/acquisition_limitedAperture.py`
Modeling for a limited-aperture acquisition defined directly in the script
(sources/receivers by hand); omit `--LA` for full aperture. See the `userguide`
docs for details.
