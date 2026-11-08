# Marmousi 2-D: from the data to the shot gather

Acoustic shots on the 2-D Marmousi model with the FUnTiDES 2-D SEM solver
(quadrilaterals in the (x, z) plane), driven by makutu. This directory prepares the
data; the modelling is done by the generic `../../shot_modeling/run_shot_modeling.py`.

| Step | Command | Result |
|---|---|---|
| 0. Environment | `source <FUnTiDES install>/funtides_env.sh` | kokkos, pyfuntides, makutu |
| 1. Data | files in `FUnTiDES-data/SEP/marmousi/` | Vp (and Rho) SEP files |
| 2. Preparation | `prepare_marmousi_2d.py [--fix]` | checked/fixed SEP files, mesh and time step, QC figure, modelling command |
| 3. Modelling | `../../shot_modeling/run_shot_modeling.py ...` | gather per shot (HDF5 or SEG-Y), figure |
| 4. Results | Python / HDF5 tools | shot gather, snapshot |

Commands below are run from `FUnTiDES-App/python`; the outputs go to the current
directory unless `--output-dir`/`--qc` give a path. Every script prints its options with
`--help`.

## 0. Environment

FUnTiDES must be installed with the Python bindings and the 2-D solver
(`-DENABLE_PYWRAP=ON -DCOMPILE_SEM_2D=ON`), and makutu installed once:

```bash
source <FUnTiDES install>/funtides_env.sh
cd FUnTiDES-App/python
pip install -e .                 # once
```

## 1. Data

The model is read from `FUnTiDES-data/SEP/marmousi/`, the `FUnTiDES-data` directory
being next to FUnTiDES-App (or given by `--data-dir` / `$FUNTIDES_DATA`):

- `Vp_xz.H`, `Vp_xz.H@`: P-wave velocity, SEP format, 500 x 174 samples at 20 m
  (10 km x 3.48 km), axis 1 = horizontal (`label1="CDP"`), axis 2 = depth
  (`label2="DEPTH"`, 0 at the surface), 1500-4767 m/s;
- `Rho_xz.H`, `Rho_xz.H@`: density (1 everywhere in this copy); optional.

To use another 2-D model, put its SEP files under these names (same orientation: axis 1
horizontal, axis 2 depth); the domain size is taken from `n1 * d1` and `n2 * d2`.

## 2. Preparation

```bash
python apps/modeling/examples/marmousi_2d/prepare_marmousi_2d.py --f0 8 --order 4          # check only
python apps/modeling/examples/marmousi_2d/prepare_marmousi_2d.py --f0 8 --order 4 --fix    # check and fix
```

- **SEP files**: the script checks the header fields, the binary named by `in=` and the
  byte order (`data_format`). With `--fix` it points `in=` to the binary next to the
  header when the declared one is missing, sets `data_format` to the byte order that
  gives physical values, and writes a density of 1 when `Rho_xz.H` is missing; modified
  headers are saved as `*.H.orig`. (The copy of FUnTiDES-data was fixed this way: it
  pointed to removed proxy-FUn files and declared the density as `xdr_float`.)
- **Mesh**: the shortest wavelength of a Ricker wavelet of peak frequency f0 is
  lambda_min = vmin / (2.5 f0) (75 m for 8 Hz). The element size is chosen so that it
  holds `--ppw` GLL points (default 6): h = order * lambda_min / ppw (50 m for order 4).
  Elements smaller than the 20 m sampling are fine (the model is then piecewise constant).
- **Time step**: dt = cfl * smallest GLL spacing / vmax (`--cfl`, default 0.3).
- The script prints the matching modelling command and writes `marmousi_2d_qc.png`
  (velocity model and element grid).

Example output (f0 = 8 Hz, order 4):

```
Mesh for f0 = 8 Hz, order 4, 6 points per wavelength (lambda_min = 75 m): element size <= 50.0 m -> 200 x 70 elements (50.0 x 49.7 m), 225081 nodes
Time step (CFL 0.3): 0.540 ms -> 7403 steps for 4 s

Run:
  python apps/modeling/shot_modeling/run_shot_modeling.py --vp-file <FUnTiDES-data>/SEP/marmousi/Vp_xz.H --rho-file <FUnTiDES-data>/SEP/marmousi/Rho_xz.H --order 4 --f0 8 --elem-size 50.00 --cfl 0.3 --tmax 4 --plot
```

## 3. Modelling

```bash
python apps/modeling/shot_modeling/run_shot_modeling.py \
    --vp-file ../../FUnTiDES-data/SEP/marmousi/Vp_xz.H --rho-file ../../FUnTiDES-data/SEP/marmousi/Rho_xz.H \
    --order 4 --f0 8 --elem-size 50 --cfl 0.3 --tmax 4 --dt-seismo 0.002 --rcv-dx 20 \
    --format hdf5 --plot --snap-time 1.5 --output-dir runs/marmousi_2d
```

- Mesh: 2-D quadrilaterals, z = 0 at the surface and positive downwards; the free
  surface is the top edge, the other edges are absorbing (first-order paraxial condition;
  `--sponge` adds a sponge of that width, useful from about two wavelengths).
- Source: Ricker wavelet (second derivative of a Gaussian) of peak frequency `--f0`,
  point source at `--src-x` (default lx / 2 = 5000 m) and `--src-depth` (default 10 m);
  `--n-shots N` gives a line of N shots.
- Receivers: a line at `--rcv-depth` (default 20 m), every `--rcv-dx` (default: element
  size), sampled every round(`--dt-seismo` / dt) time steps: 2.161 ms here for 2 ms
  requested, the value written in the outputs.
- Cost: about 3 s on an RTX 40xx GPU for the command above (225 000 nodes, 7 400 steps).

## 4. Results

- `runs/marmousi_2d/shot_00001.h5`: `pressure` (time samples x receivers), `source`,
  `receivers` ([x, y, z] in m), and the attributes `dt` (actual sampling), `f0`, `order`,
  `dim`, `vp_file`; `--format segy` writes `shot_00001.segy` instead;
- `runs/marmousi_2d/shot_00001.png`: velocity model with the source, pressure snapshot at
  `--snap-time`, shot gather.

```python
import h5py
with h5py.File("runs/marmousi_2d/shot_00001.h5") as f:
    gather, rcv, dt = f["pressure"][:], f["receivers"][:], f.attrs["dt"]
```

The gather shows the direct wave, the refractions (linear moveout) and the reflections of
the Marmousi layers, symmetric about the source.
