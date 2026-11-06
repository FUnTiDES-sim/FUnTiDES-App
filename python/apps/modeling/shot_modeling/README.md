# Shot modelling: 2-D or 3-D, one or several shots

`run_shot_modeling.py` models acoustic shots on a velocity model read from SEP files.
The mesh and the solver are built once, then the shots are run one after the other.

- **2-D or 3-D** from the SEP file: `n3 = 1` gives a 2-D model (axis 1 = x, axis 2 = z),
  otherwise 3-D (axes x, y, z).
- **Acquisition**: a decision file (`--decision`), or a line of `--n-shots` sources along x
  (one shot at `--src-x`, default lx / 2) with receivers every `--rcv-dx`, at y = ly / 2
  in 3-D.
- **Outputs** per shot in `--output-dir`: `<root>_<shot>.segy` (or `.h5` with
  `--format hdf5`) and, with `--plot`, `<root>_<shot>.png` (model section, pressure
  snapshot at `--snap-time`, gather).

Coordinates are in metres, z = 0 at the surface and positive downwards (seismic
convention). Commands below are run from `FUnTiDES-App/python` after
`source <FUnTiDES install>/funtides_env.sh`; `--help` lists all the options.

## Mesh and time step

`makutu.mesh.suggest_mesh`:

- shortest wavelength of the Ricker wavelet: lambda_min = vp_min / (2.5 f0);
- element size h = order * lambda_min / `--ppw` (GLL points per wavelength, default 6),
  or `--elem-size`;
- dt = `--cfl` (default 0.3) * smallest GLL spacing / vp_max;
- free surface on top; the other edges are absorbing, with a sponge of `--sponge` m
  (default 5 % of the smallest extent; `--sponge 0` keeps only the absorbing condition);
- orders: 1 to 3 in 3-D, up to the order compiled in FUnTiDES in 2-D.

The seismograms are recorded every round(`--dt-seismo` / dt) steps; the actual sampling is
printed and written in the outputs.

## Examples

**2-D, one shot** (Marmousi; see `../examples/marmousi_2d/` for the data preparation):

```bash
python apps/modeling/shot_modeling/run_shot_modeling.py \
    --vp-file ../../FUnTiDES-data/SEP/marmousi/Vp_xz.H --rho-file ../../FUnTiDES-data/SEP/marmousi/Rho_xz.H \
    --f0 8 --order 4 --elem-size 50 --tmax 4 --dt-seismo 0.002 --rcv-dx 20 \
    --format hdf5 --plot --output-dir runs/marmousi_2d
```

**2-D, a line of shots**: `--n-shots 3` (sources from 10 % to 90 % of lx, or
`--src-xmin`/`--src-xmax`).

**3-D, one shot** on a synthetic model:

```bash
python apps/modeling/tools/create_sep_model.py --output-dir sep_models
python apps/modeling/shot_modeling/run_shot_modeling.py \
    --vp-file sep_models/vp_layered.H --rho-file sep_models/vp_layered_rho.H \
    --f0 10 --order 2 --tmax 1 --plot --output-dir runs/one_shot_3d
```

**3-D, shots from a decision file**:

1. Acquisition: a regular survey in the 1 km x 1 km model, 2 sources at z = 10 m and
   40 receivers at z = 5 m; writes the decision files `acquisition/survey.*` and one empty
   SEG-Y file per shot:

   ```bash
   python -m makutu.acquisition.create_equispaced_segy \
       --xmin 0 --xmax 1000 --ymin 0 --ymax 1000 --margin 100 \
       --n-src-x 2 --n-rcv-x 40 --output-dir acquisition --out survey
   ```

   An existing SEG-Y acquisition is turned into decision files with
   `python -m makutu.acquisition.decision_create --from-segy ...`.

2. Check (optional): `python -m makutu.acquisition.check_acquisition --dir acquisition --out survey`.

3. Modelling (`--first-shot`/`--last-shot` select a subset):

   ```bash
   python apps/modeling/shot_modeling/run_shot_modeling.py \
       --vp-file sep_models/vp_layered.H --decision acquisition/survey --tmax 0.6 \
       --output-dir runs/survey
   ```

   Output: `runs/survey/survey_00001.segy`, `survey_00002.segy`; the result can be
   checked with `check_acquisition --dir runs/survey`.

Run times on an RTX 40xx GPU: Marmousi one shot (225 000 nodes, 7 400 steps) about 3 s;
3-D one shot (1 million nodes, 1 200 steps) about 2 s.

## Files

- SEG-Y: IEEE float traces; integer coordinates in metres, no scalar; source x, y, z in
  the trace-header words 22, 23, 14 and receiver x, y, z in 24, 25, 13 (`--word-*`
  options, the same as the decision files). These are the makutu decision-file words,
  not the standard SEG-Y byte positions read by `makutu.acquisition.SEGYAcquisition`.
- HDF5: `pressure` (samples x receivers), `source`, `receivers` ([x, y, z]); attributes
  `dt` (actual sampling), `f0`, `order`, `dim`, `vp_file`.
