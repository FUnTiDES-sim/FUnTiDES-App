# Applications

Scripts built on the `makutu` library (`../makutu`) and the FUnTiDES solvers. Run them
after `source <FUnTiDES install>/funtides_env.sh` and `pip install -e python/` (see
`../README.md`); each script prints its options with `--help`. Example data and models
are in the `FUnTiDES-data` directory next to FUnTiDES-App.

| Path | Content |
|---|---|
| `run_funtides.py` | Runs a C++ driver of FUnTiDES-App (`funtides-<method>`, found in `build*/bin`) from a YAML file of `examples/<method>/` |
| `modeling/` | Forward modelling: `shot_modeling` (2-D or 3-D, one or several shots, decision files), `examples/marmousi_2d` (data preparation), and `tools/` (synthetic SEP models); see its README |
| `mesh/seismesh_from_sep.py` | Mesh adapted to a SEP model and to the source frequency (2-D quadrilaterals, 3-D hexahedra, 2:1 doubling layers), for `GmshMesh` |
| `mesh/gmsh_mesh.py` | Gmsh meshes: conversion to MSH 2.2, Hex8 box and Q4 rectangle generators, description through FUnTiDES; see its README |
| `acquisition/` | `create_equispaced_segy.py`, `check_acquisition.py`, `decision_create.py`: regular acquisitions, checks and maps, decision files |
| `display/` | `showVelocityModel.py`, `showSeismicTraces.py`, `showFullGradient.py`: plots of models, traces, gradients and illumination (`--png` to save them) |
| `fullWaveformInversion/fwi.py` | Full-waveform inversion (acoustic; shots dealt out to the MPI ranks with `mpirun`); see its README |

```bash
python apps/run_funtides.py ../examples/sem2d/sem2d_acoustic.yaml   # from FUnTiDES-App/python
python apps/modeling/examples/marmousi_2d/prepare_marmousi_2d.py --fix
python apps/modeling/shot_modeling/run_shot_modeling.py --vp-file ../../FUnTiDES-data/SEP/marmousi/Vp_xz.H --f0 8 --order 4 --plot
```

## Status

See `../docs/source/status.rst` (or the *Status* page of the documentation).
