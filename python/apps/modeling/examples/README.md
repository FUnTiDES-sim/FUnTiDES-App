# Modeling examples

Example workflows showing how to drive the FUnTiDES SEM solvers (through
`pyfuntides`) with makutu. They use simplified setups and are **not** meant to be
physically accurate.

Source the environment first — see the top-level `README.md`
(`source ../../../env.sh` from this directory). Each script prints its options
with `--help`.

| Script | Content |
|---|---|
| `create_sep_model.py` | Creates synthetic SEP velocity models (layered, salt) in `sep_models/` and reads them back |
| `model_from_sep.py` | One acoustic shot on a heterogeneous mesh built from a SEP Vp file; seismogram in HDF5 |
| `acoustic_one_shot_unstruct_sep.py` | Same, with the mesh and time step derived from the SEP model |
| `acquisition_acoustic_modeling.py` | Multi-shot acoustic modeling driven by a decision file; one SEG-Y file per shot |

For example:

```bash
python3 create_sep_model.py --output-dir sep_models
python3 model_from_sep.py --vp-file sep_models/vp_layered.H
```

Minimal forward shots (acoustic, elastic, MPI) are also in `tests/integratedTests/shots.py`.
