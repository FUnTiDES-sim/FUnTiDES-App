# Modelling applications

Forward modelling with makutu and the FUnTiDES SEM solvers. Run the scripts from
`FUnTiDES-App/python` after `source <FUnTiDES install>/funtides_env.sh` (and
`pip install -e .` once); every script prints its options with `--help`. makutu uses the
seismic convention: z = 0 at the surface, positive downwards.

```
modeling/
  shot_modeling/     application: acoustic shots on a SEP model
  examples/          cases built on it (data and their preparation)
    marmousi_2d/
  tools/             data preparation
```

## Application

[`shot_modeling/run_shot_modeling.py`](shot_modeling/README.md): acoustic shots on a SEP
velocity model, 2-D or 3-D from the file; one shot, a line of shots or the shots of a
decision file; one gather per shot (SEG-Y or HDF5) and a figure. Its README gives the
acquisition chain (create, check, model).

## Examples

| Case | Content |
|---|---|
| [`examples/marmousi_2d/`](examples/marmousi_2d/README.md) | 2-D Marmousi: data check and fix, mesh and time-step choice (`prepare_marmousi_2d.py`), then modelling with `shot_modeling` |

Minimal forward shots (acoustic, elastic, MPI) are also in `tests/integratedTests/shots.py`.

## Tools

| Tool | Content |
|---|---|
| `python apps/modeling/tools/create_sep_model.py --output-dir sep_models` | Synthetic 3-D SEP models (layered, salt body) with their densities |
| `python apps/acquisition/create_equispaced_segy.py ...` | Regular acquisition: one SEG-Y file per shot (geometry headers, zero traces) and the decision files |
| `python apps/acquisition/check_acquisition.py ...` | Summary and map of an acquisition read from SEG-Y files |
| `python apps/acquisition/decision_create.py ...` | Decision files from SEG-Y files or from a list of traces |

These command lines are in `apps/acquisition`; their functions are in the library (`makutu.acquisition`).
