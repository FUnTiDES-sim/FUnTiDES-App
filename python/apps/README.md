# Applications

Scripts built on the `makutu` library (`../makutu`) and the FUnTiDES solvers. Run them
after `source <FUnTiDES install>/funtides_env.sh` and `pip install -e python/` (see
`../README.md`); each script prints its options with `--help`. Example data and models
are in the `FUnTiDES-data` directory next to FUnTiDES-App.

| Path | Content |
|---|---|
| `run_funtides.py` | Runs a C++ driver of FUnTiDES-App (`funtides-<method>`, found in `build*/bin`) from a YAML file of `examples/<method>/` |
| `modeling/` | Forward modelling: `shot_modeling` (2-D or 3-D, one or several shots, decision files), `examples/marmousi_2d` (data preparation), and `tools/` (synthetic SEP models); see its README |
| `display/` | `showVelocityModel.py`, `showSeismicTraces.py`, `showFullGradient.py`: plots of models, traces and gradients |
| `fullWaveformInversion/fwi.py` | Full-waveform inversion workflow; **not yet adapted** to the current solver API (see below) |

```bash
python apps/run_funtides.py ../examples/sem2d/sem2d_acoustic.yaml   # from FUnTiDES-App/python
python apps/modeling/examples/marmousi_2d/prepare_marmousi_2d.py --fix
python apps/modeling/shot_modeling/run_shot_modeling.py --vp-file ../../FUnTiDES-data/SEP/marmousi/Vp_xz.H --f0 8 --order 4 --plot
```

## Status

The modelling path (`makutu.mesh`, `makutu.solvers`, acquisition, outputs) runs on
FUnTiDES and is covered by `tests/integratedTests` (3-D acoustic and elastic, MPI, 2-D).
The inversion chain runs on FUnTiDES in makutu (acoustic, model per element):
- forward part (`makutu.forwardBackward`): shots of an acquisition, SEG-Y export, residual
  against observed SEG-Y data, cost function, `func` for the optimiser;
- gradient (adjoint-state method): `solver.runForward(snapshotStep)` keeps the second time
  derivative of the pressure, `solver.computeGradient(residual, "c" | "1/c" | "1/c2")`
  runs the adjoint (residual injected at the receivers) and applies the imaging
  condition; `makutu.forwardBackward.backward` and `makutu.imaging.gradf` use it over the
  shots. It matches finite differences of the cost function to about 1 % (2-D orders 1-4,
  3-D) and gives the same result with the MPI domain decomposition.

Tests: `test_forward_cost.py`, `test_gradient.py`, `test_mpi_decomposition.py`.
Limits: `func`, `callback` and `gradf` write the model to HDF5 in a single process (with
the MPI decomposition each rank holds its part of the model); the absorbing terms are not
differentiated (gradient exact away from the absorbing boundaries); elastic gradients are
not available. `fullWaveformInversion/fwi.py` still builds `AcousticSolver` with its former
positional signature (no model or mesh) and is not adapted yet.
