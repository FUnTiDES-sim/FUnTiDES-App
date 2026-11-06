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
The forward part of the inversion chain (`makutu.forwardBackward`: shots of an
acquisition, SEG-Y export, residual against observed SEG-Y data, cost function, `func`
for the optimiser) runs on FUnTiDES and is tested (`test_forward_cost.py`, and with MPI in
`test_mpi_decomposition.py`); `func` and `callback` write the model to HDF5 in a single
process only. The adjoint part is not: `backward`/`adjointSimulation` and the gradients
of `makutu.imaging` need a gradient computation that does not exist yet
(`computePartialGradient` raises NotImplementedError), and `fullWaveformInversion/fwi.py`
still builds `AcousticSolver` with its former positional signature (no model or mesh).
