# Applications

Scripts built on the `makutu` library (`../makutu`) and the FUnTiDES solvers. Run them
after `source <FUnTiDES install>/funtides_env.sh` and `pip install -e python/` (see
`../README.md`); each script prints its options with `--help`. Example data and models
are in the `FUnTiDES-data` directory next to FUnTiDES-App.

| Path | Content |
|---|---|
| `run_funtides.py` | Runs a C++ driver of FUnTiDES-App (`funtides-<method>`, found in `build*/bin`) from a YAML file of `examples/<method>/` |
| `modeling/examples/` | Forward modelling examples (3-D SEP models, multi-shot acquisition from a decision file, 2-D Marmousi); see its README |
| `display/` | `showVelocityModel.py`, `showSeismicTraces.py`, `showFullGradient.py`: plots of models, traces and gradients |
| `modelProcessing/SepToVtk.py` | Converts SEP models to VTK (vtu / pvtu) |
| `fullWaveformInversion/fwi.py` | Full-waveform inversion workflow; **not yet adapted** to the current solver API (see below) |

```bash
python apps/run_funtides.py ../examples/sem2d/sem2d_acoustic.yaml   # from FUnTiDES-App/python
python apps/modeling/examples/acoustic_marmousi_2d.py
cd apps/modelProcessing && python SepToVtk.py --infile=/path/to/model.H --out vtu --pn=mediumVelocity
```

## Status

The modelling path (`makutu.mesh`, `makutu.solvers`, acquisition, outputs) runs on
FUnTiDES and is covered by `tests/integratedTests` (3-D acoustic and elastic, MPI, 2-D).
The inversion chain is not: `fullWaveformInversion/fwi.py` still builds `AcousticSolver`
with its former positional signature (no model or mesh), and the `makutu.forwardBackward`
and `makutu.imaging` modules it drives (forward/backward runs, gradients, cost function)
have not been run on FUnTiDES yet.
