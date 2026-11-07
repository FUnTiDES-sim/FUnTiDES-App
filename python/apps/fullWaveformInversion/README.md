# Full-waveform inversion

`fwi.py` inverts acoustic velocities from observed SEG-Y data with makutu on the FUnTiDES
SEM solvers: least-squares cost function, gradient by the adjoint-state method
(`makutu.imaging.gradf`), optimisers of `makutu.imaging.Minimizer`.

```bash
# from the run directory, after source <FUnTiDES install>/funtides_env.sh
python <FUnTiDES-App>/python/apps/fullWaveformInversion/fwi.py --param_file fwi_parameters.txt
python .../fwi.py --param_file fwi_parameters.txt --restart --rfile velocityModel3.hdf5   # restart
```

Single process (the model files are not distributed over MPI ranks yet).

## Inputs

- **Initial model**: SEP file `vpFile` (2-D if n3 = 1: axis 1 = x, axis 2 = z; else
  x, y, z; z = 0 at the surface, downwards). It sets the domain, the mesh (`order`, `ppw`
  or `elemSize`) and the time step (`dt`, or the CFL condition).
- **Observed data**: one SEG-Y file per shot in `dataDirectory`, named
  `<rootname><shot id><extension>` (`dinfo`, default `seismoTrace_shot;5;.sgy`). The
  acquisition is read from their trace headers (sx, sy, sdepth, gx, gy, gelev with their
  scalars, as written by makutu's `SEGYTraceOutput`), or from a decision file
  (`decision`). The shot ids follow the sorted file names.
- **Parameters**: `fwi_parameters_example.txt` lists them; `minimizerOptions` is a Python
  dict with the options of the optimiser and the velocity bounds `minModel`, `maxModel`
  [m/s].

## Method

- Parametrisation `model`: `c`, `1/c` or `1/c2` (the bounds are converted).
- `freqFilterList`: the data and the source are low-pass filtered at each frequency in
  turn, each stage starting from the model of the previous one (multi-scale).
- `method`: `steepestDescent` (default) or `CGPR2`, heuristic line search (first step 1 %
  of the model, option `alpha`); `SD2` or `CGPR`, Wolfe line search
  (`scipy.optimize.line_search`); or any `scipy.optimize.minimize` method, such as
  `L-BFGS-B`. The Wolfe and scipy methods need `gDn=0` (true gradient).

## Outputs

- `velocityModel<n>.hdf5` (dataset `velocity`, parametrisation `model`) after each
  iteration, `finalModel.hdf5` at the end; flat per element, z layer 0 at the surface.
- `fullCostFunction.txt`: cost function per iteration.
- With `--dbg`: full gradients in `gradDir`; `--dbg+` also exports the simulated traces
  and the residuals.

## Status

Checked on a small 2-D case (3 shots): the cost function decreases. The gradient is not
preconditioned yet (it is dominated by the source and receiver areas) and the absorbing
terms are not differentiated; see `../README.md`.
