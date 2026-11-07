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
  turn, each stage starting from the model of the previous one (multi-scale). The filter
  is a causal Butterworth (-3 dB at the frequency), so that the filtered data and the data
  simulated with the filtered source agree (0.1 %; the former zero-phase FFT filter rang
  before t = 0 and gave 24 % of difference).
- `method`: `steepestDescent` (default) or `CGPR2`, heuristic line search (first step 1 %
  of the model, option `alpha`; the best trial step is kept if it lowers J); `SD2` or
  `CGPR`, Wolfe line search (`scipy.optimize.line_search`, step 1 scaled on the last
  decrease of J; J is computed with the gradient at each trial step and the gradient of
  the accepted step is reused: about one gradient, i.e. two forward runs, per iteration;
  stop when J changes by less than `Jtol` times the first J); or any
  `scipy.optimize.minimize` method, such as `L-BFGS-B`. The Wolfe and scipy methods need
  `gDn=0` (true gradient).
- Gradient conditioning (methods of makutu only), `makutu.imaging.GradientConditioner`:
  no update above `minDepth` and within `gradEdge` of the sides and the bottom (the
  absorbing terms are not differentiated: the gradient is exact inside, wrong on the
  edge elements), cosine ramps of `gradTaper`; `illumination=true` divides by the energy
  of the forward field (pseudo-Hessian, source side, stabilised by `illumEps`).
- Absorbing edges: the absorbing condition of FUnTiDES (about 2 % error on the
  seismograms), and optionally a sponge of width `sponge` [m] on the sides and the bottom
  (not along the free surface), useful from about two wavelengths (0.5 % with two).

## Outputs

- `velocityModel<n>.hdf5` (dataset `velocity`, parametrisation `model`) after each
  iteration, `finalModel.hdf5` at the end; flat per element, z layer 0 at the surface.
- `fullCostFunction.txt`: cost function per iteration.
- With `--dbg`: full gradients in `gradDir`; `--dbg+` also exports the simulated traces
  and the residuals.

## Status

The phases are tested separately (`tests/`): acquisition, modelling, filtering (source
and data of a multi-scale stage), residual and cost, gradient (finite differences, Taylor
test, reciprocity, edges, MPI), conditioning, optimisers (quadratic cost, number of
evaluations).

First complete inversions on `examples/anomaly_2d` (two Gaussian anomalies of +-250 m/s
above a known reflector, 8 shots, 73 receivers at the surface; `make_case.py` writes the
models, the observed data and the parameters, `check_result.py` compares the result with
the true model), CGPR, `minDepth=50`, `gradEdge=50`, `gradTaper=50`, `illumination=true`,
same mesh for the data and the inversion:

| Run | Time (GPU) | J / J0 | Anomalies recovered | Model error above 750 m |
|---|---|---|---|---|
| no filter, 15 iterations | 4 min | 0.012 | +92, -104 m/s | 0.71 |
| `freqFilterList=5,8,12`, 10 iterations each | 6.5 min | 0.05, 0.44, 0.49 | +162, -155 m/s | 0.66 |

The anomalies are found at the right place (within 15 m), smeared vertically (surface
acquisition); part of the traveltime shifts goes into the reflector (velocities up to
2830 m/s there).
