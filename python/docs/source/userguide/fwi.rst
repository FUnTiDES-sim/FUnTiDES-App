Full-waveform inversion
#######################

.. figure:: ../example/schema_fwi.png
    :scale: 60
    :align: center

The FWI minimises the least-squares misfit between observed and simulated seismograms,
J(m) = sum over the shots of dt / 2 * sum(residual\ :sup:`2`), from an initial model m0.
The application is :file:`apps/fullWaveformInversion/fwi.py` (:doc:`../example/fwi`); its
building blocks are below.

Cost and gradient
=================

The state of an inversion is a :class:`makutu.imaging.FWIProblem`: solver, acquisitions,
model bounds, filter frequencies, run mode, current model, illumination, and the counters
that number the output files. Its methods have the signatures of the optimisers:

- :code:`problem.func(x, data directory, ...)` (:func:`makutu.forwardBackward.func`) returns J;
- :code:`problem.gradf(x, ..., computeFullCF=True)` (:func:`makutu.imaging.gradf`) returns the
  gradient and J (one forward run with snapshots and one adjoint run per shot), and sums the
  illumination in :code:`problem.illumination`;
- :code:`problem.callback(x)` writes :code:`velocityModel<problem.iteration>.hdf5` after an
  iteration.

Both read the observed data (one SEG-Y file per shot, named by :code:`dinfo`), filter the
source and the data at the current frequency, and apply the model x in memory.

.. code-block:: python

    from makutu.imaging import FWIProblem, GradientConditioner, Minimizer

    problem = FWIProblem(solver, acquisition, minModel=1400.0, maxModel=5000.0, freqList=[4.0, 6.0])
    precond = GradientConditioner(solver, minDepth=50.0, illumination=True, problem=problem)
    minimizer = Minimizer(problem.func, method="CGPR", jac=problem.gradf, callback=problem.callback,
                          bounds=[[1400.0, 5000.0]] * len(x0), options={"maxiter": 10}, precond=precond)

To resume a run, give the model file, the frequency and the number of the next model file
(:code:`problem.iteration`); with :file:`fwi.py`: :code:`--restart --rfile velocityModel23.hdf5
--restart_freq 3 --restart_iter 24`. The optimisation of that frequency starts again from
the model (the conjugate-gradient history is not kept).

Verified (integrated tests): gradient against finite differences (2-D orders 1 to 3, 3-D,
the three parametrisations), Taylor test (second-order remainder), source-receiver
reciprocity, MPI.

Multi-scale
===========

:code:`freqFilterList` gives the filter frequencies of successive stages, each starting
from the model of the previous one. The source and the data are low-pass filtered by the
same causal Butterworth filter (:func:`makutu.tools.frequencyFilter`, -3 dB at the
frequency): filtering the source before the simulation then gives the filtered data (0.1 %).

Optimisers
==========

:class:`makutu.imaging.Minimizer` (method name):

- :code:`steepestDescent`, :code:`CGPR2`: heuristic line search (trial steps, the best
  kept if it lowers J);
- :code:`SD2`, :code:`CGPR` (Polak-Ribière+): Wolfe line search
  (:func:`scipy.optimize.line_search`), the first step scaled on the last decrease of J; J
  is computed with the gradient and the gradient of the accepted step is reused (about one
  gradient per iteration);
- any :func:`scipy.optimize.minimize` method (e.g. :code:`L-BFGS-B`).

The model is kept within the bounds; the iterations stop on the relative decrease of J
(:code:`Jtol`), on a stagnation of J, or after :code:`maxiter`.

Conditioning
============

:class:`makutu.imaging.GradientConditioner` builds the descent direction (the line searches
keep the true gradient): no update above :code:`minDepth` (water, sources and receivers)
and within :code:`edge` of the sides and the bottom (absorbing terms not differentiated),
cosine ramps of :code:`taper`, and optionally division by the illumination
(pseudo-Hessian, stabilised by :code:`eps` times its maximum).
