Solvers
#######

:class:`makutu.solvers.AcousticSolver` (pressure) and :class:`makutu.solvers.ElasticSolver`
(displacement) drive the FUnTiDES SEM solvers; both derive from
:class:`makutu.solvers.WaveSolver`. The Kokkos views of the model, the sources and the
seismograms are shared with FUnTiDES without copy; the time loop runs in Python, one
FUnTiDES step per call.

+-----------------+---------------------------------------------+-----------------------------------+
| Solver          | FUnTiDES                                    | Status                            |
+=================+=============================================+===================================+
| AcousticSolver  | SEM acoustic, 2-D (SEMsolver2D) and 3-D     | modelling, gradient, FWI; MPI     |
+-----------------+---------------------------------------------+-----------------------------------+
| ElasticSolver   | SEM elastic, 3-D                            | modelling only (no gradient)      |
+-----------------+---------------------------------------------+-----------------------------------+

Construction
============

.. code-block:: python

    import pyfuntides.solver as FTSolver
    from makutu.solvers import AcousticSolver

    solver = AcousticSolver(implem_type=FTSolver.ImplemType.MAKUTU, mesh_type=FTSolver.MeshType.UNSTRUCT,
                            model_location=FTSolver.ModelLocationType.ONELEMENTS,
                            dt=m.dt, minTime=0.0, maxTime=3.0, dtSeismo=2e-3,
                            sourceType="ricker2", sourceFreq=8.0, **m.mesh.solver_kwargs())
    solver.initialize()

The seismograms are recorded every round(:code:`dtSeismo` / dt) steps; the actual sampling
is :code:`solver.dtSeismoRecorded`.

One shot
========

.. code-block:: python

    solver.updateSourceAndReceiversFromCoords(shot.getSourceCoords(), shot.getReceiverCoords())
    solver.evaluateSource()          # Ricker wavelet
    solver.filterSource(6.0)         # optional low-pass (multi-scale FWI)
    solver.resetWaveField()
    solver.runForward()              # runForward(snapshotStep) keeps snapshots for the gradient
    p = solver.getPressureAtReceivers()     # (time samples, receivers)

:meth:`AcousticSolver.updateVelocityModel` changes the model (flat per element, seismic
order) and rebuilds the mass and damping matrices.

Absorbing boundaries
====================

The sides and the bottom have the first-order paraxial condition of FUnTiDES (about 2 %
error on the seismograms of a 2-D homogeneous case). A sponge can be added:
:code:`sponge_size=[sx, sy, sz]` (m), :code:`surface_sponge=True` to keep it off the free
surface, :code:`taper_delta` the attenuation of a wave crossing the sponge and coming back
(default 0.01). The sponge reflects little from about two wavelengths: 0.5 % with 400 m and
0.14 % with 600 m at 10 Hz and 2000 m/s; thinner, the paraxial condition alone does better.

Gradient
========

After :code:`runForward(snapshotStep)`, :meth:`AcousticSolver.computeGradient` runs the
adjoint simulation with the residual (observed - simulated) injected at the receivers and
returns the gradient of J = dt / 2 * sum(residual\ :sup:`2`) for the parametrisation
:code:`c`, :code:`1/c` or :code:`1/c2`; :code:`solver.illumination` is the energy of the
forward field (source side of the diagonal pseudo-Hessian).
:meth:`AcousticSolver.snapshotStepFor` gives a safe snapshot step for a maximum frequency.
The terms of the absorbing boundaries are not differentiated: the gradient is exact inside
the domain, not on the edge elements.
