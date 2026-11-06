Solver application
###################

:program:`makutu` wraps the **FUnTiDES** spectral-element wave solvers (via
:code:`pyfuntides`) behind simplified Python
classes:

    - :code:`AcousticSolver` — acoustic wave equation (SEM), scalar pressure
    - :code:`ElasticSolver`  — elastic wave equation (SEM), 3-component displacement

Both build on a common :code:`WaveSolver` base and inject Kokkos views zero-copy
into the FUnTiDES :code:`SEMsolver`. Mesh building, source/receiver placement,
the explicit time loop and seismogram extraction are driven from Python.

+--------------------------------+------------------------------------------------+---------------------------+
| :program:`makutu` solver       | FUnTiDES kernel                                | Comment                   |
+================================+================================================+===========================+
|       AcousticSolver           |  SEM acoustic (leapfrog)                        | 2-D/3-D, orders 1..3      |
+--------------------------------+------------------------------------------------+---------------------------+
|       ElasticSolver            |  SEM elastic (leapfrog)                         | VTI-parameterised         |
+--------------------------------+------------------------------------------------+---------------------------+

.. autoclass:: makutu.solvers.AcousticSolver.AcousticSolver
    :members:

.. autoclass:: makutu.solvers.ElasticSolver.ElasticSolver
    :members:
