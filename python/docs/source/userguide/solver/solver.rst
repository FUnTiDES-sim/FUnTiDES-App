Solver application
###################

The repository currently have 4 GEOS solvers wrapped:

    - :code:`AcousticSolver`
    - :code:`ElasticSolver`
    - :code:`ReservoirSolver`
    - :code:`GeoMechanicsSolver`

These classes gives simplified access to :program:`GEOS` fields of interest. As GEOS has deployed a generic :code:`PySolver` interface, this list can be extended to other :program:`GEOS` solvers easily.

+--------------------------------+------------------------------------------------+---------------------------+
|:program:`Makutu` solver name   |  Currently wrapped solvers in :program:`GEOS`  | Comment                   |
|                                |                                                |                           |
+================================+================================================+===========================+
|       AcousticSolver           |  - AcousticFirstOrderWaveEquationSEM           | First and second order    |
|                                |  - AcousticWaveEquationSEM                     |                           |
+--------------------------------+------------------------------------------------+---------------------------+
|       ElasticSolver            |  - ElasticFirstOrderWaveEquationSEM            |                           |
|                                |  - ElasticWaveEquationSEM                      |                           |
+--------------------------------+------------------------------------------------+---------------------------+
|       ReservoirSolver          |  - CompositionalMultiphaseFVM                  |                           |
+--------------------------------+------------------------------------------------+---------------------------+
|       GeoMechanicsSolver       |  - SinglePhasePoromechanics                    |                           |
+--------------------------------+------------------------------------------------+---------------------------+

.. autoclass:: utilities.solvers.AcousticSolver.AcousticSolver
    :members: