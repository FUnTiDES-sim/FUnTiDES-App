Shot modelling
##############

Application
===========

:file:`apps/modeling/shot_modeling/run_shot_modeling.py` computes shots on a SEP model,
2-D or 3-D: mesh and time step from the model and the source frequency, one shot, a line
of shots or the shots of a decision file; gathers in HDF5 or SEG-Y, figures. See its
README for the options and :doc:`../example/shot_modelling` for an example.

Library
=======

:func:`makutu.forwardBackward.directSimulation` computes the shots of an acquisition of a
:class:`makutu.imaging.FWIProblem` (solver, acquisitions, model bounds, filter
frequencies), exports the seismograms (SEG-Y, :code:`seismoTrace_shot<id>_<f0>.sgy`) and
optionally the cost function against observed data:

.. code-block:: python

    from makutu.imaging import FWIProblem

    problem = FWIProblem(solver, acquisition)
    problem.directSimulation(0, exportSeismicTrace=True, traceDirectory="obs")

With several MPI ranks the shots are dealt out to the ranks (see :doc:`parallel`).
