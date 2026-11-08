makutu documentation
####################

:code:`makutu` is a Python library for seismic applications on the **FUnTiDES**
spectral-element solvers (through their :code:`pyfuntides` bindings): forward modelling
of shots in 2-D and 3-D, gradient of the least-squares misfit by the adjoint-state method,
and acoustic full-waveform inversion (FWI).

It is the :file:`python/` directory of FUnTiDES-App:

- :file:`makutu/`: the library (mesh and model, acquisition, solvers, forward and adjoint
  runs, imaging and optimisation, outputs, display);
- :file:`apps/`: applications (shot modelling, FWI, display scripts) and their examples;
- :file:`tests/`: unit tests (no solver) and integrated tests (FUnTiDES on the GPU, MPI).

.. toctree::
   :maxdepth: 2
   :caption: Contents

   configuration/configuration
   userguide/userguide
   example/example
   api/api
   status

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
