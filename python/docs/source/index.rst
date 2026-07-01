.. Makutu documentation master file, created by
   sphinx-quickstart on Wed Oct 26 19:34:28 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MAKUTU repository documentation
=========================================

The :code:`makutu` code is a library focused on seismic applications (forward
modelling, imaging and full-waveform inversion). It drives the **FUnTiDES**
spectral-element solvers — provided by proxy-FUn's :code:`FUn-Libs` — through the
:code:`pyfuntides` Python bindings. You will find a variety of functions and
classes to build your own applications.



Organisation
-----------------------
The repository is organized into different sub-modules:
   - :code:`utilities`: seismic-specific objects and utilities (mesh/model, solvers, acquisition, imaging, I/O). Drives FUnTiDES via :code:`pyfuntides`.
   - :code:`workflows`: example use-case "main" scripts and basic visualization scripts.






.. toctree::
   :maxdepth: 2
   :caption: Contents:

   configuration/configuration
   userguide/userguide
   example/example



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
