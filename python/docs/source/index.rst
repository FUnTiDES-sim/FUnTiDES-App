.. Makutu-pygeosx documentation master file, created by
   sphinx-quickstart on Wed Oct 26 19:34:28 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MAKUTU repository documentation
=========================================

The :code:`makutu` code is a library focused on seismic application. It is relying on the Python interface of `GEOS <https://geosx-geosx.readthedocs-hosted.com>`_ multiphysics code. You will find a variety of functions and classes to build your own applications.



Organisation
-----------------------
The repository is organized into different sub-modules:
   - :code:`utilities`: module that contains seismic specifics PyObjects and utilities. This module relies on the use of PyGEOSX, the python generic interface developed by GEOS.
   - :code:`workflows`: examples of use-case "main" scripts and basic visualization scripts.
   - :code:`launcher`: contains scripts that help the launching of jobs on HPC clusters.






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
