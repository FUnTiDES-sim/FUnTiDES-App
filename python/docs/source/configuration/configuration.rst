Quickstart
############
Requirements
==============
Python 3.10 or above, with:

.. code-block:: bash

    python3 -m pip install numpy scipy matplotlib mpi4py h5py segyio

(a few workflows additionally use :code:`vtk` / :code:`pyevtk` / :code:`numba`).

:program:`makutu` drives the **FUnTiDES** solvers provided by proxy-FUn's
:code:`FUn-Libs`, imported as :code:`pyfuntides`. These — and the :code:`kokkos`
runtime — are **not** pip packages: they come from the FUn-Libs build and the
FUn-TPL prefix.


Building FUnTiDES (FUn-Libs)
=============================
Clone :code:`proxy-FUn` as a **sibling** of this repository and build
:code:`FUn-Libs` with the :code:`pyfuntides` wrappers::

    numpex/
    ├── proxy-FUn/     # FUnTiDES: FUn-TPL + FUn-Libs (+ pyfuntides)
    └── makutu-lib/    # this repository

.. code-block:: bash

    cd ../proxy-FUn
    FUN_BACKEND=cuda FUN_CUDA_ARCH=<your-arch> ./bootstrap.sh   # SEM (GPU)
    # ./bootstrap.sh                                            # serial CPU (FD only)

.. IMPORTANT:: The spectral-element (SEM) acoustic/elastic solvers require the
    **CUDA** FUn-Libs build (the mesh builder allocates CudaUVMSpace views).


Environment
=============
Source :file:`env.sh` — it wires :code:`pyfuntides` + :code:`kokkos` and puts
this repository on :code:`PYTHONPATH` so :code:`import utilities` /
:code:`import workflows` resolve:

.. code-block:: bash

    source env.sh
    # target a specific FUn-Libs build (the CUDA build for SEM):
    FUN_LIBS_BUILD=../proxy-FUn/FUn-Libs/build-cuda source env.sh

There is no :code:`pip install` / :code:`setup.py` — :code:`env.sh` alone makes
the package importable.

Tutorials and examples are available in the `next section
<../example/example.html>`_ to help you start.


Building the documentation
============================
Install Sphinx, then build the HTML docs:

.. code-block:: bash

    python3 -m pip install sphinx sphinx_rtd_theme sphinx_prompt
    make -C docs html

Open :file:`docs/build/html/index.html` and navigate through the documentation.
