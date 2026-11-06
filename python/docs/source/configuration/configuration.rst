Quickstart
############
Requirements
==============
Python 3.10 or above, with:

.. code-block:: bash

    python3 -m pip install numpy scipy matplotlib mpi4py h5py segyio vtk pyevtk numba

(:code:`utilities.model` imports :code:`vtk` and :code:`numba`).

:program:`makutu` drives the **FUnTiDES** solvers, imported as
:code:`pyfuntides`. These — and the :code:`kokkos` runtime — are **not** pip
packages: they come from the FUnTiDES installation and its third-party libraries
(FUnTiDES-TPL, which also provides the Python virtual environment).


Building FUnTiDES
==================
Install FUnTiDES-TPL and FUnTiDES (with :code:`-DENABLE_PYWRAP=ON`) into the
same prefix, next to this repository (see the FUnTiDES README)::

    FUnTiDES-dir/
    ├── FUnTiDES-TPL/      # third-party libraries
    ├── FUnTiDES/          # FUnTiDES sources
    ├── FUnTiDES-install/  # TPL + FUnTiDES install, with funtides_env.sh
    └── makutu-lib/        # this repository

.. code-block:: bash

    FUnTiDES-TPL/install.sh --prefix=$(pwd)/FUnTiDES-install --enable-cuda --enable-venv
    source FUnTiDES-install/setup_env.sh
    cmake -S FUnTiDES -B FUnTiDES/build -DCMAKE_BUILD_TYPE=Release -DENABLE_PYWRAP=ON \
          -DCMAKE_INSTALL_PREFIX=$(pwd)/FUnTiDES-install
    cmake --build FUnTiDES/build -j 8 && cmake --install FUnTiDES/build

.. NOTE:: The Kokkos arrays of makutu follow the FUnTiDES build
    (:code:`CudaUVMSpace` + :code:`LayoutLeft` with CUDA). makutu is tested with
    the CUDA build.


Environment
=============
Source :file:`env.sh` — it sources :file:`FUnTiDES-install/funtides_env.sh`
(:code:`kokkos`, :code:`pyfuntides`, virtual environment) and puts this
repository on :code:`PYTHONPATH` so :code:`import utilities` /
:code:`import workflows` resolve:

.. code-block:: bash

    source env.sh
    # use another FUnTiDES installation:
    FUNTIDES_ENV=/path/to/funtides_env.sh source env.sh

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
