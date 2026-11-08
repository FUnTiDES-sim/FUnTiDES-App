Quickstart
############
Requirements
==============
Python 3.10 or above, with:

.. code-block:: bash

    python3 -m pip install numpy scipy matplotlib mpi4py h5py segyio

:program:`makutu` drives the **FUnTiDES** solvers, imported as
:code:`pyfuntides`. These — and the :code:`kokkos` runtime — are **not** pip
packages: they come from the FUnTiDES installation and its third-party libraries
(FUnTiDES-TPL, which also provides the Python virtual environment).


Building FUnTiDES
==================
Install FUnTiDES-TPL and FUnTiDES (with :code:`-DENABLE_PYWRAP=ON`) into the
same prefix (see the FUnTiDES README); makutu is the :file:`python/` directory of
FUnTiDES-App::

    FUnTiDES-dir/
    ├── FUnTiDES-TPL/      # third-party libraries
    ├── FUnTiDES/          # FUnTiDES sources
    ├── FUnTiDES-install/  # TPL + FUnTiDES install, with funtides_env.sh
    ├── FUnTiDES-App/      # C++ drivers + python/ (makutu library, apps, tests)
    └── FUnTiDES-data/      # example data and models

.. code-block:: bash

    FUnTiDES-TPL/install.sh --prefix=$(pwd)/FUnTiDES-install --enable-cuda --enable-venv
    source FUnTiDES-install/setup_env.sh
    cmake -S FUnTiDES -B FUnTiDES/build -DCMAKE_BUILD_TYPE=Release -DENABLE_PYWRAP=ON \
          -DCOMPILE_SEM_2D=ON -DCMAKE_INSTALL_PREFIX=$(pwd)/FUnTiDES-install
    cmake --build FUnTiDES/build -j 8 && cmake --install FUnTiDES/build

.. NOTE:: The Kokkos arrays of makutu follow the FUnTiDES build
    (:code:`CudaUVMSpace` + :code:`LayoutLeft` with CUDA). makutu is tested with
    the CUDA build. The 2-D solvers need :code:`-DCOMPILE_SEM_2D=ON`; with CUDA, limit the
    parallel build (ptxas needs about 10 GB for an order-7 solver file) or the compiled
    orders (:code:`-DMAX_SOLVER_ORDER=4`).


Environment
=============
Source the FUnTiDES environment (:code:`kokkos`, :code:`pyfuntides`, virtual
environment), then install makutu once, in editable mode:

.. code-block:: bash

    source FUnTiDES-install/funtides_env.sh
    cd FUnTiDES-App/python
    pip install -e ".[test]"

:code:`import makutu` then works from any directory; the applications are in
:file:`apps/`.

Tests
=====

From :file:`FUnTiDES-App/python`, after sourcing the environment:

.. code-block:: bash

    make unitTests          # about 2 s, no solver
    make integratedTests    # about 3 min on the GPU (FUnTiDES built with COMPILE_SEM_2D=ON, mpirun)

On the GPU the float32 sums are not reproducible from one run to the next (about
1e-5 on the traces): the tests compare with thresholds relative to the energy of the data.


Building the documentation
============================
Install Sphinx in the same environment, then build the HTML pages:

.. code-block:: bash

    python3 -m pip install sphinx sphinx_rtd_theme sphinx-prompt
    make docs        # from FUnTiDES-App/python

Open :file:`docs/build/index.html`. FUnTiDES is not needed for the documentation (the
:code:`kokkos` and :code:`pyfuntides` imports are mocked).
