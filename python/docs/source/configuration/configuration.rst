Quickstart
############
Requirements
==============
Python 3.10 or above, with:

.. code-block:: bash

    python3 -m pip install numpy scipy matplotlib mpi4py h5py segyio vtk pyevtk numba

(:code:`makutu.model` imports :code:`vtk` and :code:`numba`).

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
    └── FUn-examples/      # example data and models

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
Source the FUnTiDES environment (:code:`kokkos`, :code:`pyfuntides`, virtual
environment), then install makutu once, in editable mode:

.. code-block:: bash

    source FUnTiDES-install/funtides_env.sh
    cd FUnTiDES-App/python
    pip install -e ".[test]"

:code:`import makutu` then works from any directory; the applications are in
:file:`apps/`.

Tutorials and examples are available in the `next section
<../example/example.html>`_ to help you start.


Building the documentation
============================
Install Sphinx, then build the HTML docs:

.. code-block:: bash

    python3 -m pip install sphinx sphinx_rtd_theme sphinx_prompt
    make docs        # from FUnTiDES-App/python

Open :file:`docs/build/html/index.html` and navigate through the documentation.
