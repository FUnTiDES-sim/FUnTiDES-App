Quickstart
############
Requirements
==============
Python 3.8 or above is suggested for a correct use of the codes.

The first step is to install :program:`Python`. Please download the package source from the official website, then you can configure and install it with the following command lines:

.. code-block:: bash

    cd Python3.8.13/
    ./configure --enable-shared --prefix=/your/desired/path/for/this/python/install
    make
    make install

**Note :** You need to adapt the paths to your own device.

Then you can install the required packages:

.. code-block:: bash
    
    cd /your/desired/path/for/this/python/install
    ./bin/python3 -m pip install scipy numpy mpi4py setuptools segyio virtualenv xmltodict numba pyevtk matplotlib

Some of the codes are using the proprietary :code:`sabl` module (client-server scripts, decision files reader).


Installing the PyGEOSX environment
===================================
The use of this repository's modules requires the :code:`PyGEOSX` environment to be build beforehand.

You should follow the steps of GEOS installation from the main code `documentation <https://geosx-geosx.readthedocs-hosted.com/en/latest/docs/sphinx/QuickStart.html>`_ 


**The following lines should be added to the host config file in order to activate the build of PyGEOSX enviromnent**

::

    set(ENABLE_PYGEOSX ON CACHE BOOL "")
    set(Python3_EXECUTABLE "/path/to/your/python/bin/python3.8" CACHE PATH "")

    set(ENABLE_VTK ON CACHE BOOL "")


Once that :program:`GEOS` has been build, make the :code:`PYGEOSX` environment (adapt the number of parallel processes with :code:`-j N` to speed up the build according to your machine's resources):

.. code-block:: bash

    make -j 4 pygeosx

.. IMPORTANT:: The makutu codes rely on :code:`PYGEOSX` environment, you have to activate it every time you want to use it with the following command:

    .. code-block:: bash

        source /path/to/geos/build/dir/lib/PYGEOSX/bin/activate


Installing the module
=========================
Installing the :code:`utilities` module is done by executing the following command line

.. code-block:: bash

    cd makutu
    python3 -m pip install -e .


You are now ready to use our modules for your application !

Tutorials and examples are availables in the `next section <../example/example.html>`_ to help you start.



Building the documentation
============================

In order to build the sphinx documentation, the following modules are required in your :program:`Python` enviromnent:

.. code:: bash

    python3 -m pip install sphinx sphinx_rtd_theme sphinx_prompt


Then **activate** the :code:`PYGEOSX` enviromnent, and execute the command :code:`make docs` in :file:`makutu` repository.

Open the file :file:`makutu/docs/build/index.html` and navigate through the documentation.