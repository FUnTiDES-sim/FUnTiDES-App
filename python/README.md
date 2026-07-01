# Welcome to makutu repository for pygeosx

This code is an example of Python interface of GEOSX multiphysics [code](https://geosx-geosx.readthedocs-hosted.com/en/latest/docs/sphinx/QuickStart.html), focused on the seismic applications. 


As we are currently at the development stage, don't hesitate to give us feedback or create an issue if you encounter any problem !

## Requirements
The first step is to install `Python`. Python version 3.8 or above is suggested for a correct usage of the codes. 
Please download the source package from the official website, then configure and install it with the following command lines:

```bash
cd Python3.8.13/
./configure --enable-shared --prefix=/your/desired/path/for/this/python/install
make
make install
```

Then install the required packages:

```bash
cd /your/desired/path/for/this/python/install
./bin/python3 -m pip install scipy numpy mpi4py setuptools segyio virtualenv xmltodict numba pyevtk matplotlib
```

Note that some codes are using the proprietary `sabl` module (client-server scripts, decision files reader).

## Install the PyGEOSX environment
The use of this repository's modules requires the PyGEOSX module to be build beforehand.

You should follow the steps of GEOS installation from the main code [documentation](https://geosx-geosx.readthedocs-hosted.com/en/latest/docs/sphinx/QuickStart.html).

**You need to add the following lines to the host config file in order to activate the build of PyGEOSX enviromnent**

```bash
set(ENABLE_PYGEOSX ON CACHE BOOL "")
set(Python3_EXECUTABLE "/path/to/your/python/bin/python3.8" CACHE PATH "")

set(ENABLE_VTK ON CACHE BOOL "")
```

Once that `GEOS` has been build, make the `PYGEOSX` environment (adapt the number of parallel processes `-j` as you like):

```bash
make -j 8 pygeosx
```

Then before executing the codes, you need to activate the `PYGEOSX` environment.

```bash
source /path/to/geosx/build/dir/lib/PYGEOSX/bin/activate
```

## Install the module
----------------------
Installing the `utilities` module is done by executing the following command line

```bash
cd makutu
python3 -m pip install -e .
```

**Remember that the PyGEOSX environment should be activated**

## How to generate the documentation
The documentation can be generated using Sphinx using the following command line:

```bash
source /path/to/GEOSX/build/dir/lib/PYGEOSX/bin/activate
cd makutu
make docs
```

You need to activate `PYGEOSX` environment beforehand OR give the path as argument in the make command line such that:

```bash
make docs venv=/path/to/GEOSX/build/dir/lib/PYGEOSX
```


Note: the `sphinx` and `sphinx-prompt` python packages are required to generate the doc.
  

The documentation is generated in the `docs/build` folder in html format. You can open the file `index.html` to navigate through the docs.
