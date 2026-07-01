Model processing
#####################

1. Conversion SEP to VTK
===========================
The script :code:`SeptoVtk.py` in the :code:`workflows` directory can be used to convert easily models from SEP format (.H) to VTK format (.vtu, .vts, .pvtu, .pvts) thanks to a parser. 

You can access the available options and parameters by calling the parser help:

.. code-block:: bash

    python3 SeptoVtk.py --help


.. code-block:: bash

    usage: Model Conversion Parser [-h] --infile INFILE --out {vts,vtu,pvts,pvtu,H} [--outdir OUTDIR] [--or OUTROOT]
                               [--mf | --pn PROPNAME] [--nb NB] [--nb1 NB1] [--nb2 NB2] [--nb3 NB3] [--gids] [--v]

    optional arguments:
    -h, --help            show this help message and exit

    Format options:
    --infile INFILE       Input file. Name of the input file for the dataset OR file containing the headers and model
                            names required
    --out {vts,vtu,pvts,pvtu,H}, --outformat {vts,vtu,pvts,pvtu,H}
                            Output file format
    --outdir OUTDIR       Output directory if different from input directory
    --or OUTROOT, --outrootname OUTROOT
                            Root for the output file. Required if no infile is given
    --mf                  Flag indicating if the models should be extracted from several files
    --pn PROPNAME, --propname PROPNAME
                            Name of the model property. Only if not multifiles

    Partition options:
    --nb NB, --nblocks NB
                            Number of blocks
    --nb1 NB1             Number of blocks for the dimension 1
    --nb2 NB2             Number of blocks for the dimension 2
    --nb3 NB3             Number of blocks for the dimension 3

    Other options:
    --gids, --globalIds   Generate global IDs for the output files
    --v, --verbose        Print conversion information


You will find below some examples of command line for the different types of model conversion, for sequential and parallel execution. Please refer to `Tutorial 1 <../tutorial/tutorial1.html>`_ for a practical example.


**Don't forget to install the `utilities` package of the makutu repository before trying the following command lines.**
(see `Configuration <../configuration/configuration.html>`_)


1.1 Convert several dataset from SEP to VTK format
---------------------------------------------------

.. important:: Here we consider the case where different properties are depicted on the same SEP mesh grid. The mesh grid of all input files should be identical !!

Sequential case
++++++++++++++++

.. code-block:: bash

    python3 SeptoVtk.py --out vtu --outrootname=bicouche100x100x100 --mf \
    --infile=datafile.txt --v

with :code:`datafile.txt` a file containing the names of the datasets in the first columns and the relative or absolute paths to the corresponding files in the second column:

.. code-block:: bash

    cat > datafile.txt <<!
    mediumVelocity velBicouche100x100x100.H
    mediumDensity densBicouche100x100x100.H
    !

If no :code:`outdir` argument is given, the files are created at the same place as the input file (:code:`datafile.txt`). Otherwise, the :code:`outdir` and :code:`outrootname` will be joined for the creation of the file.

For the conversion to a partitioned VTK file, the number of blocks in each direction should be given by the arguments :code:`nb1`, :code:`nb2` and :code:`nb3`. 

.. code-block:: bash

    python3 SeptoVtk.py --out pvtu --outrootname=bicouche100x100x100 --mf \
    --infile=datafile.txt  --nb1 2 --nb2 2 --nb3 2 --v


Parallel execution
+++++++++++++++++++

Finally, for a faster conversion, it is possible to parallelize the conversion by using mpirun

.. code-block:: bash

    mpirun -np 4 python3 SeptoVtk.py --out pvtu --outrootname=bicouche100x100x100 --mf \
    --infile=datafile.txt --nb1 2 --nb2 2 --nb3 2 --v

.. Note:: As you can see, the number of processes, given by :code:`np` and the number of blocks is not linked. However, please note that if you plan to use the resulting PVTK file in GEOSX, the number of processes at that time will have to be a multiple of the number of blocks.


1.2. Convert one model from SEP to VTK format
------------------------------------------------

If you need to convert only one model from SEP (.H) format to VTK format, a shortcut to the creation of the datafile.txt is to give directly the input file and property name in the command line. 

.. code-block:: bash

    python3 SeptoVtk.py --infile=velBicouche100x100x100.H --out pvtu \
     --pn=mediumVelocity --nb1 2 --nb2 2 --nb3 2 --v

Although they are not required arguments in this case, you can still define a new name and path *via* the :code:`outrootname` and :code:`outdir` arguments. By default, the output path and rootname will be those of the input file: :code:`velBicouche100x100x100.vtu` in the above example.

.. code-block:: bash

    python3 SeptoVtk.py --infile=velBicouche100x100x100.H --out vtu --pn=mediumVelocity --v


2. Invert hexahedra cells
===========================
As :program:`GEOS` cannot handle negative cell volume, you may need to invert the hexahedral cells of your mesh to obtain only positive values. The script :file:`invertHexCells.py` is the way to do it.

.. code-block:: bash

    python3 invertHexCells.py --infile inputMesh.vtu --outfile invertedMesh.vtu


3. Add SEP data to an existing VTU mesh
=========================================
The script :file:`sepDataToCellData.py` adds data array to a VTU file. The array is extracted from a SEP file and included in the VTU file as a cell data array.

An example of command line:

.. code-block:: bash

    python3 sepDataToCellData.py --infile mediumVelocity.H --outfile mesh.vtu --propname mediumVelocity


4. Convert tetrahedral to hexagonal cells
===========================================
The script :file:`tetToHex.py` transforms a tetrahedral cells mesh into a hexahedral cells one.

.. code-block:: bash

    python3 tetToHex.py --infile tetMesh.vtu --outfile hexMesh.vtu


5. Transfer cell data between two VTU files
=============================================
The transfer of a cell data array from one VTU file to another is possible **via** the script :file:`transferCellData.py`. It extracts all the cell data from the input mesh and add it to the existing cell data of the output file. 

.. code-block:: bash

    python3 transferCellData.py --infile meshIn.vtu --outfile meshOut.vtu