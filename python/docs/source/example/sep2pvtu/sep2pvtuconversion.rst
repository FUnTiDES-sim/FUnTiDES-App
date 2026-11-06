Convert a bilayer velocity model from SEP to PVTU format
======================================================================

In this first tutorial, you will learn how to convert a velocity model initially saved in SEP format to a VTK parallel unstructured grid format (pvtu).

::

    makutu
    ├── examples/
        ├── models/
            ├── velBicouche100x100x100.H
            └── velBicouche100x100x100.H@
    ├── apps/modelConversion
        ├── SeptoVtk.py

This is a bilayer model of 100 cells in each dimension, and step sizes of 10 m. You can see the files of interest in the above tree. 

As there is a single dataset to convert, we will provide directly the SEP model input file as well as the name of the dataset in the command line of conversion.

.. code-block:: bash

    cd apps/modelConversion
    mpirun -np 4 python3 SeptoVtk.py --infile=../../examples/models/velBicouche100x100x100.H --outdir=../../modelOut --pn=mediumVelocity --out pvtu --nb1 2 --nb2 3 --nb3 2 --gids --v

With this command line:
    - 4 MPI processes are requested for a parallel writing of the files.
    - The input model will be read from the file `velBicouche100x100x100.H`, and will be named `mediumVelocity` in the output `velBicouche100x100x100.pvtu` file.
    - This velocity model will be split into 12 blocks: 2 blocks in the first dimension, 3 in the second and 2 in the third. Therefore, there will be 12 `vtu` created in addition of the `pvtu` header file.
    - The files created will be saved in the repertory `makutu/modelOut/`
    - The cell and point global ids will be calculated and included in the final VTK.
    - We ask for exhaustive information with the `--v` argument ("verbose" option).

After the execution, the file tree looks like:

::

    makutu
    ├── examples/
        ├── models/
            ├── velBicouche100x100x100.H
            └── velBicouche100x100x100.H@
    ├── apps/modelConversion/
        ├── SeptoVtk.py
    ├── modelOut/
        ├── velBicouche100x100x100_0.vtu
        ├── velBicouche100x100x100_1.vtu
        ├── velBicouche100x100x100_2.vtu
        ...
        ├── velBicouche100x100x100_10.vtu
        ├── velBicouche100x100x100_11.vtu
        └── velBicouche100x100x100.pvtu

You can check with Paraview (or directly in the `.pvtu` file) that the files have been correctly created, with three arrays: the velocity model named `mediumVelocity`, and the global cells and points Ids names respectively `GlobalCellIds` and `GlobalsPointsIds`.

.. Important:: Remember that if you want to use a pvtu file in GEOSX (or pvts), you need to choose the number of MPI processes in coherence with the total number of sub-blocks (here, you will need 12 or a multiple of 12 MPI processes as there are 12 sub-blocks).