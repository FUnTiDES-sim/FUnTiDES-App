Shot modelling on Marmousi 2-D
##############################

Data: :file:`FUnTiDES-data/SEP/marmousi/Vp_xz.H` (500 x 174 samples at 20 m, 10 km x 3.48 km,
1500-4767 m/s). :file:`apps/modeling/examples/marmousi_2d/prepare_marmousi_2d.py` checks (and
with :code:`--fix` repairs) the SEP files and prints the mesh, the time step and the
modelling command:

.. code-block:: bash

    cd FUnTiDES-App/python
    python apps/modeling/examples/marmousi_2d/prepare_marmousi_2d.py --f0 8 --order 4
    python apps/modeling/shot_modeling/run_shot_modeling.py \
        --vp-file ../../FUnTiDES-data/SEP/marmousi/Vp_xz.H --rho-file ../../FUnTiDES-data/SEP/marmousi/Rho_xz.H \
        --order 4 --f0 8 --elem-size 50 --cfl 0.3 --tmax 4 --dt-seismo 0.002 --rcv-dx 20 \
        --format hdf5 --plot --snap-time 1.5 --output-dir runs/marmousi_2d

200 x 70 elements of order 4 (225 000 nodes), 7 400 time steps: about 3 s per shot on an
RTX 40xx GPU. The outputs are :file:`runs/marmousi_2d/shot_00001.h5` (pressure, source,
receivers, sampling) and a figure (model, snapshot, gather). :code:`--n-shots N` gives a line
of shots, :code:`--decision` the shots of a decision file, :code:`--format segy` SEG-Y files
for :class:`makutu.acquisition.SEGYAcquisition`. The README of
:file:`apps/modeling/shot_modeling` lists all the options.
