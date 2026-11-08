Parallel runs (MPI)
###################

Two modes, chosen by the construction of the solver:

- **Domain decomposition** (modelling, gradient): the mesh and the solver are built with
  :code:`comm=MPI.COMM_WORLD` (3-D, :class:`makutu.mesh.StructuredMesh`); every rank holds a
  slab along x and computes every shot. The seismograms and the gradient match the
  single-process ones (tests with 2 and 3 ranks).
- **Shot parallelism** (modelling, FWI): the solver is built without communicator; every
  rank holds the whole model and the shots are dealt out to the ranks
  (:code:`shots[rank::size]`). The cost, the gradient and the illumination are summed over
  the ranks, so that every rank takes the same optimisation steps; the rank that computes
  a shot exports it, rank 0 writes the models and the cost history
  (:mod:`makutu.forwardBackward.parallel`).

.. code-block:: bash

    mpirun -np 4 python apps/fullWaveformInversion/fwi.py --param_file fwi_parameters.txt

The FWI refuses the domain decomposition (the model would be distributed).

The shot parallelism pays with one GPU per rank. On one GPU the ranks compete for it and
the run is slower: anomaly_2d 78 s on 2 ranks against 38 s on one, Marmousi 2-D about
95 s per iteration on 4 ranks against 43 s (same cost history within the float32 noise).
:file:`fwi.py` and :file:`run_shot_modeling.py` give each rank one GPU before Kokkos starts
(:func:`makutu.tools.gpu.select_gpu`): the GPU of its rank on the node (OpenMPI, MPICH, Slurm
variables) among the devices listed in :code:`CUDA_VISIBLE_DEVICES` by the scheduler, or
among the GPUs of the node; a single device set by the user is kept.
