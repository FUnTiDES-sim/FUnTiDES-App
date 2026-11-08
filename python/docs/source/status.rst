Status
######

State of makutu on FUnTiDES (November 2026).

Working and tested
==================

- Meshes and models: 2-D (:class:`~makutu.mesh.CartesianMesh2D`) and 3-D
  (:class:`~makutu.mesh.StructuredMesh`, MPI domain decomposition along x), from SEP files,
  sizing for a source frequency.
- Acquisitions: coordinates, SEG-Y headers, decision files, regular grids; DAS geometry.
- Acoustic modelling, 2-D and 3-D, on one or several MPI ranks (domain decomposition or
  shots dealt out to the ranks); elastic modelling in 3-D.
- Acoustic gradient by the adjoint-state method (finite differences, Taylor test,
  reciprocity, MPI); illumination.
- FWI: cost and gradient over the shots, multi-scale filtering, optimisers (heuristic and
  Wolfe line searches, scipy), gradient conditioning, shot parallelism over MPI ranks;
  synthetic inversions of :doc:`example/fwi`.
- Tests: 139 unit tests (no solver) and 31 integrated tests (GPU, MPI), 2 to 4 min.

Limits
======

- Field data: not tried yet; the synthetic Marmousi case also runs without inverse crime
  (observed data on a finer mesh).
- 3-D FWI: the chain is the same, but no complete 3-D inversion has been run; the
  snapshots of the forward run are kept in host memory.
- The absorbing terms are not differentiated (gradient wrong on the edge elements, masked
  by the conditioning).
- No elastic gradient; elastic solver in 3-D only.
- FWI with the domain decomposition (distributed model) is refused.
- No source estimation, no data selection (offsets, time windows) for field data.
- Limited-aperture acquisitions are not implemented.
