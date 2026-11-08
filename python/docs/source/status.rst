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
  restart at any frequency; data selection and source estimation; one GPU per MPI rank; synthetic inversions of :doc:`example/fwi` (2-D, 3-D, Marmousi
  without inverse crime).
- Tests: 146 unit tests (no solver) and 34 integrated tests (GPU, MPI), 2 to 4 min.

Limits
======

- No field data inverted yet; the synthetic Marmousi case also runs without inverse crime
  (observed data on a finer mesh).
- Memory of the gradient: the snapshots of the forward run are kept in host memory (int16
  compression halves it); a large 3-D model needs checkpointing (not implemented).
- The absorbing terms are not differentiated (gradient wrong on the edge elements, masked
  by the conditioning).
- No elastic gradient; elastic solver in 3-D only.
- FWI with the domain decomposition (distributed model) is refused.
- Field data: data selection (offsets, mute, window) and source estimation are tested on
  synthetic data only; no other processing (noise, amplitude corrections, 3-D to 2-D).
- Limited-aperture acquisitions are not implemented.
