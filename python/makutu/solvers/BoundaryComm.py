import numpy as np


class BoundaryComm:
    """
    Synchronizes boundary force values for 1D X-axis domain decomposition.

    Each MPI rank holds a local X-subdomain. After ``compute_forces``, the
    shared boundary nodes between adjacent X-neighbours must be summed
    (assembled) before calling ``update_solution``.  This matches the
    synchronisation pattern in the FUnTiDES MPI example.

    Parameters
    ----------
    comm : mpi4py communicator
        MPI communicator (e.g. ``MPI.COMM_WORLD``).
    nx : int
        Number of nodes in X on this rank's local subdomain.
    ny : int
        Number of nodes in Y (global — not decomposed).
    nz : int
        Number of nodes in Z (global — not decomposed).
    """

    def __init__(self, comm, nx, ny, nz):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        # Left face: ix = 0  →  linear index = 0 + j*nx + k*nx*ny
        left = np.array(
            [j * nx + k * nx * ny for k in range(nz) for j in range(ny)],
            dtype=np.int32,
        )
        self.idx_left = left
        # Right face: ix = nx-1  →  linear index = (nx-1) + j*nx + k*nx*ny
        self.idx_right = left + (nx - 1)
        self.buf_size = len(left)

    def sync_accumulate(self, field_view):
        """
        Exchange and sum shared boundary node values with X-neighbours.

        Phase 1: rank *i* sends its right face to rank *i+1*, which adds the
                 received values onto its own left face.
        Phase 2: rank *i* sends its left face to rank *i-1*, which adds the
                 received values onto its own right face.

        Parameters
        ----------
        field_view : np.ndarray or kokkos array
            1-D float32 array, or the Kokkos view itself (e.g.
            ``solver.get_force_vector(c)`` or ``solver.get_mass_matrix()``).
            Modified **in-place**.
        """
        # Zero-copy numpy view (FUnTiDES returns Kokkos views, which do not
        # support fancy indexing)
        field_view = np.array(field_view, copy=False)
        # Right neighbour
        if self.rank < self.size - 1:
            recv = np.zeros(self.buf_size, dtype=np.float32)
            self.comm.Sendrecv(
                sendbuf=field_view[self.idx_right].copy(),
                dest=self.rank + 1,
                recvbuf=recv,
                source=self.rank + 1,
            )
            field_view[self.idx_right] += recv

        # Left neighbour
        if self.rank > 0:
            recv = np.zeros(self.buf_size, dtype=np.float32)
            self.comm.Sendrecv(
                sendbuf=field_view[self.idx_left].copy(),
                dest=self.rank - 1,
                recvbuf=recv,
                source=self.rank - 1,
            )
            field_view[self.idx_left] += recv
