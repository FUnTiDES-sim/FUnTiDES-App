"""
Distribution of the shots over the MPI ranks.

Two parallel modes:

- domain decomposition: the solver is built with a communicator (``solver.comm``) and
  every rank computes its part of the domain for every shot;
- shot parallelism: the solver is not decomposed and every rank holds the whole model;
  the shots are dealt out to the ranks (``shots[rank::size]``), and the costs, gradients
  and illuminations are summed over the ranks. Every rank then holds the same values, so
  that the optimisers take the same steps on every rank.

Without MPI (one rank) both are the serial run.
"""

import numpy as np
from mpi4py import MPI


def isDecomposed(solver):
    """True if the solver is distributed over several ranks (domain decomposition)."""
    comm = getattr(solver, "comm", None)
    return comm is not None and comm.Get_size() > 1


def shotParallel(solver):
    """True if the shots are dealt out to several ranks (shot parallelism)."""
    return MPI.COMM_WORLD.Get_size() > 1 and not isDecomposed(solver)


def checkShotParallel(solver):
    """The optimisation needs the whole model on every rank: shot parallelism, not domain
    decomposition."""
    if isDecomposed(solver):
        raise NotImplementedError("FWI with the domain decomposition (the model is distributed); "
                                  "build the solver without comm: the shots are then dealt out to the ranks")


def localShots(shots, solver):
    """Shots computed by this rank: every shot, or ``shots[rank::size]`` with shot parallelism."""
    shots = list(shots)
    if not shotParallel(solver):
        return shots
    return shots[MPI.COMM_WORLD.Get_rank()::MPI.COMM_WORLD.Get_size()]


def outputComm(solver):
    """Communicator of the outputs of a shot (SEG-Y export, partial cost): its rank 0 writes.
    With shot parallelism, the rank that computes the shot (COMM_SELF)."""
    return MPI.COMM_SELF if shotParallel(solver) else MPI.COMM_WORLD


def sumOverShots(value, solver):
    """Sum over the ranks of a value computed on the local shots (float, array, or None for
    a rank without shot) with shot parallelism; the value itself otherwise."""
    if not shotParallel(solver):
        return value
    comm = MPI.COMM_WORLD
    shapes = comm.allgather(None if value is None else np.shape(value))
    shape = next((sh for sh in shapes if sh is not None), None)
    if shape is None:                                  # no rank has a value
        return None
    local = np.zeros(shape) if value is None else np.array(value, dtype=np.float64)
    total = np.empty_like(local)
    comm.Allreduce(local, total, op=MPI.SUM)
    return float(total) if shape == () else total
