"""
One GPU per MPI rank: choose ``CUDA_VISIBLE_DEVICES`` from the rank of the process on its
node, before Kokkos is initialised.

    from makutu.tools.gpu import select_gpu
    select_gpu()          # before kokkos.initialize()
    import kokkos
    kokkos.initialize()
"""

import os

# Rank of the process on its node, by launcher
_LOCAL_RANK_VARIABLES = ("OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID", "MV2_COMM_WORLD_LOCAL_RANK",
                         "PMI_LOCAL_RANK", "SLURM_LOCALID")


def local_rank(environ=None):
    """Rank of this process on its node (0 without a known MPI launcher)."""
    environ = os.environ if environ is None else environ
    for name in _LOCAL_RANK_VARIABLES:
        if environ.get(name, "").isdigit():
            return int(environ[name])
    return 0


def gpu_count(driver_dir="/proc/driver/nvidia/gpus"):
    """Number of NVIDIA GPUs of the node (directories of the driver; 1 if unknown)."""
    try:
        return max(1, len(os.listdir(driver_dir)))
    except OSError:
        return 1


def select_gpu(environ=None, ngpus=None):
    """
    Set ``CUDA_VISIBLE_DEVICES`` to one GPU for this process: the GPU ``local_rank % n`` among
    the devices already listed in ``CUDA_VISIBLE_DEVICES`` (e.g. by the job scheduler), or
    among the ``ngpus`` GPUs of the node (default: :func:`gpu_count`). A single device
    already set is kept. Call it before ``kokkos.initialize()``.

    Returns
    -------
        str : the device set
    """
    environ = os.environ if environ is None else environ
    visible = [d for d in environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    if len(visible) == 1:
        return visible[0]
    devices = visible or [str(i) for i in range(ngpus or gpu_count())]
    device = devices[local_rank(environ) % len(devices)]
    environ["CUDA_VISIBLE_DEVICES"] = device
    return device
