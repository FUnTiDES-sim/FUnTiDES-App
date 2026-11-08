"""makutu.tools.gpu: one GPU per MPI rank."""

from makutu.tools.gpu import gpu_count, local_rank, select_gpu


def test_local_rank_from_the_launchers():
    assert local_rank({}) == 0
    assert local_rank({"OMPI_COMM_WORLD_LOCAL_RANK": "3"}) == 3
    assert local_rank({"SLURM_LOCALID": "2"}) == 2
    assert local_rank({"MPI_LOCALRANKID": "x", "PMI_LOCAL_RANK": "1"}) == 1


def test_gpu_of_each_rank():
    env = {"OMPI_COMM_WORLD_LOCAL_RANK": "5"}
    assert select_gpu(env, ngpus=4) == "1" and env["CUDA_VISIBLE_DEVICES"] == "1"
    env = {"OMPI_COMM_WORLD_LOCAL_RANK": "1", "CUDA_VISIBLE_DEVICES": "4,6"}     # from the scheduler
    assert select_gpu(env) == "6" and env["CUDA_VISIBLE_DEVICES"] == "6"
    env = {"OMPI_COMM_WORLD_LOCAL_RANK": "1", "CUDA_VISIBLE_DEVICES": "2"}       # set by the user
    assert select_gpu(env) == "2"
    assert select_gpu({}, ngpus=2) == "0"


def test_gpu_count(tmp_path):
    for name in ("0000:01:00.0", "0000:02:00.0"):
        (tmp_path / name).mkdir()
    assert gpu_count(str(tmp_path)) == 2
    assert gpu_count(str(tmp_path / "missing")) == 1
