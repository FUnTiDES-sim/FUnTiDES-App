#!/bin/bash
#SBATCH --job-name=strong-scaling
#SBATCH --output=strong_%j.txt
#SBATCH --error=strong_err_%j.txt
#SBATCH --time=01:00:00
#SBATCH --nodes=4
#SBATCH --nodelist=sirocco[07-13]
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --exclusive

# Usage:
#   FUNTIDES_INSTALL=<FUnTiDES install dir> \
#   FUNTIDES_APP_PATH=<FUnTiDES-App clone> \
#   FUNTIDES_VENV=<Python env providing pykokkos> \
#   sbatch plafrim_strong_scaling.sbatch
# The modules must match the ones FUnTiDES was built with.

module purge
module load compiler/cuda/12.3
module load compiler/gcc/12.2.0
module load mpi/openmpi/5.0.1

: "${FUNTIDES_INSTALL:?set FUNTIDES_INSTALL to the FUnTiDES install directory}"
: "${FUNTIDES_APP_PATH:?set FUNTIDES_APP_PATH to the FUnTiDES-App clone}"
: "${FUNTIDES_VENV:?set FUNTIDES_VENV to a Python env providing pykokkos}"

source $FUNTIDES_VENV/bin/activate

export PYTHONPATH=$FUNTIDES_INSTALL/python${PYTHONPATH:+:$PYTHONPATH}
export LD_LIBRARY_PATH=$FUNTIDES_INSTALL/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# Reset CSV
rm -f results_strong.csv

# Strong Scaling Params (Fits in 1 GPU)
GLOBAL_EX=400
EY=300
EZ=300
STEPS=1000

echo "=== STARTING STRONG SCALING ==="

for TASKS in 1 2 4 8; do
    echo "Running with $TASKS GPUs"

    mpirun -np $TASKS \
           --bind-to socket \
           python3 -u $FUNTIDES_APP_PATH/examples/python/solver_multigpu_multinode.py \
           --ex $GLOBAL_EX --ey $EY --ez $EZ --steps $STEPS \
           --benchmark --type-scaling strong

done
