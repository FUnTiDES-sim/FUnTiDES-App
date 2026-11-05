source env.sh && python3 -c "
import kokkos; kokkos.initialize()
from utilities.mesh.InternalMesh import _partition_1d

# Simulate 30-element X domain across 4 ranks
ex = 30
for rank in range(4):
    local_ex, offset = _partition_1d(ex, rank, 4)
    origin_x = offset * (1500.0 / ex)
    local_lx = local_ex * (1500.0 / ex)
    print(f'Rank {rank}: local_ex={local_ex}, x_offset={offset}, origin_x={origin_x:.1f}, end_x={origin_x+local_lx:.1f}')

print()
# Test coordToNodeIndex off-rank detection
import pyfuntides.model as FTModel
mesh = FTModel.CartesianStructBuilder_f32_i32_O2(8, 1500*8/30, 30, 1500, 30, 1500, False, False, 0.0, 0.0, 0.0).get_model()

import pyfuntides.solver as FTSolver
solver_obj = FTSolver.create_solver(FTSolver.MethodType.SEM, FTSolver.ImplemType.MAKUTU,
    FTSolver.MeshType.STRUCT, FTSolver.ModelLocationType.ONELEMENTS, FTSolver.PhysicType.ACOUSTIC, 2)
solver_obj.compute_fe_init(mesh, [0]*3, False, 0.0)
print('Split-step methods available:', hasattr(solver_obj, 'compute_forces'), hasattr(solver_obj, 'update_solution'), hasattr(solver_obj, 'get_force_vector'), hasattr(solver_obj, 'get_num_components'))
print('num_components:', solver_obj.get_num_components())
" 2>&1 | grep -v "WARNING\|without\|penalties\|CUDA_MANAGED\|CUDA_VISIBLE\|performance\|Kokkos::"
