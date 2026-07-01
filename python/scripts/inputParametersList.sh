source ~/.local/venv/proxy/bin/activate && source /home/henri/src/proxyApp/FUnTiDES/envpython.sh && python3 -c "
import kokkos; kokkos.initialize()
import pyfuntides.solver as FTSolver
import pyfuntides.model as FTModel

# Solver creation
print('=== create_solver ===')
help(FTSolver.create_solver)

print('=== compute_fe_init ===')
help(FTSolver.Solver.compute_fe_init)

print('=== ImplemType ===')
print(list(FTSolver.ImplemType.__members__))

print('=== MeshType ===')
print(list(FTSolver.MeshType.__members__))

print('=== PhysicType ===')
print(list(FTSolver.PhysicType.__members__))

print('=== ModelLocationType ===')
print(list(FTSolver.ModelLocationType.__members__))

print('=== MethodType ===')
print(list(FTSolver.MethodType.__members__))
" 2>&1 | grep -v "WARNING\|without\|penalties\|CUDA_MANAGED\|CUDA_VISIBLE\|performance\|Kokkos::"
