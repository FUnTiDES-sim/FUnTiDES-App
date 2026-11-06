source env.sh && python3 -c "
import kokkos; kokkos.initialize()
import pyfuntides.solver as FTSolver
import pyfuntides.model as M
import numpy as np

# Build model and inspect Kokkos arrays from ModelManagement
b = M.CartesianStructBuilder_f32_i32_O2(4, 375.0, 4, 375.0, 4, 375.0, False, False)
m = b.get_model()

# Check ModelManagement
import sys; sys.path.insert(0, '.')
from utilities.solvers.ModelManagement import geos_params_from_model
kk_vp, kk_rho = geos_params_from_model(m)
vp_arr = np.array(kk_vp, copy=False)
rho_arr = np.array(kk_rho, copy=False)
print('kk_vp shape:', vp_arr.shape, 'default:', vp_arr[0])
print('kk_rho shape:', rho_arr.shape, 'default:', rho_arr[0])
print('vp writable:', vp_arr.flags.writeable)
" 2>&1 | grep -v "WARNING\|without\|penalties\|CUDA_MANAGED\|CUDA_VISIBLE\|performance\|Kokkos::"
