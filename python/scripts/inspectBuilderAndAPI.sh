source env.sh && python3 -c "
import kokkos; kokkos.initialize()
import pyfuntides.model as FTModel

print('=== CartesianStructBuilder O2 ===')
help(FTModel.CartesianStructBuilder_f32_i32_O2.__init__)

print('=== CartesianParams ===')
help(FTModel.CartesianParams_f32_i32.__init__)

print('=== ModelApi (get_* methods) ===')
import pyfuntides.model as M
# Build a small model to inspect its API
b = M.CartesianStructBuilder_f32_i32_O2(2, 100.0, 2, 100.0, 2, 100.0, False, False)
m = b.get_model()
print([x for x in dir(m) if not x.startswith('_')])
" 2>&1 | grep -v "WARNING\|without\|penalties\|CUDA_MANAGED\|CUDA_VISIBLE\|performance\|Kokkos::"
