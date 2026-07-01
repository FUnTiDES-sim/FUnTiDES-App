source ~/.local/venv/proxy/bin/activate && source /home/henri/src/proxyApp/FUnTiDES/envpython.sh && python3 -c "
import kokkos; kokkos.initialize()
import pyfuntides.model as M
b = M.CartesianStructBuilder_f32_i32_O2(2, 100.0, 2, 100.0, 2, 100.0, False, False)
m = b.get_model()

# Check model physical parameter methods
print('=== get_model_vp (acoustic velocity) ===')
help(m.get_model_vp_on_element)

print('=== get_max_speed ===')
help(m.get_max_speed)

print('=== get_min_spacing ===')
help(m.get_min_spacing)

print('=== domain_size ===')
help(m.domain_size)
" 2>&1 | grep -v "WARNING\|without\|penalties\|CUDA_MANAGED\|CUDA_VISIBLE\|performance\|Kokkos::"
