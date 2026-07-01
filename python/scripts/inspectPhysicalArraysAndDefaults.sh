source ~/.local/venv/proxy/bin/activate && source /home/henri/src/proxyApp/FUnTiDES/envpython.sh && python3 -c "
import kokkos; kokkos.initialize()
import pyfuntides.model as M
import numpy as np

b = M.CartesianStructBuilder_f32_i32_O2(2, 100.0, 2, 100.0, 2, 100.0, False, False)
m = b.get_model()

print('n_elements:', m.get_number_of_elements())
print('n_nodes:', m.get_number_of_nodes())
print('n_pts_per_elem:', m.get_number_of_points_per_element())
print('order:', m.get_order())
print('domain_size x:', m.domain_size(0), 'y:', m.domain_size(1), 'z:', m.domain_size(2))
print('max_speed:', m.get_max_speed())
print('min_spacing:', m.get_min_spacing())
# Access element 0 vp
print('vp elem 0:', m.get_model_vp_on_element(0))
print('rho elem 0:', m.get_model_rho_on_element(0))
print('node coord 0:', m.node_coord(0))
" 2>&1 | grep -v "WARNING\|without\|penalties\|CUDA_MANAGED\|CUDA_VISIBLE\|performance\|Kokkos::"
