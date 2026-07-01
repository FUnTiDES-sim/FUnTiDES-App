# FUnTiDES Input Parameters Reference

## 1. Mesh / Model Construction

### Structured Builder (`CartesianStructBuilder_f32_i32_O{1,2,3}`)

| Parameter | Type | Description |
|-----------|------|-------------|
| `ex, ey, ez` | int | Number of elements per direction |
| `hx, hy, hz` | float | Element size [m] per direction |
| `on_nodes` | bool | Physical model on nodes (`True`) or elements (`False`, default) |
| `is_elastic` | bool | Elastic (`True`) or acoustic (`False`, default) |

```python
builder = FTModel.CartesianStructBuilder_f32_i32_O2(
    ex, hx, ey, hy, ez, hz,
    on_nodes=False,
    is_elastic=False
)
model = builder.get_model()
n_dof  = model.get_number_of_nodes()
n_pts  = model.get_number_of_points_per_element()
```

### Unstructured Builder (`CartesianUnstructBuilder_f32_i32`)

Uses a `CartesianParams_f32_i32` struct:

| Field | Type | Description |
|-------|------|-------------|
| `origin_x/y/z` | float | Domain origin offset [m] — required for MPI decomposition |
| `lx/ly/lz` | float | Local domain length [m] |
| `ex/ey/ez` | int | Local element count per direction |
| `order` | int | Polynomial order: 1, 2, or 3 |
| `is_model_on_nodes` | bool | Physical model on nodes vs elements |
| `is_elastic` | bool | Elastic flag |
| `free_surface_on_top` | bool | Free surface BC on top face (default `True`) |

```python
params = FTModel.CartesianParams_f32_i32()
params.origin_x = ox
params.lx, params.ly, params.lz = lx, ly, lz
params.ex, params.ey, params.ez = ex, ey, ez
params.order = order
params.is_model_on_nodes  = False
params.is_elastic         = False
params.free_surface_on_top = True

builder = FTModel.CartesianUnstructBuilder_f32_i32(params)
model = builder.get_model()
```

---

## 2. Solver Creation

```python
solver = FTSolver.create_solver(
    FTSolver.MethodType.SEM,                  # Only SEM supported
    FTSolver.ImplemType.MAKUTU,               # Only MAKUTU supported
    FTSolver.MeshType.STRUCT,                 # or UNSTRUCT
    FTSolver.ModelLocationType.ONELEMENTS,    # or ONNODES
    FTSolver.PhysicType.ACOUSTIC,             # or ELASTIC
    order                                     # int: 1, 2, or 3
)
```

### Available Enum Values

| Enum | Values |
|------|--------|
| `MethodType` | `SEM` |
| `ImplemType` | `MAKUTU` |
| `MeshType` | `STRUCT`, `UNSTRUCT` |
| `ModelLocationType` | `ONELEMENTS`, `ONNODES` |
| `PhysicType` | `ACOUSTIC`, `ELASTIC` |

---

## 3. FE Initialization

```python
solver.compute_fe_init(model, sponge_size, sponge_surface, taper_delta)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | pyfuntides model | — | Output of `builder.get_model()` |
| `sponge_size` | list[3 float] | `[0, 0, 0]` | Absorbing boundary thickness [m] per axis (x, y, z) |
| `sponge_surface` | bool | `True` | Apply sponge on free surface |
| `taper_delta` | float | `0.0` | Sponge taper width |

> **MPI note**: After `compute_fe_init`, call `boundary_comm.sync_accumulate(solver.get_mass_matrix())` once before the time loop.

---

## 4. Physical Model Arrays (Kokkos)

Set velocity and density via mutable Kokkos arrays **before** `compute_fe_init`.

```python
n_elem = ex * ey * ez   # if model on elements (default)
# n_elem = n_dof        # if model on nodes

kk_model_vp  = kokkos.array([n_elem], dtype=kokkos.float32,
                              space=kokkos.CudaUVMSpace, layout=kokkos.LayoutLeft)
kk_model_rho = kokkos.array([n_elem], dtype=kokkos.float32,
                              space=kokkos.CudaUVMSpace, layout=kokkos.LayoutLeft)

kk_model_vp[:]  = 1500.0   # P-wave velocity [m/s]
kk_model_rho[:] = 1.0      # density (normalized)
```

- For FWI, update `kk_model_vp` in-place between iterations.
- Default values: `vp = 1500.0 m/s`, `rho = 1.0`.

---

## 5. Time-Loop Arrays (Kokkos)

All arrays must use `space=kokkos.CudaUVMSpace, layout=kokkos.LayoutLeft`.

### Wavefield (2 arrays)

```python
kk_pn_prev = kokkos.array([n_dof], dtype=kokkos.float32, ...)
kk_pn_curr = kokkos.array([n_dof], dtype=kokkos.float32, ...)
wavefield  = FTSolver.WavefieldAcoustic(kk_pn_prev, kk_pn_curr)
```

### Source RHS (3 arrays)

```python
kk_RHSTerm    = kokkos.array([n_rhs, n_steps], dtype=kokkos.float32, ...)
kk_RHSElement = kokkos.array([n_rhs],          dtype=kokkos.int32,   ...)
kk_RHSWeights = kokkos.array([n_rhs, n_pts],   dtype=kokkos.float32, ...)
rhs = FTSolver.RhsAcoustic(kk_RHSTerm, kk_RHSElement, kk_RHSWeights)
```

| Array | Shape | dtype | Description |
|-------|-------|-------|-------------|
| `kk_RHSTerm` | `[n_rhs, n_steps]` | float32 | Source time series |
| `kk_RHSElement` | `[n_rhs]` | int32 | Element index for each source |
| `kk_RHSWeights` | `[n_rhs, n_pts]` | float32 | Interpolation weights within element |

Where `n_rhs` = number of sources, `n_pts` = `model.get_number_of_points_per_element()`.

### Solver Data Wrapper

```python
data = FTSolver.SEMsolverDataAcoustic(wavefield, rhs)
```

---

## 6. Time Stepping

| Parameter | Description |
|-----------|-------------|
| `dt` | Time step [s] |
| `n_time_steps` | Total number of steps |
| `time_sample` | 0-based integer step index |

### Single-rank

```python
for time_sample in range(n_time_steps):
    solver.compute_one_step(dt, time_sample, data)
    data.swap_wavefields()
```

### MPI multi-rank (split step)

```python
for time_sample in range(n_time_steps):
    solver.compute_forces(dt, time_sample, data)
    for c in range(solver.get_num_components()):
        boundary_comm.sync_accumulate(solver.get_force_vector(c))
    solver.update_solution(dt, data)
    data.swap_wavefields()
```

---

## 7. Seismogram Recording

Read from `kk_pn_curr` **before** `swap_wavefields()`:

```python
pn = np.array(kk_pn_curr, copy=False)
seismogram[time_sample, :] = pn[receiver_node_indices]
```

For MPI, assemble across ranks:

```python
comm.Allreduce(MPI.IN_PLACE, seismogram, op=MPI.SUM)
```

(Receivers on other ranks contribute zero; local receivers contribute their values.)

---

## 8. CFL Stability Constraint

```
dt ≤ h_min / (vp_max × sqrt(3))
```

Where `h_min` is the minimum node spacing:

```
h_min = element_size / order    (uniform structured grid)
```

---

## 9. Kokkos Initialization Order

`kokkos.initialize()` **must** be called before any pyfuntides constructor:

```python
import kokkos
kokkos.initialize()          # FIRST — before any pyfuntides import or call

import pyfuntides.model  as FTModel
import pyfuntides.solver as FTSolver
```
