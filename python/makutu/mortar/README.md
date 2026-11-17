# makutu.mortar: non-conforming blocks and hybrid SEM + DG (library, 2-D acoustics)

Regular rectangular blocks, each with its own mesh size, coupled by a symmetric interior penalty
(SIPG); a hybrid variant uses discontinuous Galerkin triangles along model interfaces.
NumPy / SciPy (plus gmsh, shapely and scikit-image for hybrid zone placement);
`funtides2d.py` and `mesh2d.py` use pyfuntides (GPU).

Applications, examples, instructions for reproducing simulations, results and reference document:
[`apps/mortar`](../../apps/mortar/README.md).

## Modules

| module | contents |
|---|---|
| `solver2d.py` | `Mortar2D`: acoustic 2-D SEM (GLL, diagonal mass), SIPG at interfaces, leapfrog, paraxial boundaries + sponge, element-wise or nodal properties (`nodal`, `nodal_inset`), exact `dt_max()` |
| `blocks2d.py` | `plan_blocks`, `plan_blocks_global`: tiling of a SEP model (dynamic programming over binary cuts, cuts at sharp jumps); `uniform_plan`; `grid_sampler` (bilinear sampling) |
| `hybrid2d.py` | `Hybrid2D`: `QuadZone` (SEM block), `GridZone` (conforming rectilinear grid with mask), `TriZone` (DG triangles, Dubiner basis, properties at quadrature points), SIPG coupling; `draw_hybrid`, `mesh_cmap` |
| `hybrid_plan.py` | `plan_hybrid` (mortar blocks + triangle band) and `plan_cg_hybrid` (without mortar: rectilinear background grid, body grid, DG band) from a SEP model |
| `funtides2d.py` | `FuntidesBlocks2D`: mortar blocks computed by FUnTiDES sem2d (interface operator, receivers on GPU) |
| `mesh2d.py` | `BlocksMesh2D`: mortar blocks in makutu solvers |
| `lts2d.py` | per-block time stepping (abandoned prototype) |
| `plots2d.py` | diagnostic plots of block layouts |

Tests: `tests/unitTests/test_mortar2d.py`, `tests/unitTests/test_hybrid2d.py`,
`tests/integratedTests/test_mortar_{funtides,makutu_solver,modeling_validation}.py`.
