# makutu.mortar : blocs non conformes et hybride SEM + DG (bibliothèque, 2-D acoustique)

Blocs rectangulaires réguliers, chacun à sa propre taille de maille, recollés par une pénalité intérieure
symétrique (SIPG) ; variante hybride avec des triangles en Galerkin discontinu le long des interfaces du
modèle. NumPy / SciPy (et gmsh, shapely, scikit-image pour le placement des zones hybrides) ;
`funtides2d.py` et `mesh2d.py` utilisent pyfuntides (GPU).

Applications, exemples, reproduction des modélisations, résultats et document de référence :
[`apps/mortar`](../../apps/mortar/README.md).

## Modules

| module | contenu |
|---|---|
| `solver2d.py` | `Mortar2D` : SEM 2-D acoustique (GLL, masse diagonale), SIPG sur les interfaces, saute-mouton, bords paraxiaux + éponge, propriétés par élément ou aux nœuds (`nodal`, `nodal_inset`), `dt_max()` exact |
| `blocks2d.py` | `plan_blocks`, `plan_blocks_global` : pavage d'un modèle SEP (programmation dynamique sur les coupes binaires, coupes sur les sauts nets) ; `uniform_plan` ; `grid_sampler` (lecture bilinéaire) |
| `hybrid2d.py` | `Hybrid2D` : zones `QuadZone` (bloc SEM), `GridZone` (grille rectiligne conforme avec masque), `TriZone` (triangles DG, base de Dubiner, propriétés aux points de quadrature), couplage SIPG ; `draw_hybrid`, `mesh_cmap` |
| `hybrid_plan.py` | `plan_hybrid` (blocs mortar + bande de triangles) et `plan_cg_hybrid` (sans mortar : grille rectiligne du fond, grille du corps, bande DG) à partir d'un modèle SEP |
| `funtides2d.py` | `FuntidesBlocks2D` : blocs mortar calculés par FUnTiDES sem2d (opérateur d'interface, récepteurs sur GPU) |
| `mesh2d.py` | `BlocksMesh2D` : blocs mortar dans les solveurs makutu |
| `lts2d.py` | pas de temps par bloc (prototype abandonné) |
| `plots2d.py` | figures de contrôle des pavages |

Tests : `tests/unitTests/test_mortar2d.py`, `tests/unitTests/test_hybrid2d.py`,
`tests/integratedTests/test_mortar_{funtides,makutu_solver,modeling_validation}.py`.
