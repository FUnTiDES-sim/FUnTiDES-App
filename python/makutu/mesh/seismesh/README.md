# seismesh — SEM meshes adapted to a velocity model (makutu.mesh.seismesh)

Conforming all-quadrilateral (2-D, (x, z) section) or all-hexahedral (3-D) meshes whose element
size follows the velocity model and source frequency; export for the FUnTiDES gmsh readers
and modelling with makutu (`GmshMesh`, `AcousticSolver`). Derived from `seismesh_agent`
(without the LLM agent component).

> **Next steps in 2-D (11–12 November 2026).** A conforming quadrilateral mesh cannot both follow
> the velocity locally and remain regular (dyadic levels: 2.16 times the ideal count). The selected
> 2-D approach tiles the model with regular non-conforming blocks coupled by SIPG, with a hybrid
> variant using DG triangles along interfaces: library `makutu.mortar`, applications and results in
> `apps/mortar/`. seismesh remains the conforming approach (2-D and 3-D) for the FUnTiDES gmsh readers.

## Usage: SEP model + source frequency
```bash
python apps/mesh/seismesh_from_sep.py --vp-file vp.H --f0 8 --order 4 --ppw 5 --out maillage   # [--rho-file rho.H] [--lts]
```
```python
from makutu.mesh import GmshMesh
from makutu.mesh.seismesh import build_mesh_from_sep
r = build_mesh_from_sep("vp.H", f0=8.0, out_dir="maillage", order=4, ppw=5)
mesh = GmshMesh(r["msh"], 4, r["materials"])            # then AcousticSolver(..., **mesh.solver_kwargs())
```

- `f_max = 2.5 f0` for a Ricker wavelet with peak frequency `f0` (as in
  `makutu.mesh.suggest_mesh`); required size `h = N·v/(f_max·ppw)`.
- 2-D SEP (`n3 = 1`): axis 1 = x, axis 2 = depth → quadrilaterals; strategies tried by default:
  `tensor`, `doubling`, `quadtree` (`bands`, `hybrid` and `adaptive` on request, through `strategies`);
  3-D SEP: axes x, y, z → hexahedra; defaults: `tensor`, `doubling`, `hybrid`, `refine3tz`,
  `extrude` (`refine3`, `tetsplit` on request). `--strategies` option of `seismesh_from_sep.py`.
- Candidates are generated and evaluated; the least expensive valid candidate
  (`ppw` reached, Jacobian > 0.2) is written to `out_dir`:
  - `mesh.msh`: MSH 2.2, **z upwards** (height = maximum depth − depth), free surface
    on top (group 1 `free_surface`), other faces absorbing (group 2 `absorbing`),
    one physical group per velocity class (≥ 100);
  - `materials.csv`: `physical_tag,vp,rho`, vp = mean of the element's SEP samples;
  - `report.json`: quality, sampling, CFL time step and cost of the selected mesh.
- makutu depths (z downwards from the surface) match the SEP model
  (including the `o3` origin).
- Validation (makutu tests): shots on doubling meshes against the makutu Cartesian mesh
  of the same model, relative trace difference ≤ 3.3 % in 2-D and ≤ 2.5 % in 3-D at
  10 points per wavelength; it decreases as the mesh is refined.
- Interfaces: element faces pass through sharp horizontal interfaces (velocity jump
  > 10 % between two samples over at least half the plane), halfway between the two
  samples. Inclined or curved interfaces (inclusions) still cut through elements: a straddling
  element receives the mean slowness of its samples (`average="slowness"`, default,
  preserves travel times; `"velocity"`: mean velocity).
- Three-layer model with a slow inclusion (2-D, 8 pts/λ): without faces on the interfaces,
  trace differences of 52 to 80 % and arrivals up to 18 ms early; with them, 6 to 40 % and
  8 ms, including 5 ms due to the interface position (sample midpoint, 495 m, versus 500 m
  for the reference Cartesian mesh); the remainder comes from the inclusion (20 m steps
  versus 70 m elements).

Grid convention: array (ix, iy, iz), z = depth downwards, z = zmin = free surface;
in 2-D, ny = 1. Other `VelocityGrid.load` inputs: `.npy`, `.npz`, raw binary
(+ metadata `.json`).

## Pipeline

1. **Criterion** `h_req = N·v_min/(f_max·ppw)` (N = SEM order; N=1 and ppw≈10 for an FD-like criterion).
2. **Optimal segmentation** of each axis by dynamic programming into ≤K constant-spacing
   segments (minimum element count, bounded grading between segments).
3. **gmsh**:
   - `tensor` — structured point→line→surface→volume extrusions with layers: 100 % hexahedra, Jacobian 1.
   - `doubling` — tensor grid + **conforming 2:1 doubling layers** (SPECFEM-style bricks):
     whenever h_req allows it over the remaining depth, horizontal spacing doubles through two
     sublayers (2-D “4 fine → 2 coarse” template, 6 convex quads, extruded in y then x).
     100 % hexahedra, no hanging nodes; mesh built in numpy then injected into gmsh (discrete
     entities + physical groups). Parameters: `max_levels`, `trans_layers`.
   - `hybrid` — `doubling` + **local conforming 3-refinement** (27-tree, reconstructed Schneiders
     templates: vertex 4 hexes, edge 11, face 22; strong balancing; minimum Jacobian 0.46 on
     a cubic grid). The doubling layout follows a background h_bg (lateral quantile `q_bg` of h_req);
     local slow anomalies are refined 1:3 in constant-level zones, with 2:1 transitions kept away
     from anomalies (replanning). Conforming assembly verified.
   - `extrude` — unstructured quads (Frontal-Delaunay + Blossom) driven by a size callback
     on min_z(h_req), extruded in z layers. Useful when low velocities are laterally localized.
4. **Evaluation**: scaled Jacobian, aspect ratio, effective ppw per element (v_min over the
   element's FD nodes), CFL time step (minimum GLL spacing), `cost_index ∝ GLL points × number of steps`,
   efficiency versus ideal bound ∫dV/h_req³, elements limiting dt.
5. **Export**: `.msh` (physical groups `medium`, `free_surface`, `absorbing`), `.vtk` (vp, ppw,
   Jacobian, per-element dt for ParaView), `_sem.npz` (nodes, connectivity, per-element min/max/mean vp).

## Automatic search

`Session.autopilot` (3-D) and `plane.autopilot_2d` (2-D) sweep strategy parameters
and rank candidates by `cost_index` (global step) or `cost_index_lts` (ideal local time
stepping in powers of 2, without the correction-support overhead measured in FUnTiDES).

## SEP model convention

As in makutu, sample i represents cell [o + i d, o + (i+1) d): the meshed domain is
[o, o + n d] and interfaces lie on cell edges (`grid_from_sep(cell_centred=True)`,
default). An element's velocity is the mean cell slowness, weighted by overlap;
its required size is that of the slowest cell it overlaps.

## 2-D: coarse to fine (quadtree_2d.py, `quadtree` strategy)

The base grid is sized for the fastest velocity (blocks of 2 x 2 elements), then blocks
are split 2:1 while their elements exceed the required size: fast zones (salt) retain large
elements, slow zones (water) receive small ones. Conforming transitions within each block depend
on its sides adjacent to finer blocks: doubling template (one side, two opposite sides),
refined quarter + two 3-quad corner templates (two adjacent sides, three sides); four sides:
split. The search sweeps the base size (1 to 0.5 times the largest required size, so that levels
base / 2^l fall near the model sizes) and an offset of the base grid. Validation: SEG/EAGE 2-D
salt model (f0 = 5 Hz, order 4) against the makutu Cartesian mesh (20 m): 15 % overall difference
at 8 pts/λ (3 663 elements), 5.5 % at 16 pts/λ; synthetic salt model (tests): 9.6 % then 4.1 %.

## 3-D: coarse to fine (`refine3tz` strategy)

The base grid is sized for the fastest velocity (`size_factor` × largest required size),
with 3-refinement (27-tree) of cells exceeding the required size (1/3, 1/9), followed by
conforming transitions using the “element-saving” templates of **Hua Tong and Yongjie
Jessica Zhang**:

> H. Tong, Y. J. Zhang, *Element-saving hexahedral 3-refinement templates*, Computer Aided Geometric
> Design (2026) 102550, doi:[10.1016/j.cagd.2026.102550](https://doi.org/10.1016/j.cagd.2026.102550).
> Original code: [CMU-CBML/Element-Saving-Hexahedral-3-Refinement-Templates](https://github.com/CMU-CBML/Element-Saving-Hexahedral-3-Refinement-Templates)
> (MIT license, Copyright (c) 2026 CMU-CBML).

`templates3_es.py` is a Python translation (method 3, vertex-based: a vertex is “black”
when a finer cell touches it; 256 codes, 15 classes). Unlike Schneiders templates
(`refine3`, where a cell bordered by two or three refined faces must itself be refined,
causing a cascade), all codes are admissible: a **coarse island** (salt) retains its base
cells, with transitions occupying its boundary cells. Conditions: contact balancing (two
leaves that touch differ by at most one level, `Refine3Mesh.balance_touch`); the fast body
must be at least 3 base cells thick for a cell to remain intact.
Quality: minimum scaled Jacobian ≈ 0.23 in transitions, 1 elsewhere.

`salt3d` model (2 × 2 × 1.5 km, 200 m of water at 1500 m/s, sediments 2000 + 0.3 (z − 200) m/s,
salt ellipsoid 1400 × 1400 × 700 m at 4500 m/s; f0 = 5 Hz, order 3, 8 pts/λ): 73 540 hexahedra
(sizes 44 m outside salt, 133 m inside), 19 s meshing time, versus 657 096 irregular
hexahedra for `tetsplit`.

## 2-D: comparison with gmsh (10 November 2026)

Reference models from `FUnTiDES-data/SEP` (seg2d, Marmousi, Sigsbee 2A), f0 = 5 Hz, order 4, 8 pts/λ;
cost = elements / time step (`cost_index`); best valid mesh from each approach:

| model (ideal elements) | selected seismesh | gmsh, frontal quadrilaterals + recombination | gmsh, subdivided triangles |
|---|---|---|---|
| seg2d (1 704) | `quadtree` 3 663, dt 1.3 ms, cost 0.18 | 5 326, dt 0.45 ms, cost 0.77 | 6 120, cost 0.93 |
| Marmousi (3 513) | `tensor` 7 348, dt 1.1 ms, cost 0.43 | 11 590, dt 0.55 ms, cost 1.36 | 9 904, cost 1.11 |
| Sigsbee 2A (31 106) | `doubling` 49 440, dt 1.1 ms, cost 2.88 | 89 468, dt 0.20 ms, cost 28.1 | 104 884, cost 17.4 |

gmsh follows the requested size within ±40 %: to meet the sampling requirement everywhere,
it must be asked for 0.6 times the required size (achieved median size 0.6 times the required
size, versus 0.73 to 0.91 for seismesh), and its irregular elements have short edges that reduce
the time step by a factor of 2 to 5. The gmsh quasi-structured algorithm (11) does not follow
the size map. → In 2-D, structured strategies are retained; automatic search selects `quadtree`
when a fast body occupies little space (seg2d), and `tensor` / `doubling` when velocity mainly
increases with depth.

### Approaches not pursued

- Tolerance at fast-body boundaries (allowing a coarse block to extend into sediments):
  the salt / sediment contrast (≈ 2.2) prevents this without falling below 4 pts/λ.
- gmsh meshing by regions following major interfaces (velocity thresholds, simplified
  contours): the same shortcomings as gmsh above.
- Quadtree nodes snapped to interfaces (Zhang and Bajaj 2006), then optimized (Knupp):
  on seg2d, 4 544 elements, dt 0.70 ms, shot difference 16.0 %, versus 3 663 elements,
  1.3 ms and 15.1 % for the plain quadtree. A ring of fine elements remains in the salt (2 x 2 blocks).
- Structured mesh + gmsh inside fast bodies (hole around the salt, imposed boundary nodes,
  conforming connection): gmsh coarsens the salt well, but its irregular elements reduce the time
  step by a factor of 2 to 4 (seg2d: cost 0.28 to 0.33 versus 0.18; Sigsbee: 10.9 versus 2.9);
  even with local stepping, no better than the quadtree alone (0.19 versus 0.17).
- Moving quadtree nodes towards the required size (r-adaptation, Knupp target matrix):
  at fixed element count, it only equalizes resolution (5 % gain in time step) and cannot
  make a coarser quadtree valid (fixed connectivity).
- Regions meshed with regular grids connected by gmsh bands (`patches`, initially recombined
  quadrilaterals, then subdivided triangles as in the FUnTiDES local-step validation):
  on that validation's bilayer model (10 m SEP), 1 847 elements, some 1.36 times too large,
  versus 1 716 for the valid quadtree; connections are wide and gmsh fails to meet the size there.
- Required-size zones meshed with regular grids, connected by optimized gmsh transitions
  (`zones`): fails on thin salt and gradient sediments; module removed on 16 November 2026.

Interface extraction (`interfaces.py`) and 2-D node optimization (`smoothing.py`) are retained.
The `bands` strategy (depth bands with their own horizontal spacing, connected by rows of
3 -> 1 templates, `bands2d.py`) is no longer in the default search (16 November 2026);
it remains available on request (`--strategies bands`). For a size ratio below 2 between bands,
transition cells become sheared (up to 1.6 times too large) and are rejected by evaluation.
What was retained: the search also sweeps quadtree base sizes above the largest required
size (1.2 to 1.6), and ``ppw_min`` (``--ppw-min`` option) accepts sampling slightly below
the target in a few elements. The 2-D `hybrid` strategy (doubling + local 3-refinement) is no
longer tried by default: never selected (cost 2 to 5 times the best) and slow (140 s on Marmousi
versus 6 s for the others); it remains available through ``strategies`` / ``--strategies``.

## 2-D meshes (plane.py and 2-D modules)

- `tensor`: piecewise-constant-spacing grid in x and z;
- `doubling`: horizontal spacing doubles with depth through a conforming transition layer
  (“4 fine → 2 coarse” template, 6 quadrilaterals) whenever `h_req` allows it below;
  `allow_refine` refines again below a slow layer (reversed template, minimum Jacobian ≈ 0.5);
- `quadtree` (`quadtree_2d.py`): coarse to fine, see above;
- `bands` (`bands2d.py`): regular depth bands with their own horizontal spacing, on request;
- `hybrid`: `doubling` + local 3-refinement (`refine3_2d.py`, `Refine3Mesh2D`), on request;
- `adaptive`: unstructured gmsh quadrilaterals (Frontal-Delaunay + recombination) of size
  `size_factor × h_req` (`mesh_adaptive_2d`), on request; this is the gmsh approach compared above;
- model values on the actual element geometry, including deformed elements (`elements2d.py`).

## Results (synthetic 5×4×6 km model, fmax 10 Hz, order 4, 5 ppw)

| strategy | elements | CFL dt | cost_index | minimum Jacobian |
|---|---|---|---|---|
| tensor | 49 491 | 1.2 ms | 10.6 | 1.00 |
| doubling (L=3, doublings at 1025 and 3350 m) | 44 160 | 2.7 ms | **4.26** | 0.68 |

The gain mainly comes from the time step: no more small elements in deep fast zones.

## Local refinement (hybrid): when to use it

Large 20×16×6 km domain, 12 slow pockets (2000 m/s) in the third layer, fmax 10 Hz:

| strategy | elements | cost (global dt) | cost (ideal LTS) |
|---|---|---|---|
| doubling + lateral segmentation | 869 760 | **160** | 121 |
| hybrid (doubling + 3-refinement) | 691 272 | 271 | **92** |

−20 % elements, but refined/transition elements lie in the fast host medium and reduce
global dt by ~2. → `hybrid` is only worthwhile with local time stepping (LTS);
otherwise `doubling` remains the best choice. With `--lts`, the search ranks by LTS cost
(power-of-2 levels, capped by `lts_max_level`) and reports `lts_levels`
(element count per level, useful for load balancing).

## Limitations / next steps

- Global doubling across the whole plane (no local doubling in x,y); fine-cell count
  rounded to a multiple of 2^(L+1).
- Re-refinement with depth (`allow_refine`) through reversed bricks, still across the whole plane:
  a localized slow inclusion imposes the fine level throughout the corresponding depth slice.
- `trans_layers=2` may undersample in z (left side of the brick over its full height).
- FUnTiDES local time stepping has two levels and is 2-D only: `cost_index_lts`
  (multiple levels, 3-D, no overhead) is an optimistic bound.
- Modelling only: makutu FWI still assumes a Cartesian grid.
- 2-D: no `extrude` strategy; `hybrid` is only tried on request (never selected).
- 2-D: to follow velocity locally without deformed elements, see `makutu.mortar`
  (non-conforming blocks), at the top of this document.
