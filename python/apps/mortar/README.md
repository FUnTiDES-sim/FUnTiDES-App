# Mortar and hybrid SEM + DG applications (2-D acoustics)

Scripts and examples built on the `makutu.mortar` library (`../../makutu/mortar`). Run from
`FUnTiDES-App/python` after `source <FUnTiDES install>/funtides_env.sh`; each script lists its options
with `--help`. Models: `FUnTiDES-data/SEP/<model>/Vp_xz.H`.

Reference document (mathematics, implementation, validation, results):
[doc/mortar-maths-et-implementation.md](doc/mortar-maths-et-implementation.md) (and its PDF).

## Scripts

| script | contents | computation |
|---|---|---|
| `hybrid_compare_2d.py` | hybrid mortar, CG + DG without mortar, blocks, uniform grid against a fine reference; seismograms on a common scale | CPU |
| `mesh_cost_2d.py` | meshes of the four approaches (full view and zoom) and run times at their own time steps | CPU |
| `hybrid_disk_2d.py` | fast disk: hybrid versus staircase | CPU |
| `interface_error_study.py` (+ `interface_study.py`) | error due to non-conforming interfaces (plane wave, R and T) | CPU |
| `blocks_shots_2d.py` | FUnTiDES shots: blocks versus uniform grid versus reference | GPU |
| `blocks_convergence_2d.py` | ppw convergence of blocks and uniform grid (FUnTiDES) | GPU |

## Reproducing the simulations

Cases (model, options, expected values) have been in the **FUnTiDES-examples** repository,
under `mortar/` (13 examples, 134 checks), since 16 November 2026:

```bash
cd FUnTiDES-examples
python run.py --list
python run.py "mortar/*"            # CPU examples (NumPy prototypes)
python run.py "mortar/*" --gpu      # including FUnTiDES shots
```

The FUnTiDES shot tables below (11 November) predate the latest block partitioning adjustments
(cuts at sharp jumps); current values are in the `mortar/*_shots` and
`mortar/marmousi_convergence` examples.

# Results

## Validation (../../tests/unitTests/test_mortar2d.py, figures in tests/figures/test_mortar2d)

- K is symmetric and positive semidefinite with the default penalty
  alpha = 1.25 N / (2 (N + 1)) (measured minimum alpha 0.22 / 0.32 / 0.37 at orders 2 / 4 / 6);
  time-step penalty of a conforming interface: 8 % at order 4 (alpha = 1 cost 37 %).
- Discrete energy conserved to within 1e-15 (6000 steps, 4 blocks with ratios from 1.37 to 3.5,
  velocity and density contrasts, nodal properties).
- Spurious reflection at an interface in a homogeneous medium (order 4, coarse side at 5 points /
  wavelength): 1e-5 (r = 1), 2e-5 (r = 1.37), 3e-4 (r = 2) of the direct wave.
- Spectral convergence (h, 1.5 h); difference from a fine reference dominated by the coarse side.

## Element counts (order 4, 5 GLL points / wavelength, f0 = 8 Hz; bilayer: order 2, fmax 5 Hz, 6 ppw)

| model | ideal | blocks | ratio | previous best conforming mesh |
|---|---|---|---|---|
| seg2d | 1704 | 2345 (50 blocks) | 1.38 | bands 2808, tensor 4042 |
| Marmousi | 3513 | 4960 (67) | 1.41 | bands 6294, doubling 7224 |
| Sigsbee 2A | 31106 | 36159 (276) | 1.16 | bands 44314, doubling 49440 |
| Sigsbee + gas | 31548 | 36722 (298) | 1.16 | bands 45474, doubling 51088 |
| bilayer + inclusion | 1209 | 1356 (27) | 1.12 | quadtree 1716 |
| 3 layers + slow inclusion | 328 | 492 (24) | 1.50 | bands 606, doubling 624 |

Tiling takes 0.1 to 9 s (conforming autopilot: 7 s to 11 min). Salt receives large elements,
the gas pocket a fitted fine block; only blocks straddling the salt boundary retain the sediment size.

## Shots against a reference aligned with SEP cells (properties at GLL nodes)

The SEP model must be read as a **continuous model** (`grid_sampler`, bilinear interpolation of
samples at cell centres), identical for the reference and all meshes.
Read cell by cell, it has a small jump at each cell boundary; nodes placed arbitrarily within
cells then see a locally shifted model, and the error no longer converges
(Marmousi: 1.6 to 7.2 % depending on the mesh, with no consistent ordering).

Checks: Marmousi layout in a homogeneous medium, difference 0.3 % (5 ppw) then 0.07 % (7 ppw);
K p = 0 at interior nodes for linear p (1e-14); smooth analytical model: blocks 0.75 / 0.20 /
0.12 / 0.04 % from 5 to 8 ppw, uniform 0.74 % at 5 ppw with twice as many elements.

Relative L2 trace difference (receivers at 80 m, offsets > 300 m, 2 s), bilinear model:

| model | mesh | elements | stable dt | difference |
|---|---|---|---|---|
| Marmousi | uniform 5 ppw | 9686 | 1.33 ms | 2.31 % |
| Marmousi | blocks 5 ppw | 4960 | 1.46 ms | 2.39 % |
| Marmousi | blocks 6 ppw | 6917 | 1.40 ms | 1.42 % |
| seg2d | uniform 5 ppw | 3698 | 1.39 ms | 2.30 % |
| seg2d | blocks 5 ppw | 2345 | 1.14 ms | 2.78 % |
| seg2d | blocks 7 ppw | 4388 | 0.90 ms | 1.58 % |

At equal accuracy, blocks require about 2.2 times less computation on Marmousi and 1.3 times less
on seg2d (global time step; more with per-block time stepping). The plateau around 1.5–2 % comes
from actual discontinuities (salt, seabed) cutting through elements, for all meshes.
The block time step can be smaller than that of the uniform grid: rounding
ceil(W / h) in small blocks produces thinner elements (next step: per-block cost
n_e / dt_block in dynamic programming, natural with local time stepping).

## Computational cost (rank, dt_min)

Block time step: dt = C min(hx, hz) / vp_max, C = 0.1044 at order 4 (exact eigenvalue of a
regular grid; the estimate is exact for a single block and overestimates by about 8 % with interfaces,
due to the penalty). `rank="lts"` minimizes the sum of elements / dt_block (each block's own time
step); `dt_min` rejects blocks requiring a smaller step than a target global step.

Work (millions of element-steps per simulated second, order 4, 5 ppw, f0 = 8 Hz):

| model | uniform | blocks, global step (dt_min = uniform dt) | blocks, per-block step (rank="lts") |
|---|---|---|---|
| seg2d | 2.66 | 1.61 (x1.65) | 0.88 (x3.0) |
| Marmousi | 7.38 | 3.12 (x2.4) | 1.73 (x4.3) |
| Sigsbee 2A | 51.3 | 30.5 (x1.7) | 10.9 (x4.7) |

## FUnTiDES (GPU): funtides2d.py, apps/mortar/blocks_shots_2d.py

FUnTiDES (dev-2d branch, not installed in FUnTiDES-install): `Solver::setInterfaceOperator(dofs, P, G,
gamma, m, n_q)` in SEMsolver2D, SIPG form factorized per Gauss point (two kernels: products P.u and
G.u, then rows with atomic additions); single-element edges containing an interior node
(interface edges) are no longer damped. `FuntidesBlocks2D` builds the 2-D ModelUnstruct with
duplicated nodes using Mortar2D numbering. FUnTiDES (float32) reproduces the prototype to within 1e-6
(tests/integratedTests/test_mortar_funtides.py).

FUnTiDES shots, global time step (plan_blocks_global), nodal properties, reference aligned with
SEP cells; L2 difference of traces (offsets > 300 m) / snapshot at 0.6 T:

| model | blocks 5 ppw | blocks 6 ppw | uniform 5 ppw |
|---|---|---|---|
| Sigsbee 2A | 37 596 el., 5.5 / 5.4 % | 53 249 el., 4.5 / 4.3 % | 67 840 el., 6.7 / 5.9 % |
| Marmousi | 5 476 el., 7.0 / 4.8 % | 7 642 el., 1.1 / 3.1 % | 9 686 el., 2.7 / 3.4 % |
| seg2d | 2 724 el., 3.0 / 8.2 % | 3 707 el., 2.6 / 5.4 % | 3 698 el., 2.3 / 4.9 % |
| bilayer + inclusion (order 2) | 1 398 el. (6 ppw), 8.8 / 9.3 % | 1 939 el. (7 ppw), 7.3 / 7.6 % | 11 250 el., 10.3 / 11.4 % |
| 3 layers + slow inclusion | 674 el., 4.7 / 12.9 % | 789 el., 4.7 / 11.0 % | 850 el., 4.8 / 12.3 % |

GPU times (laptop, variable clock: warm-up then best of three): Sigsbee 0.18 ms per step
(blocks 5) versus 0.23 ms (uniform), i.e. 1.0 s versus 1.2 s for 6 simulated seconds at their stable
steps, with better accuracy. Below 10 000 elements, the GPU is limited by kernel launch latency
and timings do not distinguish the meshes. The interface operator still costs the equivalent of
1 to 2 elements per segment: next improvements, one kernel per team (one launch, no intermediate
array) and especially each block's own time step (estimated work reduced by a factor of 3 to 4.7).

## Interface kernel and receivers (FUnTiDES, 2026-11-11)

- Structured interface operator (`Mortar2D.interface_faces`): traces and normal derivatives at face
  nodes (derivative along the element line normal to the face), interpolated at Gauss points; one
  thread per segment. Sigsbee (4663 segments): 0.024 ms per step, versus 0.106 ms with full rows in two
  kernels and 0.187 ms with one team per segment. Blocks 0.128 ms per step versus 0.258 ms for the
  uniform grid (37 596 versus 67 840 elements).
- `pyfuntides.solver.FieldSampler`: weighted sums at receivers on the GPU; 392 receivers cost
  0.019 ms per step, versus about 0.9 ms when reading the field (unified memory) from Python.

## Per-block time stepping (lts2d.py, prototype abandoned for now)

Diaz and Grote LF-LTS applied recursively by level (level l block: step dt / 2^l), stabilized
version (Grote, Michel, Sauter), levels based on exact stability (largest eigenvalue of the
operator restricted to each node group, including interfaces: `assign_levels`).

- Stability: elements on both sides of an interface must be promoted to the finest level
  (1 / h penalty on the fine side), and levels must be graded (at most one level difference between
  coupled nodes: a two-level jump diverges, as measured); energy then remains bounded.
- Gain: at the stability step, leapfrog already has a temporal error of about 5 to 8 % over 2 to
  3 s (compared with dt / 4); increasing the large step worsens accuracy. At the step required for
  accuracy (that of the uniform grid), LTS gains nothing on Marmousi and 4 to 14 % on Sigsbee, with
  one unstable case: not ported to FUnTiDES. The `plan_blocks_global` layout (global step) is retained.
- Figure: tests/figures/mortar/lts_diagnostic.png.

## In makutu solvers (mesh2d.py)

`BlocksMesh2D(vp_file, f0=...)`: tiling, mesh, FUnTiDES model; `solver_kwargs()` for
`AcousticSolver` (source and receiver locator, `solver_setup` hook installing the interface
operator after each `compute_fe_init`, nodal properties). For FWI, the variable is the SEP model
grid: `updateVelocityModel` reinterpolates the nodes (bilinear interpolation B), the gradient is
computed at nodes then mapped back to the grid by the transpose of B. Tests
(tests/integratedTests/test_mortar_makutu_solver.py): AcousticSolver shot = direct shot to 1.4e-5;
gradient = finite differences to 0.2 % (c and 1/c^2 parameterizations).

## Sharp interfaces and modelling validation

- `plan_blocks(z_cuts="auto")` first slices the model at sharp horizontal interfaces
  (jump > 10 % over half the width, `sharp_interfaces`, interfaces at least one element size apart),
  then optimizes each slice: no element crosses these interfaces.
- Local sharp boundaries (`edge_cuts`, `force_jump=0.3`): in each rectangle, dynamic programming
  also tries cuts at sharp boundaries (jump > 10 % over half the rectangle) and forbids retaining
  a single block across a jump > 30 %: a sharp axis-aligned body is bounded by block edges.
  Diagnostic (bilayer + square inclusion at 600 m/s, order 2, 6 ppw): a uniform grid whose elements
  happen to align with the inclusion edges (40 m) gives 1.6 %, an unaligned grid (38.7 m) 7.7 %;
  blocks improve from 8.5 % to 4.1 % (1254 elements). Grading the required size (`grading`,
  Lipschitz size field) only helps at 9 ppw (2.5 %): not enabled by default.
- `Mortar2D(nodal_inset=...)` (BlocksMesh2D: half a grid spacing): nodes on a block boundary sample
  the model inside their own block; an interface on a block boundary is sampled from one side on
  each side (exact jump, possible because blocks share no nodes).
- tests/integratedTests/test_mortar_modeling_validation.py (makutu AcousticSolver + FUnTiDES, GPU):
  exact 2-D solution across 9 blocks with ratios from 1.15 to 1.9 (orders 2 to 4, source at a block
  corner) < 3e-3; free surface, layered model, against a fine grid split at the interface: 2.6 %
  (18 % before interface cuts, 9.6 % with cuts but a reference smoothing the jump);
  8 simulated seconds without growth.

## References and SEP model sampling

A SEP grid can be read as a “staircase” (constant per cell: exact reference = uniform grid with
one element per cell, element-wise properties) or “smoothed” (bilinear interpolation: reference =
fine uniform grid, nodal properties). Trace differences between the two readings range from 2 %
(bilayer) to 8 % (Sigsbee, curved salt boundaries): the reference must sample the model like the
tested mesh. Trace differences (offsets > 300 m), tests/figures/mortar/references_*.png:

- Bilayer + square inclusion (order 2, jumps at cell edges, followed by blocks): against the
  staircase reference, blocks 1.9 / 0.85 / 0.34 / 0.12 % from 6 to 12 ppw; unaligned uniform grid
  14 % / 7 % (6 / 9 ppw). The 3.8 % “ceiling” was the difference between the two references.
- Sigsbee: against the smoothed one-cell reference (3.8 million elements), blocks 5.1 / 3.9 / 2.7 /
  2.4 % from 5 to 8 ppw, uniform 5.6 / 2.4 % (5 / 7 ppw); the old reference (3 cells) differed by
  2.8 %. The remainder (~2.4 %, common to blocks and the uniform grid) comes from sharp salt
  boundaries at cell scale, which every mesh coarser than a cell sees in the same way.
- Marmousi: blocks cut at the seabed and see it as a sharp jump; smoothed sampling turns it into
  a one-cell ramp: blocks differ by 0.7 % from the smoothed 2-cell reference (8 ppw), 1.9 % from
  the one-cell reference; the uniform grid (7 ppw) shows the reverse (1.9 / 0.6 %).

Sampling used by BlocksMesh2D: smoothed, except at block boundaries carrying a sharp jump
(one-sided sampling, exact jump).

## FUnTiDES port (estimate)

FUnTiDES 3-D DG pairs faces through shared nodes (FaceConnectivityUnstruct: one face = two
elements): h-non-conforming faces require a rewrite. The 2-D SEM (sem2d) is the appropriate base:
element loop + atomic additions into a global vector. To add: mesh with duplicated nodes at block
boundaries (injected from Python), interface segment table (element, edge, parameters on both sides),
a Kokkos SIPG flux kernel per segment (traces and normal derivatives interpolated at Gauss points),
and per-block time steps for local time stepping.
