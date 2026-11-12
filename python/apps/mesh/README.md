# Gmsh meshes

`gmsh_mesh.py` prepares Gmsh meshes for the FUnTiDES SEM solvers (library:
`makutu.mesh.gmshmesh`; needs the `gmsh` Python module, `pip install gmsh`). The FUnTiDES
readers take linear quadrilaterals Q4 of the (x, z) plane (2-D) or hexahedra Hex8 (3-D) in
the MSH 2.2 ASCII format; cell physical groups are materials, boundary groups are free
surfaces or absorbing boundaries.

```bash
python apps/mesh/gmsh_mesh.py box box.msh --lengths 1500 1500 1500 --cells 15   # + box.csv
python apps/mesh/gmsh_mesh.py box ubox.msh --lengths 1500 1500 1500 --cells 30 --unstructured
python apps/mesh/gmsh_mesh.py rectangle rect.msh --lengths 2000 1000 --size 20
python apps/mesh/gmsh_mesh.py convert input41.msh output22.msh        # --plane xy for an (x, y) 2-D mesh
python apps/mesh/gmsh_mesh.py info box.msh box.csv --order 2
```

- `box`: volume group 10 (material, `box.csv` = `physical_tag,vp,rho`), surface 1 = top
  (free surface), surface 2 = the other five sides (absorbing). `--unstructured`:
  unstructured quadrilaterals in (x, y) extruded along z.
- `rectangle`: surface group 10, line 1 = top edge (free surface), line 2 = the other edges.
- `convert`: keeps the physical groups; triangles, tetrahedra and curved elements are
  rejected, not converted.

Modelling on a Gmsh mesh: `makutu.mesh.GmshMesh(file, order, materials)` and
`AcousticSolver(..., **mesh.solver_kwargs())` (see the *Mesh and model* page of the
documentation and `tests/integratedTests/test_gmsh_mesh.py`).

## Meshes adapted to a SEP model (seismesh)

```bash
python apps/mesh/seismesh_from_sep.py --vp-file vp.H --f0 8 --order 4 --ppw 5 --out mesh_dir
```

Element size `order * v / (2.5 f0 * ppw)`; writes `mesh_dir/mesh.msh` (z up, free surface on
top), `materials.csv` (one class per element) and `report.json`. See
`makutu/mesh/seismesh/README.md`.

## Non-conforming blocks and hybrid SEM + DG (2-D)

Conforming quadrilateral meshes cannot both follow the velocity locally and stay regular. The 2-D
alternative is in `makutu.mortar`: regular blocks, each at its own element size, glued by an interior
penalty, and a hybrid variant with DG triangles along the model interfaces (`plan_blocks`,
`plan_hybrid`, `plan_cg_hybrid`). Their applications, mesh figures and costs are in `apps/mortar/`
(`mesh_cost_2d.py`, `reproduce.py`; README there).

