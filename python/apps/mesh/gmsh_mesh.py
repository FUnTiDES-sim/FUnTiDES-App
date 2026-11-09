"""
Gmsh meshes for the FUnTiDES SEM solvers (makutu.mesh.gmshmesh).

  convert   linear Q4/Hex8 mesh (any Gmsh format) -> MSH 2.2 ASCII of the FUnTiDES readers
  box       Hex8 box with a free surface on top and absorbing sides (+ materials CSV)
  rectangle Q4 rectangle of the (x, z) plane
  info      read a mesh through FUnTiDES at an order and print its size and bounding box
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("convert", help="convert to MSH 2.2 ASCII")
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--plane", choices=["xz", "xy"], default="xz",
                   help="2-D: plane of the input mesh (xy is rotated into xz)")

    p = sub.add_parser("box", help="Hex8 box")
    p.add_argument("output", type=Path)
    p.add_argument("--lengths", type=float, nargs=3, default=[1.0, 1.0, 1.0], metavar=("LX", "LY", "LZ"))
    p.add_argument("--cells", type=int, nargs="+", default=[2], help="cells per axis (1 or 3 values)")
    p.add_argument("--unstructured", action="store_true",
                   help="unstructured quadrilaterals in (x, y), extruded along z")
    p.add_argument("--vp", type=float, default=1500.0)
    p.add_argument("--rho", type=float, default=1000.0)
    p.add_argument("--msh-version", type=float, choices=[2.2, 4.1], default=2.2)

    p = sub.add_parser("rectangle", help="Q4 rectangle of the (x, z) plane")
    p.add_argument("output", type=Path)
    p.add_argument("--lengths", type=float, nargs=2, default=[1000.0, 500.0], metavar=("LX", "LZ"))
    p.add_argument("--size", type=float, required=True, help="element size [m]")
    p.add_argument("--structured", action="store_true")
    p.add_argument("--no-free-surface", action="store_true", help="absorbing top edge")
    p.add_argument("--msh-version", type=float, choices=[2.2, 4.1], default=2.2)

    p = sub.add_parser("info", help="read through FUnTiDES and describe")
    p.add_argument("mesh", type=Path)
    p.add_argument("materials", type=Path, help="CSV physical_tag,vp,rho")
    p.add_argument("--order", type=int, default=2)

    args = parser.parse_args()
    from makutu.mesh import gmshmesh

    if args.command == "convert":
        gmshmesh.convertToMsh22(args.input, args.output, plane=args.plane)
    elif args.command == "box":
        if len(args.cells) not in (1, 3):
            parser.error("--cells takes 1 or 3 values")
        cells = args.cells[0] if len(args.cells) == 1 else tuple(args.cells)
        gmshmesh.generateHexBox(args.output, tuple(args.lengths), cells, msh_version=args.msh_version,
                                vp=args.vp, rho=args.rho, structured=not args.unstructured)
        print(f"{args.output} and {args.output.with_suffix('.csv')}")
    elif args.command == "rectangle":
        gmshmesh.generateQuadRectangle(args.output, *args.lengths, args.size, structured=args.structured,
                                       free_surface=not args.no_free_surface, msh_version=args.msh_version)
    else:
        import kokkos
        kokkos.initialize()
        try:
            mesh = gmshmesh.GmshMesh(args.mesh, args.order, args.materials)
            print(mesh)
            print("boundaries:", {t: str(f) for t, f in mesh.boundaries.items()})
            print("materials:", mesh.materials)
            del mesh
        finally:
            import gc
            gc.collect()
            kokkos.finalize()


if __name__ == "__main__":
    main()
