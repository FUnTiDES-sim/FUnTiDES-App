"""
Mesh adapted to a SEP velocity model and to the source frequency (makutu.mesh.seismesh).

The element size follows the velocity: h = order * v / (fmax * ppw), with fmax = 2.5 f0 for
a Ricker wavelet of peak frequency f0 (shortest wavelength v_min / (2.5 f0), as
makutu.mesh.suggest_mesh). Several meshes are generated and evaluated (2-D: tensor grid and
2:1 doubling layers; 3-D: also hybrid local 3-refinement and extruded unstructured quads);
the cheapest valid one is written to the output directory:

  mesh.msh        MSH 2.2, z up, free surface on top (group 1), absorbing sides (group 2),
                  one physical group per velocity class
  materials.csv   physical_tag,vp,rho (vp: mean of the SEP samples inside each element)
  report.json     quality, sampling, CFL time step, cost of the selected mesh

Modelling on it:  mesh = makutu.mesh.GmshMesh("out/mesh.msh", order, "out/materials.csv")
                  AcousticSolver(..., **mesh.solver_kwargs())
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vp-file", required=True, type=Path, help="SEP velocity (2-D: n3 = 1, axis 2 = depth)")
    parser.add_argument("--f0", required=True, type=float, help="peak frequency of the Ricker source [Hz]")
    parser.add_argument("--out", type=Path, default=Path("seismesh_out"), help="output directory")
    parser.add_argument("--order", type=int, default=4, help="SEM polynomial order")
    parser.add_argument("--ppw", type=float, default=5.0, help="GLL points per shortest wavelength")
    parser.add_argument("--rho-file", type=Path, help="SEP density (default: constant --rho)")
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--t-sim", type=float, default=4.0, help="simulated time for the cost index [s]")
    parser.add_argument("--lts", action="store_true", help="rank the meshes by their cost with local time stepping")
    parser.add_argument("--strategies", nargs="+", choices=["tensor", "doubling", "hybrid", "extrude"],
                        help="strategies to try (default: all those of the dimension)")
    args = parser.parse_args()

    from makutu.mesh.seismesh import build_mesh_from_sep
    result = build_mesh_from_sep(args.vp_file, args.f0, args.out, order=args.order, ppw=args.ppw,
                                 rho_file=args.rho_file, rho=args.rho, t_sim=args.t_sim, lts=args.lts,
                                 strategies=args.strategies)
    sel = result["selected"]
    print(json.dumps({"mesh": str(result["msh"]), "materials": str(result["materials"]),
                      "strategy": sel["strategy"], "elements": sel["n_elements"],
                      "dt_cfl_s": sel["dt_cfl"], "ppw_min": sel["ppw_effective"]["min"]}, indent=1))


if __name__ == "__main__":
    main()
