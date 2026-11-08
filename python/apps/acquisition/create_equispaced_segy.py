"""
Regular acquisition: one SEG-Y file per shot (geometry headers, zero traces) and the
decision files.

Command line of :mod:`makutu.acquisition.create_equispaced_segy`.

Usage
-----
    python apps/acquisition/create_equispaced_segy.py --xmin 0 --xmax 2000 --ymin 0 --ymax 0 \
        --n-src-x 8 --n-rcv-x 73 --output-dir acquisition --out survey
"""

import argparse

from makutu.acquisition.create_equispaced_segy import equispaced_shots, write_survey


def _build_parser():
    p = argparse.ArgumentParser(
        description=__doc__.strip().splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Domain
    p.add_argument("--xmin", type=float, required=True, help="Domain xmin [m]")
    p.add_argument("--xmax", type=float, required=True, help="Domain xmax [m]")
    p.add_argument("--ymin", type=float, required=True, help="Domain ymin [m]")
    p.add_argument("--ymax", type=float, required=True, help="Domain ymax [m]")

    # Depth
    p.add_argument("--z-src", type=float, default=10.0,
                   help="Source depth [m] (default: 10)")
    p.add_argument("--z-rcv", type=float, default=5.0,
                   help="Receiver depth [m] (default: 5)")

    # Source grid
    p.add_argument("--n-src-x", type=int, default=20,
                   help="Number of sources along X (default: 20)")
    p.add_argument("--n-src-y", type=int, default=1,
                   help="Number of source lines along Y (default: 1)")

    # Receiver grid
    p.add_argument("--n-rcv-x", type=int, default=200,
                   help="Number of receivers along X (default: 200)")
    p.add_argument("--n-rcv-y", type=int, default=1,
                   help="Number of receiver lines along Y (default: 1)")

    # Margin
    p.add_argument("--margin", type=float, default=0.0,
                   help="Inset margin from domain edges [m] (default: 0)")

    # Trace parameters
    p.add_argument("--n-samples", type=int, default=1001,
                   help="Number of time samples per trace (default: 1001)")
    p.add_argument("--dt", type=float, default=2.0,
                   help="Sample interval [ms] (default: 2.0)")

    # Decision-file word numbers
    p.add_argument("--word-sx", type=int, default=22)
    p.add_argument("--word-sy", type=int, default=23)
    p.add_argument("--word-sz", type=int, default=14)
    p.add_argument("--word-rx", type=int, default=24)
    p.add_argument("--word-ry", type=int, default=25)
    p.add_argument("--word-rz", type=int, default=13)

    # Output
    p.add_argument("--out", default="survey",
                   help="Rootname for SEG-Y files and decision files (default: survey)")
    p.add_argument("--output-dir", default="acquisition",
                   help="Output directory (default: acquisition)")

    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)

    print("Building equispaced acquisition geometry ...")
    shots = equispaced_shots(
        xmin=args.xmin, xmax=args.xmax,
        ymin=args.ymin, ymax=args.ymax,
        n_src_x=args.n_src_x, n_src_y=args.n_src_y,
        n_rcv_x=args.n_rcv_x, n_rcv_y=args.n_rcv_y,
        z_src=args.z_src, z_rcv=args.z_rcv,
        margin=args.margin,
    )

    print(f"Writing {len(shots)} SEG-Y shot files ...")
    write_survey(
        shots=shots,
        out=args.out,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        dt_ms=args.dt,
        w_sx=args.word_sx, w_sy=args.word_sy, w_sz=args.word_sz,
        w_rx=args.word_rx, w_ry=args.word_ry, w_rz=args.word_rz,
    )
    print("Done.")


if __name__ == "__main__":
    main()
