"""
Summary and map of an acquisition read from SEG-Y files or decision files.

Command line of :mod:`makutu.acquisition.check_acquisition`.

Usage
-----
    python apps/acquisition/check_acquisition.py --dir acquisition --out survey
    python apps/acquisition/check_acquisition.py --from-decision acquisition/survey --no-plot
"""

import argparse
import glob
import os

from makutu.acquisition.check_acquisition import check_acquisition, check_from_decision, plot_acquisition, print_report


def _build_parser():
    p = argparse.ArgumentParser(
        description=__doc__.strip().splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("files", nargs="*",
                   help="SEG-Y files to check (positional).")
    p.add_argument("--dir", metavar="DIR",
                   help="Directory containing SEG-Y files (*.segy / *.sgy).")
    p.add_argument("--from-decision", metavar="ROOTNAME",
                   help="Decision-file rootname (e.g. acquisition/survey). "
                        "Reads <rootname>.mapped.data — much faster than "
                        "scanning SEG-Y files for large surveys.")

    p.add_argument("--out", default=None,
                   help="Rootname for output plot PNG.  "
                        "Default: <dir>/acquisition_check.png or ./acquisition_check.png")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip plotting (report only).")
    p.add_argument("--show", action="store_true",
                   help="Display the plot interactively (implies matplotlib GUI).")

    # decision-file words (--from-decision); the SEG-Y files use the standard header words
    p.add_argument("--word-sx", type=int, default=22)
    p.add_argument("--word-sy", type=int, default=23)
    p.add_argument("--word-sz", type=int, default=14)
    p.add_argument("--word-rx", type=int, default=24)
    p.add_argument("--word-ry", type=int, default=25)
    p.add_argument("--word-rz", type=int, default=13)
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)

    kw = dict(w_sx=args.word_sx, w_sy=args.word_sy, w_sz=args.word_sz,
              w_rx=args.word_rx, w_ry=args.word_ry, w_rz=args.word_rz)

    n_src = bool(args.from_decision) + bool(args.dir) + bool(args.files)
    if n_src == 0:
        raise SystemExit("Provide SEG-Y files, --dir, or --from-decision.")
    if n_src > 1:
        raise SystemExit("Use only one of: positional files, --dir, --from-decision.")

    if args.from_decision:
        summary  = check_from_decision(args.from_decision, **kw)
        plot_dir = os.path.dirname(args.from_decision) or "."
    elif args.dir:
        patterns = [os.path.join(args.dir, "*.segy"),
                    os.path.join(args.dir, "*.sgy")]
        paths = sorted(p for pat in patterns for p in glob.glob(pat))
        if not paths:
            raise SystemExit(f"No SEG-Y files found in {args.dir}")
        print(f"Found {len(paths)} SEG-Y file(s).")
        summary  = check_acquisition(paths)
        plot_dir = args.dir
    else:
        paths = sorted(args.files)
        print(f"Found {len(paths)} SEG-Y file(s).")
        summary  = check_acquisition(paths)
        plot_dir = "."

    print_report(summary)

    if not args.no_plot:
        import matplotlib
        if not args.show:
            matplotlib.use("Agg")
        out_png = args.out if args.out else os.path.join(plot_dir, "acquisition_check.png")
        if not out_png.endswith(".png"):
            out_png += ".png"
        plot_acquisition(summary, out_path=out_png, show=args.show)


if __name__ == "__main__":
    main()
