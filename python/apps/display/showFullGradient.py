"""
Display HDF5 gradients of makutu (fullGradient<n>.hdf5, written in debug mode), or the
illumination (``--fname illumination``).

The files written by makutu carry their shape; older files need ``--n nx nz`` (2-D) or
``--n nx ny nz`` (3-D). 3-D fields are shown as a slice (``--axis``, ``--index``).

    python showFullGradient.py fullGradient0.hdf5 --png
"""

import argparse
import os

import matplotlib

parser = argparse.ArgumentParser(description="HDF5 gradient display")
parser.add_argument("files", nargs="+", help="HDF5 files (same shape for --diff)")
parser.add_argument("--dir", default=None, help="Directory of the files")
parser.add_argument("--fname", dest="fkey", default="fullGradient",
                    help="Dataset name: fullGradient (default) or illumination")
parser.add_argument("--n", nargs="+", type=int, default=None,
                    help="Numbers of elements nx nz or nx ny nz (files without shape attribute)")
parser.add_argument("--lengths", nargs="+", type=float, default=None,
                    help="Domain lengths lx lz or lx ly lz [m] (files without lengths attribute)")
parser.add_argument("--axis", default="y", choices=("x", "y", "z"), help="3-D: constant axis of the slice")
parser.add_argument("--index", type=int, default=None, help="3-D: element index of the slice (default middle)")
parser.add_argument("--extent", nargs=4, type=float, default=None, help="Image extent left right bottom top")
parser.add_argument("--vmin", type=float, default=None, help="Default -max|g| (symmetric)")
parser.add_argument("--vmax", type=float, default=None, help="Default +max|g|")
parser.add_argument("--nf", "--norm_factor", dest="nf", type=float, default=None, help="Divides the colour bounds")
parser.add_argument("--diff", action="store_true", help="Also show the second field minus the first")
parser.add_argument("--png", action="store_true", help="Save <file>.png (and diff.png) instead of showing")


def main(args):
    if args.png:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from makutu.display import Hdf5Field

    fields = []
    for file in args.files:
        path = os.path.join(args.dir, file) if args.dir else file
        g = Hdf5Field(path, fieldname=args.fkey, n=args.n, lengths=args.lengths)
        vmin, vmax, cmap = args.vmin, args.vmax, None
        if args.fkey != "illumination":              # signed field: symmetric bounds
            lim = max(abs(g.getMinValue()), abs(g.getMaxValue()))
            vmin = -lim if vmin is None else vmin
            vmax = lim if vmax is None else vmax
            cmap = "seismic"
        png = os.path.splitext(os.path.basename(path))[0] + f"_{args.fkey}.png" if args.png else None
        fields.append((g, g.display(axis=args.axis, index=args.index, extent=args.extent, vmin=vmin,
                                    vmax=vmax, normFactor=args.nf, cmap=cmap, savefig=png)))
        if png:
            print(f"{png} written")
    if args.diff and len(fields) == 2:
        (g, a), (_, b) = fields
        hlabel, hlen, vlabel, vlen = g._axes(args.axis)
        fig = plt.figure()
        d = b - a
        lim = abs(d).max()
        plt.imshow(d, extent=args.extent or (0.0, hlen, vlen, 0.0), cmap="seismic", vmin=-lim, vmax=lim)
        plt.xlabel(hlabel)
        plt.ylabel(vlabel)
        plt.title(f"{os.path.basename(args.files[1])} - {os.path.basename(args.files[0])}")
        plt.colorbar()
        if args.png:
            fig.savefig("diff.png", dpi=150, bbox_inches="tight")
            print("diff.png written")
    if not args.png:
        plt.show()


if __name__ == "__main__":
    main(parser.parse_args())
