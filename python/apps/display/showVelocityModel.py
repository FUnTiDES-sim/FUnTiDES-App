"""
Display HDF5 velocity models of makutu (velocityModel<n>.hdf5, finalModel.hdf5).

The files written by makutu carry their shape; older files need ``--n nx nz`` (2-D) or
``--n nx ny nz`` (3-D). 3-D models are shown as a slice (``--axis``, ``--index``).

    python showVelocityModel.py finalModel.hdf5 velocityModel0.hdf5 --diff
    python showVelocityModel.py finalModel.hdf5 --png        # writes finalModel.png
"""

import argparse
import os

import matplotlib

parser = argparse.ArgumentParser(description="HDF5 velocity model display")
parser.add_argument("files", nargs="+", help="HDF5 files (same shape for --diff)")
parser.add_argument("--dir", default=None, help="Directory of the files")
parser.add_argument("--fname", dest="fkey", default="velocity", help="Dataset name (default velocity)")
parser.add_argument("--vmodel", default="c", choices=("c", "1/c", "1/c2"),
                    help="Parametrisation of the file (displayed as velocity)")
parser.add_argument("--n", nargs="+", type=int, default=None,
                    help="Numbers of elements nx nz or nx ny nz (files without shape attribute)")
parser.add_argument("--lengths", nargs="+", type=float, default=None,
                    help="Domain lengths lx lz or lx ly lz [m] (files without lengths attribute)")
parser.add_argument("--axis", default="y", choices=("x", "y", "z"), help="3-D: constant axis of the slice")
parser.add_argument("--index", type=int, default=None, help="3-D: element index of the slice (default middle)")
parser.add_argument("--extent", nargs=4, type=float, default=None, help="Image extent left right bottom top")
parser.add_argument("--vmin", type=float, default=None)
parser.add_argument("--vmax", type=float, default=None)
parser.add_argument("--nf", "--norm_factor", dest="nf", type=float, default=None, help="Divides the colour bounds")
parser.add_argument("--diff", action="store_true", help="Also show the second model minus the first")
parser.add_argument("--png", action="store_true", help="Save <file>.png (and diff.png) instead of showing")


def main(args):
    if args.png:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from makutu.display import VelocityModel

    models = []
    for file in args.files:
        path = os.path.join(args.dir, file) if args.dir else file
        m = VelocityModel(path, n=args.n, fieldname=args.fkey, type=args.vmodel, lengths=args.lengths)
        png = os.path.splitext(os.path.basename(path))[0] + ".png" if args.png else None
        models.append((m, m.display(axis=args.axis, index=args.index, extent=args.extent, vmin=args.vmin,
                                    vmax=args.vmax, normFactor=args.nf, savefig=png)))
        if png:
            print(f"{png} written")
    if args.diff and len(models) == 2:
        (m, a), (_, b) = models
        hlabel, hlen, vlabel, vlen = m._axes(args.axis)
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
