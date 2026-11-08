"""
Display seismograms: SEG-Y files written by makutu (seismoTrace_shot<id>.sgy), or SEP
files (.H, axis 1 = time).

    python showSeismicTraces.py seismoTrace_shot0.sgy --png
"""

import argparse
import os

import matplotlib

parser = argparse.ArgumentParser(description="Seismic traces display (.sgy/.segy or .H)")
parser.add_argument("files", nargs="+", help="Files")
parser.add_argument("--dir", default=None, help="Directory of the files")
parser.add_argument("--vmin", type=float, default=None, help="Default -max|data| (symmetric)")
parser.add_argument("--vmax", type=float, default=None, help="Default +max|data|")
parser.add_argument("--nf", "--norm_factor", dest="nf", type=float, default=None, help="Divides the colour bounds")
parser.add_argument("--png", action="store_true", help="Save <file>.png instead of showing")


def show_sep(path, args, png):
    import matplotlib.pyplot as plt
    import numpy as np
    from makutu.input import read_sep
    sep = read_sep(path)
    data = sep.data.reshape(-1, sep.n1).T                  # (time, traces)
    amax = np.max(np.abs(data))
    vmin = -amax if args.vmin is None else args.vmin
    vmax = amax if args.vmax is None else args.vmax
    if args.nf is not None:
        vmin, vmax = vmin / args.nf, vmax / args.nf
    fig = plt.figure()
    plt.imshow(data, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax,
               extent=(0.5, data.shape[1] + 0.5, sep.o1 + sep.d1 * sep.n1, sep.o1))
    plt.xlabel("Traces", fontsize=12)
    plt.ylabel("Time [s]", fontsize=12)
    plt.title(os.path.basename(path), fontsize=12)
    plt.colorbar()
    if png:
        fig.savefig(png, dpi=150, bbox_inches="tight")
        plt.close(fig)


def main(args):
    if args.png:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from makutu.display import SEGYSeismicTrace

    for file in args.files:
        path = os.path.join(args.dir, file) if args.dir else file
        root, ext = os.path.splitext(path)
        png = os.path.basename(root) + ".png" if args.png else None
        if ext == ".H":
            show_sep(path, args, png)
        elif ext in (".sgy", ".segy"):
            d, f = os.path.split(path)
            SEGYSeismicTrace(filename=f, directory=d or ".").display(vmin=args.vmin, vmax=args.vmax,
                                                                     normFactor=args.nf, savefig=png)
        else:
            print(f"{path}: format {ext} not handled (.sgy, .segy or .H)")
            continue
        if png:
            print(f"{png} written")
    if not args.png:
        plt.show()


if __name__ == "__main__":
    main(parser.parse_args())
