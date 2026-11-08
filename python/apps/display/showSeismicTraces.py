import os
import argparse
import matplotlib.pyplot as plt

from makutu.display import SEGYSeismicTrace


if __name__ == "__main__":
    # Parse parameters
    parser = argparse.ArgumentParser(description="Basic seismic traces display. Only .H and .sgy/.segy files handled")
    parser.add_argument("files", nargs="*", default=[],
                        help="Filenames")
    parser.add_argument("--dir",
                        default=None, type=str,
                        help="File directory. Should be common to all files")
    parser.add_argument("--vmin",
                        default=None, type=float,
                        help="Bound min")
    parser.add_argument("--vmax",
                        default=None, type=float,
                        help="Bound max")
    parser.add_argument("--nf", "--norm_factor",
                        default=None, type=float,
                        help="Normalisation factor")
    

    args, _ = parser.parse_known_args()

    # Extract and display each field individually
    for i, file in enumerate(args.files):
        if args.dir is not None:
            file = os.path.join(args.dir, file)
        _, ext = os.path.splitext(file)
        
        if ext == ".H":
            # SEP traces: axis 1 is time, the other axes are the traces
            from makutu.input.SEPReader import read_sep
            sep = read_sep(file)
            data = sep.data.reshape(-1, sep.n1).T
            vmin, vmax = args.vmin, args.vmax
            if args.nf is not None:
                vmin = (data.min() if vmin is None else vmin) / args.nf
                vmax = (data.max() if vmax is None else vmax) / args.nf
            plt.figure()
            plt.imshow(data, aspect="auto", vmin=vmin, vmax=vmax,
                       extent=[0, data.shape[1], sep.o1 + sep.d1 * sep.n1, sep.o1])
            plt.xlabel("Traces", fontsize=12)
            plt.ylabel("Time [s]", fontsize=12)
            plt.title(f"Shot {i}", fontsize=10)

        elif ext in (".sgy", ".segy"):
            segpath, segfile = os.path.split(file)
            seismos = SEGYSeismicTrace(filename=segfile, directory=segpath)

            seismos.display(vmin=args.vmin, vmax=args.vmax, normFactor=args.nf)

        else:
            print(f"Format {ext} not handled. Only .H and .sgy/.segy files can be accepted")

    plt.show()