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
            fformat = "SEP"
            from makutu.model import SEPModel
            sepmodel = SEPModel(file)

            sepmodel.display2d(vmin=args.vmin, vmax=args.vmax, normFactor=args.nf)
            plt.title(f"Shot {i}", fontsize = 10)

        elif ext in (".sgy", ".segy"):
            fformat = "SEGY"
            
            segpath, segfile = os.path.split(file)
            seismos = SEGYSeismicTrace(filename=segfile, directory=segpath)

            seismos.display(vmin=args.vmin, vmax=args.vmax, normFactor=args.nf)

        else:
            print(f"Format {ext} not handled. Only .H and .sgy/.segy files can be accepted")

    plt.show()