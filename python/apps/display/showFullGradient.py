import os
import argparse
import matplotlib.pyplot as plt

from utilities.display import Hdf5Field

  
if __name__ == "__main__":
    # Parse parameters
    parser = argparse.ArgumentParser(description="HDF5 full gradient display")
    parser.add_argument("files", nargs="*", default=[],
                        help="Filenames. If >1, should have the same shape and extent")
    parser.add_argument("--dir",
                        default=None, type=str,
                        help="File directory. Should be common to all files")
    parser.add_argument("--n", nargs=3, type=int,
                        default=(127,96,122),
                        help="Shape of the model. Default is (127,96,122)")
    parser.add_argument("--extent", nargs=4,
                        default=None, type=int,
                        help="Extent of the graph. Default is None")
    parser.add_argument("--axis", "--slice_axis",
                        dest="axis", default="y",
                        help="Slice axis")
    parser.add_argument("--index", "--slice_index",
                        type=int, dest="index",
                        default=None,
                        help="Slice index")
    parser.add_argument("--fname",
                        default="fullGradient",
                        type=str, dest="fkey",
                        help="Name of the field in the HDF5 file. Default is `fullGradient`")
    parser.add_argument("--vmin",
                        default=None, type=float,
                        help="Graph normalization min bound")
    parser.add_argument("--vmax",
                        default=None, type=float,
                        help="Graph normalization max bound")
    parser.add_argument("--nf", "--norm_factor",
                    default=None, type=float,
                    help="Graph normalization factor")
    parser.add_argument("--diff",
                        default=False, action="store_true",
                        help="Compute and display difference between two fields")

    args, _ = parser.parse_known_args()

    if args.diff == True and len(args.files) == 2:
        diff = True
        grad = []
    else:
        diff = False

    # Extract and display each field individually
    for i, file in enumerate(args.files):
        if args.dir is not None:
            file = os.path.join(args.dir, file)
        fullGradient = Hdf5Field(filename=file, fieldname=args.fkey, n=args.n)

        gradSlice = fullGradient.display(axis=args.axis, index=args.index, extent=args.extent, vmin=args.vmin, vmax=args.vmax, normFactor=args.nf)

        if diff:
            grad.append(gradSlice)

    # Display difference between two if requested
    if diff:
        plt.figure()
        plt.imshow(grad[1]-grad[0],
                   extent=args.extent,
                   vmin=args.vmin, vmax=args.vmax)
        plt.title("difference")
        plt.colorbar()

    plt.show()