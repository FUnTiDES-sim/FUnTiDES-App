import os
import argparse
import matplotlib.pyplot as plt

from utilities.display import VelocityModel


if __name__ == "__main__":
    # Parse parameters
    parser = argparse.ArgumentParser(description="HDF5 velocity model display")
    parser.add_argument("files", nargs="*", default=[],
                        help="Filenames. If >1, should have the same shape and extent")
    parser.add_argument("--dir",
                        default=None, type=str,
                        help="File directory. Should be common to all files")
    parser.add_argument("--n", nargs=3, type=int, default=(127, 96, 122),
                        help="Shape of the model")
    parser.add_argument("--extent", nargs=4,
                        default=None, type=int,
                        help="Extent of the graph")
    parser.add_argument("--axis", "--slice_axis",
                        default="y", type=str,
                        dest="axis",
                        help="Slice axis")
    parser.add_argument("--index", "--slice_index",
                        type=int, dest="index",
                        default=None,
                        help="Slice index")
    parser.add_argument("--vmodel", default="c",
                        help="Model, 'c', '1/c' or '1/c2'")
    parser.add_argument("--fname", 
                        default="velocity", type=str, dest="fkey",
                        help="Name of the field in the HDF5 file. Default is `velocity`")
    parser.add_argument("--vmin",
                        default=None, type=float,
                        help="Graph min bound value")
    parser.add_argument("--vmax",
                        default=None, type=float,
                        help="Graph max bound value")
    parser.add_argument("--nf", "--norm_factor",
                        default=None, type=float,
                        help="Graph normalization factor")
    parser.add_argument("--diff",
                        default=False, action="store_true",
                        help="Compute and display difference between two velocity models")

    args, _ = parser.parse_known_args()
    
    if args.diff == True and len(args.files) == 2:
        diff = True
        vel = []
    else:
        diff = False

    # Extract and display
    for i, file in enumerate(args.files):
        if args.dir is not None:
            file = os.path.join(args.dir, file)
        velocityModel = VelocityModel(filename=file, fieldname=args.fkey, type=args.vmodel, n=args.n)

        velSlice = velocityModel.display(axis=args.axis, index=args.index, extent=args.extent, vmin=args.vmin, vmax=args.vmax, normFactor=args.nf)

        if diff:
            vel.append(velSlice)

    # Display difference between two models if requested
    if diff:
        plt.figure()
        plt.imshow(vel[1]-vel[0],
                   extent=args.extent,
                   vmin=args.vmin, vmax=args.vmax)
        plt.title("difference")
        plt.colorbar()
        
    plt.show()