"""
Result of the Marmousi 2-D case of make_case.py: cost function per stage, model error below
the water against the true model, optional figure (true, initial, inverted models and a
vertical profile).

Usage
-----
    python check_result.py --rundir marmousi_2d_run [--model finalModel.hdf5] [--png result.png]
"""

import argparse
import glob
import os

import h5py
import numpy as np

from makutu.input.SEPReader import read_sep
from makutu.mesh.CartesianModel2D import sample_sep_2d


def main():
    p = argparse.ArgumentParser("Check the Marmousi 2-D FWI result")
    p.add_argument("--rundir", default="marmousi_2d_run")
    p.add_argument("--model", default="finalModel.hdf5")
    p.add_argument("--png", default=None)
    a = p.parse_args()

    sep = read_sep(os.path.join(a.rundir, "true.H"))
    lx, lz = sep.n1 * sep.d1, sep.n2 * sep.d2
    with open(os.path.join(a.rundir, "fwi_parameters.txt")) as f:
        params = dict(line.split("#")[0].strip().split("=", 1) for line in f if "=" in line.split("#")[0])
    elem = float(params["elemSize"])
    shape = (max(1, int(round(lz / elem))), max(1, int(round(lx / elem))))    # as makutu.mesh.suggest_mesh
    ez, ex = shape                                              # same sampling as the mesh
    vtrue = np.asarray(sample_sep_2d(sep, ex, ez, lx, lz)).reshape(shape)
    v0 = np.asarray(sample_sep_2d(read_sep(os.path.join(a.rundir, "init.H")), ex, ez, lx, lz)).reshape(shape)
    with h5py.File(os.path.join(a.rundir, a.model), "r") as f:
        v = f["velocity"][:].reshape(shape)

    for cf in sorted(glob.glob(os.path.join(a.rundir, "fullCostFunction*.txt"))):
        it, J = np.loadtxt(cf, ndmin=2).T
        starts = list(np.flatnonzero(it == 0)) + [len(J)]           # one stage per filter frequency
        for k, (i0, i1) in enumerate(zip(starts[:-1], starts[1:])):
            print(f"stage {k + 1}: {i1 - i0 - 1} iterations, J {J[i0]:.4g} -> {J[i1 - 1]:.4g} "
                  f"(J / J0 = {J[i1 - 1] / J[i0]:.3g})")
    below = vtrue > 1500.5
    err = np.linalg.norm((v - vtrue)[below]) / np.linalg.norm((v0 - vtrue)[below])
    print(f"Model error below the water |v - vtrue| / |v0 - vtrue| = {err:.3f}")
    print(f"Model range {v.min():.0f} - {v.max():.0f} m/s")

    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 1, figsize=(8, 12))
        ext = (0, lx, lz, 0)
        for ax, m, title in zip(axes, (vtrue, v0, v), ("True", "Initial", "Inverted")):
            im = ax.imshow(m, extent=ext, cmap="jet", vmin=1500, vmax=vtrue.max(), aspect="auto")
            ax.set_title(title)
            ax.set_ylabel("z [m]")
        fig.colorbar(im, ax=axes[:3], label="vp [m/s]")
        z = (np.arange(shape[0]) + 0.5) * elem
        for x, ls in ((3000.0, "-"), (6500.0, "--")):
            i = int(x / elem)
            axes[3].plot(z, vtrue[:, i], "k" + ls, z, v0[:, i], "b" + ls, z, v[:, i], "r" + ls)
        axes[3].set_title("Profiles at x = 3000 m (solid) and 6500 m (dashed): true (black), initial (blue), "
                          "inverted (red)", fontsize=8)
        axes[3].set_xlabel("z [m]")
        axes[3].set_ylabel("vp [m/s]")
        fig.savefig(a.png, dpi=80)
        print(f"Figure: {a.png}")


if __name__ == "__main__":
    main()
