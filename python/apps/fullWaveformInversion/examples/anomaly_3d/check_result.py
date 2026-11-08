"""
Result of the synthetic 3-D case of make_case.py: cost function history, model error and
position of the recovered anomaly; optional figure (sections through the anomaly).

Usage
-----
    python check_result.py --rundir anomaly_3d_run [--model finalModel.hdf5] [--png result.png]
"""

import argparse
import os

import h5py
import numpy as np

from case import ANOMALIES, ELEM, LX, LY, LZ, Z_REFLECTOR, models


def main():
    p = argparse.ArgumentParser("Check the 3-D FWI result")
    p.add_argument("--rundir", default="anomaly_3d_run")
    p.add_argument("--model", default="finalModel.hdf5")
    p.add_argument("--png", default=None)
    a = p.parse_args()

    X, Y, Z, v0, vtrue = models(ELEM)                      # one value per element
    with h5py.File(os.path.join(a.rundir, a.model), "r") as f:
        v = f["velocity"][:].reshape(v0.shape)

    cf = os.path.join(a.rundir, "fullCostFunction.txt")
    if os.path.exists(cf):
        it, J = np.loadtxt(cf, ndmin=2).T
        print(f"{len(J) - 1} iterations, J {J[0]:.4g} -> {J[-1]:.4g} (J / J0 = {J[-1] / J[0]:.3g})")
    above = Z < Z_REFLECTOR - 50.0
    err = np.linalg.norm((v - vtrue)[above]) / np.linalg.norm((v0 - vtrue)[above])
    print(f"Model error above the reflector |v - vtrue| / |v0 - vtrue| = {err:.3f}")
    for x, y, z, r, dv in ANOMALIES:
        near = ((X - x) ** 2 + (Y - y) ** 2 + (Z - z) ** 2 < (2 * r) ** 2) & above   # not the reflector imprint
        k = np.unravel_index(np.argmax(np.where(near, (v - v0) * np.sign(dv), -np.inf)), v.shape)
        c = np.unravel_index(np.argmin((X - x) ** 2 + (Y - y) ** 2 + (Z - z) ** 2), v.shape)
        print(f"Anomaly {dv:+g} m/s at ({x:g}, {y:g}, {z:g}) m: {v[c] - v0[c]:+.0f} m/s at its centre, "
              f"extremum {v[k] - v0[k]:+.0f} m/s at ({X[k]:g}, {Y[k]:g}, {Z[k]:g}) m")

    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x, y, z = ANOMALIES[0][:3]
        iy, iz = int(y // ELEM), int(z // ELEM)
        fig, axes = plt.subplots(2, 3, figsize=(15, 7))
        for col, (m, title) in enumerate(((vtrue, "True"), (v0, "Initial"), (v, "Inverted"))):
            kw = dict(cmap="jet", vmin=v0.min(), vmax=vtrue[above].max())
            axes[0, col].imshow(m[:, iy, :], extent=(0, LX, LZ, 0), **kw)
            axes[0, col].set_title(f"{title}, y = {Y[0, iy, 0]:g} m")
            axes[0, col].set_xlabel("x [m]")
            axes[0, col].set_ylabel("z [m]")
            im = axes[1, col].imshow(m[iz], extent=(0, LX, LY, 0), **kw)
            axes[1, col].set_title(f"{title}, z = {Z[iz, 0, 0]:g} m")
            axes[1, col].set_xlabel("x [m]")
            axes[1, col].set_ylabel("y [m]")
        fig.colorbar(im, ax=axes, shrink=0.8, label="vp [m/s]")
        fig.savefig(a.png, dpi=110, bbox_inches="tight")
        print(f"Figure: {a.png}")


if __name__ == "__main__":
    main()
