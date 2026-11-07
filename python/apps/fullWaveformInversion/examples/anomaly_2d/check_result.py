"""
Result of the synthetic 2-D case of make_case.py: cost function history, model error and
position of the recovered anomalies; optional figure (true, initial, final models).

Usage
-----
    python check_result.py --rundir anomaly_2d_run [--model finalModel.hdf5] [--png result.png]
"""

import argparse
import glob
import os

import h5py
import numpy as np

from case import LX, LZ, ELEM, ANOMALIES, models


def summary(v, X, Z, v0, vtrue):
    """Relative model error, and value and position of the extremum near every anomaly."""
    err = np.linalg.norm(v - vtrue) / np.linalg.norm(v0 - vtrue)
    lines = []
    for x, z, r, dv in ANOMALIES:
        near = (X - x) ** 2 + (Z - z) ** 2 < (2 * r) ** 2
        dvr = np.where(near, (v - v0) * np.sign(dv), -np.inf)
        k = np.unravel_index(np.argmax(dvr), v.shape)
        lines.append((x, z, dv, X[k], Z[k], (v - v0)[k]))
    return err, lines


def main():
    p = argparse.ArgumentParser("Check the synthetic 2-D FWI result")
    p.add_argument("--rundir", default="anomaly_2d_run")
    p.add_argument("--model", default="finalModel.hdf5")
    p.add_argument("--png", default=None)
    a = p.parse_args()

    X, Z, v0, vtrue = models(ELEM)          # the model of fwi.py is per element
    with h5py.File(os.path.join(a.rundir, a.model), "r") as f:
        v = f["velocity"][:].reshape(X.shape)

    for cf in sorted(glob.glob(os.path.join(a.rundir, "fullCostFunction*.txt"))):
        J = np.loadtxt(cf)[:, 1]
        print(f"{os.path.basename(cf)}: {len(J) - 1} iterations, J {J[0]:.4g} -> {J[-1]:.4g} "
              f"(J / J0 = {J[-1] / J[0]:.3g})")
    err, lines = summary(v, X, Z, v0, vtrue)
    print(f"Model error |v - vtrue| / |v0 - vtrue| = {err:.3f}")
    print(f"Model range {v.min():.0f} - {v.max():.0f} m/s")
    for x, z, dv, xr, zr, dvr in lines:
        print(f"Anomaly {dv:+.0f} m/s at ({x:.0f}, {z:.0f}) m: recovered {dvr:+.0f} m/s at ({xr:.0f}, {zr:.0f}) m")

    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
        ext = (0, LX, LZ, 0)
        for ax, m, title in zip(axes, (vtrue, v0, v), ("True", "Initial", "Inverted")):
            im = ax.imshow(m, extent=ext, cmap="jet", vmin=vtrue.min(), vmax=vtrue.max())
            ax.set_title(title)
            ax.set_ylabel("z [m]")
        axes[-1].set_xlabel("x [m]")
        fig.colorbar(im, ax=axes, label="vp [m/s]")
        fig.savefig(a.png, dpi=100)
        print(f"Figure: {a.png}")


if __name__ == "__main__":
    main()
