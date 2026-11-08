"""Definition of the synthetic 3-D case (shared by make_case.py and check_result.py)."""

import numpy as np

LX, LY, LZ = 1000.0, 1000.0, 600.0        # model extent [m]
V0 = 2000.0                               # background velocity [m/s]
Z_REFLECTOR, V_BELOW = 450.0, 2600.0      # known reflector (in both models): depth [m], velocity below
ANOMALIES = [(500.0, 500.0, 250.0, 120.0, +250.0)]       # x, y, z, radius, dv
F0, ORDER, ELEM = 8.0, 2, 25.0            # source peak frequency [Hz], order, element size [m]
SOURCES_XY = [(x, y) for y in (250.0, 500.0, 750.0) for x in (250.0, 500.0, 750.0)]
RECEIVERS_XY = [(x, y) for y in np.linspace(100.0, 900.0, 9) for x in np.linspace(100.0, 900.0, 9)]
DEPTH_SRC, DEPTH_RCV = 25.0, 25.0         # [m]
MAXTIME = 1.0                             # [s]


def models(h):
    """Cell centres X, Y, Z (z down) of a grid of step h, initial and true models, shape
    (nz, ny, nx). Initial model: the background with the reflector; true model: plus the
    anomaly."""
    c = lambda length: (np.arange(int(round(length / h))) + 0.5) * h   # noqa: E731
    Z, Y, X = np.meshgrid(c(LZ), c(LY), c(LX), indexing="ij")
    v0 = np.where(Z < Z_REFLECTOR, V0, V_BELOW)
    vtrue = v0.copy()
    for x, y, z, r, dv in ANOMALIES:
        vtrue += dv * np.exp(-((X - x) ** 2 + (Y - y) ** 2 + (Z - z) ** 2) / r ** 2)
    return X, Y, Z, v0, vtrue
