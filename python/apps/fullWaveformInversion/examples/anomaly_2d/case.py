"""Definition of the synthetic 2-D case (shared by make_case.py and check_result.py)."""

import numpy as np

LX, LZ = 2000.0, 1000.0                   # model extent [m]
V0 = 2000.0                               # background velocity [m/s]
Z_REFLECTOR, V_BELOW = 800.0, 2600.0      # known reflector (in both models): depth [m], velocity below
ANOMALIES = [(700.0, 450.0, 120.0, +250.0), (1300.0, 550.0, 120.0, -250.0)]   # x, z, radius, dv
F0, ORDER, ELEM = 10.0, 2, 25.0           # source peak frequency [Hz], order, element size [m]
SOURCES_X = np.linspace(250.0, 1750.0, 8)
RECEIVERS_X = np.linspace(100.0, 1900.0, 73)
DEPTH_SRC, DEPTH_RCV = 25.0, 25.0         # [m]
MAXTIME = 1.2                             # [s]


def models(h):
    """Cell centres X, Z (z down) of a grid of step h, initial and true models, shape (nz, nx).
    Initial model: the background with the reflector; true model: plus the anomalies."""
    X, Z = np.meshgrid((np.arange(int(round(LX / h))) + 0.5) * h, (np.arange(int(round(LZ / h))) + 0.5) * h)
    v0 = np.where(Z < Z_REFLECTOR, V0, V_BELOW)
    vtrue = v0.copy()
    for x, z, r, dv in ANOMALIES:
        vtrue += dv * np.exp(-((X - x) ** 2 + (Z - z) ** 2) / r ** 2)
    return X, Z, v0, vtrue
