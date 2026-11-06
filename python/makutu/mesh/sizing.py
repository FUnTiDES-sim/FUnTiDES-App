"""Mesh and time-step sizing for the Cartesian SEM meshes (2-D and 3-D)."""

import math

import numpy as np
from numpy.polynomial import legendre


def gll_min_spacing(order):
    """Smallest gap between the GLL points of an element, as a fraction of the element size."""
    inner = legendre.Legendre.basis(order).deriv().roots().real if order > 1 else np.array([])
    points = np.sort(np.concatenate(([-1.0], inner, [1.0])))
    return float(np.min(np.diff(points))) / 2.0


def suggest_mesh(lengths, vp_min, vp_max, f0, order, ppw=6.0, cfl=0.3, elem_size=None):
    """
    Element counts and time step for a box of the given lengths.

    - Shortest wavelength of a Ricker wavelet of peak frequency f0:
      lambda_min = vp_min / (2.5 f0).
    - Element size: h = order * lambda_min / ppw (ppw GLL points per shortest
      wavelength), or ``elem_size`` when given.
    - Time step: dt = cfl * (smallest GLL spacing) / vp_max.

    Parameters
    ----------
    lengths : sequence of float
        Domain lengths (lx, lz) in 2-D or (lx, ly, lz) in 3-D [m].
    vp_min, vp_max : float
        Velocity range [m/s].
    f0 : float
        Peak frequency [Hz].
    order : int
        Polynomial order.
    ppw : float
        GLL points per shortest wavelength (5 is a usual minimum).
    cfl : float
        CFL number.
    elem_size : float, optional
        Element size [m]; the counts are then round(length / elem_size).

    Returns
    -------
    counts : list of int
        Number of elements along each length.
    dt : float
        Time step [s].
    lambda_min : float
        Shortest wavelength [m].
    """
    lambda_min = vp_min / (2.5 * f0)
    if elem_size is None:
        h = order * lambda_min / ppw
        counts = [max(1, math.ceil(length / h)) for length in lengths]
    else:
        counts = [max(1, int(round(length / elem_size))) for length in lengths]
    h_min = min(length / n for length, n in zip(lengths, counts))
    dt = cfl * gll_min_spacing(order) * h_min / vp_max
    return counts, dt, lambda_min
