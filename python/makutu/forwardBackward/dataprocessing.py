"""
Processing of the data for the inversion of field data: selection of the residual (offset
range, mute, time window) and estimation of the source wavelet.

Selection: weights W(t, receiver) between 0 and 1; the cost of a shot becomes
J = dt / 2 * sum((W r)^2) and its adjoint source W^2 r (:meth:`DataSelection.apply` applied
twice), so that the gradient stays the derivative of J.

Source: least-squares filter c(f) such that c * simulated ~ observed, summed over the
traces, the shots and the MPI ranks (Pratt, 1999):
``c(f) = sum conj(S) D / (sum |S|^2 + eps max sum |S|^2)``, applied to the source wavelet
(:func:`applySourceCorrection`), at each filter frequency.
"""

import math

import numpy as np


def _ramp(x, width):
    """0 for x <= 0, cosine ramp to 1 at x = width, 1 beyond."""
    if width <= 0:
        return (x > 0).astype(float)
    t = np.clip(x / width, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * t)


class DataSelection:
    """
    Weights of the residual per time sample and receiver.

    Parameters
    ----------
        offsetMin, offsetMax : float, optional
            Horizontal source-receiver offsets kept [m] \
            Default is all
        muteVelocity : float, optional
            Top mute: samples before ``muteTime + offset / muteVelocity`` are set to 0
            (e.g. noise before the first arrivals) \
            Default is None (no mute)
        muteTime : float, optional
            Time of the mute line at zero offset [s] \
            Default is 0
        window : float, optional
            With a mute: length [s] of the window kept after the mute line (e.g. the
            early arrivals only) \
            Default is None (to the end of the traces)
        taper : float, optional
            Length [s] of the cosine ramps of the mute and of the window \
            Default is 0.05
    """
    def __init__(self, offsetMin=0.0, offsetMax=math.inf, muteVelocity=None, muteTime=0.0, window=None,
                 taper=0.05):
        if window is not None and muteVelocity is None:
            raise ValueError("DataSelection: the time window follows the mute line (give muteVelocity)")
        self.offsetMin, self.offsetMax = float(offsetMin), float(offsetMax)
        self.muteVelocity = None if muteVelocity is None else float(muteVelocity)
        self.muteTime, self.window, self.taper = float(muteTime), window, float(taper)

    def __repr__(self):
        return (f"DataSelection(offsets {self.offsetMin:g}-{self.offsetMax:g} m, mute "
                f"{'none' if self.muteVelocity is None else f'{self.muteTime:g} s + offset / {self.muteVelocity:g} m/s'}"
                f", window {self.window}, taper {self.taper:g} s)")

    def weights(self, nt, dt, source, receivers):
        """
        Weights, shape (nt, n_receivers), for traces of nt samples every dt [s] from t = 0.

        Parameters
        ----------
            source : sequence [x, y, z]
            receivers : array-like, shape (n_receivers, 3)
        """
        src = np.asarray(source, dtype=float).ravel()
        rcv = np.asarray(receivers, dtype=float).reshape(-1, 3)
        offset = np.hypot(rcv[:, 0] - src[0], rcv[:, 1] - src[1])
        w = np.ones((nt, len(rcv)))
        w[:, (offset < self.offsetMin) | (offset > self.offsetMax)] = 0.0
        if self.muteVelocity is not None:
            t = (np.arange(nt) * dt)[:, None]
            tm = self.muteTime + offset[None, :] / self.muteVelocity
            w *= _ramp(t - tm + self.taper, self.taper)                 # 1 from the mute line
            if self.window is not None:
                w *= _ramp(tm + float(self.window) - t, self.taper)    # 0 from the end of the window
        return w

    def apply(self, data, dt, source, receivers):
        """W * data, data of shape (nt, n_receivers) sampled every dt from t = 0."""
        data = np.asarray(data)
        return self.weights(data.shape[0], dt, source, receivers) * data


def sourceCorrectionTerms(simulated, observed, dt):
    """
    Terms of the least-squares source filter for one shot: (frequencies, sum over the traces
    of conj(S) D, sum of ``|S|^2``), with S and D the spectra of the simulated and observed
    traces (n_samples, n_traces) sampled every dt.
    """
    n = 2 * simulated.shape[0]                             # zero padding: no wrap-around
    S = np.fft.rfft(simulated, n, axis=0)
    D = np.fft.rfft(observed, n, axis=0)
    return np.fft.rfftfreq(n, dt), np.sum(np.conj(S) * D, axis=1), np.sum(np.abs(S) ** 2, axis=1)


def sourceCorrection(freqs, cross, power, eps=1e-3):
    """Filter c(f) = cross / (power + eps max power), as (frequencies, complex values)."""
    return freqs, cross / (power + eps * np.max(power))


def applySourceCorrection(solver, correction):
    """
    Filter the source wavelet of the solver by ``correction`` = (frequencies, c(f)) (linear
    interpolation of c, 0 above its last frequency) and update the right-hand side.
    """
    freqs, c = correction
    src = np.asarray(solver.sourceValue, dtype=float)[:, 0]
    n = 2 * len(src)
    f = np.fft.rfftfreq(n, solver.dt)
    cc = np.interp(f, freqs, c.real, right=0.0) + 1j * np.interp(f, freqs, c.imag, right=0.0)
    filtered = np.fft.irfft(np.fft.rfft(src, n) * cc, n)[:len(src)]
    solver.updateSourceValue(filtered[:, None])
