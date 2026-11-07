import numpy as np
from scipy.signal import butter, sosfilt


def frequencyFilter(signal, fmax, dt, padding=False, order=4):
    """Causal low-pass Butterworth filter along the time axis (axis 0).

    The multi-scale inversion filters the source (``solver.filterSource``) and the observed
    data (``ComputeResidual``) with this function. The filter is causal: it is zero before
    the first sample, so that filtering the source before the simulation gives the
    simulated data filtered after it (up to the difference of sampling). A zero-phase FFT
    filter with a sharp cut-off rings before t = 0: that part of the filtered source is
    lost, and the filtered data and the data simulated with the filtered source then
    differ (24 % at 6 Hz on a 10 Hz Ricker wavelet).

    Parameters:
    -----------
        signal : np.ndarray
            Signal, time along axis 0 (1-D, or 2-D: n_samples x n_traces)
        fmax : float or "all"
            Cut-off frequency [Hz] (-3 dB); "all": no filter
        dt : float
            Sampling [s]
        padding : bool
            Unused (kept for compatibility)
        order : int
            Order of the Butterworth filter \
            Default is 4

    Returns
    -------
        np.ndarray : filtered signal, same shape (float64)
    """
    if str(fmax) == "all":
        return signal
    fmax = float(fmax)
    nyquist = 0.5 / dt
    if not 0.0 < fmax < nyquist:
        raise ValueError(f"filter frequency {fmax} Hz outside (0, {nyquist:g}) Hz")
    sos = butter(order, fmax, btype="low", fs=1.0 / dt, output="sos")
    return sosfilt(sos, np.asarray(signal, dtype=np.float64), axis=0)
