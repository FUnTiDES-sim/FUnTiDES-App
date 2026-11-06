import numpy as np
from scipy.fftpack import fft, ifft, fftfreq


def frequencyFilter(signal, fmax, dt, padding=False):
    """Filter the desired 2d np.array signal. It filters the signal in the interval [0, fmax+1]

    Parameters:
    -----------
        signal : 2d np.array
            2d numpy array
        fmax : float
            Max frequency to keep in the signal
        dt : float
            Time step between values in signal
        padding : bool
            Whether or not the return signal have padding or not to avoid discontinuity

    Returns
    -------
        y : 2d np.array
            Real part of filtered array
    """
    # TODO: trace-by-trace filtering adding a cosinus window on the edges (Dangerous!). IIR + padding.
    if str(fmax) == "all":
        return signal

    pad = int(round(signal.shape[0]/2))
    n = signal.shape[0] + 2*pad

    tf = fftfreq(n, dt)
    y_fft = np.zeros((n,signal.shape[1]), dtype=np.complex128)
    y = np.zeros(y_fft.shape, dtype=np.complex128)

    for i in range(y_fft.shape[1]):
        y_fft[pad:n-pad,i] = signal[:,i]
        y_fft[:,i] = fft(y_fft[:,i])# Perform fourier transform

    isup = np.where(tf>=fmax)[0]
    imax = np.where(tf[isup]>=fmax+1)[0][0]
    i1 = isup[0]
    i2 = isup[imax]

    iinf = np.where(tf<=-fmax)[0]
    imin = np.where(tf[iinf]<=-fmax-1)[0][-1]

    i3 = iinf[imin]
    i4 = iinf[-1]

    for i in range(y_fft.shape[1]):
        y_fft[i1:i2,i] = np.cos((isup[0:imax] - i1)/(i2-i1) * np.pi/2)**2 * y_fft[i1:i2,i]
        y_fft[i3:i4,i] = np.cos((iinf[imin:-1] - i4)/(i3-i4) * np.pi/2)**2 * y_fft[i3:i4,i]
        y_fft[i2:i3,i] = 0

    for i in range(y.shape[1]):
        y[:,i] = ifft(y_fft[:,i])# Perform inverse fourier transform

    if padding:
        return np.real(y)
    else:
        return np.real(y[pad:n-pad])
