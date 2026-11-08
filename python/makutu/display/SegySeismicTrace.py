"""Display of the SEG-Y seismograms written by makutu."""

import os

import matplotlib.pyplot as plt
import numpy as np
import segyio


class SEGYSeismicTrace:
    """
    SEG-Y seismic traces (one file per shot).

    Attributes
    -----------
        filename : str
            Full filename
        data : np.ndarray
            Traces, shape (n_samples, n_traces), after :meth:`read`
        dt : float
            Sample interval [s] (from the binary header), after :meth:`read`
    """
    def __init__(self, filename, directory="./", **kwargs):
        """
        Parameters
        -----------
            filename : str
                File containing the data
            directory : str, optional
                Path to directory \
                Default is current directory
        """
        self.rootname, self.format = os.path.splitext(filename)
        self.directory = directory
        self.filename = os.path.join(self.directory, self.rootname + self.format)
        self.data = None
        self.dt = None

    def read(self):
        """Read the traces and the sample interval."""
        with segyio.open(self.filename, "r", ignore_geometry=True) as f:
            self.data = f.trace.raw[:].T.astype(np.float64)           # (n_samples, n_traces)
            self.dt = segyio.tools.dt(f) * 1e-6          # microseconds in the header
        return self.data

    def display(self, vmin=None, vmax=None, normFactor=None, xlabel="Receivers", ylabel="Time [s]",
                savefig=None):
        """
        Image of the traces, time down.

        Parameters
        -----------
            vmin, vmax : float, optional
                Colour bounds \
                Default is symmetric, +-max|data|
            normFactor : float, optional
                Divides the colour bounds
            xlabel, ylabel : str, optional
                Axis labels
            savefig : str, optional
                Save the figure to this file (and close it) instead of leaving it open
        """
        if self.data is None:
            self.read()
        amax = np.max(np.abs(self.data))
        vmin = -amax if vmin is None else vmin
        vmax = amax if vmax is None else vmax
        if normFactor is not None:
            vmin, vmax = vmin / normFactor, vmax / normFactor
        nt, ntr = self.data.shape
        fig = plt.figure()
        plt.imshow(self.data, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax,
                   extent=(0.5, ntr + 0.5, nt * self.dt, 0.0))
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(os.path.basename(self.filename), fontsize=12)
        plt.colorbar()
        if savefig is not None:
            fig.savefig(savefig, dpi=150, bbox_inches="tight")
            plt.close(fig)

    def getMinValue(self):
        """Minimum of the traces."""
        if self.data is None:
            self.read()
        return np.min(self.data)

    def getMaxValue(self):
        """Maximum of the traces."""
        if self.data is None:
            self.read()
        return np.max(self.data)

    def getMinAndMaxValues(self):
        """(minimum, maximum) of the traces."""
        return self.getMinValue(), self.getMaxValue()
