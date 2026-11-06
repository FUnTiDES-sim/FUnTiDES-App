import os
import numpy as np
import segyio
import matplotlib.pyplot as plt


class SEGYSeismicTrace:
    """
    Class containing methods for SEGY seismic traces reading

    Attributes
    -----------
        rootname : str
            File rootname
        format : str
            Extension of the file
        directory : str
            File path
        filename : str
            Full filename
        data : array-like
            Seismic traces array
    """
    def __init__(self, filename, directory="./", **kwargs):
        """
        Parameters
        -----------
            filename : str
                File containing the data
            directory : str, optional
                Path to directory\
                Default is current directory
        """
        self.rootname, self.format = os.path.splitext(filename)
        self.directory = directory

        self.filename = os.path.join(self.directory, self.rootname+self.format)
        self.data = None


    def read(self):
        """
        Extract the seismic traces
        """
        with segyio.open(self.filename, 'r+', ignore_geometry=True) as f:
                self.data = np.zeros((len(f.trace[0]),len(f.trace)))
                for i in range(len(f.trace)):
                        self.data[:,i] = f.trace[i]


    def display(self, vmin=None, vmax=None, normFactor=None, xlabel="Receivers", ylabel="Time [s]"):
        """
        Generate a graph with the seismic traces

        Parameters
        -----------
            vmin : float, optional
                Minimal bound\
                Default is None
            vmax : float, optional
                Maximal bound\
                Default is None
            normFactor : float, optional
                Normalization factor\
                Default is None
            xlabel : str, optional
                x label\
                Default is `Receivers`
            ylabel : str, optional
                y label\
                Default is `Time [s]`
        """
        if self.data is None:
             self.read()

        if normFactor is not None:
            if vmin is None:
                vmin = self.getMinValue()  
            if vmax is None:
                vmax = self.getMaxValue()
            vmin /= normFactor
            vmax /= normFactor
        
        plt.figure()
        plt.imshow(self.data,
                   aspect="auto",
                   vmin=vmin, vmax=vmax,
                   )
        
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)


    def getMinValue(self):
        """
        Return the global minimal value

        Returns
        --------
            float
                Seismic traces minimal value
        """
        if self.data is None:
            self.read()

        return np.min(self.data)
    

    def getMaxValue(self):
        """
        Return the global maximal value

        Returns
        --------
            float
                Seismic traces maximal value
        """
        if self.data is None:
            self.read()
        
        return np.max(self.data)


    def getMinAndMaxValues(self):
        """
        Return the global minimal and maximal values

        Returns
        --------
            minValue : float
                Seismic traces minimal value
            maxValue : float
                Seismic traces maximal value
        """
        minValue = self.getMinValue()
        maxValue = self.getMaxValue()

        return minValue, maxValue