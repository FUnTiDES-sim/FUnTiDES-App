import os
import numpy as np
import h5py
import matplotlib.pyplot as plt


class Hdf5Field:
    def __init__(self, filename, fieldname=None, n=None):
        self.filename = filename
        self.init = False
        self.fieldname = fieldname
        if n is not None:
            self.n1, self.n2, self.n3 = n
        self.model = None


    def getSlice(self, axis="y", index=None):
        """
        Get the requested model slice

        Parameters
        -----------
            axis : str, optional
                Slice axis, "x", "y" or "z"\
                Default is `y`
            index : int, optional
                Slice index\
                Default is middle of axis

        Returns
        --------
            array-like
                Model slice
        """
        assert axis.lower() in ("x", "y", "z"), "Unknown axis"
        if not self.init:
            self.read()
        elif self.init and self.model is None:
            return

        if axis == "x":
            if index is None:
                index = int(self.model.shape[0]/2)
            return self.model[index,:,:]
        elif axis == "y":
            if index is None:
                index = int(self.model.shape[1]/2)
            return self.model[:,index,:]
        else:
            if index is None:
                index = int(self.model.shape[2]/2)
            return self.model[:,:,index]


    def read(self):
        """
        Read the model from the HDF5 file
        """
        with h5py.File( self.filename, "r" ) as f:
            field = f[self.fieldname]
            self.model = np.reshape(field, (self.n3, self.n2, self.n1))

        self.init = True


    def getMinValue(self):
        """
        Return the field minimal value

        Returns
        --------
            float
                Field minimal value
        """
        if self.model is None:
            self.read()

        return np.min(self.model)


    def getMaxValue(self):
        """
        Return the field minimal value

        Returns
        -------
            float
                Field maximal value
        """
        if self.model is None:
            self.read()

        return np.max(self.model)
    

    def display(self, axis="y", index=None, extent=None, vmin=None, vmax=None, normFactor=None):
        """
        Generate a 2d graph for a given slice

        Parameters
        -----------
            axis : str, optional
                Slice axis\
                Default is `y`
            index : int, optional
                Slice index\
                Default is 0
            extent : None,
                Graph extent\
                Default is None
            vmin : float, optional
                Minimal bound\
                Default is None
            vmax : float, optional
                Maximal bound\
                Default is None
            normFactor : float, optional
                Normalization factor\
                Default is None

        Returns
        --------
            data : array-like
                Data slice
        """
        if normFactor is not None:
            if vmin is None:   
                vmin = self.getMinValue()
            if vmax is None:
                vmax = self.getMaxValue()
            vmin /= normFactor
            vmax /= normFactor

        data = self.getSlice(axis=axis, index=index)

        plt.figure()
        plt.imshow(data, extent=extent, vmin=vmin, vmax=vmax)

        ax = ["x", "y", "z"]
        ax.remove(axis)

        plt.xlabel(ax[0])
        plt.ylabel(ax[1])
        title = os.path.split(self.filename)[1]
        plt.title(title, fontsize=12)
        plt.colorbar()

        return data



class VelocityModel(Hdf5Field):
    """
    Class containing methods for HDF5 velocity models

    Attributes
    -----------
        filename = filename
        init = False
        fieldname = fieldname
        n : int
        modelType : str
            Model type
            Default is `c`
    """
    def __init__(self, filename, n, fieldname="velocity", type="c"):
        super().__init__(filename=filename, n=n, fieldname=fieldname)
        self.modelType = type

    def convert(self, data=None):
        """
        Convert the data to `c` model

        Parameters
        -----------
            data : array-like, optional
                Velocity data to convert\
                Default is using `model` attribute
    
        Returns
        --------
            array-like
                Converted data
        """
        assert self.modelType in ("c", "1/c", "1/c2"), "Unkwown model type"

        if data is None:
            if not self.init: 
                self.read()
            data = self.model.copy()

        if self.modelType == "1/c":
            return 1/data
        elif self.modelType == "1/c2":
            return np.sqrt(1/data)
        else:
            return data


    def getSlice(self, axis="y", index=None, convert=True):
        """
        Get the requested model slice

        Parameters
        -----------
            axis : str, optional
                Slice axis, "x", "y" or "z"\
                Default is `y`
            index : int, optional
                Slice index\
                Default is middle of axis
            convert : bool
                Flag to convert the data to `c` model

        Returns
        --------
            array-like
                Model slice
        """
        velocity = super().getSlice(axis=axis, index=index)
        
        if convert:
            velocity = self.convert(velocity)

        return velocity
        
    
    
    


