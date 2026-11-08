"""
Display of the HDF5 fields of makutu (velocity models, gradients, illumination).

The fields are flat per element in seismic order: x fastest, then y, then z (z down), i.e.
``field.reshape(nz, nx)`` in 2-D and ``field.reshape(nz, ny, nx)`` in 3-D. The files written
by makutu carry this shape and the domain lengths as dataset attributes (``shape``,
``lengths``, see :func:`makutu.forwardBackward.modelio.gridAttributes`); otherwise give the
numbers of elements ``n`` = (nx, nz) or (nx, ny, nz).
"""

import os

import h5py
import matplotlib.pyplot as plt
import numpy as np


class Hdf5Field:
    """
    Field of an HDF5 file, flat per element in seismic order.

    Attributes
    -----------
        filename : str
            HDF5 file
        fieldname : str
            Dataset name
        model : np.ndarray
            Field, shape (nz, nx) or (nz, ny, nx), after :meth:`read`
        lengths : tuple or None
            Domain lengths [m] in the order of ``model.shape``, if known
    """
    def __init__(self, filename, fieldname=None, n=None, lengths=None):
        """
        Parameters
        -----------
            filename : str
                HDF5 file
            fieldname : str
                Dataset name
            n : sequence of int, optional
                Numbers of elements (nx, nz) or (nx, ny, nz); default: the ``shape``
                attribute of the dataset
            lengths : sequence of float, optional
                Domain lengths (lx, lz) or (lx, ly, lz) [m]; default: the ``lengths``
                attribute of the dataset, else the axes are in elements
        """
        self.filename = filename
        self.fieldname = fieldname
        self._n = None if n is None else tuple(int(v) for v in n)
        self._lengths = None if lengths is None else tuple(float(v) for v in lengths)
        self.model = None
        self.lengths = None

    @property
    def dim(self):
        if self.model is None:
            self.read()
        return self.model.ndim

    def read(self):
        """Read the field and reshape it to (nz, nx) or (nz, ny, nx)."""
        with h5py.File(self.filename, "r") as f:
            d = f[self.fieldname]
            data = d[...]
            shape = tuple(self._n[::-1]) if self._n is not None else d.attrs.get("shape")
            lengths = self._lengths[::-1] if self._lengths is not None else d.attrs.get("lengths")
        if shape is None:
            if data.ndim in (2, 3):
                shape = data.shape
            else:
                raise ValueError(f"{self.filename}:{self.fieldname}: no 'shape' attribute, give n=(nx, nz) "
                                 "or (nx, ny, nz)")
        shape = tuple(int(v) for v in shape)
        if int(np.prod(shape)) != data.size:
            raise ValueError(f"{self.filename}:{self.fieldname}: {data.size} values, shape {shape[::-1]} "
                             "(x first) does not match")
        self.model = np.reshape(data, shape)
        self.lengths = None if lengths is None else tuple(float(v) for v in lengths)
        return self.model

    def getMinValue(self):
        """Minimum of the field."""
        if self.model is None:
            self.read()
        return np.min(self.model)

    def getMaxValue(self):
        """Maximum of the field."""
        if self.model is None:
            self.read()
        return np.max(self.model)

    def getSlice(self, axis="y", index=None):
        """
        Slice of the field.

        Parameters
        -----------
            axis : str, optional
                3-D: the slice is at a constant ``axis`` ("x", "y" or "z") \
                Default is `y` (vertical x-z section). Ignored in 2-D (the whole field).
            index : int, optional
                Element index along ``axis`` \
                Default is the middle of the axis

        Returns
        --------
            array-like
                2-D array, rows along the vertical axis of the image (z, or y for a z slice)
        """
        if self.model is None:
            self.read()
        if self.model.ndim == 2:
            return self.model
        axis = axis.lower()
        if axis not in ("x", "y", "z"):
            raise ValueError(f"Unknown axis {axis!r}: 'x', 'y' or 'z'")
        k = {"z": 0, "y": 1, "x": 2}[axis]
        if index is None:
            index = self.model.shape[k] // 2
        return np.take(self.model, index, axis=k)

    def _axes(self, axis):
        """(horizontal label, its length, vertical label, its length) of the image."""
        L = self.lengths
        unit = "m" if L is not None else "elements"
        size = L if L is not None else self.model.shape
        if self.model.ndim == 2:                       # (z, x)
            return f"x [{unit}]", size[1], f"z [{unit}]", size[0]
        return {"x": (f"y [{unit}]", size[1], f"z [{unit}]", size[0]),
                "y": (f"x [{unit}]", size[2], f"z [{unit}]", size[0]),
                "z": (f"x [{unit}]", size[2], f"y [{unit}]", size[1])}[axis.lower()]

    def display(self, axis="y", index=None, extent=None, vmin=None, vmax=None, normFactor=None,
                cmap=None, savefig=None):
        """
        Image of a slice (the whole field in 2-D), z down.

        Parameters
        -----------
            axis, index :
                See :meth:`getSlice`
            extent : sequence of 4 floats, optional
                Image extent (left, right, bottom, top) \
                Default is the domain (or the element indices)
            vmin, vmax : float, optional
                Colour bounds
            normFactor : float, optional
                Divides the colour bounds (default bounds: the min and max of the field)
            cmap : str, optional
                Colour map
            savefig : str, optional
                Save the figure to this file (and close it) instead of leaving it open

        Returns
        --------
            data : array-like
                The slice
        """
        if normFactor is not None:
            vmin = (self.getMinValue() if vmin is None else vmin) / normFactor
            vmax = (self.getMaxValue() if vmax is None else vmax) / normFactor
        data = self.getSlice(axis=axis, index=index)
        hlabel, hlen, vlabel, vlen = self._axes(axis)
        if extent is None:
            extent = (0.0, hlen, vlen, 0.0)

        fig = plt.figure()
        plt.imshow(data, extent=extent, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
        plt.xlabel(hlabel)
        plt.ylabel(vlabel)
        title = os.path.split(self.filename)[1]
        if self.model.ndim == 3:
            k = {"z": 0, "y": 1, "x": 2}[axis.lower()]
            title += f" ({axis} index {self.model.shape[k] // 2 if index is None else index})"
        plt.title(title, fontsize=12)
        plt.colorbar()
        if savefig is not None:
            fig.savefig(savefig, dpi=150, bbox_inches="tight")
            plt.close(fig)
        return data


class VelocityModel(Hdf5Field):
    """
    HDF5 velocity model, in the parametrisation ``type`` ("c", "1/c" or "1/c2"), displayed
    as a velocity.
    """
    def __init__(self, filename, n=None, fieldname="velocity", type="c", lengths=None):
        super().__init__(filename=filename, fieldname=fieldname, n=n, lengths=lengths)
        if type not in ("c", "1/c", "1/c2"):
            raise ValueError(f"Unknown model type {type!r}: 'c', '1/c' or '1/c2'")
        self.modelType = type

    def convert(self, data):
        """Velocity from values in the parametrisation of the file."""
        if self.modelType == "1/c":
            return 1.0 / data
        if self.modelType == "1/c2":
            return np.sqrt(1.0 / data)
        return np.array(data, copy=True)

    def read(self):
        """Read the model and convert it to velocity."""
        super().read()
        self.model = self.convert(self.model)
        return self.model
