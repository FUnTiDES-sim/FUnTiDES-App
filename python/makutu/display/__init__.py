"""Display submodule (needs matplotlib)"""

try:
    import matplotlib.pyplot as plt  # noqa: F401
except ImportError:                  # matplotlib is optional: no display classes
    pass
else:
    from .Hdf5Field import VelocityModel, Hdf5Field
    from .SegySeismicTrace import SEGYSeismicTrace
