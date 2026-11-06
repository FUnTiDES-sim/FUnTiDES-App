"""Display submodule"""

try:
    import matplotlib.pyplot as plt
    from .Hdf5Field import VelocityModel, Hdf5Field
    from .SegySeismicTrace import SEGYSeismicTrace
except:
    pass