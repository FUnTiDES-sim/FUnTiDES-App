import numpy as np


def linearInterpolation( data, tmax, dt, dtNew ):
    """Interpolate field for each simulation timestep

    Parameters
    ----------
        data : array of float
            Field to interpolate
        maxTime : float
            End time of simulation
        dt : float
            Current field time step
        dtNew : float
            New time step
        

    Returns
    --------
        dinterp : array
            Interpolated array with dtNew time step
    """

    dinterp = np.zeros( ( int( tmax / dtNew ) + 1, np.size( data, 1 )))
    tnew = np.linspace( 0, tmax, int( tmax / dtNew ) + 1 )
    t = np.linspace( 0, tmax, int( tmax / dt ) + 1 )

    for i in range( np.size( data, 1 )):
        dinterp[:,i] = np.interp(tnew, t, data[:,i])

    return dinterp