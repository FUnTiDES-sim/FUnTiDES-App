#!/usr/bin/env python
# coding: utf-8

"""
-----------------  Author C.Shen
-----------------  Makutu Inria & Totalenergies, Pau

Functions for DAS acquisition :
get_acquiDAS : calculate a discretized DAS gauge
projee_DAS   : projection of strain tensor along gauge direction
projeu_DAS   : projection of diff(u) vector along gauge direction
"""
import numpy as np
from math import sin, cos


def get_acquiDAS(*gauge_center, **das_geom):
    """
    Discretize a single DAS gauge.
    The discretization is necessary for integration on strain
                      and the differenciation on displacement
    
    Parameters
    -----------
        gauge_center : 1D array-like, the center of a gauge in cartesian coord
        das_geom     : keyword args
             DAS geometry setting in a local spheric coord
                     gauge_length     : float, length of the gauge
                     dip              : float, DAS gauge latitude in spheric coord,
                                        dip>0, projection pointing to +Z
                                        dip=0, projection is the origin
                                        dip<0, projection pointing to -Z
                     azimuth          : float, DAS gauge longitude in spheric coord,
                                        azimuth=]-90,90[ degrees, projection pointing to +X
                                        azimuth={-90,90} degrees, projection pointing to origin
                                        azimuth=]90,270[ degrees, projection pointing to -X
                                        +X ? North ?
                     quadrature_points: int, number of points for integral evaluation
    Returns
    -------- 
        gd, gx, gz OR gd, gx, gy, gz
            gd : 1D numpy array, gauge direction
            gx : x coords of discretized gauge, including the 2 terminals
            gy : y coords of discretized gauge, including the 2 terminals
            gz : z coords of discretized gauge, including the 2 terminals

    Example
    -------
        number of quadrature points N=7,
        quadrature point length=dl
        strain gauge length=gl
        gauge descretization : >---x1--|---x2--|---x3--|---x4--|---x5--|---x6--|---x7-->
                               |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
                            gstart |   |   |   |   |   |   gc  |   |   |   |   |   | gend
                               U0--E1--|---E2--|---E3--|---E4--|---E5--|---E6--|---E7--U1
                               |                       |<--+-->|                       |
                               |                           |___|quadrature point for   |
                             |                               |integral on strain     |
                             |<--------------------------+-------------------------->|
                                                           |______|strain-gauge length
                                                                  |
    Discretized DAS strain calculation :
    1) \frac{\sum_{n=1}^{N} \epsilon_n\,dl}{gl}
       in case of regularly sampled quadratures,
       it is equivalent to the mean value of \epsilon
    2) U1(gend)-U0(gstart)
       U is the primitive of \epsilon
    """
    dimension = len(gauge_center)
    gc = np.zeros(dimension)                # strain gauge center
    gl = das_geom['gauge_length']           # strain gauge length
    dip = das_geom['dip']

    if 'quadrature_points' in das_geom.keys():
        Nquad = das_geom['quadrature_points']
    else:
        Nquad = 7
    Nrcv = Nquad+2                          # plus the 2 terminal points

    for i in range(dimension):
        gc[i] = gauge_center[i]
    if dimension == 2:
        gd = np.array([cos(dip), sin(dip)]) # gauge direction unit vector
        gstart = gc - 0.5 * gl * gd               # gauge starts at this coord
        gend = gc + 0.5 * gl * gd                 # gauge ends at this coord
        gx = np.zeros(Nrcv)                 # x coords of discretized gauge
        gz = np.zeros(Nrcv)                 # z coords of discretized gauge
        gx[0] = gstart[0]                   # x coord of the start point
        gz[0] = gstart[1]                   # z coord of the start point
        gx[-1] = gend[0]                    # x coord of the end point
        gz[-1] = gend[1]                    # z coord of the end point

        for i in range(Nquad):
            gmid = gstart + gl * (i+0.5) / Nquad * gd
            gx[i+1], gz[i+1] = gmid[0], gmid[1]

        return gd, gx, gz

    if dimension == 3:
        azimuth = das_geom['azimuth']
        gd = np.array([cos(dip)*cos(azimuth),\
                     cos(dip)*sin(azimuth),\
                     sin(dip)])           # gauge direction unit vector
        gstart = gc - 0.5 * gl * gd               # gauge starts at this coord
        gend = gc + 0.5 * gl * gd                 # gauge ends at this coord
        gx = np.zeros(Nrcv)                 # x coords of discretized gauge
        gy = np.zeros(Nrcv)                 # y coords of discretized gauge
        gz = np.zeros(Nrcv)                 # z coords of discretized gauge
        gx[0] = gstart[0]                   # x coord of the start point
        gy[0] = gstart[1]                   # y coord of the start point
        gz[0] = gstart[2]                   # z coord of the start point
        gx[-1] = gend[0]                    # x coord of the end point
        gy[-1] = gend[1]                    # y coord of the end point
        gz[-1] = gend[2]                    # z coord of the end point

        for i in range(Nquad):
            gmid = gstart + gl * (i+0.5) / Nquad * gd
            gx[i+1], gy[i+1], gz[i+1]=gmid[0], gmid[1], gmid[2]

        return gd, gx, gy, gz


def projee_DAS(*gd, **epsilons):
    """
    Projection for DAS measurement along a certain (dip, azimuth)
               for one gauge, at one time sample, using the strains

    Parameters
    -----------

        gd      : array-like
            DAS gauge direction vector
        epsilons: keyword args for strains
                   exx, exz, ezz in 2D
                   exx, exy, exz, eyy, eyz, ezz in 3D
    Returns
    --------
         das_dir_ee
    """
    dim = len(gd)
    das_dir_ee = None

    if dim == 2:
        dasdir = np.zeros((1, 2))
        for i in range(dim):
            dasdir[0][i] = gd[i]
        emat = np.array([[epsilons['exx'], epsilons['exz']],\
                       [epsilons['exz'], epsilons['ezz']]])
        das_dir_ee = dasdir@emat@dasdir.T

    if dim == 3:
        dasdir = np.zeros((1, 3))
        for i in range(dim):
            dasdir[0][i] = gd[i]
        emat=np.array([[epsilons['exx'], epsilons['exy'], epsilons['exz']],\
                       [epsilons['exy'], epsilons['eyy'], epsilons['eyz']],\
                       [epsilons['exz'], epsilons['eyz'], epsilons['ezz']]])
        das_dir_ee = dasdir@emat@dasdir.T

    return das_dir_ee


def projeu_DAS(*gd,**uds):
    """
    Projection for DAS measurement along a certain (dip, azimuth)
               for one gauge, at one time sample, using the diff(u)

    Parameters
    -----------
        gd      : das gauge direction vector
        uds     : keyword args for differences of displacements
                   uxd and uzd in 2D
                   uxd, uyd, and uzd in 3D

    Returns
    -------
        das_dir_eu
    """
    dim = len(gd)
    das_dir_eu = None

    if dim == 2:
        dasdir = np.zeros((1, 2))
        for i in range(dim):
            dasdir[0][i] = gd[i]
        evec = np.array([uds['uxd'], uds['uzd']])
        das_dir_eu = evec@dasdir.T

    if dim == 3:
        dasdir = np.zeros((1, 3))
        for i in range(dim):
            dasdir[0][i] = gd[i]
        evec = np.array([uds['uxd'], uds['uyd'], uds['uzd']])
        das_dir_eu = evec@dasdir.T

    return das_dir_eu
