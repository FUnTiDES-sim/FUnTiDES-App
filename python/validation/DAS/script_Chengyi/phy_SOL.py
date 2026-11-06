#!/usr/bin/env python
# coding: utf-8

"""
-----------------  Author C.Shen
-----------------  Makutu Inria & Totalenergies, Pau

Functions for Solid Physics
C_EHI : calculate stiffness C for an Elastic Homogeneous Isotropic (EHI) medium
"""

import numpy as np

def C_EHI(vp, vs, rho=1, dim=3):
    """
    Return the stiffness matrix C of an Elastic Homogeneous Isotropic medium

    Parameters
    -----------
        vp  : float
            P-wave velocity
        vs  : float
            S-wave velocity
        rho : float
            Density
        dim : int
            Problem dimension

    Returns
    ---------
        C_stiffness : array-like
            Stiffness matrix C of an Elastic Homogeneous Isotropic medium
    """
    # Lame parameters
    lame_lambda = (vp**2 - 2 * vs**2) * rho
    lame_mu = vs**2 * rho

    if dim == 2:
        csize = 3 * (dim-1)
        C_stiffness = np.zeros((csize, csize))
        for i in range(2):
            C_stiffness[i, i] = lame_lambda + 2 * lame_mu
            C_stiffness[i, 1-i] = lame_lambda
        C_stiffness[2, 2] = lame_mu

    if dim == 3:
        csize = 3 * (dim-1)
        C_stiffness = np.zeros((csize, csize))
        for i in range(3):
            dims = [0, 1, 2]
            dims.remove(i)
            C_stiffness[i, i] = lame_lambda + 2 * lame_mu
            for j in dims:
                C_stiffness[i, j] = lame_lambda
        for i in range(3, 6):
            C_stiffness[i, i] = lame_mu

    return C_stiffness
