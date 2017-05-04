from __future__ import division

import numpy as np


HEIGHT = 1400  # detector height in [m]
SPACING = 1500  # detector spacing in [m]


def rectangular_array(n=11):
    """ Coordinates for rectangular array with n^2 stations and given spacing.
        Returns: n^2 x 3 array of x,y,z coordinates for each station.
    """
    n0 = (n - 1) / 2
    x, y = (np.mgrid[0:n, 0:n].astype(float) - n0) * SPACING
    z = np.ones_like(x) * HEIGHT  # z-position
    return np.dstack([x, y, z]).reshape(-1, 3)


def triangular_array(n=11, offset=True):
    """ Coordinates for triangular array with n^2 stations and given spacing.
        Returns: n^2 x 3 array of x,y,z coordinates for each station.
    """
    n0 = (n - 1) / 2
    x, y = np.mgrid[0:n, 0:n].astype(float) - n0
    if offset:  # offset coordinates
        x += 0.5 * (y % 2)
    else:  # axial coordinates
        x += 0.5 * y
    x *= SPACING
    y *= np.sin(np.pi / 3) * SPACING
    z = np.ones_like(x) * HEIGHT  # z-position
    return np.dstack([x, y, z]).reshape(-1, 3)


def rand_zenith(N=1, zmax=np.pi / 3):
    """ Sample random zenith angles z for air shower surface detector.
        Returns zenith angles z sampled pdf f(z) ~ sin(z) cos(z) in range [0, zmax].
    """
    b = 1 / (1 - np.cos(zmax)**2)
    r = np.random.rand(N)
    return np.arccos(np.sqrt(1 - r / b))


def ang2vec(phi, zenith):
    """ Get 3-vector from spherical angles.
    Args:
        phi (array): azimuth (pi, -pi), 0 points in x-direction, pi/2 in y-direction
        zenith (array): zenith (0, pi), 0 points in z-direction
    Returns:
        array of 3-vectors
    """
    x = np.sin(zenith) * np.cos(phi)
    y = np.sin(zenith) * np.sin(phi)
    z = np.cos(zenith)
    return np.array([x, y, z]).T


def vec2ang(v):
    """ Get spherical angles phi and zenith from 3-vector
    Args:
        array of 3-vectors
    Returns:
        phi, zenith
        phi (array): azimuth (pi, -pi), 0 points in x-direction, pi/2 in y-direction
        zenith (array): zenith (0, pi), 0 points in z-direction
    """
    x, y, z = v.T
    phi = np.arctan2(y, x)
    zenith = np.pi / 2 - np.arctan2(z, (x**2 + y**2)**.5)
    return phi, zenith


def distance2showerplane(v, va):
    """ Get shortest distance to shower plane
    Args:
        v (Nx3 array): array of positions
        va (3 array): shower axis = normal vector of the shower plane
    """
    return np.dot(v, va)


def distance2showeraxis(v, va):
    """ Shortest distance to shower axis.
    Args:
        v (Nx3 array): array of positions
        va (3 array): shower axis
    """
    d = distance2showerplane(v, va)
    vp = v - np.outer(d, va)
    return np.linalg.norm(vp, axis=-1)


def distance2showermaximum(v, vm):
    """ Shortest distance to shower maximum
    Args:
        v (Nx3 array): array of vectors
        vm (3 array): position of the shower maximum
    """
    return np.linalg.norm(v - vm, axis=-1)


def arrival_time_planar(v, vc, va):
    """ Get arrival times for a planar wavefront.
    Note: The shower core is not considered here and as it only adds a constant time offset to all stations.
    Args:
        v (N x 3 array)
        vc (3 array): shower core
        va (3 array): shower axis, pointing upwards
    Return:
        array: arrival times [s]
    """
    d = distance2showerplane(v - vc, -va)
    return d / 3E8
