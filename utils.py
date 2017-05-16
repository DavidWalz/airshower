from __future__ import division

import numpy as np


def rectangular_array(n=11):
    """ Return x,y coordinates for rectangular array with n^2 stations. """
    n0 = (n - 1) / 2
    return (np.mgrid[0:n, 0:n].astype(float) - n0)


def triangular_array(n=11, offset=True):
    """ Return x,y coordinates for triangular array with n^2 stations. """
    n0 = (n - 1) / 2
    x, y = np.mgrid[0:n, 0:n].astype(float) - n0
    if offset:  # offset coordinates
        x += 0.5 * (y % 2)
    else:  # axial coordinates
        x += 0.5 * y
    y *= np.sin(np.pi / 3)
    return x, y


def station_coordinates(n=11, layout='axial', spacing=1500, height=1400):
    """ Return array of n^2*(x,y,z) coordinates of SD stations for given layout. """
    if layout == 'axial':
        x, y = triangular_array(n, offset=False)
    elif layout == 'offset':
        x, y = triangular_array(n, offset=True)
    elif layout == 'cartesian':
        x, y = rectangular_array(n)
    else:
        raise ValueError('layout must be one of axial, offset, cartesian')
    x = x.reshape(n**2) * spacing
    y = y.reshape(n**2) * spacing
    z = np.ones_like(x) * height
    return np.c_[x, y, z]


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


def distance2showerplane(v, vc, va):
    """ Get shortest distance to shower plane
    Args:
        v (Nx3 array): array of positions
        va (3 array): shower axis = normal vector of the shower plane
        vc (3 array): shower core
    """
    return np.dot(v - vc, va)


def distance2showeraxis(v, vc, va):
    """ Shortest distance to shower axis.
    Args:
        v (Nx3 array): array of positions
        va (3 array): shower axis
        vc (3 array): shower core
    """
    d = distance2showerplane(v, vc, va)
    vp = v - vc - np.outer(d, va)
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
        v (N x 3 array): array of vectors
        vc (3 array): shower core
        va (3 array): shower axis, pointing upwards
    Return:
        array: arrival times [s]
    """
    d = distance2showerplane(v, vc, -va)
    return d / 3E8


def arrival_time_spherical(v, v0, vc=None):
    """ Get arrival times for a spherical wavefront.
    Args:
        v (N x 3 array): array of vectors
        v0 (3 array): virtual shower origin
        vc (3 array, optional): shower core
    Return:
        array: arrival times [s]
    """
    d = np.linalg.norm(v - v0, axis=1)
    if vc is not None:
        d -= np.linalg.norm(vc - v0)
    return d / 3E8
