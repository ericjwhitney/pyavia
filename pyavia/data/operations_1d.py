from __future__ import annotations
from typing import TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Written by Eric J. Whitney, 2022.

_T = TypeVar('_T')


# =============================================================================

def sine_spacing(x1: float, x2: float, n: int,
                 spacing: float) -> NDArray[float]:
    r"""
    Generates a non-linear distribution of values in the interval
    :math:`[x_1, x_2]`.

    .. note:: This procedure is adapted from a Fortran subroutine contained
       within the `AVL` source code written by M. Drela and H. Youngren
       (2002).

    Parameters
    ----------
    x1 : float
        Startpoint (inclusive) of the interval.

    x2 : float
        Endpoint (inclusive) of the interval.

    n : int
        Number of points to generate.

    spacing : float
        Distribution type [-3..+3]:

            - *0* = Equal spacing.
            - *1* = Cosine spacing.
            - *2* = Sine spacing (points concentrated toward `x1`).
            - *3* = Equal spacing.

        A negative value of `dist` produces reversed spacing (only applicable
        to sine spacing).  Intermediate / fractional values produce a spacing
        which is a combination of the adjacent integer values.

    Returns
    -------
    x : ndarray[float]
        Points in the interval [`x1`, `x2`] clustered to the distribution.

    Examples
    --------
    >>> sine_spacing(0.0, 10.0, 5, spacing=0)  # Equal spacing.
    array([ 0. ,  2.5,  5. ,  7.5, 10. ])
    >>> sine_spacing(0.0, 10.0, 5, spacing=1)  # Cosine spacing.
    array([ 0.        ,  1.46446609,  5.        ,  8.53553391, 10.        ])
    >>> sine_spacing(0.0, 10.0, 5, spacing=-2)  # Sine spacing toward end.
    array([ 0.        ,  3.82683432,  7.07106781,  9.23879533, 10.        ])

    """
    if not (-3 <= spacing <= 3):
        raise ValueError("Invalid spacing parameter.")

    # Compute fractions of each spacing type to use.
    spc_abs = np.abs(spacing)
    if 0 <= spc_abs < 1:
        f_eq, f_cos, f_sin = 1 - spc_abs, spc_abs, 0
    elif 1 <= spc_abs < 2:
        f_eq, f_cos, f_sin = 0, 2 - spc_abs, spc_abs - 1
    else:
        f_eq, f_cos, f_sin = spc_abs - 2, 0, 3 - spc_abs

    # Compute spacing due to equal and cosine distributions.
    u = np.linspace(0.0, 1.0, n)
    u_spc = (f_eq * u) + (0.5 * f_cos * (1 - np.cos(u * np.pi)))

    # Add spacing due to sine distribution.
    if spacing >= 0:
        u_spc += f_sin * (1 - np.cos(0.5 * u * np.pi))
    else:
        u_spc += f_sin * np.sin(0.5 * u * np.pi)

    return (x2 - x1) * u_spc + x1


# ----------------------------------------------------------------------------
def subdivide_series(x: ArrayLike[_T], ndiv: int,
                     keep_original: str = 'all') -> NDArray[_T]:
    """
    Subdivide a list / array by inserting `ndiv` intermediate points equally
    into each interval between consequtive `x` values. See `Notes` section
    for a detailed explanation.

    .. note::  This function is commonly used with a series of function
               coordinates or grid values to produce a finer discretisation
               / grid.

    Parameters
    ----------
    x : array_like, shape (npoints,)
        Values to subdivide.  These are not necesarily required to be in
        sorted order or unique.

        .. note:: If adjacent `x` values are repeated / duplicates,
                  subdividing points will be inserted which are also equal
                  to that `x` value.

    ndiv : int
        Number of divisions (>= 1) to perform on the intervals between
        existing `x` points.

    keep_original : str, default = 'all'
        This option determines which existing `x` values to retain in
        the result, in addition to the new subdividing points:

        - ``keep_original='all'``: Retain all original grid points.
        - ``keep_original='ends'``: Retain only the first and last original
          points.
        - ``keep_original='interior'``: Retain all the original points
          *except* the first and last points.
        - ``keep_original='none'``: Discard all original grid points.

    Returns
    -------
    ndarray
        Subdivided and original `x` values (depending on options).

    Notes
    -----
    The following is a visual depiction of different subdivisions,
    also noting the end and interior points::

                           `x`:  E ----- I ----- I ----- E
        Subdivisions, ndiv = 1:  E - x - I - x - I - x - E
                  ... ndiv = 2:  E -x-x- I -x-x- I -x-x- E
                  ... ndiv = 3:  E x-x-x I x-x-x I x-x-x E

        Where:  x - Subdiving points.
                E - End points.
                I - Interior points.

    The subdividing points divide each interval based on its individual
    length, in the case where the `x` spacing is irregular.

    To demonstrate one use-case, a call of ``subdivide(x, 1,
    keep_original='ends')`` would gives an offset subdividing grid that
    retains the original boundary::

            `x`:  E ----- I ----- I ----- E
        Returns:  E - x ----- x ----- x - E

    Examples
    --------
    Subdivide some typical `x`-coordinates:

    >>> x_coords = [1, 2.5, 5.5, 10]
    >>> subdivide_series(x_coords, 2)
    array([ 1. ,  1.5,  2. ,  2.5,  3.5,  4.5,  5.5,  7. ,  8.5, 10. ])

    >>> subdivide_series(x_coords, 3)
    array([ 1.   ,  1.375,  1.75 ,  2.125,  2.5  ,  3.25 ,  4.   ,  4.75 ,
            5.5  ,  6.625,  7.75 ,  8.875, 10.   ])

    Produce an offset subdivision that surrounds each existing `x`-
    coordinate and keeps the boundary points:

    >>> subdivide_series(x_coords, 1, keep_original='ends')
    array([ 1.  ,  1.75,  4.  ,  7.75, 10.  ])
    """
    if ndiv < 1:
        raise ValueError("Require 'ndiv' >= 1.")

    x = np.asarray(x)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("Require 'x' to be of shape (>= 2,).")

    x = x[:, np.newaxis]  # Make column vector [np, 1]
    Δx = np.diff(x, axis=0)  # Δx along axis [np - 1, 1]

    # Compute subdividing fractions as row vector.  The compute Δx and x'
    # for each subdivision.  Resulting x' values are in columns.
    frac = ((np.arange(ndiv) + 1) / (ndiv + 1))[np.newaxis, :]  # [1, ndiv]
    offset = Δx @ frac
    x_sub = x[:-1, :] + offset  # [np - 1, ndiv]

    if keep_original == 'none':
        return x_sub.ravel()  # Interleave / return only subdividing points.

    elif keep_original == 'ends':
        return np.r_[x[0], x_sub.ravel(), x[-1]]  # Tack on ends.

    # Otherwise generate 'all' result by interleaving original 'x' values
    # with subdivisions [x_0, x_sub, ..., x_1, x_sub, ..., x_np-2, x_sub,
    # ...] and finally tack on RH end [x_np-1].
    x_all = np.r_[np.c_[x[:-1, :], x_sub].ravel(), x[-1]]

    if keep_original == 'all':
        return x_all

    elif keep_original == 'interior':
        return x_all[1:-1]  # Dock ends.

    else:
        raise ValueError(f"Unknown 'keep_original' option: '{keep_original}'.")
