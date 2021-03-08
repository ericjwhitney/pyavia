"""
**pyavia.interpolate** provides functions for interpolation that are
specific to engineering tasks, for convenience or not otherwise covered by
NumPy.
"""

# Last updated: 20 January 2021 by Eric J. Whitney
from __future__ import annotations

from collections import Sequence
from copy import copy
from typing import Union

import numpy as np
from scipy.interpolate import interp1d

__all__ = ['smooth_array2d', 'subdivide_list']


# =============================================================================

def smooth_array2d(arr: np.array, axis, x_idx: int,
                   x_new: Union[int, Sequence], kind: str = 'linear',
                   extrapolate: bool = False):
    """
    Produces a new 2D array by doing 1D interpolation of rows / columns in
    input ``array2d``.  This can be used to regularise data, or make a larger
    input array from a smaller amount of data.

    The returned array is produced by interpolating along ``axis`` and the
    other axis is unchanged.  The row / column index in ``x_idx`` is treated
    as the independent variable and is replaced by  ``x_new``.  Remaining
    rows / columns are interpolated as the dependent variables (e.g. 'y').

    Examples
    --------
    >>> a = np.array([[1, 1, 3, 5], [3, 4, 5, 6], [10, 8, 5, 4]],
    ... dtype=np.float64)
    >>> a
    array([[ 1.,  1.,  3.,  5.],
           [ 3.,  4.,  5.,  6.],
           [10.,  8.,  5.,  4.]])

    >>> smooth_array2d(a, axis=0, x_idx=0, x_new=np.linspace(1, 10, 4))
    array([[ 1.        ,  1.        ,  3.        ,  5.        ],
           [ 4.        ,  4.57142857,  5.        ,  5.71428571],
           [ 7.        ,  6.28571429,  5.        ,  4.85714286],
           [10.        ,  8.        ,  5.        ,  4.        ]])

    >>> smooth_array2d(a, axis=0, x_idx=0, x_new=4, kind='quadratic')
    array([[ 1.        ,  1.        ,  3.        ,  5.        ],
           [ 4.        ,  5.19047619,  5.66666667,  6.23809524],
           [ 7.        ,  7.52380952,  6.33333333,  5.9047619 ],
           [10.        ,  8.        ,  5.        ,  4.        ]])

    >>> smooth_array2d(a, axis=1, x_idx=1, x_new=[3, 3.5, 4, 4.5, 5, 5.5, 6],
    ... kind='cubic')
    array([[ 1.    ,  0.625 ,  1.    ,  1.875 ,  3.    ,  4.125 ,  5.    ],
           [ 3.    ,  3.5   ,  4.    ,  4.5   ,  5.    ,  5.5   ,  6.    ],
           [10.    ,  9.3125,  8.    ,  6.4375,  5.    ,  4.0625,  4.    ]])

    Parameters
    ----------
    arr : np.array or [[a, b, c, ...], ...]
        2D array of data to be smoothed.
    axis : int
        Interpolation axis.
    x_idx : int
        Row / column index along 'other' axis to treat as independent variable.
    x_new : array-like or int
        Depending on argument type:
            - array-like: New values replacing the previous 'independent'
              variable along ``array2d[:, x_idx]`` if ``axis == 0`` or
              ``array2d[x_idx, :]`` if ``axis == 1``.

            - int: Use an number of equally spaced points between min and max
              values of the independent axis.  Equivalent to: ``x_new =
              np.linspace(np.min(x), np.max(x), num=x_new)``
    kind : str, optional
        Interpolation method to use (default = 'linear').  Any 'kind'
        acceptable to the underlying interpolator
        ``scipy.interpolate.interp1d`` can be used.

        ..note:: Sufficient data rows / columns must be available to compute
          higher order interpolations `quadratic` or `cubic`, i.e. 3 and 4
          respectively.

    extrapolate : bool
        If True, then any ``x_new`` value outside the original data range
        will have the dependent parts extrapolated (default = False).
        Otherwise ``x_new`` values outside the original data range raise
        ValueError.

        ..note:: This is not normally recommended for any higher order
          interpolations due to potential kinks.
    Returns
    -------
    result : np.array
        A new 2D array with specified row / columb replaced with x_new and
        remaining rows / columns replaced with interpolated values.

    Raises
    ------
    AttributeError if input array is not 2D.
    ValueError on incorrect interpolation axis, independent index, number
    of available datapoints of ``x_new`` outside data range (for extrapolate
    == False).
    """
    arr = np.asarray(arr)
    orig_shape = arr.shape
    if len(orig_shape) != 2:
        raise AttributeError("Can only smooth 2D arrays.")

    if axis != 0 and axis != 1:
        raise ValueError("Invalid smoothing axis.")

    if not 0 <= x_idx < orig_shape[1 - axis]:
        raise ValueError("Invalid row / column for independent variable.")

    if (orig_shape[axis] < 2 or
            (kind == 'quadratic' and orig_shape[axis] < 3) or
            (kind == 'cubic' and orig_shape[axis] < 4)):
        raise ValueError(f"Insufficient data for smoothing by {kind}.")

    x = arr[:, x_idx] if axis == 0 else arr[x_idx, :]
    y = np.delete(arr, x_idx, axis=1 - axis)
    interp_fn = interp1d(x, y, axis=axis, kind=kind,
                         bounds_error=False if extrapolate else True,
                         fill_value='extrapolate' if extrapolate else np.nan)

    if isinstance(x_new, int):
        x_new = np.linspace(np.min(x), np.max(x), num=x_new)

    y_new = interp_fn(x_new)
    return np.insert(y_new, x_idx, x_new, axis=1 - axis)


# -----------------------------------------------------------------------------

def subdivide_list(li: Sequence, max_size: int) -> list:
    """
    Expands a sorted list of values by repeatedly subdividing the intervals
    in the initial list, stopping before max_size is exceeded.  This keeps
    the originally specified values in position.

    Parameters
    ----------
    li : list or array-like
        Starting values.
    max_size : int
        Upper limit on len(list).

    Returns
    -------
    li_dense : list
        New list containing subdivided values.
    """
    li_dense, li_add = list(li), []
    while len(li_add) + len(li_dense) <= max_size:
        li_dense += li_add
        li_dense.sort()
        li_add = [0.5 * (x1 + x2) for x1, x2 in zip(li_dense, li_dense[1:])]
    return li_dense
