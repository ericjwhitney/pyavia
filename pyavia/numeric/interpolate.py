
from __future__ import annotations
from typing import Sequence, Union

import numpy as np
from scipy.interpolate import interp1d

from pyavia.util import find_bracket
from pyavia.numeric.lines import line_pt

# Written by Eric J. Whitney, March 2021.


# ======================================================================

# TODO This may end up being deleted.
def linear_int_ext(data_pts, p, scale=None, allow_extrap=False):
    """
    Interpolate data points to find remaining unknown values absent from
    `p` with optionally scaled axes. If `p` is not in the range and
    `allow_extra` == True, a linear extrapolation is done using the two data
    points at the end corresponding to the `p`.

    Parameters
    ----------
    data_pts : list_like(tuple)
        [(a_1, ... a_n), ...] sorted on the required axis (either direction).
    p : list_like
        Required point to interpolate / extrapolate with at least a single
        known component, i.e. :math:`(..., None, p_i, None, ...)`. If
        more than one is supplied, the first is used.
    scale :
        Same as ``line_pt`` scale.
    allow_extrap : bool, optional
        If True linear extrapolation from the two adjacent endpoints is
        permitted. Default = False.

    Returns
    -------
    list :
        Interpolated / extrapolated point :math:`[q_1, ..., q_n]` where
        :math:`q_i = p_i` from above.
    """
    if len(data_pts) < 2:
        raise ValueError("At least two data points required.")
    if scale is None:
        scale = [None] * len(data_pts[0])

    # Get working axis.
    for ax, x in enumerate(p):
        if x is not None:
            break
    else:
        raise ValueError("Requested point must include at least one known "
                         "value.")

    def on_axis(li):  # Return value along required axis.
        return li[ax]

    # Get two adjacent points for straight line.
    try:
        # Try interpolation.
        # noinspection PyTypeChecker
        l_idx, r_idx = find_bracket(data_pts, p, key=on_axis)
    except ValueError:
        if not allow_extrap:
            raise ValueError(f"Point not within data range.")

        if ((on_axis(data_pts[0]) < on_axis(data_pts[-1])) != (
                on_axis(p) < on_axis(data_pts[0]))):
            l_idx, r_idx = -2, -1  # RHS extrapolation.
        else:
            l_idx, r_idx = 0, 1  # LHS extrapolation.

    return line_pt(data_pts[l_idx], data_pts[r_idx], p, scale)


# -----------------------------------------------------------------------------

def smooth_array2d(arr: [Sequence], axis, x_idx: int,
                   x_new: Union[int, Sequence], kind: str = 'linear',
                   fill_value: Union[Sequence, (Sequence, Sequence), str, None]
                   = None):
    """
    Produces a new 2D array by doing 1D interpolation of rows / columns in
    given input array.  This can be used to regularise data, or make a larger
    input array from a smaller amount of data.

    The returned array is produced by interpolating `arr` along `axis`
    and the other axis is unchanged.  The row / column index in `x_idx` is
    treated as the independent variable and is replaced by  `x_new`.
    Remaining rows / columns are interpolated as the dependent variables
    (e.g. 'y').

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
              variable along `array2d[:, x_idx]` if ``axis == 0`` or
              `array2d[x_idx, :]` if ``axis == 1``.

            - int: Use an number of equally spaced points between min and max
              values of the independent axis.  Equivalent to: ``x_new =
              np.linspace(np.min(x), np.max(x), num=x_new)``
    kind : str, optional
        Interpolation method to use (default = 'linear').  Any 'kind'
        acceptable to the underlying interpolator
        ``scipy.interpolate.interp1d`` can be used.

        .. Note:: Sufficient data rows / columns must be available to compute
           higher order interpolations `quadratic` or `cubic`, i.e. 3 and 4
           respectively.

    fill_value : None, array-like, (array-like, array_like) or 'extrapolate'
                 (optional)
        This argument provided determines if / how extrapolation outside the
        given range is handled:

            - None: Extrapolation is not performed.
            - Single value: This value will be used to fill in for requested
              points outside of the data range. If not provided, then the
              default is NaN.
            - Two-element tuple:  The first element is used as a fill value
              for x_new < x[0] and the second element is used for x_new >
              x[-1]. Anything that is not a 2-element tuple (e.g., list or
              ndarray, regardless of shape) is taken to be a single
              array-like argument meant to be used for both bounds as below,
              above = fill_value, fill_value.
            - If 'extrapolate', then points outside the data range will be
              extrapolated using the interpolation function.

        ..note:: 'extrapolate' is not normally recommended using higher order
          interpolations due to potential kinks.

    Returns
    -------
    result : np.array
        A new 2D array with specified row / columb replaced with x_new and
        remaining rows / columns replaced with interpolated values.

    Raises
    ------
    AttributeError
        If input array is not 2D.
    ValueError
        On incorrect interpolation axis, independent index, number of
        available datapoints of `x_new` outside data range (for fill_value
        == None).
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
                         bounds_error=True if fill_value is None else False,
                         fill_value=fill_value)

    if isinstance(x_new, int):
        x_new = np.linspace(np.min(x), np.max(x), num=x_new)

    y_new = interp_fn(x_new)
    return np.insert(y_new, x_idx, x_new, axis=1 - axis)


def smooth_multi(x: Sequence, y: [Sequence], x_new: Sequence,
                 kind: str = 'linear',
                 fill_value: Union[Sequence, (Sequence, Sequence), str, None]
                 = None) -> [Sequence]:
    """
    Smooth multiple equal-length arrays / lists `y` (dependent variables) using
    interpolation / extrapolation, taking `x` as the independent variable.

    Examples
    --------
    >>> xi = [1, 4, 7, 10]
    >>> yd = [[3, 4, 5, 6], [10, 8, 5, 4]]
    >>> new_x = [0, 5, 10, 15]
    >>> smooth_multi(xi, yd, new_x, kind='cubic', fill_value=([3, 10], [6, 4]))
    [[3.0, 4.333333333333334, 6.0, 6.0], [10.0, 6.962962962962963, 4.0, 4.0]]

    Parameters
    ----------
    x : array-like
        Independent variable array.
    y : [array-like, ...]
        One or more equal-length arrays / lists of data which depend on `x`.
        Length must be equal to `x`.
    x_new : array-like
        The result will be interpolated / extrapolated so that the first
        array will equal this value.
    kind : str
        See ``smooth_array2d()``.
    fill_value
        See ``smooth_array2d()``.  If provided then ``len(fill_value) ==
        len(y)`` is required.

    Returns
    -------
    result : [array-like, ...]
        List of smoothed arrays in the same layout as `data`.
    """
    res = smooth_array2d(np.vstack((x, *y)), axis=1, x_idx=0, x_new=x_new,
                         kind=kind, fill_value=fill_value)
    return [subarr.flatten().tolist() for subarr in np.vsplit(res[1:], len(y))]


# -----------------------------------------------------------------------------


# TODO This may be a duplicate of 'subdivide_series'
def subd_num_list(li: Sequence, max_size: int) -> list:
    """
    Expands a sorted list of values by repeatedly subdividing the intervals
    in the initial list, stopping before `max_size` is exceeded.  This keeps
    the originally specified values in position.

    Parameters
    ----------
    li : list or array-like
        Starting values.
    max_size : int
        Upper limit on ``len(li_dense)``.

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


# -----------------------------------------------------------------------------

class Interp2Level:
    """
    Two step interpolator based on SciPy interp1d, for interpolating data
    where multiple single-Y-valued curves are given.  Interpolation
    proceeds as follows for an intermediate level:

    - A parameterised curve is generated from the provided data on either
      side.
    - The parameterised curve is interrogated for the specific X-value
      required.
    """

    def __init__(self, x, y_lvls, lvl_vals, *, y_kind='linear',
                 y_bounds_error=None, y_fill_value=np.nan,
                 lvl_kind='linear', lvl_bounds_error=None,
                 lvl_fill_value=np.nan, copy=True, assume_sorted=False):
        """

        Parameters
        ----------
        x : (N,) array_like
            A 1-D array of `x` real values.

        y_lvls : (N,M) array_like
            A ``N x M`` array of real values, where columns represent the `y`
            data at different values of the level curves:

                - ``N`` must equal to the length of `x`.
                - ``M`` must equal to the length of `val_lvl`.

        lvl_vals : (M,) array_like
            A 1-D array of values corresponding to each level curve.
            Minimum `M` = 2.

        y_kind, y_bounds_error, y_fill_value :
            Interpolation kind, bounds handling and fill value to
            use during the first interpolation step along level curves.
            For details refer to SciPy interp1d.

        lvl_kind, lvl_bounds_error, lvl_fill_value :
            Interpolation kind, bounds handling and fill value to
            use during the second interpolation step between level curves.
            For details refer to SciPy interp1d.

        copy, assume_sorted : bool (optional)
            Refer to SciPy interp1d.
        """
        # Check inputs.
        y_lvls = np.asarray(y_lvls)
        if y_lvls.ndim != 2:
            raise ValueError("y values along level curves must be 2D array.")

        n, m = y_lvls.shape
        if n != len(x):
            raise ValueError(f"Number of y rows ({n}) does not match the "
                             f"number of x values ({len(x)}).")

        if m != len(lvl_vals):
            raise ValueError(f"Number of y columns ({m}) does not match "
                             f"the number of levels ({len(lvl_vals)}).")

        if len(lvl_vals) < 2:
            raise ValueError(f"At least two levels required, got "
                             f"{len(lvl_vals)}.")

        #  Build the first step interpolators.
        self._lvl_interp = []
        for j in range(m):
            self._lvl_interp.append(
                interp1d(x, y_lvls[:, j], kind=y_kind, copy=copy,
                         bounds_error=y_bounds_error,
                         fill_value=y_fill_value,
                         assume_sorted=assume_sorted))
        self._lvl_vals = lvl_vals
        self._lvl_kind = lvl_kind
        self._lvl_bounds_error = lvl_bounds_error
        self._lvl_fill_value = lvl_fill_value
        self._copy, self._assume_sorted = copy, assume_sorted

    def __call__(self, x, lvl):
        """Determine the interpolated value.  Similar to implemention in
        SciPy _Interpolator1D, except we require both an `x` and level
        curve value (or sequence of values).

        .. Note::  Data for each level curve may not overlap the required
           area.  Points where interpolation are not available are simply
           excluded from the second level of interpolation and we attempt to
           continue.  This could cause higher order interpolations to fail
           if there are insufficient points remaining.

        Parameters
        ----------
        x : array_like
            `x` values where interpolation is required.
        lvl : array_like
            Level curve values where interpolation is required. Requires
            len(`x`) == len(`y`).

        Returns
        -------
        y : array_like
            Interpolated values. Shape is determined by replacing
            the interpolation axis in the original array with the shape of
            `x`.
        """
        x, lvl = np.atleast_1d(x), np.atleast_1d(lvl)
        if x.ndim != 1 or lvl.ndim != 1:
            raise ValueError(f"Incorrect dimensions for x or level curve "
                             f"values.")

        if x.shape[0] != lvl.shape[0]:
            raise ValueError(f"Number of x values ({x.shape[0]}) must match "
                             f"the number of level curve values "
                             f"({lvl.shape[0]}).")

        # First step:  Interpolate y for each level curve.
        m = len(self._lvl_vals)
        y_lvls = []
        for j in range(m):
            # TODO allow fails
            y_lvls.append(self._lvl_interp[j](x))

        # Second step:  Build interpolator using these points and evaluate.
        interp_lvl = interp1d(self._lvl_vals, y_lvls,
                              kind=self._lvl_kind, copy=self._copy,
                              bounds_error=self._lvl_bounds_error,
                              fill_value=self._lvl_fill_value,
                              assume_sorted=self._assume_sorted)
        y_final = interp_lvl(lvl)

        return y_final
