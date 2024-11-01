"""
Polynomials (:mod:`pyavia.numeric.polynomial`)
==============================================

.. currentmodule:: pyavia.numeric.polynomial

Useful polynomial operations not otherwise covered by `NumPy` / `SciPy`.
"""

import numpy as np
import numpy.typing as npt


# Written by Eric J. Whitney, April 2024.


# ======================================================================

def divided_difference(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """
    Calculate a single divided divided difference value [y0, y1, ...,
    yn] (also written as `f[x0, x1, ..., xn]`), based on the number of
    `(x, y)` points provided.

    Parameters
    ----------
    x, y : array_like of float, shape (n,)
        Arrays of `x` and `y` values.  The order corresponds to the divided
        difference required.

    Returns
    -------
    float
        The value of the divided difference.

    Notes
    -----
    - Divided differences are arranged as follows [1]_::

        Points      1st Div. Diff.  2nd Div. Diff. ...
        (x0, y0)
                    [y1, y0]
        (x1, y1)                    [y2, y1, y0]
                    [y2, y1]
        (x2, y2)                    [y3, y2, y1]
                    [y3, y2]
        (x3, y3)
        ...

      Where :math:`[y_i, y_j] = f[x_i, x_j] = (y_i - y_j) /
      (x_i - x_j)`, and so on.

    - Divided differences are symmetric, e.g. :math:`[y_i, y_j] =
      [y_j, y_i]`.  The first divided difference is :math:`[y0] = f[x0]
      = y0`.

    - Working values are cast to `float` as purely integer parameters
      may result in integer division giving incorrect results.

    References
    ----------
    .. [1] Divided Differences: https://en.wikipedia.org/wiki/Divided_differences

    Examples
    --------
    Given a series of `(x, y)` points such as `(-1, 11.3)`, `(0, 2)`,
    `(1, -2.7)`:
    >>> divided_difference([-1], [11.3])  # 0th D-D.
    11.3
    >>> divided_difference([-1, 0], [11.3, 2])  # 1st D-D.
    -9.3
    >>> divided_difference([-1, 0, 1], [11.3, 2, -2.7])  # 2nd D-D.
    2.3000000000000003
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    if n < 1 or np.ndim(x) != 1 or x.shape != y.shape:
        raise ValueError("'x' and 'y' must be 1D arrays of equal "
                         "length >= 1.")

    elif n == 1:
        return float(y[0])

    elif n == 2:
        return float(y[1] - y[0]) / float(x[1] - x[0])

    else:
        return (divided_difference(x[1:], y[1:]) -
                divided_difference(x[:-1], y[:-1])) / float(x[-1] - x[0])


# ----------------------------------------------------------------------


# This code heavily inspired by https://stackoverflow.com/a/49547968

def newton_poly_coeff(x: npt.ArrayLike,
                      y: npt.ArrayLike) -> np.ndarray[float]:
    """
    Generate an array of increasing divided differences for multiple
    points `(x, y)`. These are the coefficients of the interpolating
    polynomial in Newton form.

    The array contains the divided differences arranged as follows::

        `[[y0], [y0, y1], [y0, y1, y2], ...]`

    Where `[y_i, ..., y_j]` is the divided difference operator that also
    depends on the `x` values.  This can also be written `f[x_i, ...,
    x_j]`.

    Parameters
    ----------
    x, y : array_like of float, shape (n,)
        Arrays of `x` and `y` values.  `x` values are assumed to be
        sorted in increasing order.

    Returns
    -------
    np.ndarray of float, shape (n,)
        Array of divided differences, as the coefficients of the
        interpolating polynomial in Newton form `[f[x0], f[x1, x0],
        f[x2, x1, x0], ...]`.

    Notes
    -----
    - For detailed explanation of divided differences see function
      ``divided_differences``.  The array returned by this function is
      the same as the top row of the complete upper triangular matrix of
      divided differences [1]_.

    - Inputs are cast to `float` as purely integer parameters may result
      in integer division giving incorrect results.

    References
    ----------
    .. [1] Divided differences (matrix form):
           https://en.wikipedia.org/wiki/Divided_differences#Matrix_form
    """
    x = np.asarray(x, dtype=float)
    a = np.array(y, dtype=float, copy=True)
    if (np.ndim(x) != 1) or (x.shape != a.shape):
        raise ValueError("'x' and 'y' must have the same shape (n,).")

    n = len(x)
    for i in range(1, n):
        a[i:n] = (a[i:n] - a[i - 1]) / (x[i:n] - x[i - 1])

    return a


# ----------------------------------------------------------------------

def newton_poly(x_pts: npt.ArrayLike, y_pts: npt.ArrayLike,
                x: npt.ArrayLike) -> np.ndarray:
    # noinspection PyShadowingNames
    """
    Generate Newton interpolating polynomial at the given points `x`,
    based on the known points defined by `x_pts` and `y_pts`.

    The Newton polynomial of degree `n` is given by::

        :math:`P_{n-1}(x) = [y_0] + [y_0, y_1](x - x_0) + ... +
        [y_0, ..., y_n](x - x_0)(x - x_1)...(x - x_{n-1})`

    Parameters
    ----------
    x_pts, y_pts : array_like, shape (n + 1,)
        `x` and `y` coordinates of the data points.  `x` values are
        assumed to be sorted in increasing order.

    x : array_like, shape (m,)
        New `x`-coordinates at which to evaluate the polynomial.

    Returns
    -------
    np.ndarray, shape (m,)
        The interpolated values at the given x-coordinates.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> x = [-5, -1, 0, 2, 4]
    >>> y = [-2, 6, 1, 3, -2]
    >>> x_interp = np.linspace(-5, 4, 50)
    >>> y_interp = newton_poly(x, y, x_interp)
    >>> plt.plot(x, y, 'ks')  # doctest:+ELLIPSIS
    [<matplotlib.lines.Line2D object at ...>]
    >>> plt.plot(x_interp, y_interp, 'b-')  # doctest:+ELLIPSIS
    [<matplotlib.lines.Line2D object at ...>]
    >>> plt.show()
    """
    x = np.asarray(x)
    if np.ndim(x) != 1:
        raise ValueError("'x' must be a 1D array.")

    a = newton_poly_coeff(x_pts, y_pts)
    n = len(x_pts) - 1  # Degree of interpolating polynomial.
    p = a[n]

    for k in range(1, n + 1):
        p = a[n - k] + (x - x_pts[n - k]) * p

    return p
