"""
Filters (:mod:`pyavia.numeric.filter`)
======================================

.. currentmodule:: pyavia.numeric.filter

Data filtering algorithms.

.. autosummary::
    :toctree:

    J211_2pole
    J211_4pole
    MovingAverage
    pinv_simple
    savgol_variable
"""

# Last updated: 17 March 2023 by Eric J. Whitney.
from collections import deque
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike


# ===========================================================================

# noinspection PyPep8Naming
def J211_2pole(X, T: float, CFC: int, *, axis: int = 0,
               fwd_pass: bool = True):
    """
    This a 2-pole Butterworth filter, implemented using the exact algorithm
    of SAE J211-1 Appendix C.  This algorithm is designed for filtering
    impact test data and does not rely on `scipy.signal`.  The 2-pole
    filter will phase shift the result in the direction of filtering.  See
    `J211_4pole()` for a phaseless implementation,

    Filter startup is acheived by directly copying the first two data
    points to the output.  Filter startup effects can be avoided by having
    at least 10 ms of data before or after the area of interest with
    direction depending on `fwd_pass`.

    Parameters
    ----------

    X : (N,…) or (…,N) array_like
        Input data stream as an N-D array.
    T : float
        Sample period (i.e. time step) in seconds.  Typical order might be
        around `T` = 1e-3 (i.e. milliseconds).
    CFC : float
        Channel Frequency Class is a number corresponding to the channel
        frequency response lies and is numerically equal to :math:`F_H` the
        filter high-pass frquency (in Hz).  Examples are shown in SAE J211
        Figure 1 for CFCs of 1000 and 600.
    axis : int
        For N-dimensional data, this represents the time-like axis (default
        `axis` = 0).
    fwd_pass : bool (optional)
        If False, the data is passed over in reverse.

    Returns
    -------
    Y : (N,…) or (…,N) array_like
        Filtered output data stream with shape matching `X`.
    """
    X = np.asarray(X)
    if X.ndim != 1 and X.ndim != 2:
        raise ValueError("Data stream must be scalars or vectors.")
    if axis != 0 and axis != 1:
        raise ValueError("Axis must be 0 or 1.")

    N = X.shape[axis]
    if N < 10:
        raise ValueError("Too few data points provided.")
    if axis == 1:
        X = X.reshape((N, -1))  # Internally time is axis 0.
    if not fwd_pass:
        X = X[::-1]

    sqrt2 = np.sqrt(2)
    w_d = 2 * np.pi * CFC * 2.0775
    w_a = np.sin(w_d * T / 2) / np.cos(w_d * T / 2)
    a_0 = (w_a ** 2) / (1 + sqrt2 * w_a + w_a ** 2)
    a_1 = 2 * a_0
    a_2 = a_0
    b_1 = -2 * (w_a ** 2 - 1) / (1 + sqrt2 * w_a + w_a ** 2)
    b_2 = (-1 + sqrt2 * w_a - w_a ** 2) / (1 + sqrt2 * w_a + w_a ** 2)

    Y = np.empty_like(X)  # Filtered result.
    Y[0] = X[0]
    Y[1] = X[1]
    for i in range(2, N):
        Y[i] = (a_0 * X[i] + a_1 * X[i - 1] + a_2 * X[i - 2] +
                b_1 * Y[i - 1] + b_2 * Y[i - 2])  # SAE J211 Equation C1.

    if not fwd_pass:
        Y = Y[::-1]
    if axis == 1:
        Y = Y.reshape((-1, N))
    return Y


# ---------------------------------------------------------------------------

# noinspection PyPep8Naming
def J211_4pole(X, T: float, CFC: int, axis: int = 0):
    """
    This a 4-pole phaseless Butterworth filter, implemented using the exact
    algorithm of SAE J211-1 Appendix C.  This algorithm is designed for
    filtering impact Test data and does not rely on `scipy.signal`.  The
    4-pole filter is simply a forwards and backwards pass of the 2-pole
    filter of the data (see `J211_2pole()`) for a phaseless implementation,

    Filter startup is acheived by directly copying the first two data points
    at each end to the output.  Filter startup effects can be avoided by
    having at least 10 ms of data beyond each end of the area of interest.

    Parameters
    ----------

    X : (N,…) or (…,N) array_like
       Input data stream as an N-D array.
    T : float
       Sample period (i.e. time step) in seconds.  Typical order is around
       1e-3 (milliseconds).
    CFC : float
       Channel Frequency Class is a number corresponding to the channel
       frequency response lies and is numerically equal to :math:`F_H`
       the filter high-pass frquency (in Hz).  Examples are shown in SAE
       J211 Figure 1 for CFCs of 1000 and 600.
    axis : int
       For N-dimensional data, this represents the time-like axis (default
       `axis` = 0).

    Returns
    -------
    Y : (N,…) or (…,N) array_like
       Filtered output data stream with shape matching `X`.
    """
    Y_pass1 = J211_2pole(X, T, CFC, axis=axis, fwd_pass=True)
    Y_pass2 = J211_2pole(Y_pass1, T, CFC, axis=axis, fwd_pass=False)
    return Y_pass2


# =============================================================================

class MovingAverage:
    """
    This is a simple on-line moving average filter with a given fixed length.
    Each time a new data point is added, the oldest is dropped and the
    current output value is updated.  The output value is the equal
    (unweighted) average of the stored data points.

    Examples
    --------
    Setup a filter of length 3 using some initial values:
    >>> my_filter = MovingAverage([1, 2, 3, 4, 5], length=3)
    >>> my_filter.value
    4.0

    Note the filter value is 4.0 because only the last three values are
    stored. Add a new data point:
    >>> my_filter.add(7)
    5.333333333333333

    ``add()`` also returns the updated moving average, in this case
    ``(4 + 5 + 7) / 3 = 5.333...``.
    """

    def __init__(self, init_vals: Sequence, length: int = None):
        """
        Parameters
        ----------
        init_vals : list or array-like
            Values used to initialise the filter.
        length : int (optional)
            Set the filter length (default = None).  It is required that
            ``length <= len(init_vals)``.  If `length` is not given,
            the filter length will be equal to the length of `init_vals`.
        """
        if len(init_vals) <= 1:
            raise ValueError("Initial values required to start the filter.")

        length = length if length is not None else len(init_vals)
        if length > len(init_vals) or length < 1:
            raise ValueError("Invalid filter length.  Must be >= 1 and <= "
                             "length of initial values.")

        self._pts = deque(maxlen=length)  # Stores values pre-multiplied.
        for x in init_vals:
            self._pts.appendleft(x)

        self._current_sum = sum(self._pts)

    def add(self, x):
        """
        Adds a new data point to the filter, discards the oldest and updates
        the current moving average.

        Parameters
        ----------
        x : number-like
            New data point to add to the filter.

        Returns
        -------
        average : number-like
            The new moving average value of the filter.

        """
        self._current_sum -= self._pts.pop()  # Subtract oldest.
        self._current_sum += x  # Add newest.
        self._pts.appendleft(x)
        return self.value

    @property
    def value(self):
        """
        Returns the current value of the moving average.
        """
        return self._current_sum / len(self._pts)


# ===========================================================================

def pinv_simple(A: ArrayLike, side: str = 'left') -> np.ndarray:
    """
    Compute the pseudoinverse (or `Moore–Penrose` inverse) of matrix `A`.
    This is simpler than the version contained in `numpy.linalg.pinv` as it
    requires `A` to be a full-rank matrix containing only real values.

    Parameters
    ----------
    A : shape (m, n), array_like
        Input matrix.  See description above for allowed element types.

    side : str, default = 'left'
        Which inverse `side` to compute:
            - ``side='left'``: :math:`A^+ = (A^T A)^{-1} A^T`.  This is
               used when when `A` has linearly independent columns.
            - ``side='right'``: :math:`A^+ = A^T (A A^T)^{-1}`.  This is
               used when when `A` has linearly independent rows.
    Returns
    -------
    shape (n, m), np.ndarray
        `A+` the pseudoinverse of `A`.
    """
    A = np.atleast_2d(A)
    if A.ndim != 2:
        raise ValueError(f"Input array not 2D.")

    if side == 'left':
        return np.linalg.inv(A.T @ A) @ A.T

    elif side == 'right':
        return A.T @ np.linalg.inv(A @ A.T)

    else:
        raise ValueError(f"Invalid side '{side}'.")

# ---------------------------------------------------------------------------


def savgol_variable(x: ArrayLike, y: ArrayLike, window: int,
                    order: int, passes: int = 1) -> np.ndarray:
    """
    A Savitzky–Golay digital filter for *unequally spaced* datapoints.
    This type of filter is typically used to smooth general datapoints
    without distorting the signal phase / behaviour [1]_.

    This implementation is more general than
    `scipy.signal.savgol_filter` as it allows variable step sizes in
    `x` [2]_ [3]_.

    Parameters
    ----------
    x : shape(n), array_like
        Sequence of floats giving `x` values.
    y : shape(n), array_like
        Sequence of floats giving `y` values.
    window : int
        Odd number to use as window / filter length, with ``window <
        len(x)``.
    order : int
        Order of local interpolating polynomial, with ``order <
        window``.
    passes : int, default = 1
        Number of times to pass the filter over the data.  If
        ``passes < 1`` the input `y` values are returned unchanged.

    Returns
    -------
    numpy.ndarray
        Smoothed `y`-values.

    Notes
    -----
    .. [1] Savitzky-Golay filter:
           https://en.wikipedia.org/wiki/Savitzky-Golay_filter
    .. [2] Endpoints of data set are interpolated in the same manner as
           `scipy.signal.savgol_filter`.
    .. [3] Adapted from code shown here: https://dsp.stackexchange.com/a/64313
    """
    x, y = np.atleast_1d(x), np.atleast_1d(y)

    # Multiple passes - apply recursively.
    if passes > 1:
        y_filt = np.array(y, copy=True)
        for _ in range(passes):
            y_filt = savgol_variable(x, y_filt, window, order, passes=1)

        return y_filt

    elif passes < 1:
        return y
    
    # Single pass method from this point on.
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError("'x' and 'y' must be 1D arrays of the same size.")

    n_pts = len(x)
    if window % 2 == 0 or window >= n_pts or not isinstance(window, int):
        raise ValueError("'window' must be an odd integer < len(x).")

    if order >= window or not isinstance(order, int):
        raise ValueError("'order' must be an integer < window.")

    half_window = window // 2
    order += 1

    # Initialize variables.
    t = np.empty(window)  # Local `x` coordinates.
    A = np.empty((window, order))  # Design matrix.
    lh_coeffs = np.zeros(order)  # Interpolating coefficients for ...
    rh_coeffs = np.zeros(order)  # ... each end of data set.
    y_filt = np.full(n_pts, np.nan)  # Result vector.

    # Step through intermediate points, each time analysing a window centred
    # on x[i].
    for i in range(half_window, n_pts - half_window):

        # Compute local 'x' coordinate (t).
        for j in range(window):
            t[j] = x[i + j - half_window] - x[i]

        # Compute the design matrix (A).
        for j in range(window):
            r = 1.0
            for k in range(order):
                A[j, k] = r
                r *= t[j]

        # Calculate the left pseudoinverse of the design matrix and get c0 /
        # y[i] using the top row.
        A_pinv = pinv_simple(A, side='left')
        y_filt[i] = 0
        for j in range(window):
            y_filt[i] += A_pinv[0, j] * y[i + j - half_window]

        # If at the LH or RH end, retain interpolating coefficients.
        if i == half_window:
            # LH end.
            for j in range(window):
                for k in range(order):
                    lh_coeffs[k] += A_pinv[k, j] * y[j]

        elif i == n_pts - half_window - 1:
            # RH end.
            for j in range(window):
                for k in range(order):
                    rh_coeffs[k] += A_pinv[k, j] * y[n_pts - window + j]

    # Do LH end interpolations.
    for i in range(half_window):
        y_filt[i] = 0
        x_i = 1
        for j in range(order):
            y_filt[i] += lh_coeffs[j] * x_i
            x_i *= x[i] - x[half_window]

    # Do RH end interpolations.
    for i in range(n_pts - half_window, n_pts):
        y_filt[i] = 0
        x_i = 1
        for j in range(order):
            y_filt[i] += rh_coeffs[j] * x_i
            x_i *= x[i] - x[-half_window - 1]

    return y_filt

# ---------------------------------------------------------------------------
