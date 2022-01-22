"""
Data filtering algorithms.
"""

# Last updated: 16 March 2021 by Eric J. Whitney.

import numpy as np

__all__ = ['J211_2pole', 'J211_4pole']


# -----------------------------------------------------------------------------

# noinspection PyPep8Naming
def J211_2pole(X, T: float, CFC: int, *, axis: int = 0, fwd_pass: bool = True):
    """
    This a 2-pole Butterworth filter, implemented using the exact algorithm of
    SAE J211-1 Appendix C.  This algorithm is designed for filtering impact
    test data and does not rely on `scipy.signal`.  The 2-pole filter will
    phase shift the result in the direction of filtering.  See `J211_4pole()`
    for a phaseless implementation,

    Filter startup is acheived by directly copying the first two data points
    to the output.  Filter startup effects can be avoided by having at least
    10 ms of data before or after the area of interest with direction
    depending on `fwd_pass`.

    Parameters
    ----------

    X : (N,…) or (…,N) array_like
        Input data stream as an N-D array.
    T : float
        Sample period (i.e. time step) in seconds.  Typical order might be
        around `T` = 1e-3 (i.e. milliseconds).
    CFC : float
        Channel Frequency Class is a number corresponding to the channel
        frequency response lies and is numerically equal to :math:`F_H`
        the filter high-pass frquency (in Hz).  Examples are shown in SAE J211
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


# noinspection PyPep8Naming
def J211_4pole(X, T: float, CFC: int, axis: int = 0):
    """
    This a 4-pole phaseless Butterworth filter, implemented using the exact
    algorithm of SAE J211-1 Appendix C.  This algorithm is designed for
    filtering impact test data and does not rely on `scipy.signal`.  The
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

# -----------------------------------------------------------------------------
