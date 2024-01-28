import numpy as np
from scipy.special import binom


# =============================================================================

def bernstein_poly(t: np.ndarray, i: int, n: int):
    """
    Evaluate Bernstein polynomial for t (array) and i, n.
    """
    return binom(n, i) * (t ** i) * (1 - t) ** (n - i)


# -----------------------------------------------------------------------------

def bezier(t: np.ndarray, ctrl_pts: np.ndarray) -> np.ndarray:
    """
    Compute a Bézier curve with given length parameter and control points.

    Parameters
    ----------
    t : ndarray, shape (N,)
        Length parameters along the curve 0 <= `t` <= 1.

    ctrl_pts: ndarray, shape(M, D)
        Quantity 'M' Bezier control points of dimension `D`.

    Returns
    -------
    x : ndarray, shape(N, D)
        Points of dimension `D` corresponding to each `t` value.
    """
    x = np.zeros((t.shape[0], ctrl_pts.shape[1]))
    n = ctrl_pts.shape[0] - 1
    for i in range(n + 1):
        x += bernstein_poly(t, i, n)[:, np.newaxis] * ctrl_pts[i, :]
    return x


# -----------------------------------------------------------------------------

def bezier_deriv(t: np.ndarray, ctrl_pts: np.ndarray) -> np.ndarray:
    """
    Compute derivative along a Bézier curve with given length parameter and
    control points.

    Parameters
    ----------
    Same as ``bezier()``.

    Returns
    -------
    dxdt : ndarray, shape(N, D)
        Derivative of curve with respect to `t` at each point.
    """
    dxdt = np.zeros((t.shape[0], ctrl_pts.shape[1]))
    n = ctrl_pts.shape[0] - 1
    for i in range(n):
        dxdt += bernstein_poly(t, i, n - 1)[:, np.newaxis] * (
                ctrl_pts[i + 1, :] - ctrl_pts[i, :])
    return n * dxdt
