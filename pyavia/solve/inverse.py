from typing import Callable, Tuple

from scipy.optimize import root_scalar


# =============================================================================

def y_to_x(fn: Callable[[float], float], y: float,
           x_range: Tuple[float, float], xtol: float = 2e-12,
           rtol: float = 8.881784197001252e-16, maxiter: int = 100) -> float:
    """
    Given a single-valued scalar function ``y = fn(x)``, find the `x` value
    that would result in the `y` value provided.  This is a simple method
    used when the inverse of a function is not available.  Internally,
    the `brentq` method of SciPy is used to find the value of `x`.

    Parameters
    ----------
    fn : Callable[[Scalar], Scalar]
        Function ``y = fn(x)`` to invert / solve for `x`.
    y : Scalar
        Objective value.
    x_range : [Scalar, Scalar]
        Bracket in which to search for `x`.
    xtol, rtol : Scalar
        Convergence parameters - Refer to SciPy `brentq` for more information.
    maxiter : int
        Maximum number of iterations permitted - Refer to SciPy `brentq` for
        more information.

    Returns
    -------
    x : Scalar
        The value giving the `y` value provided via ``y = fn(x)``.
    """

    def trial_fn(x: float) -> float:
        return y - fn(x)

    sol = root_scalar(trial_fn, method='brentq', bracket=x_range,
                      xtol=xtol, rtol=rtol, maxiter=maxiter)
    if sol.converged:
        return sol.root
    else:
        raise RuntimeError(f"Could not find 'x' value for y = fn(x) = "
                           f"{y:.5G} in interval {x_range}:  {sol.flag}")
