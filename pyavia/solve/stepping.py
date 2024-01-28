from collections.abc import Callable

import numpy as np

from pyavia.solve.exception import SolverError


# Written by Eric J. Whitney, January 2023.


# ======================================================================

def step_bracket_min(func: Callable[..., float], x1: float,
                     x2: float, func_args=(), x_limit: float = None,
                     max_steps: int = 50) -> (float, float):
    """
    Find an initial bracket (`x1`, `x2`) of the minimum of a function
    `func` by stepping in the `x` direction. The search direction is
    towards `x2`, i.e. if ``x2 > x1`` then positive `x` steps are taken,
    otherwise if ``x2 < x1`` then negative `x` steps are taken.

    A minimum bracket means ``func(x1) >= func(xm)`` and ``func(x2) >=
    func(xm)`` where `xm` is a midpoint between `x1` and `x2`. Only the
    first bracket found is returned.

    Parameters
    ----------
    func : Callable[[float, ...], float]
        Scalar function to bracket root.
    x1 : float
        First starting point for bracket.
    x2 : float
        Second starting point for bracket (in search direction).
    func_args : optional
        Extra arguments passed to be passed to `func`.
    x_limit : float, optional
        Stops if the next step will exceed this value.
    max_steps : int, default = 50
        Stops once this number of steps has been completed.

    Returns
    -------
    x1, x2 : (float, float)
        `x`-values bracketing the minimum.

    Raises
    ------
    ValueError
        Illegal starting conditions.

    SolverError
        Failure to converge raises a `SolverError` exception including
        the following attributes:
            - `x1`, `x2`: Most recent bracket values used.
            - `y1`, `y2`: Function values corresponding to `x1`, `x2`.
            - `flag` and `detail`:
                - 1: Reached max_steps.
                - 2: Reached x_limit.
            - 'xm': Midpoint between `x1` and `x2`.
            - 'ym': Function value corresponding to `xm`.
            - 'steps': Number of steps taken.
            - 'fevals': Number of function evaluations.
    """
    if x1 == x2:
        raise ValueError("x1, x2 must have different values.")

    if x_limit is None:
        x_limit = (x2 - x1) * np.inf

    if (x2 > x1 and x_limit <= x2) or (x2 < x1 and x_limit >= x2):
        raise ValueError("x_limit must be outside initial bracket in "
                         "direction of search.")

    y1, y2 = func(x1, *func_args), func(x2, *func_args)
    xm = 0.5 * (x1 + x2)  # Midpoint.
    ym = func(xm, *func_args)
    steps, fevals = 0, 0

    # Main loop.
    while y1 < ym or y2 < ym:
        if steps >= max_steps:
            raise SolverError("step_bracket_min() failed to converge:",
                              flag=1, details="Reached max_steps.",
                              x1=x1, x2=x2, y1=y1, y2=y2, xm=xm, ym=ym,
                              steps=steps, fevals=fevals)

        # Check if we had stopped at the boundary on the last step.
        if x2 == x_limit:
            raise SolverError("step_bracket_min() failed to converge:",
                              flag=2, details="Reached x_limit.",
                              x1=x1, x2=x2, y1=y1, y2=y2, xm=xm, ym=ym,
                              steps=steps, fevals=fevals)

        # Advance a half step.
        x1, xm = xm, x2
        y1, ym = ym, y2
        x2 = 2.0 * xm - x1
        steps += 1

        # Check if this will step over the boundary. If so we shrink the
        # interval to allow this to be the last step.
        if (x2 > x1 and x2 > x_limit) or (x2 < x1 and x2 < x_limit):
            x2 = x_limit
            xm = 0.5 * (x2 - x1)
            ym = func(xm, *func_args)
            fevals += 1

        y2 = func(x2, *func_args)
        fevals += 1

    # Completed successfully.
    return x1, x2


# ======================================================================

def step_bracket_root(func: Callable[..., float], x1: float,
                      x2: float, func_args=(), x_limit: float = None,
                      max_steps: int = 50) -> (float, float):
    """
    Find an initial bracketing interval (`x1`, `x2`) of the root of
    function `func` by stepping in the `x` direction. The search
    direction is towards `x2`, i.e. if ``x2 > x1`` then positive `x`
    steps are taken, otherwise if ``x2 < x1`` then negative `x` steps
    are taken.

    A bracket will give opposite signs for ``func(x1)`` and
    ``func(x2)``. Only the first bracket encountered is returned.

    Parameters
    ----------
    func : Callable[[float, ...], float]
        Scalar function to bracket root.
    x1 : float
        First starting point for bracket.
    x2 : float
        Second starting point for bracket (in search direction).
    func_args : optional
        Extra arguments passed to be passed to `func`.
    x_limit : float, optional
        Stops if the next step will exceed this value.
    max_steps : int, default = 50
        Stops once this number of steps has been completed.

    Returns
    -------
    x1, x2 : float, float
        `x`-values bracketing the root.

    Raises
    ------
    ValueError
        Illegal starting conditions.

    SolverError
        Failure to converge raises a `SolverError` exception including
        the following attributes:
            - `x1`, `x2`: Most recent bracket values used.
            - `y1`, `y2`: Function values corresponding to `x1`, `x2`.
            - `flag` and `detail`:
                - 1: Reached max_steps.
                - 2: Reached x_limit.
            - 'steps': Number of steps taken.
            - 'fevals': Number of function evaluations.
    """
    if x1 == x2:
        raise ValueError("x1, x2 must have different values.")

    if x_limit is None:
        x_limit = (x2 - x1) * np.inf

    if (x2 > x1 and x_limit <= x2) or (x2 < x1 and x_limit >= x2):
        raise ValueError("x_limit must be outside initial bracket in "
                         "direction of search.")

    y1, y2 = func(x1, *func_args), func(x2, *func_args)
    steps, fevals = 0, 0

    # Main loop.
    while np.sign(y1) == np.sign(y2):
        if steps >= max_steps:
            raise SolverError("step_bracket_root() failed to converge:",
                              flag=1, details="Reached max_steps.",
                              x1=x1, x2=x2, y1=y1, y2=y2, steps=steps,
                              fevals=fevals)

        # Check if we had stopped at the boundary on the last step.
        if x2 == x_limit:
            raise SolverError("step_bracket_root() failed to converge:",
                              flag=2, details="Reached x_limit.",
                              x1=x1, x2=x2, y1=y1, y2=y2, steps=steps,
                              fevals=fevals)

        # Advance one step.
        x1, x2 = x2, 2.0 * x2 - x1
        y1 = y2
        steps += 1

        # Check if this is a step over the boundary. If so we trim x2 to
        # allow this to be the last step.
        if (x2 > x1 and x2 > x_limit) or (x2 < x1 and x2 < x_limit):
            x2 = x_limit

        y2 = func(x2, *func_args)
        fevals += 1

    # Completed successfully.
    return x1, x2
