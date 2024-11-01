from collections.abc import Callable

import numpy as np

from pyavia.numeric.solve import SolverError


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

# TODO This would be better named 'walking' as the bracket size doesn't
#  increase in this case.
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
        x1, x2 = x2, 2.0 * x2 - x1  # TODO align with NR
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


# ----------------------------------------------------------------------


def bracket_root(f: Callable[..., float], x1: float, x2: float, *,
                 f_args=(),
                 x_limits: tuple[float, float] = (-np.inf, np.inf),
                 grow_factor: float = 0.5,
                 Δx_max: float = np.inf,
                 max_steps: int = 50) -> tuple[float, float]:
    """
    Given an initial guessed range `x1` to `x2`, the range is expanded
    geometrically until a root of the function `f(x)` is bracketed or
    until the stopping criteria are met.

    Parameters
    ----------
    f : Callable[[float, ...], float]
        Scalar function taking a float as the first argument. May accept
        additional arguments (see parameter `args`).
    x1, x2 : float
        Starting points for bracket, with `x1` < `x2`.
    f_args : optional
        Extra arguments passed to be passed to `f()`.
    x_limits : tuple[float, float], default = (-∞, +∞)
        Stops if the next step will exceed this value.
    grow_factor : float, default = 0.5
        Size factor that determines the amount `x1` or `x2` are moved in
        each step to expand the range (`Δx`), with
        ``Δx = grow_factor * (x2 - x1)`` (unless `Δx_max` is reached -
        see below).
    Δx_max : float, default = ∞
        Maximum magnitude of movement permitted for `x1` or `x2` when
        expanding the range.
    max_steps : int, default = 100
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
        - `f1`, `f2`: Function values corresponding to `x1`, `x2`.
        - `flag` and `detail`:
            - 1: Reached max_steps.
            - 2: Reached x_limit.
        - 'steps': Number of steps taken.
        - 'fevals': Number of function evaluations.

    Notes
    -----
    - A bracket is found when `f(x1)` and `f(x2)` have opposite signs,
      or if either `f(x1)` or `f(x2)` become exactly zero.
    - Basic stepping can easily fail for functions that have extrema
      near the area of interest. Quoting [1]_: `'The procedure “go
      downhill until your function changes sign,” can be foiled by a
      function that has a simple extremum.  Nevertheless, if you are
      prepared to deal with a “failure” outcome, this procedure is
      often a good first start; success is usual if your function has
      opposite signs in the limit x → ±∞.'`

    References
    ----------
    .. [1] Press, W. H.; Flannery, B. P.; Teukolsky, S. A.; and
       Vetterling, W. T. *Numerical Recipes: The Art of Scientific
       Computing*, 3rd ed. Cambridge, England: Cambridge University
       Press, pp. 447, 2007. Section 9.1: "Bracketing and Bisection".

    Examples
    --------
    Equation :math:`y = x^2 -3x + 2` has roots at `x` = 1 and `x` = 2.
    >>> def example_fn(x):
    ...     return x**2 - 3 * x + 2

    Find bracket starting from left side:
    >>> bracket_root(example_fn, -2, -1)
    (-2, 1.375)

    Find bracket starting from right side:
    >>> bracket_root(example_fn, 3, 4)
    (1.75, 4)

    """
    if x1 >= x2:
        raise ValueError("Requires x1 < x2.")

    if Δx_max <= 0:
        raise ValueError("Requires Δx_max > 0.")

    f1 = f(x1, *f_args)
    f2 = f(x2, *f_args)
    steps, fevals = 0, 2

    # TODO Add case where relatively magnitude of f1, f2 swaps,
    #  indicating we probably stepped over a root and are on the way
    #  back out.  Perhaps in this case we could try a midpoint between
    #  x1 and x2 and see if that produces a better result.

    while np.sign(f1) == np.sign(f2):  # False if f1 or f2 == 0.
        if steps >= max_steps:
            raise SolverError("bracket_root() failed to converge:",
                              flag=1, details="Reached max_steps.",
                              x1=x1, x2=x2, f1=f1, f2=f2, steps=steps,
                              fevals=fevals)

        # Check if we had stopped at the boundary on the last step.
        if (x1 <= x_limits[0]) or (x2 >= x_limits[1]):
            raise SolverError("bracket_root() failed to converge:",
                              flag=2, details="Reached x_limit.",
                              x1=x1, x2=x2, f1=f1, f2=f2, steps=steps,
                              fevals=fevals)

        # Advance one step; clip to limits if necessary.
        Δx = min(grow_factor * (x2 - x1), Δx_max)
        if np.abs(f1) < np.abs(f2):
            x1 = max(x1 - Δx, x_limits[0])  # <- Grow left
            f1 = f(x1, *f_args)
        else:
            x2 = min(x2 + Δx, x_limits[1])  # Grow right ->
            f2 = f(x2, *f_args)

        steps += 1
        fevals += 1

    return x1, x2


# ----------------------------------------------------------------------
