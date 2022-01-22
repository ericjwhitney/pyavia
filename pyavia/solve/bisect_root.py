"""
**pyavia.solve** provides functions for finding solutions to various types of
equations. These are included when not covered by NumPy or different variant
is required.
"""


# Last updated: 6 January 2022 by Eric J. Whitney


# ----------------------------------------------------------------------------


def bisect_root(func, x_a, x_b, maxits: int = 50, ftol=1e-6, verbose=False):
    # noinspection PyUnresolvedReferences
    r"""
    Approximate solution of :math:`f(x) = 0` on interval :math:`x \in [x_a,
    x_b]` by the bisection method. For bisection to work :math:`f(x)` must
    change sign across the interval, i.e. ``func(x_a)`` and ``func(x_b)`` must
    return values of opposite sign.

    .. note: This function is able to be used with arbirary units.

    Examples
    --------
    >>> f = lambda x: x**2 - x - 1
    >>> bisect_root(f, 1, 2, 25)  # This will take 17 iterations.
    1.6180343627929688
    >>> f = lambda x: (2*x - 1)*(x - 3)
    >>> bisect_root(f, 0, 1, 10)  # Only 1 it. (soln was in centre).
    0.5

    Parameters
    ----------
    func : Callable[scalar]
        Function which we are searching for root.
    x_a,x_b : scalar
        Each end of the search interval, in any order.
    maxits : int
        Maximum number of iterations.
    ftol : scalar
        End search when :math:`|f(x)| < f_{tol}`.
    verbose : bool
        If True, print progress statements.

    Returns
    -------
    x_m : scalar
        Best estimate of root found i.e. :math:`f(x_m) \approx 0`.

    Raises
    ------
    RuntimeError
        If maxits is reached before a solution is found.
    """
    if verbose:
        print(f"Bisecting Root:")
    x_a_next, x_b_next = x_a, x_b
    f_a_next, f_b_next = func(x_a_next), func(x_b_next)
    try:
        # We use division approach to compare signs instead of
        # multiplicationas it neatly cancels units if present.  But this
        # requires a check for func(x_b) == 0 first.
        if f_a_next/f_b_next >= 0:   # Sign comp. via division.
            raise ValueError(f"f(x_a) and f(x_b) must have opposite "
                             f"sign.")
    except ZeroDivisionError:
        raise ValueError(f"One of the start points is already zero.")

    it = 0
    while True:
        # Compute midpoint.
        x_m = (x_a_next + x_b_next) / 2
        f_m = func(x_m)
        it += 1

        if verbose:
            print(f"... Iteration {it}: x = [{x_a_next}, {x_m}, {x_b_next}], "
                  f"f = [{f_a_next}, {f_m}, {f_b_next}]")

        # Check stopping criteria.
        if abs(f_m) < ftol:
            return x_m

        if it >= maxits:
            raise RuntimeError(f"Reached {maxits} iteration limit.")

        # Check which side root is on, narrow interval.
        if f_a_next / f_m < 0:  # Sign comp. via division.
            x_b_next = x_m
            f_b_next = f_m
        else:
            x_a_next = x_m
            f_a_next = f_m
