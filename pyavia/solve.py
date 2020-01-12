"""
**pyavia.solve** provides functions for finding solutions to various types of
equations. These are included when not covered by NumPy or different variant
is required.
"""

# Last updated: 11 January 2020 by Eric J. Whitney

import numpy as np

__all__ = ['bisect_root', 'fixed_point', 'solve_dqnm']


# ----------------------------------------------------------------------------


def bisect_root(func, x_a, x_b, maxits: int = 50, ftol=1e-6, verbose=False):
    # noinspection PyUnresolvedReferences
    r"""
    Approximate solution of :math:`f(x) = 0` on interval :math:`x \in [x_a,
    x_b]` by the bisection method. For bisection to work :math:`f(x)` must
    change sign across the interval, i.e. ``func(x_a)`` and ``func(x_b)`` must
    return values of opposite sign.

    .. note: This function is able to be used with arbirary units.

    ..
        >>> import pyavia as pa

    Examples
    --------
    >>> f = lambda x: x**2 - x - 1
    >>> pa.solve.bisect_root(f, 1, 2, 25)  # This will take 17 iterations.
    1.6180343627929688
    >>> f = lambda x: (2*x - 1)*(x - 3)
    >>> pa.solve.bisect_root(f, 0, 1, 10)  # Only 1 it. (soln was in centre).
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
        x_m = (x_a_next + x_b_next)/2
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


# ----------------------------------------------------------------------------

def fixed_point(func, x0, xtol, h: float = 1.0, maxits: int = 15,
                verbose=False):
    # noinspection PyTypeChecker
    r"""
    Find the fixed point of a function :math:`x = f(x)` by iterating a
    damped second-order ODE.  The ODE is solved as two equations using the
    forward Euler method:

        1. :math:`x' = x + uh`
        2. :math:`u' = u + h(f(x') - x) - 2hu`

    Note that equation 2 for `u'` above is a simplification of the following:

        2. :math:`u' = u + (h / m)(f(x') - x) - (2{\zeta}h / \sqrt{m})u`

    Where:

        - m: Fictitious 'mass' to give inertia to the solution x.
        - :math:`\zeta`: Damping ratio.

    For practical problems we take :math:`m = 1` because the 'force'
    (correction size :math:`f(x') - x`) is of the same magnitude as :math:`x`.
    We take :math:`\zeta = 1` because critical damping is generally the
    shortest path to convergence.

    ..
        >>> import pyavia as pa

    Examples
    --------
        A fixed-point iteration of a scalar function:

        >>> def f(x): return (x + 10) ** 0.25
        >>> x_scalar = pa.solve.fixed_point(f, x0=-3, xtol=1e-6, verbose=True)
        Second-Order Damped Fixed Point Iteration:
        ... Iteration 1: x = [   -3.0000]
        ... Iteration 2: x = [    1.6266]
        ... Iteration 3: x = [    1.8466]
        ... Iteration 4: x = [    1.8552]
        ... Iteration 5: x = [    1.8556]
        ... Iteration 6: x = [    1.8556]
        ... Iteration 7: x = [    1.8556]
        ... Converged.

    This example uses the same function however `x` is now a list.  Note
    that this works because internally everything is converted to NumPy
    arrays, provided component-wise operations are valid and `func(x)` can
    also return a list:

        >>> x_vector = pa.solve.fixed_point(f, x0=[-3, -4], xtol=[1e-6]*2,
        ... verbose=True)
        Second-Order Damped Fixed Point Iteration:
        ... Iteration 1: x = [   -3.0000    -4.0000]
        ... Iteration 2: x = [    1.6266     1.5651]
        ... Iteration 3: x = [    1.8466     1.8441]
        ... Iteration 4: x = [    1.8552     1.8551]
        ... Iteration 5: x = [    1.8556     1.8556]
        ... Iteration 6: x = [    1.8556     1.8556]
        ... Iteration 7: x = [    1.8556     1.8556]
        ... Converged.

    Parameters
    ----------
    func : Callable[scalar or list_like]
        Function that returns a better estimate of `x`.
    x0 : scalar or list_like
        Starting value for `x`. Any numeric type including user types may be
        used, provided they support component-wise mathematical operations.
        Individual elements need not be the same type.  Internally they are
        converted to NumPy arrays.
    xtol : scalar or list_like
        Stop when ``abs(x' - x) < xtol``.  The type/s or element/s of `xtol`
        should correspond to `x`.
    h : float
        Step size (time-like) to advance `x` to next estimate.  The default
        value of 1.0 should be acceptable in most cases.  Reduce if
        instability is suspected (e.g. 0.5, 0.25, etc).
    maxits : int
        Iteration limit.
    verbose : bool
        If True, print iterations.

    Returns
    -------
    scalar or list_like
        Converged x value.

    Raises
    ------
        RuntimeError
            If maxits is exceeded.
    """
    x, xtol_ = np.atleast_1d(x0), np.atleast_1d(xtol)
    u = x - x  # This makes a zero array of arbitrary objects.
    its = 0
    if verbose:
        print(f"Second-Order Damped Fixed Point Iteration:")

    while its < maxits:
        x_n = x + u * h
        fx_n = np.atleast_1d(func(x_n) if len(x_n) > 1 else func(x_n.item()))
        delta = fx_n - x
        u_n = u + h * delta - 2 * h * u
        x, u = x_n, u_n
        its += 1

        if verbose:
            with np.printoptions(threshold=5,
                                 formatter={'float_kind':
                                            lambda num: f'{num:#10.5G}'}):
                print(f"... Iteration {its}: x = {x}")

        if np.all(abs(delta) < xtol):
            if verbose:
                print(f"... Converged.")
            return x.tolist() if len(x) > 1 else x.item()

    raise RuntimeError(f"Limit of {maxits} iterations exceeded.")


# ----------------------------------------------------------------------------

def solve_dqnm(func, x0, xtol=1e-6, ftol=None, bounds=None, maxits=25, order=2,
               jacob_diag=None, verbose=False):
    r"""
    Solve nonlinear system of equations using the diagonal quasi-Newton method
    of [1]_.

    Notes
    -----
    - This method only estimates the diagonal elements of the Jacobian. As
      such it only needs O(N) storage and does not require any matrix
      solution steps.
    - Additional to [1]_: Optional bounds check and adaptive scaling of move
      :math:`s`.  If bounds are exceeded the move is scaled back to a factor
      of 0.75 of the distance remaining to the boundary. In this way a
      solution on the boundary can stil be approached via a number of steps
      without the solver getting immediately stuck on the edge.  Iteration
      stops if the multiplier becomes smaller than :math:`\epsilon = 1
      \times 10^{-30}`.
    - Additional to [1]_: There is a check for extremely small moves where
      :math:`\nu_0 \approx \nu_1`, evaluating :math:`|\nu_1 - \nu_0| <
      \epsilon`.  We drop back to first order for this step if this is the
      case.
    - Additional to [1]_: Drops back to first order if :math:`\|F(x)\|` is
      escaping upwards at this step with :math:`\|F(x')\| > 2\|F(x)\|`.

    Parameters
    ----------
    func : Callable[list_like]
        Vector valued function taking `x` and returning `F(x)`.
    x0 : list_like
        Vector of numeric types as starting `x` value.  Not suitable for use
        with user types due to matricies and norms, etc.
    xtol : float
        Stop when :math:`\|x' - x\| < x_{tol}`.
    ftol : float
        When present we also require :math:`\|F(x)\| <= f_{tol}` before
        stopping.
    bounds : tuple(list_like, list_like)
        A tuple of low and high bounds respectively i.e. :math:`([x_{low},
        ...], [x_{high}, ...])` that activates bounds checking.  If specific
        bounds are not required these can be individually set to +/-inf.
    maxits : int
        Maximum number of iterations allowed.
    order : {2, 1}
        Next `x` position determined via a linear (``order = 1``) or quadratic
        (``order = 2``) estimate.
    jacob_diag : list_like
        Initial estimate of diagonal elements of Jacobian.  If None, assumes
        :math:`D = I`.
    verbose : bool
        If True, print status updates during run.

    Returns
    -------
    list
        Converged solution.

    Raises
    ------
    ValueError
        Invalid parameters.
    RuntimeError
        Maximum iterations reached before convergence.

    References
    -----

    .. [1] Waziri, M. Y. and Aisha, H. A., "A Diagonal Quasi-Newton Method
       For Systems Of Nonlinear Equations", Applied Mathematical and
       Computational Sciences Volume 6, Issue 1, August 2014, pp 21-30.
    """

    def verbose_print(info):
        if verbose:
            print(info)

    if order != 1 and order != 2:
        raise ValueError(f"Order must be 1 or 2.")

    n = len(x0)
    tiny = 1e-30
    if jacob_diag is None:
        d = np.ones(n)  # [D] Diagonal elements of Jacobian.  D = I.
    else:
        d = np.array(jacob_diag).astype(float)
    x = np.array(x0)
    fx = np.array(func(x))
    fx_norm = np.linalg.norm(fx)
    s, y = None, None
    it = 1

    if len(fx) != n:
        raise TypeError(f"Function result size ({len(fx)}) does not match "
                        f"problem dimension ({n}).")
    if len(d) != n:
        raise TypeError(f"Number of Jacobian diagonals ({len(d)}) does not "
                        f"match problem dimension ({n}).")

    verbose_print(f"Diagonal Quasi-Newton Method - " +
                  (f"First" if order == 1 else f"Second") + f" Order - " +
                  f"Solving {n} Equations:")

    while True:
        if it >= maxits:
            raise RuntimeError(f"Reached maximum iteration limit: {maxits}")

        # Take step.
        x_prev, fx_prev, s_prev, y_prev = x, fx, s, y
        s = -fx_prev / d
        x = x_prev + s

        # EJW Addition: Bounds check.
        if bounds is not None:
            with np.errstate(invalid='ignore'):  # Handle +/-inf warning.
                lo_mult = max(s / (bounds[0] - x_prev))
                hi_mult = max(s / (bounds[1] - x_prev))

            mult = max(1.0, lo_mult, hi_mult)  # Factor stepped over bound.
            if mult > 1.0:
                mult = 0.75 / mult  # 0.75 seems a bit more stable than 0.9.
                if mult < tiny:
                    raise RuntimeError(f"Resting on boundary, terminating.")
                s *= mult
                x = x_prev + s
                verbose_print(f"*** Shortened step size x{mult:.3G} due to "
                              f"boundary condition.")

        # Evaluate new point.
        fx = np.array(func(x))
        fx_norm_prev = fx_norm
        fx_norm = np.linalg.norm(fx)
        y = fx - fx_prev
        it += 1

        # Do second order update scheme if requested.
        rho, mu, rho_norm = [None] * 3  # Reset as poss. trigger.
        while (s_prev is not None) and (order == 2):
            # Note:  ^^^ This is a 'while' masquerading as an 'if'.  There
            # is a break at the end.

            # EJW Addition: Drop back to first order if ||Fx|| is escaping.
            if fx_norm > 2 * fx_norm_prev:
                verbose_print(f"*** Skipped second order step: ||F(x)|| "
                              f"rising. ")
                break

            # Second order update scheme.
            nu1 = -np.linalg.norm(s)
            nu0 = -np.linalg.norm(s + s_prev)

            # EJW Addition: Check for extremely small moves.
            if abs(nu1 - nu0) < tiny:
                verbose_print(f"*** Skipped second order step: ν₁ ≈ ν₂.")
                break

            beta = -nu0 / (nu1 - nu0)
            alpha = (beta ** 2) / (1 + 2 * beta)
            rho = s - alpha * s_prev
            mu = y - alpha * y_prev
            rho_norm = np.linalg.norm(rho)

            # Fallback to first order.
            if np.dot(rho, mu) < (xtol * rho_norm * np.linalg.norm(mu)):
                rho = None
                verbose_print(f"*** Skipped second order step: "
                              f"ρᵀμ < xₜₒₗ.||ρ||.||μ|| ")

            break  # <<< Terminates the 'if' statement

        # Do first order update scheme (as defined or as fallback).
        if rho is None:
            rho, mu = s, y
            rho_norm = np.linalg.norm(rho)

        if n < 10:
            x_str = f", x* = " + f', '.join(f"{x_i:.6G}" for x_i in x)
        else:
            x_str = ''
        verbose_print(f"... Iteration {it}: ||F(x)|| = {fx_norm:.5G}, "
                      f"||x' - x|| = {rho_norm:.5G}{x_str}")

        # Check stopping criteria.
        if rho_norm <= xtol and (fx_norm <= ftol if ftol else True):
            verbose_print(f"... Converged.")
            return x.tolist()

        # Update [D].
        if rho_norm >= xtol:
            g = rho ** 2
            d += g * (np.dot(rho, mu) - np.dot(g, d)) / (np.dot(g, g))
