"""
Functions for finding solutions to various types of equations. These are
included when not covered by NumPy or different variant is required.

Contains:
    bisect_root     Find scalar equation root via bisection.
    fixed_point		Iterate for the fixed point of a scalar equation.
    solve_dqnm      Solve a system of nonlinear equations using the Diagonal
                    Quasi-Newton Merhod.
"""
# Last updated: 30 December 2019 ny Eric J. Whitney

import numpy as np

__all__ = ['bisect_root', 'fixed_point', 'solve_dqnm']


# ----------------------------------------------------------------------------


def bisect_root(func, x_a, x_b, maxits: int = 50, ftol=1e-6, verbose=False):
    # noinspection PyUnresolvedReferences
    """
    Approximate solution of func(x)=0 on interval [x_a, x_b] by bisection
    method. For bisection to work func(x) must change sign across the interval,
    i.e. func(x_a) and func(x_b) must have opposite sign.

    Note: This function is able to be used with arbirary units.

    Examples
    --------
    >>> f = lambda x: x**2 - x - 1
    >>> bisect_root(f, 1, 2, 25)  # This will take 17 iterations.
    1.6180343627929688
    >>> f = lambda x: (2*x - 1)*(x - 3)
    >>> bisect_root(f, 0, 1, 10)  # Only takes 1 iteration (in middle).
    0.5

    Args:
        func: Function to find root f(x_m) -> 0.
        x_a, x_b: Each end of the search interval, in any order.
        maxits:  Maximum number of iterations.
        ftol: End search when abs(f(x)) < ftol.
        verbose: (bool) If True, print progress statements.

    Returns:
        xm: Best estimate of root found i.e. f(x_m) approx = 0.0.

    Raises:
        RuntimeError is maxits is reached before a solution is found.
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

def fixed_point(func, x0, xtol, relax=1.0, maxits: int = 15, verbose=False):
    """
    Find the fixed point of a function x = f(x) by repeatedly passing x
    through the function, and return when the result stabilises.  Here is an
    example with a scalar function:

    >>> def f(x): return (x + 10) ** 0.25
    >>> x_scalar = fixed_point(f, x0=-3, xtol=1e-6, verbose=True)
    Fixed Point Iteration: x0 = -3
    ... Iteration 1: x = [1.62657656]
    ... Iteration 2: x = [1.84655805]
    ... Iteration 3: x = [1.85523123]
    ... Iteration 4: x = [1.8555707]
    ... Iteration 5: x = [1.85558399]
    ... Iteration 6: x = [1.85558451]
    ... Converged.

    This example uses the same function however x is now a list.  Note
    that this only works because internally everything is converted to NumPy
    arrays so that component-wise operations are valid and f() also returns an
    array:

    >>> x_vector = fixed_point(f, x0=[-3, -4], xtol=[1e-6]*2, verbose=True)
    Fixed Point Iteration: x0 = [-3, -4]
    ... Iteration 1: x = [1.62657656 1.56508458]
    ... Iteration 2: x = [1.84655805 1.84411162]
    ... Iteration 3: x = [1.85523123 1.85513544]
    ... Iteration 4: x = [1.8555707  1.85556696]
    ... Iteration 5: x = [1.85558399 1.85558384]
    ... Iteration 6: x = [1.85558451 1.8555845 ]
    ... Converged.

    Args:
        func: (Scalar / List) Function that returns a better estimate of x.
        x0: (Scalar / List) Starting value for x. Any numeric type
            including user types may be used, provided they support
            component-wise mathematical operations.  Individual elements
            need not be the same type.  Internally they are converted to
            Numpy arrays.
        xtol: (Scalar / List) Stop when abs(x' - x) < xtol.  The type/s or
            element/s of xtol should correspond to x.
        relax: (Float) Relaxation factor.  The next estimate of x is
            computed as follows:
               x' = (1 - relax) * x + relax * f(x)
           If relax = 1 this is equivalent to the standard iteration of
            x' = f(x). When relax < 1 (e.g. 0.5) this is an under-
            relaxation which can add stability.
        maxits: (int) Iteration limit.
        verbose: (bool) If True, print iterations.

    Returns:
        (Scalar / List) Converged x value.
    """
    if verbose:
        print(f"Fixed Point Iteration: x0 = {x0}")
    x = np.atleast_1d(x0)
    its = 0
    while True:
        if its >= maxits:
            raise RuntimeError(f"Limit of {maxits} iterations exceeded.")

        x_old = x
        fx = np.atleast_1d(func(x) if len(x) > 1 else func(x.item()))
        x = (1 - relax) * x + relax * fx
        its += 1
        if verbose:
            print(f"... Iteration {its}: x = {x}")

        if np.all(abs(x - x_old) < xtol):
            if verbose:
                print(f"... Converged.")
            break
    return x.tolist() if len(x) > 1 else x.item()


# ----------------------------------------------------------------------------

def solve_dqnm(func, x0, xtol=1e-6, ftol=None, bounds=None, maxits=25,
               order=2, jacob_diag=None, verbose=False):
    """
    Solve nonlinear system of equations using diagonal quasi-Newton method
    from  Waziri, M. Y. and Aisha, H. A., "A Diagonal Quasi-Newton Method
    For Systems Of Nonlinear Equations", Applied Mathematical and
    Computational Sciences Volume 6, Issue 1, August 2014, pp 21-30.

    Notes:
        - This method only estimates the diagonal elements of the Jacobian.
            As such it only needs O(N) storage and does not require any
            matrix solution steps.
        - EJW Addition: Optional bounds check and adaptive scaling of move
            s.  If bounds are exceeded the move is scaled back to a factor
            of 0.75 of the distance remaining to the boundary. In this way a
            solution on the boundary can stil be approached via a number of
            steps without the solver getting immediately stuck on the edge.
        - EJW Addition: Check for extremely small moves where nu0 approx
            equals nu1 i.e. abs(nu1 - nu0) < tiny.  We drop back to first
            order for this step in this case.
        - EJW Addition: Drops back to first order if ||Fx|| is escaping
        upwards at this step with ||Fx'|| > 2*||Fx||.
        - The universally tiny number is taken to be 1e-20 for both.

    Args:
        func: Vector valued function taking list-like x and returning
            list-like F(x).
        x0: List-like of numeric types as starting x value.  Not suitable
            for use with user types due to matricies and norms,
            etc.
        xtol: (Opt) Stop when ||x' - x|| < xtol.  Default = 1e-6.
        ftol: (Opt) when present we require ||F(x)|| <= ftol before
            stopping.  Default = None.
        bounds: (Opt) A tuple of list-like objects giving low and high
            bounds respectively i.e. ([x_low, ...], [x_high, ...]) that
            activates bounds checking.  If specific bounds are not required
            these can be set to +/-inf.  Default = None.
        maxits: (Opt) Maximum number of iterations allowed.  Default = 25.
        order: (Opt) Next x position determined via a linear (order = 1) or
            quadratic (order = 2) estimate.  Default = 2.
        jacob_diag:  (Opt) Initial estimate of diagonal elements of
            Jacobian.  If None, assumes D = I.  Default = None.
        verbose: (Opt) Print status updates during run.  Default = False.

    Returns:
        x: Converged solution as list.

    Raises:
        ValueError for invalid parameters.
        RuntimeError if maximum iterations reached before convergence.
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
