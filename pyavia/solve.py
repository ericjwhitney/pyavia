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
    Find the fixed point of a scalar function x = f(x) by iteratively
    passing an estimate through the function.  Return when the point
    stabilises.  Example:
    >>> def f(x): return (x + 10) ** 0.25
    >>> x_fixed = fixed_point(f, x0=-3, xtol=1e-6, verbose=True)
    Fixed Point Iteration: x0 = -3
    ... Iteration 1: x = 1.6265765616977852
    ... Iteration 2: x = 1.8465580452920933
    ... Iteration 3: x = 1.8552312312746646
    ... Iteration 4: x = 1.855570704344644
    ... Iteration 5: x = 1.8555839877110183
    ... Iteration 6: x = 1.855584507474938
    ... Converged.

    Args:
        func: Function that returns a better estimate of 'x'.
        x0: Starting value for 'x'. Any numeric type (including user
            types) may be used.  Individual elements need not be the same type.
        xtol: Stop when abs(x' - x) < xtol.
        relax: Relaxation factor.  After the next trial value 'x_step' is
            computed, it is revised to x' as follows:
               x' = x + relax * (x_step - x)
            If relax = 1 this is equivalent to standard iteration using x_step.
            When relax < 1 (e.g. 0.5) this is an under-relaxation which can
            add stability.
        maxits: (int) Iteration limit.
        verbose: (bool) If True, print iterations.

    Returns:
        Converged x value.
    """
    if verbose:
        print(f"Fixed Point Iteration: x0 = {x0}")
    x, its = x0, 0
    while True:
        if its >= maxits:
            raise RuntimeError(f"Limit of {maxits} iterations exceeded.")

        x_step = func(x)
        x_next = x + relax * (x_step - x)
        its += 1
        if verbose:
            print(f"... Iteration {its}: x = {x_next}")

        if abs(x_next - x) < xtol:
            if verbose:
                print(f"... Converged.")
            break
        x = x_next
    return x_next


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
            of 0.9 the distance remaining to the boundary. In this way a
            solution on the boundary can stil be approached via a number of
            steps without the solver getting immediately stuck on the edge.
        - EJW Addition: Check for extremely small moves where nu_0 approx
            equals nu_1.  We drop back to first order for this step in this
            case. 1e-20 is taken as a universally tiny number for the
            comparison.

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
        maxits: (Opt) Maximum number of iterations allowed.  Default = 50.
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
    if order != 1 and order != 2:
        raise ValueError(f"Order must be 1 or 2.")

    n = len(x0)
    if jacob_diag is None:
        d = np.ones(n)  # [D] Diagonal elements of Jacobian.  D = I.
    else:
        d = np.array(jacob_diag).astype(float)
    x = np.array(x0)
    fx = np.array(func(x))
    s, y = None, None
    it = 1

    if len(fx) != n:
        raise TypeError(f"Function result size ({len(fx)}) does not match "
                        f"problem dimension ({n}).")
    if len(d) != n:
        raise TypeError(f"Number of Jacobian diagonals ({len(d)}) does not "
                        f"match problem dimension ({n}).")

    if verbose:
        print(f"Diagonal Quasi-Newton Method - "
              f"{'First' if order == 1 else 'Second'} Order - Solving {n} "
              f"Equations:")

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
                mult = 0.9 / mult
                x = x_prev + mult * s
                if verbose:
                    print(f"*** Shortened step size by factor {mult:.5G} due "
                          f"to boundary condition.")

        # Evaluate new point.
        fx = np.array(func(x))
        fx_norm = np.linalg.norm(fx)
        y = fx - fx_prev
        it += 1

        rho, mu, rho_norm = [None] * 3  # Reset for later calc.
        if s_prev is not None and order == 2:
            # Second order update scheme.
            nu1 = -np.linalg.norm(s)
            nu0 = -np.linalg.norm(s + s_prev)

            # EJW Addition: Check for extremely small moves.
            if (nu1 - nu0) > 1e-20:
                beta = -nu0 / (nu1 - nu0)
                alpha = (beta ** 2) / (1 + 2 * beta)
                rho = s - alpha * s_prev
                mu = y - alpha * y_prev
                rho_norm = np.linalg.norm(rho)

                if np.dot(rho, mu) < (xtol * rho_norm * np.linalg.norm(mu)):
                    # Fallback to first order.
                    rho = None

        if rho is None:
            # First order update scheme.
            rho, mu = s, y
            rho_norm = np.linalg.norm(rho)

        if verbose:
            if n < 10:
                x_str = f", x* = " + f', '.join(f"{x_i:.5G}" for x_i in x)
            else:
                x_str = ''
            print(f"... Iteration {it}: ||Fx|| = {fx_norm:.5G}, "
                  f"||x' - x|| = {rho_norm:.5G}{x_str}")

        # Check stopping criteria.
        if rho_norm <= xtol and (fx_norm <= ftol if ftol else True):
            if verbose:
                print(f"... Converged.")
            return x.tolist()

        # Update [D].
        if rho_norm >= xtol:  # tol:
            g = rho ** 2
            d += g * (np.dot(rho, mu) - np.dot(g, d)) / (np.dot(g, g))
