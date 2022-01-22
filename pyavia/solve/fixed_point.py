import numpy as np


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

    Examples
    --------
    A fixed-point iteration of a scalar function:

        >>> def f(x_): return (x_ + 10) ** 0.25
        >>> x_scalar = fixed_point(f, x0=-3, xtol=1e-6, verbose=True)
        Second-Order Damped Fixed Point Iteration:
        ... Iteration 1: x = -3.000000
        ... Iteration 2: x =  1.626577
        ... Iteration 3: x =  1.846558
        ... Iteration 4: x =  1.855231
        ... Iteration 5: x =  1.855571
        ... Iteration 6: x =  1.855584
        ... Iteration 7: x =  1.855585
        ... Converged.

    This example uses the same function however `x` is now a list.  Note
    that this works because internally everything is converted to NumPy
    arrays, provided component-wise operations are valid and `func(x)` can
    also return a list:

        >>> x_vector = fixed_point(f, x0=[-3, -4], xtol=[1e-6]*2,
        ... verbose=True)
        Second-Order Damped Fixed Point Iteration:
        ... Iteration 1: x = [-3.000000, -4.000000]
        ... Iteration 2: x = [ 1.626577,  1.565085]
        ... Iteration 3: x = [ 1.846558,  1.844112]
        ... Iteration 4: x = [ 1.855231,  1.855135]
        ... Iteration 5: x = [ 1.855571,  1.855567]
        ... Iteration 6: x = [ 1.855584,  1.855584]
        ... Iteration 7: x = [ 1.855585,  1.855585]
        ... Converged.

    Parameters
    ----------
    func : Callable[array-like]
        Function that returns a better estimate of `x`.
    x0 : array-like
        Starting value for `x`. Any numeric type including user types may be
        used, provided they support component-wise mathematical operations.
        Individual elements need not be the same type.  Internally they are
        converted to NumPy arrays.
    xtol : array-like
        Stop when ``abs(x' - x) < xtol``.  The type/s or element/s of `xtol`
        should be broadcastable to `x`.
    h : float, optional
        Step size (time-like) to advance `x` to next estimate.  The default
        value of 1.0 should be acceptable in most cases.  Reduce if
        instability is suspected (e.g. 0.5, 0.25, etc).
    maxits : int, optional
        Iteration limit (default = 15).
    verbose : bool, optional
        If True, print iterations (default = False).

    Returns
    -------
    result : array-like
        Converged `x` value.

    Raises
    ------
    RuntimeError
        If maxits is exceeded.
    """
    # EJW Revised 6/1/22.
    # x, xtol_ = np.atleast_1d(x0), np.atleast_1d(xtol)
    # u = x - x  # This makes a zero array of arbitrary objects.

    x, xtol_ = np.asarray(x0), np.asarray(xtol)
    u = np.zeros_like(x)

    its = 0
    if verbose:
        print(f"Second-Order Damped Fixed Point Iteration:")

    while its < maxits:
        x_n = x + u * h
        fx_n = np.asarray(func(x_n))

        delta = fx_n - x
        u_n = u + h * delta - 2 * h * u
        x, u = np.asarray(x_n), u_n  # Hold array to simplify logic.

        its += 1

        if verbose:
            print(f"... Iteration {its}: x = " +
                  np.array2string(x, precision=6, suppress_small=True,
                                  separator=', ', sign=' ', floatmode='fixed'))

        if np.all(abs(delta) < xtol_):
            if verbose:
                print(f"... Converged.")

            # EJW Revised 6/1/22.
            # return x.tolist() if len(x) > 1 else x.item()
            return x.tolist()

    raise RuntimeError(f"Limit of {maxits} iterations exceeded.")
