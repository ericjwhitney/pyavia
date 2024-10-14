import numpy as np


# ----------------------------------------------------------------------------

def solve_dqnm(func, x0, xtol=1e-6, ftol=None, bounds=None, maxits=25,
               order=2, jacob_diag=None, verbose=False) -> [float]:
    r"""
    Solve nonlinear system of equations using the diagonal quasi-Newton
    method of [1]_.

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
        Next `x` position determined via a linear (``order = 1``) or
        quadratic (``order = 2``) estimate.
    jacob_diag : list_like
        Initial estimate of diagonal elements of Jacobian.  If None, assumes
        :math:`D = I`.
    verbose : bool
        If True, print status updates during run.

    Returns
    -------
    x : [float]
        Converged solution.

    Raises
    ------
    ValueError
        Invalid parameters.
    RuntimeError
        Maximum iterations reached before convergence.

    References
    ----------
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
