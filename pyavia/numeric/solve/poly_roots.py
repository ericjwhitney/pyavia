import numpy as np


# TODO Bernstein polynomials can go here?

# Written by Eric J. Whitney, March 2024.

# ======================================================================

def quadratic_roots(a: float, b: float, c: float, *,
                    allow_complex: bool = True
                    ) -> tuple[float | complex, ...]:
    """
    Return roots of the quadratic equation given by
    :math:`0 = ax^2 + bx + c`, where `a`, `b`, `c` are real
    coefficients.  The calculation is performed in such a way as to
    minimise roundoff errors for poorly conditioned equations [1]_.
    The straight-line case (``a=0``) is also handled.

    Parameters
    ----------
    a, b, c : float
      Real-valued coefficients of the equation.

    allow_complex : bool, default = True
      If `True`, complex valued roots are included in the result (if
      present).  If `False`, these are omitted.

    Returns
    -------
    roots : tuple[float | complex, ...]
        A tuple containing the roots of the quadratic equation, with
        length depending on the result:

        - `len(roots) == 0`: No real roots if ``allow_complex=False``,
          or no root for a line parallel to the x-axis (i.e. ``a=0``
          and ``b=0``).
        - `len(roots) == 1`: One real root.
        - `len(roots) == 2`: Two real roots, or two complex roots with
        ``allow_complex=True``.

    References
    ----------
    .. [1] Press, W. H.; Flannery, B. P.; Teukolsky, S. A.; and
           Vetterling, W. T. *Numerical Recipes: The Art of Scientific
           Computing*, 3rd ed. Cambridge, England: Cambridge University
           Press, pp. 227, 2007. Section 5.6: "Quadratic and Cubic
           Equations".

    Examples
    --------
    Equation :math:`x^2 -3x + 2 = 0` has two real roots:
    >>> quadratic_roots(1, -3, 2)
    (2.0, 1.0)

    Equation :math:`x^2 -2x + 1 = 0` has a single real root:
    >>> quadratic_roots(1, -2, 1)
    (1.0,)

    Equation :math:`x^2 + 1 = 0` has two complex roots:
    >>> quadratic_roots(1, 0, 1)
    ((-0-1j), (-0+1j))

    Equation :math:`x^2 + 4x + 5 = 0` has two complex roots:
    >>> quadratic_roots(1, 4, 5)
    ((-2-1j), (-2+1j))

    Repeating this equation with ``allow_complex=False`` omits the real
    roots:
    >>> quadratic_roots(1, 4, 5, allow_complex=False)
    ()

    Equation :math:`5x - 3 = 0` is a straight line with a single real
    root:
    >>> quadratic_roots(0, 5, -3)
    (0.6,)

    Equation :math:`0x + 2 = 0` is a straight line parallel to the
    x-axis (no roots):
    >>> quadratic_roots(0, 0, 2)
    ()
    """

    # Check for the straight-line case and shortcut if possible.
    if a == 0.0:
        if b != 0.0:
            return (-c / b,)  # Line with slope, one root.
        else:
            return ()  # Line parallel to x-axis, no roots.

    # Calculate discriminant (Δ) and sign(b).  Note: np.sign(0) = 0, so
    # we calculate sign(b) manually.
    Δ = b * b - 4 * a * c
    sign_b = +1 if b >= 0 else -1

    if Δ > 0.0:  # Two real roots.
        q = -0.5 * (b + sign_b * np.sqrt(Δ))
        return q / a, c / q

    elif Δ == 0.0:  # Single real root.
        return (-0.5 * b / a,)

    else:  # Two complex roots.
        if allow_complex:
            # Note: np.sqrt() requires a complex argument.
            q = -0.5 * (b + sign_b * np.sqrt(Δ + 0j))
            return q / a, c / q

        else:
            return ()  # Omit complex roots.
