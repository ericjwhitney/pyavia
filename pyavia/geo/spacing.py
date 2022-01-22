from typing import List

import numpy as np


# =============================================================================

def sine_spacing(x1: float, x2: float, n: int, spacing: float) -> List[float]:
    r"""
    Generates a non-linear distribution of values in the interval
    :math:`[x_1, x_2]`.

    .. Note:: This procedure is adapted from a Fortran subroutine contained
       within the `AVL` source code written by M. Drela and H. Youngren (2002).

    Parameters
    ----------
    x1 : float
        Startpoint (inclusive) of the interval.

    x2 : float
        Endpoint (inclusive) of the interval.

    n : int
        Number of points to generate.

    spacing : float
        Distribution type [-3..+3]:

            - *0* = Equal spacing.
            - *1* = Cosine spacing.
            - *2* = Sine spacing (points concentrated toward `x1`).
            - *3* = Equal spacing.

        A negative value of `dist` produces reversed spacing (only applicable
        to sine spacing).  Intermediate / fractional values produce a spacing
        which is a combination of the adjacent integer values.

    Returns
    -------
    x : [float]
        Points in the interval [`x1`, `x2`] clustered to the distribution.

    Examples
    --------
    >>> sine_spacing(0.0, 10.0, 5, spacing=0)  # Equal spacing.
    array([ 0. ,  2.5,  5. ,  7.5, 10. ])
    >>> sine_spacing(0.0, 10.0, 5, spacing=1)  # Cosine spacing.
    array([ 0.        ,  1.46446609,  5.        ,  8.53553391, 10.        ])
    >>> sine_spacing(0.0, 10.0, 5, spacing=-2)  # Sine spacing toward end.
    array([ 0.        ,  3.82683432,  7.07106781,  9.23879533, 10.        ])

    """
    if not (-3 <= spacing <= 3):
        raise ValueError("Invalid spacing parameter.")

    # Compute fractions of each spacing type to use.
    spc_abs = np.abs(spacing)
    if 0 <= spc_abs < 1:
        f_eq, f_cos, f_sin = 1 - spc_abs, spc_abs, 0
    elif 1 <= spc_abs < 2:
        f_eq, f_cos, f_sin = 0, 2 - spc_abs, spc_abs - 1
    else:
        f_eq, f_cos, f_sin = spc_abs - 2, 0, 3 - spc_abs

    # Compute spacing due to equal and cosine distributions.
    u = np.linspace(0.0, 1.0, n)
    u_spc = (f_eq * u) + (0.5 * f_cos * (1 - np.cos(u * np.pi)))

    # Add spacing due to sine distribution.
    if spacing >= 0:
        u_spc += f_sin * (1 - np.cos(0.5 * u * np.pi))
    else:
        u_spc += f_sin * np.sin(0.5 * u * np.pi)

    return (x2 - x1) * u_spc + x1
