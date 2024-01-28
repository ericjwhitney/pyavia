#!usr/bin/env python3

# Example of a 1-D modelling functions.
# Written by Eric J. Whitney, May 2023.

import matplotlib.pyplot as plt
import numpy as np

from pyavia.data import PCHIP1D, SmoothPoly1D  # noqa

# Example data points.
x = [-1.2, 0.2, 1.1, 2.0, 3.9, 5.0, 5.5, 7.0]
y = [1.2, 0.2, 1.4, 4.3, 15.0, 25.0, 31.3, 35.1]

# ----------------------------------------------------------------------
# Examples of different functions.

# Fit a smooth polynomial up to order 3 through the points applicable
# to a domain slightly larger than the input points.  No extrapolation
# beyond the domain is defined.  Try R2_min = 0.5, 0.95 for lower
# quality.
f_smpoly1 = SmoothPoly1D(x, y, x_domain=(-10, 10), degree=(1, 2, 3),
                         R2_min=0.97)

# Fit a smooth quartic through the points, but fix the value at both
# ends of the given points domain.
f_smpoly2 = SmoothPoly1D(x, y, degree=4,  ext_lo=0, ext_hi=33)

# Fit a PCHIP function (passes through all point) adding a linear
# extrapolation at the low end and 'natural' continuation above x_max.
f_pchip1 = PCHIP1D(x, y, x_domain=(-2, +np.inf), ext_lo='linear')

# Fit the same PCHIP function but only use a subsection of the points
# domain and limiting it to a constant both ends.  Note however that all
# points are still used when computing the basic PCHIP function.
f_pchip2 = PCHIP1D(x, y, x_domain=(1, 6), ext_lo='constant',
                   ext_hi='constant')

show_funcs = [f_smpoly1, f_smpoly2, f_pchip1, f_pchip2]

# ----------------------------------------------------------------------

# Solve / plot each function in turn.
for f_approx in show_funcs:
    f_type = str(f_approx)
    print(f"\nFunction type: {f_type}:")
    print(f"\tLow side extraplation (< x_min): {str(f_approx.ext_lo)}")
    print(f"\tHigh side extraplation (> x_max): {str(f_approx.ext_hi)}")
    print(f"\tx_domain: {f_approx.x_domain}")
    print(f"\tx_extents: {f_approx.x_extents}")

    # Find a y value on the curve (extend search to max extents).
    y_sol = 6.0
    x_sol = f_approx.solve(y_sol)
    print(f"\tSolution of f(x) = {y_sol} -> x = {x_sol}")

    # Compute the proxy function and derivative at some typical points.
    print(f"\tPlot of f(x) and df(x)/dx ...")
    x_plot_min = np.clip(min(x) - 5, *f_approx.x_extents)
    x_plot_max = np.clip(max(x) + 5, *f_approx.x_extents)
    x_plot = np.linspace(x_plot_min, x_plot_max, num=500)
    y_plot = f_approx(x_plot)
    dydx_plot = f_approx.derivative(x_plot)

    plt.figure()
    plt.plot(x, y, '^k', label="ORIGINAL POINTS")
    plt.plot(x_plot, y_plot, '--b', label="APPROX. FUNC.")
    plt.grid(axis='both')
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title(f"FUNCTION - {f_type}")

    plt.figure()
    plt.plot(x_plot, dydx_plot, '--b')
    plt.grid(axis='both')
    plt.xlabel("$x$")
    plt.ylabel(r"${dy}/{dx}$")
    plt.title(f"DERIVATIVE - {f_type}")

    plt.show(block=False)

plt.show(block=True)
