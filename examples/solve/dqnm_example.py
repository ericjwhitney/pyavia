#!usr/bin/env python3

# Examples of the solution of systems of equations.
# Last updated: 21 December 2022 by Eric J. Whitney


import numpy as np

from pyavia.solve.dqnm import solve_dqnm


def linear_system_example(x):
    """A simple linear system of equations."""
    n = len(x)
    res = [0] * n
    for i in range(n):
        for j in range(n):
            res[i] += (1 + i * j) * x[j]
    return res


def std_problem_1(x):
    """Standard start point x0 = (0.87, 0.87, ...)"""
    return [np.cos(x_i) - 1 for x_i in x]


def std_problem_3(x):
    """Standard start point x0 = (0.5, 0.5, ...)"""
    n = len(x)
    f = [0.0] * n
    for i in range(n - 1):
        f[i] = x[i] * x[i + 1] - 1
    f[n - 1] = x[n - 1] * x[0] - 1
    return f


# Solve one of the above problems at a given size.
ndim = 500
x0 = [0.5] * ndim
bounds = ([-1] * ndim, [+np.inf] * ndim)
x_result = solve_dqnm(std_problem_1, x0=x0, ftol=1e-5, xtol=1e-6,
                      bounds=bounds, maxits=50, order=2, verbose=True)

print("\nResult x = " +
      np.array2string(np.asarray(x_result), precision=6, suppress_small=True,
                      separator=', ', sign=' ', floatmode='fixed'))
