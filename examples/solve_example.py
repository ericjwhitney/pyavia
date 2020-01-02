#!usr/bin/env python3

# Examples of the solution of systems of equations.
# Last updated: 30 December 2019 by Eric J. Whitney

from math import cos, inf
from solve import solve_dqnm


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
    return [cos(x_i) - 1 for x_i in x]


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
x0 = [0.87] * ndim
bounds = ([-1] * ndim, [+inf] * ndim)
x_result = solve_dqnm(std_problem_1, x0=x0, xtol=1e-4, bounds=bounds,
                      maxits=500, order=2, verbose=True)
print(f"\nResult x = {x_result}")
