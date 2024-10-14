
import numpy as np


# ======================================================================

# Define test functions, along with first and second derivatives.  For
# use on scalars or element-wise on arrays.

def f(x):
    return x ** 2 - x - 1


def df_dx(x):
    return 2 * x - 1


def df2_dx2(x):
    return 2 * np.ones_like(x)


def f_exact(x):
    return 1.618033988749895 * np.ones_like(x)
