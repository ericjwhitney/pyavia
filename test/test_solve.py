from unittest import TestCase

import numpy as np


# Define test function, first and second derivative.  Used for scalars or
# element-wise on arrays.
def f(x):
    return x ** 2 - x - 1


def df_dx(x):
    return 2 * x - 1


def df2_dx2(x):
    return 2 * np.ones_like(x)


def f_exact(x):
    return 1.618033988749895 * np.ones_like(x)


class TestBisectRoot(TestCase):
    def test_bisect_root(self):
        from pyavia.solve.bisect_root import bisect_root

        # Check normal operation.
        x = bisect_root(f, 1.0, 2.0, ftol=1e-15)
        self.assertAlmostEqual(x, f_exact(x), places=15)

        # Check failure to converge is flagged.
        with self.assertRaises(RuntimeError):
            bisect_root(f, 1.0, 2.0, maxits=10, ftol=1e-15)


class TestNewtonBounded(TestCase):
    def test_newton_bounded_scalar(self):
        from pyavia.solve.newton_bounded import newton_bounded

        # Check normal operation.
        x = newton_bounded(f, 1.0, bounds=(-1, 10))
        self.assertAlmostEqual(x, f_exact(x), places=12)

        # Check failure to converge is flagged.
        with self.assertRaises(RuntimeError):
            x = newton_bounded(f, 1.0, bounds=(-1, 10), maxiter=2)

        # Check convergence to boundary.
        x = newton_bounded(f, 3.0, bounds=(f_exact(x), 10))
        self.assertAlmostEqual(x, f_exact(x), places=12)

        # Check failure to converge if solution is outside boundary.
        with self.assertRaises(RuntimeError):
            newton_bounded(f, 3.0, bounds=(f_exact(x) + 1, 10))

        # Check single-sided boundary.
        x = newton_bounded(f, 3.0, bounds=(-np.inf, 10))
        self.assertAlmostEqual(x, f_exact(x), places=12)

        # Check no boundaries.
        x = newton_bounded(f, 3.0, bounds=(-np.inf, +np.inf))
        self.assertAlmostEqual(x, f_exact(x), places=12)

    # TODO add versions that check with derivatives.

    def test_newton_bounded_vector(self):
        from pyavia.solve.newton_bounded import newton_bounded
        from numpy.testing import assert_allclose

        # Check normal operation.
        n_pts = 5
        x0 = np.linspace(1.0, 5.0, num=n_pts)
        bounds = (np.full(x0.shape, -1), np.full(x0.shape, 10))
        sol = newton_bounded(f, x0, bounds=bounds)
        assert_allclose(sol, f_exact(x0))

        # Check convergence to boundary.
        x0 = np.linspace(2.0, 5.0, num=n_pts)
        bounds = (np.full(x0.shape, f_exact(x0)), np.full(x0.shape, 10))
        sol = newton_bounded(f, x0, bounds=bounds)
        assert_allclose(sol, f_exact(x0))

        # Check failure to converge if solution is outside boundary.
        x0 = np.linspace(3.0, 5.0, num=n_pts)
        bounds = (np.full(x0.shape, f_exact(x0) + 1), np.full(x0.shape, 10))
        with self.assertRaises(RuntimeError):
            newton_bounded(f, x0, bounds=bounds)

    # TODO add versions that check with derivatives.
