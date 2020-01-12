from unittest import TestCase


class TestBisectRoot(TestCase):
    def test_bisect_root(self):
        from pyavia.solve import bisect_root

        def f(g):
            return g ** 2 - g - 1

        exact = 1.618033988749895

        # Normal operation.
        x = bisect_root(f, 1.0, 2.0, ftol=1e-15)
        self.assertAlmostEqual(x, exact, places=15)

        # Failure to converge.
        with self.assertRaises(RuntimeError):
            bisect_root(f, 1.0, 2.0, maxits=10, ftol=1e-15)
