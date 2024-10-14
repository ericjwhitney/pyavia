from unittest import TestCase

from .scalar_tst_functions_early import f, f_exact


# ======================================================================


class TestBisectRoot(TestCase):
    def test_bisect_root(self):
        from pyavia.numeric.solve.bisect_root import bisect_root

        # Check normal operation.
        x = bisect_root(f, 1.0, 2.0, ftol=1e-15)
        self.assertAlmostEqual(x, f_exact(x), places=15)

        # Check failure to converge is flagged.
        with self.assertRaises(RuntimeError):
            bisect_root(f, 1.0, 2.0, maxits=10, ftol=1e-15)
