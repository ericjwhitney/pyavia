from unittest import TestCase


class TestForceType(TestCase):
    def test_force_type(self):
        from util import force_type

        x = force_type(3.5, int, float)
        self.assertIsInstance(x, int)  # int(3.5) -> 3 (int)
        self.assertEqual(x, 3)

        x = force_type("3.5+4j", float, complex)
        self.assertIsInstance(x, complex)  # complex("3.5+4j") -> (3.5+4j)
        self.assertEqual(x, 3.5 + 4j)

        x = force_type(3.5 + 4j, int, float, str)
        self.assertIsInstance(x, str)
        self.assertEqual(x, '(3.5+4j)')

        with self.assertRaises(ValueError):
            force_type(3.5 + 4j, int, float)


class TestCoaxType(TestCase):
    def test_coax_type(self):
        from util import coax_type

        x = coax_type(3.5, int, float)
        self.assertIsInstance(x, float)  # Because int(3.5) != 3.5
        self.assertEqual(x, 3.5)

        x = coax_type(3.0, int, str)
        self.assertIsInstance(x, int)
        self.assertEqual(x, 3)

        with self.assertRaises(ValueError):
            coax_type("3.0", int, float)  # Because '3.0' != 3.0

        x = 3 + 2j
        y = coax_type(x, int, float, default=x)
        self.assertIsInstance(y, complex)
        self.assertEqual(y, 3 + 2j)


class TestBisectRoot(TestCase):
    def test_bisect_root(self):
        from util import bisect_root

        def f(g):
            return g ** 2 - g - 1
        exact = 1.618033988749895

        # Normal operation.
        x = bisect_root(f, 1.0, 2.0, f_tol=1e-15)
        self.assertAlmostEqual(x, exact, places=15)

        # Failure to converge.
        with self.assertRaises(RuntimeError):
            bisect_root(f, 1.0, 2.0, max_its=10, f_tol=1e-15)


class TestLinearInterp(TestCase):
    def test_linear_interp(self):
        from util import linear_int_ext

        # Also checks line_pt by proxy.

        data = [(9, 100000),
                (8, 10000),
                (7, 1000)]

        res = linear_int_ext(data, (7.5, None))  # Linear interp.
        self.assertAlmostEqual(res[1], 5500)

        res = linear_int_ext(data, (None, 31622.7766016),  # Scaled linear.
                             scale=(None, 'log'))
        self.assertAlmostEqual(res[0], 8.5)

        with self.assertRaises(ValueError):
            linear_int_ext(data, (6, None))  # Out of bounds if no extrap.

        res = linear_int_ext(data, (None, 190000),
                             allow_extrap=True)  # High side extrap.
        self.assertAlmostEqual(res[0], 10)

        res = linear_int_ext(data, (6, None), scale=(None, 'log'),
                             allow_extrap=True)  # Low side extrap, log scale.
        self.assertAlmostEqual(res[1], 100)
